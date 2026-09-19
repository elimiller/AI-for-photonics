# %% imports
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import gdsfactory as gf
from collections.abc import Callable, Iterable
from gdsfactory.technology import LayerLevel, LayerStack
from gplugins.gmeep.get_meep_geometry import get_meep_geometry_from_component
from inspect import Parameter, signature
from itertools import product
import os
import json
import re
from pathlib import Path

from Unit_cell_generation import*
gf.gpdk.PDK.activate()
# %% Find correct directory
path = Path(__file__).parent
print(path)
save_path = path / "Unit_cell_Libraries" /  'Ellipse Pillar SiN on SiO2 532 nm KAIST' 
print(save_path)
gds_lib_path = save_path / 'GDS Library'
data_lib_path = save_path / 'Library Data'

# %% Manual params
r_x_list = [0.25,0.3]
r_y_list = [0.25]
theta_list = [0]
wavelength = 0.532             # design wavelength [um]
period_list = [0.60]                 # square-lattice pitch [um]
pillar_h_list = [0.85]
incident_angle_list = [10]      # polar angle in degrees; tilt is in the x-z plane    
RUN_SIMULATION = True
PLOT_FIRST_PILLAR_FIELD_PROFILE = True  # Set False for library sweeps without plots.
n_SiN = 2.0
n_sio2 = 1.46
resolution = 50               # pixels / um; increase after convergence test
dpml = 0.8                  # z-only absorbing boundary thickness [um]
air_padding = 1.0             # air above and below the structure [um]
substrate_h = 1.0 
# %% Generate GDS library 
gds_files = generate_unit_cell_gds_lib(
    elliptical_pillar_gds,
    gds_lib_path,
    r_x=r_x_list,
    r_y=r_y_list,
    theta=theta_list,
)

# %% Create and run simulations
PILLAR_LAYER = (1, 0)
POLARIZATION = mp.Ex  
fcen = 1 / wavelength

# %% Individual unit cell sim 
def run_unit_cell(
    extra_geometry: list[mp.GeometricObject],
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
    plot_field_profile: bool = False,
) -> dict[str, complex]:
    """Return the mean complex p-polarized fields at the transmission plane."""
    cell_z = substrate_h + pillar_h + 2 * air_padding + 2 * dpml
    cell = mp.Vector3(period, period, cell_z)
    monitor_z = 0.5 * cell_z - dpml - 0.35
    source_z = -0.5 * cell_z + dpml + 0.35
    incident_angle_rad = np.deg2rad(incident_angle_deg)

    # Meep's k_point is in inverse-layout units.  The incident medium is air,
    # so |k| = fcen and kx = fcen * sin(theta).
    k_point = mp.Vector3(fcen * np.sin(incident_angle_rad), 0, 0)

    def bloch_phase(position: mp.Vector3) -> complex:
        """Apply the x-dependent phase of the oblique Bloch plane wave."""
        return np.exp(2j * np.pi * k_point.x * position.x)

    substrate = mp.Block(
        size=mp.Vector3(period, period, substrate_h),
        center=mp.Vector3(0, 0, -substrate_h / 2),
        material=mp.Medium(index=n_sio2),
    )
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(dpml, direction=mp.Z)],
        geometry=[substrate, *extra_geometry],
        sources=[
            mp.Source(
                mp.GaussianSource(fcen, fwidth=0.05 * fcen),
                component=POLARIZATION,
                center=mp.Vector3(0, 0, source_z),
                size=mp.Vector3(period, period, 0),
                amp_func=bloch_phase,
            )
        ],
        k_point=k_point,
        resolution=resolution,
        default_material=mp.air,
    )
    transmission_plane = mp.Volume(
        center=mp.Vector3(0, 0, monitor_z),
        size=mp.Vector3(period, period, 0),
    )
    # Ex is the p-polarized source current.  Ez is monitored to confirm that
    # the resulting field is transverse to the oblique wavevector.
    dft = sim.add_dft_fields([mp.Ex, mp.Ez], fcen, 0, 1, where=transmission_plane)
    field_profile_dft = None
    if plot_field_profile:
        field_profile_plane = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(period, 0, cell_z - 2 * dpml),
        )
        field_profile_dft = sim.add_dft_fields(
            [mp.Ex], fcen, 0, 1, where=field_profile_plane
        )
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, POLARIZATION, mp.Vector3(0, 0, monitor_z), 1e-3
        )
    )
    if field_profile_dft is not None:
        ex_profile = np.squeeze(
            sim.get_dft_array(field_profile_dft, mp.Ex, 0)
        )
        z_min = -0.5 * cell_z + dpml
        z_max = 0.5 * cell_z - dpml
        fig, axis = plt.subplots(figsize=(7, 5))
        image = axis.imshow(
            np.abs(ex_profile).T,
            origin="lower",
            extent=(-0.5 * period, 0.5 * period, z_min, z_max),
            aspect="auto",
            cmap="magma",
        )
        axis.axhline(-substrate_h, color="cyan", linewidth=0.8)
        axis.axhline(0, color="cyan", linewidth=0.8)
        axis.axhline(source_z, color="white", linestyle="--", linewidth=0.8)
        axis.axhline(monitor_z, color="lime", linestyle="--", linewidth=0.8)
        axis.set(
            xlabel="x (um)",
            ylabel="z (um)",
            title=(
                f"|Ex| at {wavelength:.3f} um, "
                f"incident angle = {incident_angle_deg:.1f} deg"
            ),
        )
        fig.colorbar(image, ax=axis, label="|Ex| (arbitrary units)")
        fig.tight_layout()
        plt.show()
    return {
        "ex": complex(np.mean(sim.get_dft_array(dft, mp.Ex, 0))),
        "ez": complex(np.mean(sim.get_dft_array(dft, mp.Ez, 0))),
    }


def run_reference_unit_cell(
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
) -> dict[str, complex]:
    """Run the bare-substrate reference for one height/period/angle condition."""
    return run_unit_cell([], pillar_h, period, incident_angle_deg)


def simulate_gds_unit_cell(
    file_path: Path,
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
    reference_field: dict[str, complex],
    plot_field_profile: bool = False,
) -> dict[str, object]:
    """Simulate one pillar GDS using an already-computed bare reference."""
    # The generator has already created a cell with this GDS top-cell name.
    # Rename the temporary imported copy rather than treating it as a conflict.
    unit_cell = gf.import_gds(
        gdspath=file_path, rename_duplicated_cells=True
    )
    layer_stack = LayerStack(
        layers={
            "si_n_pillar": LayerLevel(
                layer=PILLAR_LAYER,
                thickness=pillar_h,
                zmin=0.0,
                material="si_n",
            )
        }
    )
    pillar_geometry = get_meep_geometry_from_component(
        component=unit_cell,
        layer_stack=layer_stack,
        material_name_to_meep={"si_n": n_SiN},
        wavelength=wavelength,
    )
    pillar_field = run_unit_cell(
        pillar_geometry,
        pillar_h,
        period,
        incident_angle_deg,
        plot_field_profile=plot_field_profile,
    )
    transmission_and_phase = calculate_transmission_and_phase(
        reference_field["ex"], pillar_field["ex"]
    )
    return {
        "geometry": ellipse_geometry_metadata(file_path),
        "simulation_condition": {
            "pillar_height_um": pillar_h,
            "period_um": period,
            "incident_angle_deg": incident_angle_deg,
        },
        "reference_transmission_plane_fields": reference_field,
        "pillar_transmission_plane_fields": pillar_field,
        "normalized_response": transmission_and_phase,
    }


def calculate_transmission_and_phase(
    reference_field: complex, pillar_field: complex
) -> dict[str, complex | float]:
    """Calculate normalized power transmission and phase from two DFT fields."""
    transmission_coefficient = pillar_field / reference_field
    return {
        "complex_transmission_coefficient": transmission_coefficient,
        "power_transmission": float(abs(transmission_coefficient) ** 2),
        "phase_deg": float(
            np.degrees(np.angle(transmission_coefficient) % (2 * np.pi))
        ),
    }


def ellipse_geometry_metadata(file_path: Path) -> dict[str, object]:
    """Return labeled ellipse parameters encoded in a generated GDS filename."""
    match = re.fullmatch(
        r"elliptical_pillar_rx_(.+)_ry_(.+)_theta_(.+)", file_path.stem
    )
    metadata: dict[str, object] = {
        "geometry_type": "elliptical_pillar",
        "gds_path": file_path,
    }
    if match is not None:
        radius_x, radius_y, rotation_deg = match.groups()
        metadata.update(
            {
                "radius_x_um": float(radius_x),
                "radius_y_um": float(radius_y),
                "rotation_deg": float(rotation_deg),
            }
        )
    return metadata


def p_polarization_sanity_check(
    fields: dict[str, complex], incident_angle_deg: float
) -> dict[str, complex | float]:
    """Compare the reference-field ratio to Ez / Ex = -tan(incident angle)."""
    expected_ratio = -np.tan(np.deg2rad(incident_angle_deg))
    measured_ratio = fields["ez"] / fields["ex"]
    return {
        "expected_ez_over_ex": float(expected_ratio),
        "measured_ez_over_ex": measured_ratio,
        "absolute_error": float(abs(measured_ratio - expected_ratio)),
    }


def json_value(value: object) -> object:
    """Convert simulation values, including complex DFT fields, to JSON data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def simulation_condition_key(record: dict[str, object]) -> tuple[object, ...]:
    """Return the unique key for one height, period, and incidence condition."""
    condition = record["simulation_condition"]
    if not isinstance(condition, dict):
        raise ValueError("Simulation record is missing its simulation_condition.")
    return (
        condition["pillar_height_um"],
        condition["period_um"],
        condition["incident_angle_deg"],
    )


def result_key(record: dict[str, object]) -> tuple[object, ...]:
    """Return the unique key for one geometry and simulation condition."""
    geometry = record["geometry"]
    if not isinstance(geometry, dict):
        raise ValueError("Simulation result is missing its geometry metadata.")
    return (
        geometry.get("geometry_type"),
        geometry.get("radius_x_um"),
        geometry.get("radius_y_um"),
        geometry.get("rotation_deg"),
        *simulation_condition_key(record),
    )


def upsert_records(
    existing_records: list[dict[str, object]],
    new_records: list[dict[str, object]],
    key_function: Callable[[dict[str, object]], tuple[object, ...]],
) -> list[dict[str, object]]:
    """Replace records with matching keys and retain unrelated library records."""
    records_by_key = {key_function(record): record for record in existing_records}
    for record in new_records:
        records_by_key[key_function(record)] = record
    return list(records_by_key.values())


def sweep_unit_cell_simulations(
    files: Iterable[Path],
    save_path: Path,
    pillar_heights: Iterable[float],
    periods: Iterable[float],
    incident_angles_deg: Iterable[float],
) -> list[dict[str, object]]:
    """Run shared bare references, then simulate every GDS at each condition."""
    conditions = list(dict.fromkeys(product(
        pillar_heights, periods, incident_angles_deg
    )))
    reference_fields: dict[tuple[float, float, float], dict[str, complex]] = {}
    for index, (pillar_h, period, incident_angle_deg) in enumerate(
        conditions, start=1
    ):
        print(
            f"[reference {index}/{len(conditions)}] no pillar: "
            f"height={pillar_h} um, period={period} um, "
            f"incident angle={incident_angle_deg} deg",
            flush=True,
        )
        reference_fields[(pillar_h, period, incident_angle_deg)] = (
            run_reference_unit_cell(pillar_h, period, incident_angle_deg)
        )

    combinations = list(product(files, conditions))
    results = []
    for index, (file_path, condition) in enumerate(
        combinations, start=1
    ):
        pillar_h, period, incident_angle_deg = condition
        print(
            f"[{index}/{len(combinations)}] {file_path.name}: "
            f"height={pillar_h} um, period={period} um, "
            f"incident angle={incident_angle_deg} deg",
            flush=True,
        )
        results.append(
            simulate_gds_unit_cell(
                file_path,
                pillar_h,
                period,
                incident_angle_deg,
                reference_fields[condition],
                plot_field_profile=(
                    PLOT_FIRST_PILLAR_FIELD_PROFILE and index == 1
                ),
            )
        )
        
    reference_results = [
        {
            "simulation_condition": {
                "pillar_height_um": pillar_h,
                "period_um": period,
                "incident_angle_deg": incident_angle_deg,
            },
            "transmission_plane_fields": reference_fields[
                (pillar_h, period, incident_angle_deg)
            ],
            "p_polarization_sanity_check": p_polarization_sanity_check(
                reference_fields[(pillar_h, period, incident_angle_deg)],
                incident_angle_deg,
            ),
        }
        for pillar_h, period, incident_angle_deg in conditions
    ]
    library_data = {
        "metadata": {
            "schema_version": 1,
            "wavelength_um": wavelength,
            "length_unit": "um",
            "source_current_component": "Ex",
            "polarization": "p/TM",
            "incident_plane": "x-z",
            "transmission_component": "Ex",
            "complex_value_format": {"real": "float", "imag": "float"},
        },
        "references": reference_results,
        "results": results,
    }
    if not save_path.is_dir():
        raise NotADirectoryError(
            f"Library data directory does not exist: {save_path}"
        )
    output_file = save_path / "ellipse_pillar_library_data.json"
    new_library_data = json_value(library_data)
    if not isinstance(new_library_data, dict):
        raise TypeError("Library data must serialize to a JSON object.")
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as file:
            existing_library_data = json.load(file)
        if not isinstance(existing_library_data, dict):
            raise ValueError(f"Existing library data is invalid: {output_file}")
        existing_references = existing_library_data.get("references", [])
        existing_results = existing_library_data.get("results", [])
        if not all(isinstance(record, dict) for record in existing_references):
            raise ValueError("Existing references must be JSON objects.")
        if not all(isinstance(record, dict) for record in existing_results):
            raise ValueError("Existing results must be JSON objects.")
        new_library_data["references"] = upsert_records(
            existing_references,
            new_library_data["references"],
            simulation_condition_key,
        )
        new_library_data["results"] = upsert_records(
            existing_results,
            new_library_data["results"],
            result_key,
        )
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(new_library_data, file, indent=2)
    print(f"Saved library data to {output_file}", flush=True)

    return results



# %% Run Sim
if RUN_SIMULATION:
    simulation_results = sweep_unit_cell_simulations(
        gds_files,
        data_lib_path,
        pillar_h_list,
        period_list,
        incident_angle_list,
    )
# %% Plot field profile
