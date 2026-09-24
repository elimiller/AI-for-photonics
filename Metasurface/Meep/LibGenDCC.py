# %% Imports
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from matplotlib.patches import Rectangle
import time
import gdsfactory as gf
from collections.abc import Callable, Iterable
from gdsfactory.technology import LayerLevel, LayerStack
from gplugins.gmeep.get_meep_geometry import get_meep_geometry_from_component
from inspect import Parameter, signature
from itertools import product
import os
import sys
import re
from pathlib import Path
from datetime import datetime
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Metasurface.Unit_cell_generation import*
gf.gpdk.PDK.activate()
# %%  Get correct save directory
path = Path(__file__).parent.parent
print(path)
save_path = path / "Unit_cell_Libraries" /  'Ellipse Pillar SiN on SiO2 532 nm KAIST' 
print(save_path)
gds_lib_path = save_path / 'GDS Library'
data_lib_path = save_path / 'Library Data'

# %% Manual params
r_x_list = [0.25,0.3]
r_y_list = [0.25]
theta_list = [0,5]
wavelength = 0.532             # design wavelength [um]
period_list = [0.60]                 # square-lattice pitch [um]
pillar_h_list = [0.85]
incident_angle_list = [10]       # polar angle in degrees; tilt is in the x-z plane
RUN_SIMULATION = False
n_SiN = 2.0
n_sio2 = 1.46
resolution = 75               # pixels / um; increase after convergence test
dpml = 0.8                    # z-only absorbing boundary thickness [um]
air_padding = 1.0             # air above and below the structure [um]
substrate_h = 1.0 

# %% Gen GDS library
# gds_files = generate_unit_cell_gds_lib(
#     elliptical_pillar_gds,
#     gds_lib_path,
#     r_x=r_x_list,
#     r_y=r_y_list,
#     theta=theta_list,
# )


# %% Create sim
PILLAR_LAYER = (1, 0)
POLARIZATION = mp.Ex # TM or P polarized
fcen = 1 / wavelength
MEEP_PROGRESS_INTERVAL = 5.0
DECAY_LOG_CONTEXT = {}

# %% Logging, calculation, cost reductions, and other functions to be used
def log_decay_progress(running_sim: mp.Simulation) -> None:
    """Print field decay for the currently running sweep case."""
    field_magnitude = float(
        abs(running_sim.get_field_point(POLARIZATION, DECAY_LOG_CONTEXT["point"]))
    )
    DECAY_LOG_CONTEXT["peak_field"] = max(
        DECAY_LOG_CONTEXT["peak_field"], field_magnitude
    )
    peak_field = DECAY_LOG_CONTEXT["peak_field"]
    field_ratio = field_magnitude / peak_field if peak_field else 0.0
    if mp.am_master():
        print(
            f"{DECAY_LOG_CONTEXT['label']} decay t={running_sim.meep_time():.2f}, "
            f"wall={time.perf_counter() - DECAY_LOG_CONTEXT['started']:.1f}s, "
            f"field/peak={field_ratio:.3e}, squared={field_ratio**2:.3e}",
            flush=True,
        )


def diffraction_order(field: np.ndarray, order_x: int = 0, order_y: int = 0) -> complex:
    """Extract one spatial Fourier coefficient from a transmission-plane field."""
    spectrum = np.fft.fft2(field) / field.size
    index_x = (-order_x) % field.shape[0]
    index_y = (-order_y) % field.shape[1]
    return complex(spectrum[index_x, index_y])

def calculate_transmissions(
    reference_fields: dict[str, np.ndarray],
    pillar_fields: dict[str, np.ndarray],
    reference_flux: float,
    pillar_flux: float,
    incident_angle_deg: float = 0.0,
    period: float = period_list[0],
    order_x: int = 0,
    order_y: int = 0,
    medium_index: float = 1.0,
) -> dict[str, object]:
    """Store Cartesian responses and normalized TE/TM field coefficients.

    For p-polarized (TM) light in the x-z plane, the electric field has both
    x and z components at oblique incidence.  Projecting onto the TM unit
    polarization includes both components in the reported ``tm`` response.
    """
    reference_order = {
        component: diffraction_order(reference_fields[component], order_x, order_y)
        for component in ("ex", "ey", "ez")
    }
    pillar_order = {
        component: diffraction_order(pillar_fields[component], order_x, order_y)
        for component in ("ex", "ey", "ez")
    }
    component_coefficients = {
        component: pillar_order[component] / reference_order[component]
        for component in ("ex", "ey", "ez")
    }

    # Meep uses k_point in cycles/length, hence the 2*pi conversion here.
    kx = 2 * np.pi * (
        fcen * np.sin(np.deg2rad(incident_angle_deg)) + order_x / period
    )
    ky = 2 * np.pi * order_y / period
    kz_squared = (2 * np.pi * medium_index * fcen) ** 2 - kx**2 - ky**2
    if kz_squared <= 0:
        raise ValueError("Requested diffraction order is evanescent.")
    kz = np.sqrt(kz_squared)

    tm_vector = np.array([kz, 0.0, -kx], dtype=float)
    tm_vector /= np.linalg.norm(tm_vector)
    te_vector = np.array([-ky, kx, 0.0], dtype=float)
    te_vector /= np.linalg.norm(te_vector)
    tm_reference = np.dot(tm_vector, [reference_order["ex"], reference_order["ey"], reference_order["ez"]])
    tm_pillar = np.dot(tm_vector, [pillar_order["ex"], pillar_order["ey"], pillar_order["ez"]])
    te_reference = np.dot(te_vector, [reference_order["ex"], reference_order["ey"], reference_order["ez"]])
    te_pillar = np.dot(te_vector, [pillar_order["ex"], pillar_order["ey"], pillar_order["ez"]])
    coefficients = {
        "tm": tm_pillar / tm_reference,
        "te": te_pillar / te_reference if abs(te_reference) > 1e-14 else 0j,
    }

    return {
        **{
            component: {
                "real": float(coefficient.real),
                "imag": float(coefficient.imag),
            }
            for component, coefficient in component_coefficients.items()
        },
        **{
            polarization: {
                "real": float(coefficient.real),
                "imag": float(coefficient.imag),
            }
            for polarization, coefficient in coefficients.items()
        },
        "zero_order_power_estimate": float(
            sum(abs(coefficient) ** 2 for coefficient in coefficients.values())
        ),
        "total_transmitted_power_ratio": float(pillar_flux / reference_flux),
        "zero_order_tm_magnitude": float(abs(coefficients["tm"])),
        "zero_order_tm_phase_deg": float(
            np.degrees(np.angle(coefficients["tm"])) % 360
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


def Elim_Redundincies(r_x_list, r_y_list, theta_list):
    """Return unique centered-ellipse geometries from three parameter sweeps.

    The inputs are independent sweeps: every ``r_x``, ``r_y``, and ``theta``
    combination is considered.  The returned lists are *parallel* lists, so
    ``zip(unique_r_x, unique_r_y, unique_theta, strict=True)`` yields the
    unique configurations.  They must therefore not be passed back to
    ``generate_unit_cell_gds_lib`` as independent sweeps.

    Angles are in degrees.  The canonical representation uses ``r_x >= r_y``
    and ``0 <= theta < 180``.  It removes these exact geometric duplicates:

    * ``(r_x, r_y, theta) == (r_x, r_y, theta + 180)``;
    * ``(r_x, r_y, theta) == (r_y, r_x, theta + 90)``;
    * a circle has the canonical orientation ``theta = 0``.
    """
    unique_r_x = []
    unique_r_y = []
    unique_theta = []
    seen = set()

    for r_x, r_y, theta in product(r_x_list, r_y_list, theta_list):
        r_x = float(r_x)
        r_y = float(r_y)
        theta = float(theta)
        if not (np.isfinite(r_x) and np.isfinite(r_y) and np.isfinite(theta)):
            raise ValueError("Ellipse radii and rotation angles must be finite.")
        if r_x <= 0 or r_y <= 0:
            raise ValueError("Ellipse radii must be positive.")

        # An ellipse axis is unoriented, hence theta and theta + 180 deg are
        # the same geometry.  Normalize before applying the axis convention.
        theta = theta % 180.0
        if r_x < r_y:
            r_x, r_y = r_y, r_x
            theta = (theta + 90.0) % 180.0

        # All orientations of a circle describe the same geometry.
        if r_x == r_y:
            theta = 0.0

        configuration = (r_x, r_y, theta)
        if configuration in seen:
            continue
        seen.add(configuration)
        unique_r_x.append(r_x)
        unique_r_y.append(r_y)
        unique_theta.append(theta)

    return unique_r_x, unique_r_y, unique_theta
# %% Data saving functions
def save_ellipse_pillar_plot(
    pillar_fields: dict[str, np.ndarray],
    geometry: tuple[float, float, float],
    pillar_h: float,
    period: float,
    incident_angle: float,
    plot_directory: Path,
) -> None:
    """Save monitor fields and the DatalibraryGenEllipse y=0 |Ex| profile."""
    plot_directory.mkdir(exist_ok=True)
    r_x, r_y, theta = geometry
    name = (
        f"rx_{r_x:g}_ry_{r_y:g}_theta_{theta:g}_h_{pillar_h:g}_"
        f"period_{period:g}_angle_{incident_angle:g}"
    )
    cell_z = substrate_h + pillar_h + 2 * air_padding + 2 * dpml
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    component_labels = {"ex": "Ex", "ey": "Ey", "ez": "Ez"}

    for axis, (component, label) in zip(
        axes.flat[:3], component_labels.items(), strict=True
    ):
        image = axis.imshow(
            np.abs(pillar_fields[component]).T,
            origin="lower",
            extent=(-period / 2, period / 2, -period / 2, period / 2),
            cmap="magma",
        )
        axis.set(
            xlabel="x (um)",
            ylabel="y (um)",
            title=f"|{label}| at transmission monitor",
        )
        figure.colorbar(image, ax=axis, label=f"|{label}| (arbitrary units)")

    axis = axes[1, 1]
    image = axis.imshow(
        np.abs(pillar_fields["y0_ex"]).T,
        origin="lower",
        extent=(
            -period / 2,
            period / 2,
            -cell_z / 2 + dpml,
            cell_z / 2 - dpml,
        ),
        aspect="auto",
        cmap="magma",
    )
    axis.axhline(-substrate_h, color="cyan", linewidth=0.8)
    axis.axhline(0, color="cyan", linewidth=0.8)
    source_z = -0.5 * cell_z + dpml + 0.35
    monitor_z = 0.5 * cell_z - dpml - 0.35
    axis.axhline(source_z, color="white", linestyle="--", linewidth=0.8)
    axis.axhline(monitor_z, color="lime", linestyle="--", linewidth=0.8)
    axis.set(
        xlabel="x (um)",
        ylabel="z (um)",
        title=(
            f"|Ex| at {wavelength:.3f} um, "
            f"incident angle = {incident_angle:.1f} deg"
        ),
    )
    figure.colorbar(image, ax=axis, label="|Ex| (arbitrary units)")
    figure.tight_layout()
    figure.savefig(plot_directory / f"{name}_field_profile.png", dpi=200)
    plt.close(figure)

def write_ellipse_pillar_library(
    json_path: Path, csv_path: Path, simulations: list[dict[str, object]]
) -> None:
    """Write the simulation library as matching JSON and CSV tables."""
    library = {
        "metadata": {
            "wavelength_um": wavelength,
            "length_unit": "um",
            "field_definitions": ELLIPSE_LIBRARY_FIELD_DEFINITIONS,
        },
        "simulations": simulations,
    }
    json_temp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    csv_temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    json_temp_path.write_text(json.dumps(library, indent=2))
    json_temp_path.replace(json_path)
    with csv_temp_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ELLIPSE_LIBRARY_COLUMNS)
        writer.writeheader()
        writer.writerows(simulations)
    csv_temp_path.replace(csv_path)



# %% Run an individual sim

def run_unit_cell(
    extra_geometry: list[mp.GeometricObject],
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
    theta_deg: float = 0.0,
    plot_field_profile: bool = False,
    simulation_label: str = "",
) -> dict[str, np.ndarray]:
    """Return transmission-plane fields and, when requested, a y=0 field slice."""
    cell_z = substrate_h + pillar_h + 2 * air_padding + 2 * dpml
    cell = mp.Vector3(period, period, cell_z)
    monitor_z = 0.5 * cell_z - dpml - 0.35
    source_z = -0.5 * cell_z + dpml + 0.35
    incident_angle_rad = np.deg2rad(incident_angle_deg)

    # Meep's k_point is in inverse-layout units.  The incident medium is air,
    # so |k| = fcen and kx = fcen * sin(theta).
    k_point = mp.Vector3(fcen * np.sin(incident_angle_rad), 0, 0)

    # With an unrotated ellipse, the x-z plane remains a symmetry plane for
    # all x-z incidence. The y-z plane is additionally valid at normal
    # incidence. For an Ex source, the source is even under y -> -y and odd
    # under x -> -x.
    symmetries = []
    if theta_deg == 0:
        symmetries.append(mp.Mirror(mp.Y, phase=+1))
        if incident_angle_deg == 0:
            symmetries.append(mp.Mirror(mp.X, phase=-1))

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
        symmetries=symmetries,
        resolution=resolution,
        default_material=mp.air,
    )
    transmission_plane = mp.Volume(
        center=mp.Vector3(0, 0, monitor_z),
        size=mp.Vector3(period, period, 0),
    )
    dft = sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez], fcen, 0, 1, where=transmission_plane)
    transmission_flux = sim.add_flux(
        fcen,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0, 0, monitor_z),
            size=mp.Vector3(period, period, 0),
            direction=mp.Z,
        ),
    )
    field_profile_dft = None
    if plot_field_profile:
        field_profile_dft = sim.add_dft_fields(
            [mp.Ex],
            fcen,
            0,
            1,
            where=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(period, 0, cell_z - 2 * dpml),
            ),
        )

    DECAY_LOG_CONTEXT.update(
        {
            "label": simulation_label,
            "point": mp.Vector3(0, 0, monitor_z),
            "peak_field": 0.0,
            "started": time.perf_counter(),
        }
    )
    sim.run(
        mp.at_every(MEEP_PROGRESS_INTERVAL, log_decay_progress),
        until_after_sources=mp.stop_when_fields_decayed(
            50, POLARIZATION, mp.Vector3(0, 0, monitor_z), 1e-3
        )
    )
    ex = sim.get_dft_array(dft, mp.Ex, 0)
    ey = sim.get_dft_array(dft, mp.Ey, 0)
    ez = sim.get_dft_array(dft, mp.Ez, 0)
    fields = {
        'ex' : ex,
        'ey' : ey,
        'ez' : ez,
        'transmitted_flux': float(mp.get_fluxes(transmission_flux)[0]),
    }
    if field_profile_dft is not None:
        fields["y0_ex"] = np.squeeze(
            sim.get_dft_array(field_profile_dft, mp.Ex, 0)
        )
    return fields

def run_reference_unit_cell(
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
    simulation_label: str = "",
) -> dict[str, complex | float]:
    """Run the bare-substrate reference for one height/period/angle condition."""
    return run_unit_cell(
        [], pillar_h, period, incident_angle_deg, simulation_label=simulation_label
    )


 
# %% Upload GDS to sim

def simulate_gds_unit_cell(
    file_path: Path,
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
    reference_field: dict[str, complex | float],
    plot_field_profile: bool = False,
    theta_deg: float = 0.0,
    simulation_label: str = "",
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
        theta_deg,
        plot_field_profile=plot_field_profile,
        simulation_label=simulation_label,
    )
    transmission_and_phase = calculate_transmissions(
        reference_field,
        pillar_field,
        reference_field["transmitted_flux"],
        pillar_field["transmitted_flux"],
        incident_angle_deg,
        period=period,
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
# %% Loop through sim function
ELLIPSE_LIBRARY_COLUMNS = [
    "radius_x_um", "radius_y_um", "rotation_deg", "pillar_height_um",
    "period_um", "incident_angle_deg", "gds_path", "ex_real", "ex_imag",
    "ey_real", "ey_imag", "ez_real", "ez_imag", "tm_real", "tm_imag",
    "te_real", "te_imag", "zero_order_power_estimate",
    "total_transmitted_power_ratio", "reference_transmission_plane_flux",
    "pillar_transmission_plane_flux", "zero_order_tm_magnitude",
    "zero_order_tm_phase_deg", "completed_at",
]

ELLIPSE_LIBRARY_FIELD_DEFINITIONS = {
    "zero_order_power_estimate": (
        "abs(tm)^2 + abs(te)^2 from normalized zero-order field coefficients; "
        "not an integrated flux measurement"
    ),
    "total_transmitted_power_ratio": (
        "pillar transmission-plane flux divided by bare-reference "
        "transmission-plane flux"
    ),
    "reference_transmission_plane_flux": (
        "raw integrated z-directed Poynting flux for the bare reference"
    ),
    "pillar_transmission_plane_flux": (
        "raw integrated z-directed Poynting flux for the pillar simulation"
    ),
    "zero_order_tm_magnitude": "magnitude of the normalized zero-order TM coefficient",
    "zero_order_tm_phase_deg": (
        "phase of the normalized zero-order TM coefficient, in degrees from 0 to 360"
    ),
}

ELLIPSE_LIBRARY_LEGACY_FIELDS = {
    "power_flux_transmission": "zero_order_power_estimate",
    "total_power_transmission": "total_transmitted_power_ratio",
    "reference_transmitted_flux": "reference_transmission_plane_flux",
    "pillar_transmitted_flux": "pillar_transmission_plane_flux",
    "transmission_magnitude": "zero_order_tm_magnitude",
    "transmission_phase_deg": "zero_order_tm_phase_deg",
}


def ellipse_pillar_sweeps(
    r_x_list,
    r_y_list,
    theta_list,
    pillar_h_list,
    period_list,
    incident_angle_list,
    gds_lib_path,
    data_lib_path,
) -> list[dict[str, object]]:
    """Run only new ellipse-pillar simulations from the supplied sweeps."""
    unique_r_x, unique_r_y, unique_theta = Elim_Redundincies(
        r_x_list, r_y_list, theta_list
    )
    geometries = list(zip(unique_r_x, unique_r_y, unique_theta, strict=True))
    pillar_h_list = list(dict.fromkeys(map(float, pillar_h_list)))
    period_list = list(dict.fromkeys(map(float, period_list)))
    incident_angle_list = list(dict.fromkeys(map(float, incident_angle_list)))

    json_path = data_lib_path / "ellipse_pillar_library_data.json"
    csv_path = data_lib_path / "ellipse_pillar_library_data.csv"
    plot_directory = data_lib_path / "Simulation Plots"
    plot_directory.mkdir(exist_ok=True)
    simulations = (
        json.loads(json_path.read_text())["simulations"]
        if json_path.exists()
        else []
    )
    for record in simulations:
        for old_name, new_name in ELLIPSE_LIBRARY_LEGACY_FIELDS.items():
            if old_name in record:
                record[new_name] = record.pop(old_name)
    completed = {
        (
            record["radius_x_um"], record["radius_y_um"], record["rotation_deg"],
            record["pillar_height_um"], record["period_um"],
            record["incident_angle_deg"],
        )
        for record in simulations
    }

    # ``Elim_Redundincies`` returns parallel lists, so export one geometry at
    # a time. Passing the full lists would recreate their Cartesian product.
    gds_files = {
        geometry: generate_unit_cell_gds_lib(
            elliptical_pillar_gds,
            gds_lib_path,
            r_x=[geometry[0]],
            r_y=[geometry[1]],
            theta=[geometry[2]],
        )[0]
        for geometry in geometries
    }
    reference_fields = {}
    conditions = list(product(pillar_h_list, period_list, incident_angle_list))
    total_simulations = len(geometries) * len(conditions)
    sweep_started = time.perf_counter()

    for index, (geometry, condition) in enumerate(
        product(geometries, conditions), start=1
    ):
        pillar_h, period, incident_angle = condition
        key = (*geometry, pillar_h, period, incident_angle)
        simulation_label = f"[{index}/{total_simulations}]"
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        if key in completed:
            print(
                f"[{index}/{total_simulations}] [{timestamp}] skipped {key}",
                flush=True,
            )
            continue

        print(f"{simulation_label} running {key}", flush=True)
        reference_key = (pillar_h, period, incident_angle)
        if reference_key not in reference_fields:
            reference_fields[reference_key] = run_reference_unit_cell(
                pillar_h, period, incident_angle, simulation_label
            )
        simulation = simulate_gds_unit_cell(
            gds_files[geometry],
            pillar_h,
            period,
            incident_angle,
            reference_fields[reference_key],
            plot_field_profile=True,
            theta_deg=geometry[2],
            simulation_label=simulation_label,
        )
        response = simulation["normalized_response"]
        save_ellipse_pillar_plot(
            simulation["pillar_transmission_plane_fields"],
            geometry,
            pillar_h,
            period,
            incident_angle,
            plot_directory,
        )
        record = {
            "radius_x_um": geometry[0],
            "radius_y_um": geometry[1],
            "rotation_deg": geometry[2],
            "pillar_height_um": pillar_h,
            "period_um": period,
            "incident_angle_deg": incident_angle,
            "gds_path": str(gds_files[geometry]),
            "zero_order_power_estimate": response["zero_order_power_estimate"],
            "total_transmitted_power_ratio": response[
                "total_transmitted_power_ratio"
            ],
            "reference_transmission_plane_flux": reference_fields[reference_key][
                "transmitted_flux"
            ],
            "pillar_transmission_plane_flux": simulation[
                "pillar_transmission_plane_fields"
            ]["transmitted_flux"],
            "zero_order_tm_magnitude": response["zero_order_tm_magnitude"],
            "zero_order_tm_phase_deg": response["zero_order_tm_phase_deg"],
            "completed_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        }
        for component in ("ex", "ey", "ez", "tm", "te"):
            record[f"{component}_real"] = response[component]["real"]
            record[f"{component}_imag"] = response[component]["imag"]
        simulations.append(record)
        completed.add(key)
        write_ellipse_pillar_library(json_path, csv_path, simulations)
        elapsed = time.perf_counter() - sweep_started
        print(
            f"[{index}/{total_simulations}] [{record['completed_at']}] completed; "
            f"sweep wall={elapsed:.1f}s",
            flush=True,
        )

    write_ellipse_pillar_library(json_path, csv_path, simulations)
    return simulations

# %% Initial Notebook test
ellipse_pillar_sweeps(r_x_list,r_y_list,theta_list,pillar_h_list,period_list,incident_angle_list,gds_lib_path,data_lib_path)
# %% Data processing before running sim
