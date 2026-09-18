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
from pathlib import Path

from Unit_cell_generation import*
# %% Find correct directory
path = Path(__file__).parent
print(path)
gds_directory = path / "GDS libraries" / "Elliptical_focus_library"
print(gds_directory)
# %% Manual params
r_x_list = [0.25]
r_y_list = [0.25]
theta_list = [0]
wavelength = 0.532             # design wavelength [um]
period_list = [0.60]                 # square-lattice pitch [um]
pillar_h_list = [0.85]
incident_angle_list = [0.0]          # polar angle in degrees; tilt is in the x-z plane
RUN_SIMULATION = False
n_SiN = 2.0
n_sio2 = 1.46
resolution = 50               # pixels / um; increase after convergence test
dpml = 0.8                    # z-only absorbing boundary thickness [um]
air_padding = 1.0             # air above and below the structure [um]
substrate_h = 1.0 
# %% Generate GDS library 
gds_files = generate_unit_cell_gds_lib(
    elliptical_pillar_gds,
    gds_directory,
    r_x=r_x_list,
    r_y=r_y_list,
    theta=theta_list,
)

# %% Create and run simulations
PILLAR_LAYER = (1, 0)
POLARIZATION = mp.Ey  # TE (s-polarized) for an incident beam in the x-z plane
fcen = 1 / wavelength

# %% Individual unit cell sim 
def run_unit_cell(
    extra_geometry: list[mp.GeometricObject],
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
) -> complex:
    """Return the mean complex TE field at the transmission plane."""
    cell_z = substrate_h + pillar_h + 2 * air_padding + 2 * dpml
    cell = mp.Vector3(period, period, cell_z)
    source_z = 0.5 * cell_z - dpml - 0.35
    monitor_z = -0.5 * cell_z + dpml + 0.35
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
    dft = sim.add_dft_fields([POLARIZATION], fcen, 0, 1, where=transmission_plane)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, POLARIZATION, mp.Vector3(0, 0, monitor_z), 1e-3
        )
    )
    return np.mean(sim.get_dft_array(dft, POLARIZATION, 0))


def run_reference_unit_cell(
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
) -> complex:
    """Run the bare-substrate reference for one height/period/angle condition."""
    return run_unit_cell([], pillar_h, period, incident_angle_deg)


def simulate_gds_unit_cell(
    file_path: Path,
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
    reference_field: complex,
) -> dict[str, object]:
    """Simulate one pillar GDS using an already-computed bare reference."""
    unit_cell = gf.import_gds(gdspath=file_path)
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
        pillar_geometry, pillar_h, period, incident_angle_deg
    )
    return {
        "gds_path": file_path,
        "pillar_h": pillar_h,
        "period": period,
        "incident_angle_deg": incident_angle_deg,
        "reference_field": reference_field,
        "pillar_field": pillar_field,
    }


def calculate_transmission_and_phase(
    reference_field: complex, pillar_field: complex
) -> dict[str, float]:
    """Calculate normalized power transmission and phase from two DFT fields."""
    transmission_coefficient = pillar_field / reference_field
    return {
        "power_transmission": float(abs(transmission_coefficient) ** 2),
        "phase_deg": float(
            np.degrees(np.angle(transmission_coefficient) % (2 * np.pi))
        ),
    }


def sweep_unit_cell_simulations(
    files: Iterable[Path],
    pillar_heights: Iterable[float],
    periods: Iterable[float],
    incident_angles_deg: Iterable[float],
) -> list[dict[str, object]]:
    """Run shared bare references, then simulate every GDS at each condition."""
    conditions = list(
        product(pillar_heights, periods, incident_angles_deg)
    )
    reference_fields: dict[tuple[float, float, float], complex] = {}
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
            )
        )
    return results


if RUN_SIMULATION:
    simulation_results = sweep_unit_cell_simulations(
        gds_files,
        pillar_h_list,
        period_list,
        incident_angle_list,
    )
