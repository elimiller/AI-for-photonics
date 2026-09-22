# %% Imports
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
from functools import partial
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

# %% Gen GDS library
gds_files = generate_unit_cell_gds_lib(
    elliptical_pillar_gds,
    gds_lib_path,
    r_x=r_x_list,
    r_y=r_y_list,
    theta=theta_list,
)


# %% Create sim
PILLAR_LAYER = (1, 0)
POLARIZATION = mp.Ex # TM or P polarized
fcen = 1 / wavelength
MEEP_PROGRESS_INTERVAL = 5.0  # simulation-time units between progress messages

# %% Fucntion definitions
def log_decay_progress(
    running_sim: mp.Simulation,
    monitor_point: mp.Vector3,
    progress_state: dict,
    wall_clock_start: float,
) -> None:
    field_magnitude = float(
        abs(running_sim.get_field_point(POLARIZATION, monitor_point))
    )
    progress_state["peak_field"] = max(
        progress_state["peak_field"], field_magnitude
    )
    peak_field = progress_state["peak_field"]
    relative_field = field_magnitude / peak_field if peak_field else 0.0
    if mp.am_master():
        print(
            f"[Meep decay] t={running_sim.meep_time():.2f}, "
            f"wall={time.perf_counter() - wall_clock_start:.1f}s, "
            f"field/peak={relative_field:.3e}, threshold=1.000e-03",
            flush=True,
        )
# %% Run an individual sim

def run_unit_cell(
    extra_geometry: list[mp.GeometricObject],
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
) -> complex:
    """Return the mean complex TE field at the transmission plane."""
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
    dft = sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez], fcen, 0, 1, where=transmission_plane)

    progress_state = {"peak_field": 0.0}
    wall_clock_start = time.perf_counter()
    monitor_point = mp.Vector3(0, 0, monitor_z)

    sim.run(
        mp.at_every(
            MEEP_PROGRESS_INTERVAL,
            partial(
                log_decay_progress,
                monitor_point=monitor_point,
                progress_state=progress_state,
                wall_clock_start=wall_clock_start,
            ),
        ),
        until_after_sources=mp.stop_when_fields_decayed(
            50, POLARIZATION, mp.Vector3(0, 0, monitor_z), 1e-3
        )
    )
    ex = np.mean(sim.get_dft_array(dft, mp.Ex, 0))
    ey = np.mean(sim.get_dft_array(dft,mp.Ey,0))
    ez = np.mean(sim.get_dft_array(dft,mp.Ez,0))
    return {
        'ex' : ex,
        'ey' : ey,
        'ez' : ez
    }

def run_reference_unit_cell(
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
) -> dict[str, complex | float]:
    """Run the bare-substrate reference for one height/period/angle condition."""
    return run_unit_cell([], pillar_h, period, incident_angle_deg)

# %% Upload GDS to sim
def calculate_transmissions(
    reference_fields: dict[str, complex],
    pillar_fields: dict[str, complex],
) -> dict[str, object]:
    """Store real/imaginary field responses and normalized power transmission."""
    coefficients = {
        component: pillar_fields[component] / reference_fields[component]
        for component in ("ex", "ey", "ez")
    }
    return {
        **{
            component: {
                "real": float(coefficient.real),
                "imag": float(coefficient.imag),
            }
            for component, coefficient in coefficients.items()
        },
        "tm": {
            "real": float(coefficients["ex"].real),
            "imag": float(coefficients["ex"].imag),
        },
        "te": {
            "real": float(coefficients["ey"].real),
            "imag": float(coefficients["ey"].imag),
        },
        "power_flux_transmission": float(
            sum(abs(coefficient) ** 2 for coefficient in coefficients.values())
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
 
def simulate_gds_unit_cell(
    file_path: Path,
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
    reference_field: dict[str, complex | float],
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
    transmission_and_phase = calculate_transmissions(
        reference_field,
        pillar_field,
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
