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
import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from Unit_cell_generation import*
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
resolution = 75               # pixels / um; increase after convergence test
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

# %% Logging, calculation, cost reductions, and other functions to be used
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

def diffraction_order(field: np.ndarray, order_x: int = 0, order_y: int = 0) -> complex:
    """Extract one spatial Fourier coefficient from a transmission-plane field."""
    spectrum = np.fft.fft2(field) / field.size
    index_x = (-order_x) % field.shape[0]
    index_y = (-order_y) % field.shape[1]
    return complex(spectrum[index_x, index_y])

def calculate_transmissions(
    reference_fields: dict[str, np.ndarray],
    pillar_fields: dict[str, np.ndarray],
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

# Test
r_x_list = [0.10, 0.20]
r_y_list = [0.10, 0.20]
theta_list = [0, 30, 90, 120, 180]
rx_unique, ry_unique, theta_unique = Elim_Redundincies(
    r_x_list, r_y_list, theta_list
)
# %% Run an individual sim

def run_unit_cell(
    extra_geometry: list[mp.GeometricObject],
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
    theta_deg: float = 0.0,
    plot_field_profile: bool = False,
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
    ex = sim.get_dft_array(dft, mp.Ex, 0)
    ey = sim.get_dft_array(dft, mp.Ey, 0)
    ez = sim.get_dft_array(dft, mp.Ez, 0)
    return {
        'ex' : ex,
        'ey' : ey,
        'ez' : ez
    }

def run_reference_unit_cell(
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
    theta_deg: float = 0.0,
) -> dict[str, complex | float]:
    """Run the bare-substrate reference for one height/period/angle condition."""
    return run_unit_cell([], pillar_h, period, incident_angle_deg, theta_deg)


 
# %% Upload GDS to sim

def simulate_gds_unit_cell(
    file_path: Path,
    pillar_h: float,
    period: float,
    incident_angle_deg: float,
    reference_field: dict[str, complex | float],
    plot_field_profile: bool = False,
    theta_deg: float = 0.0,
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
    )
    transmission_and_phase = calculate_transmissions(
        reference_field,
        pillar_field,
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
# %% Handle input lists
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

# Test
r_x_list = [0.10, 0.20]
r_y_list = [0.10, 0.20]
theta_list = [0, 30, 90, 120, 180]
rx_unique, ry_unique, theta_unique = Elim_Redundincies(
    r_x_list, r_y_list, theta_list
)

# Conventional spelling for new callers; retain the original function name
# because it may already be used by notebooks or scripts in this project.

