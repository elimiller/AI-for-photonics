# Trying to make everything as similar to 
# test sim of ellipse cell in meep,, a lot of script was developed by codex
# %% imports
# %% imports and set up timetag for sweep|
import sys
from datetime import datetime
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pylab as plt
import os
import json
import gdstk as gd
from pathlib import Path
import tidy3d as td
import tidy3d.web as web
from scipy.optimize import fsolve
# Import regular tidy3d.
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.web import run as run_mode_solver
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tidy3d as td
from tidy3d.plugins.mode.web import run as run_mode_solver
from tidy3d.plugins.dispersion import AdvancedFastFitterParam, FastDispersionFitter
import gdsfactory as gf
# %% Functions
def stupidFunction(
    gf_comp: gf.Component,
    temp_path: str,
    gf_name: str,
    cell_name: str,
    dx: float = 0,
    dy: float = 0,
    filter_layer: bool = False,
):
    """
    Function takes a gdsfactory component and converts it to a gdstk cell with option for translation.
    This is somewhat dumb, but needed since tidy3d only takes gdstk cells for import.
    Returns a gdstk cell.
    """

    # NOW DO THE NONSENSE DANCE TO GET A GDSTK CELL SO WE CAN IMPORT IT; ONE DAY USE GDSFACTORY PLUG IN TODO TODO TODO

    # export to temp
    gf_comp.write_gds(temp_path + "temp.gds")

    ### FOR THE MEANWHILE DO A DUMB AND IMPORT INTO GDSTK FOR CELL MANAGEMENT ### TODO
    temp_cell = gd.Cell(cell_name + "_temp")

    if filter_layer:
        importTemp = gd.read_gds(temp_path + "temp.gds", filter={(1, 0)})
    else:
        importTemp = gd.read_gds(temp_path + "temp.gds")
    gf_cell = importTemp[gf_name]

    for ii in gf_cell.get_polygons():
        temp_cell.add(ii)

    # shift to compensate for any shifts or positioning considerations
    temp_cell = temp_cell.copy(name=cell_name, translation=(dx, dy))

    return temp_cell
# %% Inpute params from meep
r_x_list = [0.25, 0.3]
r_y_list = [0.25]
theta_list = [0]
wavelength = 0.532             # design wavelength [um]
period_list = [0.60]            # square-lattice pitch [um]
pillar_h_list = [0.85]
incident_angle_list = [10]      # polar angle in degrees; tilt is in the x-z plane
RUN_SIMULATION = True
PLOT_FIRST_PILLAR_FIELD_PROFILE = True
n_SiN = 2.0
n_sio2 = 1.46
resolution = 50                 # Meep pixels / um; mapped below to Tidy3D steps / wavelength
dpml = 0.8                      # z-only absorbing-boundary spacing [um]
air_padding = 1.0               # air above and below the structure [um]
substrate_h = 1.0

# This check simulation uses the first value of each sweep list.
r_x = r_x_list[0]
r_y = r_y_list[0]
theta = theta_list[0]
period = period_list[0]
pillar_h = pillar_h_list[0]
incident_angle_deg = incident_angle_list[0]
incident_angle_rad = np.deg2rad(incident_angle_deg)

# Tidy3D uses micrometers for length and Hz for frequency.
fcen = td.C_0 / wavelength
fwidth = 0.05 * fcen
cell_z = substrate_h + pillar_h + 2 * air_padding + 2 * dpml
sim_size = (period, period, cell_z)
z_min = -0.5 * cell_z + dpml
z_max = 0.5 * cell_z - dpml
monitor_z = 0.5 * cell_z - dpml - 0.35
source_z = -0.5 * cell_z + dpml + 0.35
run_time = 200 / fcen
# %% Import params tidy3d
air = td.Medium(permittivity=1.0)
si_n = td.Medium(permittivity=n_SiN**2)
sio2 = td.Medium(permittivity=n_sio2**2)
min_steps_per_wvl = max(10, int(round(resolution * wavelength)))
# %% Generate Monitors
transmission_monitor = td.FluxMonitor(
    center=(0, 0, monitor_z),
    size=(td.inf, td.inf, 0),
    freqs=[fcen],
    name="transmission_monitor",
)
field_monitor_xz = td.FieldMonitor(
    center=(0, 0, 0),
    size=(period, 0, cell_z - 2 * dpml),
    freqs=[fcen],
    fields=["Ex", "Ez"],
    name="field_monitor_xz",
)
monDict = {"unit_cell_monitors": [transmission_monitor, field_monitor_xz]}
# %% Generate Inputs
plane_wave_input = td.PlaneWave(
    center=(0, 0, source_z),
    size=(td.inf, td.inf, 0),
    source_time=td.GaussianPulse(freq0=fcen, fwidth=fwidth),
    direction="+",
    angle_theta=incident_angle_rad,
    angle_phi=0.0,
    pol_angle=0.0,  # p/TM: Ex at normal incidence; Ex and Ez for an x-z tilt.
    angular_spec=td.FixedInPlaneKSpec(),
    name="p_polarized_plane_wave",
)
srcDict = {"plane_wave_source": [plane_wave_input]}
# %% Grid Spec
grid_spec = td.GridSpec.auto(
    min_steps_per_wvl=min_steps_per_wvl,
    wavelength=wavelength,
)

# %% Bring proper gds file
gds_directory = next(
    (
        candidate / "Unit_cell_Libraries" / "Ellipse Pillar SiN on SiO2 532 nm KAIST" / "GDS Library"
        for candidate in (Path.cwd(), *Path.cwd().parents, Path.cwd() / "Metasurface")
        if (candidate / "Unit_cell_Libraries").is_dir()
    ),
    None,
)
if gds_directory is None:
    raise FileNotFoundError("Could not locate the Metasurface/Unit_cell_Libraries directory.")
gds_path = gds_directory / f"elliptical_pillar_rx_{r_x}_ry_{r_y}_theta_{theta}.gds"
# %% Get cell name
library = gd.read_gds(gds_path)
gds_cells = [
    cell for cell in library.top_level()
    if not cell.name.startswith("$$$CONTEXT_INFO$$$")
]
if len(gds_cells) != 1:
    raise ValueError(f"Expected one physical top cell in {gds_path}, found {len(gds_cells)}.")
gds_cell = gds_cells[0]
print(gds_cell.name)
# %% Turn this to tidy3d geometry
pillar_geometries = td.PolySlab.from_gds(
    gds_cell=gds_cell,
    axis=2,
    slab_bounds=(0, pillar_h),
    gds_layer=1,
    gds_dtype=0,
    gds_scale=1.0,
)
pillar_structure = td.Structure(
    geometry=td.GeometryGroup(geometries=pillar_geometries),
    medium=si_n,
    name="sin_ellipse_pillar",
)
substrate_structure = td.Structure(
    geometry=td.Box(
        center=(0, 0, -substrate_h / 2),
        size=(td.inf, td.inf, substrate_h),
    ),
    medium=sio2,
    name="sio2_substrate",
)
structDict = {"unit_cell_structures": [substrate_structure, pillar_structure]}
# %% Build sim
all_struct = np.concatenate(list(structDict.values())).tolist()
all_mon = np.concatenate(list(monDict.values())).tolist()
all_src = np.concatenate(list(srcDict.values())).tolist()

init_sim = td.Simulation(
    size=sim_size,
    center=(0, 0, 0),
    grid_spec=grid_spec,
    structures=all_struct,
    sources=all_src,
    monitors=all_mon,
    medium=air,
    boundary_spec=td.BoundarySpec(
        x=td.Boundary.bloch_from_source(
            source=plane_wave_input, domain_size=period, axis=0, medium=air
        ),
        y=td.Boundary.bloch_from_source(
            source=plane_wave_input, domain_size=period, axis=1, medium=air
        ),
        z=td.Boundary.pml(),
    ),
    run_time=run_time,
    subpixel=True,
)
# %% Send sim to server
init_job = web.Job(
    simulation=init_sim,
    task_name="ellipse_unit_cell_check",
    folder_name="metasurface_unit_cell",
    verbose=True,
)
estimated_cost = init_job.estimate_cost()
print(f"Estimated maximum cost per sim: {estimated_cost:.3f} Flex Credits")

# ## Plot field profile of first sim

# %%
import tidy3d.web as web
web.test()
# %% Extract transmission and phase
if RUN_SIMULATION:
    # Use the identical source, boundaries, substrate, and monitors without the
    # pillar as the reference.  This removes propagation and interface phase.
    reference_sim = init_sim.updated_copy(structures=[substrate_structure])
    reference_job = web.Job(
        simulation=reference_sim,
        task_name="ellipse_unit_cell_reference",
        folder_name="metasurface_unit_cell",
        verbose=True,
    )

    data_directory = gds_directory.parent / "Library Data"
    if not data_directory.is_dir():
        raise FileNotFoundError(f"Expected existing data directory: {data_directory}")

    reference_data = reference_job.run(
        path=data_directory / "ellipse_unit_cell_reference.hdf5"
    )
    pillar_data = init_job.run(
        path=data_directory / "ellipse_unit_cell_pillar.hdf5"
    )

    reference_flux = float(
        reference_data["transmission_monitor"].flux.sel(f=fcen, method="nearest").item()
    )
    pillar_flux = float(
        pillar_data["transmission_monitor"].flux.sel(f=fcen, method="nearest").item()
    )
    power_transmission = pillar_flux / reference_flux

    # The ratio of complex Ex fields at the transmission plane is the
    # zero-order complex transmission coefficient.  Averaging x samples the
    # full unit-cell field; the common Bloch phase cancels in the ratio.
    reference_ex = complex(
        reference_data["field_monitor_xz"].Ex
        .sel(f=fcen, z=monitor_z, method="nearest")
        .mean()
        .item()
    )
    pillar_ex = complex(
        pillar_data["field_monitor_xz"].Ex
        .sel(f=fcen, z=monitor_z, method="nearest")
        .mean()
        .item()
    )
    complex_transmission = pillar_ex / reference_ex
    transmission_phase_rad = float(np.angle(complex_transmission))
    transmission_phase_deg = float(np.degrees(transmission_phase_rad) % 360)

    transmission_results = {
        "power_transmission": power_transmission,
        "complex_transmission": complex_transmission,
        "phase_rad": transmission_phase_rad,
        "phase_deg": transmission_phase_deg,
    }
    print(f"Power transmission: {power_transmission:.6f}")
    print(f"Transmission phase: {transmission_phase_rad:.6f} rad ({transmission_phase_deg:.3f} deg)")

    if PLOT_FIRST_PILLAR_FIELD_PROFILE:
        ex_profile = (
            pillar_data["field_monitor_xz"].Ex
            .sel(f=fcen, method="nearest")
            .squeeze(drop=True)
        )
        _, ax = plt.subplots()
        np.abs(ex_profile).plot(x="x", y="z", ax=ax, cmap="magma")
        ax.set_title("|Ex| field profile: elliptical-pillar unit cell")
        ax.axhline(monitor_z, color="cyan", linestyle="--", linewidth=1, label="transmission monitor")
        ax.legend()
        plt.show()

# %%
