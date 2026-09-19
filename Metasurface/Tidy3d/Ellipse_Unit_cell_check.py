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
    temp_cell = gdstk.Cell(cell_name + "_temp")

    if filter_layer:
        importTemp = gdstk.read_gds(temp_path + "temp.gds", filter={(1, 0)})
    else:
        importTemp = gdstk.read_gds(temp_path + "temp.gds")
    gf_cell = importTemp[gf_name]

    for ii in gf_cell.get_polygons():
        temp_cell.add(ii)

    # shift to compensate for any shifts or positioning considerations
    temp_cell = temp_cell.copy(name=cell_name, translation=(dx, dy))

    return temp_cell
# %% Inpute params
r_x_list = [0.25,0.3] # Will just do first radius
tidy3d_r_x = r_x_list[0]
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
# %% Generate Monitors
mon_mode_bar = td.ModeMonitor(
            center=[size_x / 2 - pml_spacing_x, -ring_radius / 2, 0],
            size=[0, mon_w, mon_h],
            freqs=freq_range,
            mode_spec=mode_spec,
            name="mode_monitor_bar",
        )  # type: ignore
mon_flux_bar = td.FluxMonitor(
            center=[size_x / 2 - pml_spacing_x, -ring_radius / 2, 0],
            size=[0, mon_w, mon_h],
            freqs=freq_range,
            name="flux_monitor_bar",
        )  # type: ignore
mon_mode_cross = td.ModeMonitor(
            center=[
                ring_radius + coupling_length / 2,
                size_y / 2 - pml_spacing_y,
                0,
            ],
            size=[mon_w, 0, mon_h],
            freqs=freq_range,
            mode_spec=mode_spec,
            name="mode_monitor_cross",
        )  # type: ignore

mon_flux_cross = td.FluxMonitor(
            center=[
                ring_radius + coupling_length / 2,
                size_y / 2 - pml_spacing_y,
                0,
            ],
            size=[mon_w, 0, mon_h],
            freqs=freq_range,
            name="flux_monitor_cross",
        )  # type: ignore

mon_field_xy = td.FieldMonitor(
            center=[center_x, center_y, 0],
            size=(td.inf, td.inf, 0),
            freqs=freq_c,
            name="field_monitor_xy",
            interval_space=(10, 10, 10),
        )  # type: ignore

monDict = {
            "mon_mode_bar": [mon_mode_bar],
            "mon_mode_cross": [mon_mode_cross],
            "mon_flux_bar": [mon_flux_bar],
            "mon_flux_cross": [mon_flux_cross],
            "mon_field_xy": [mon_field_xy],
        }
# %% Generate Inputs
mode_input = td.ModeSource(
            center=[-size_x / 2 + pml_spacing_x, -ring_radius / 2, 0],
            size=[0, mon_w, mon_h],
            source_time=td.GaussianPulse(
                freq0=freq_c,
                fwidth=freq_bw,
            ),  # type: ignore
            direction="+",
            mode_index=mode_index,
        )  # type: ignore
mode_input_src = [mode_input]

srcDict = {
            "mode_input_src": mode_input_src,
        }
# %% Grid Spec
grid_spec = td.GridSpec(grid_x = td.AutoGrid(min_steps_per_wvl = 16,max_scale = 1.9)
                                            , grid_y = td.AutoGrid(min_steps_per_wvl = 16, max_scale = 1.9)
                                            , grid_z = td.AutoGrid(min_steps_per_wvl = 16, max_scale = 1.9)
                                            # , override_structures = [override_structure]
                                            ,override_structures = [box_structure]
)

# %% Bring proper gds file
path = os.getcwd()
gds_path = path + "/Unit_cell_Libraries/Ellipse Pillar SiN on SiO2 532 nm KAIST/GDS Library/elliptical_pillar_rx_0.25_ry_0.25_theta_0.gds"
component = gf.import_gds(gds_path)
# %% Get cell name

library = gd.read_gds(gds_path)

for cell in library.cells:
    print(cell.name)
# %% Turn this to tidy3d geometry
temp_cell = gd.Cell('gdstk version of elliptical_pillar_rx_0.25_ry_0.25_theta_0')
importTemp = gd.read_gds(gds_path)
gf_cell = importTemp['elliptical_pillar_rx_0.25_ry_0.25_theta_0']
for ii in gf_cell.get_polygons():
        temp_cell.add(ii)

    # shift to compensate for any shifts or positioning considerations
test_cell = temp_cell.copy(name='test pillar')

structure = td.Structure(geometry = td.GeometryGroup(geometries = test_cell),medium = mat_device,name = 'Override Structure')
# %% Build sim 
all_struct = np.concatenate(list(structDict.values())).tolist()
all_mon = np.concatenate(list(monDict.values())).tolist()
all_src = np.concatenate(list(srcDict.values())).tolist()
init_sim = td.Simulation(
                size=[size_x, size_y, size_z],
                center=[center_x, center_y, center_z],
                grid_spec=grid_spec,
                structures=all_struct,
                sources=all_src,
                monitors=all_mon,
                medium=mat_cladding,
                boundary_spec=td.BoundarySpec(x=td.Boundary.absorber(), y=td.Boundary.absorber(), z=td.pml()),
                run_time=run_time,
                subpixel=True,
            )  # type: ignore
# %% Run sim

# ## Plot field profile of first sim
