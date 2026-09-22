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
    dft = sim.add_dft_fields([POLARIZATION], fcen, 0, 1, where=transmission_plane)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, POLARIZATION, mp.Vector3(0, 0, monitor_z), 1e-3
        )
    )
    return np.mean(sim.get_dft_array(dft, POLARIZATION, 0))
