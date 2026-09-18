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
RUN_SIMULATION = False
n_SiN = 2.0
n_sio2 = 1.46
resolution = 50               # pixels / um; increase after convergence test
dpml = 0.8                    # z-only absorbing boundary thickness [um]
air_padding = 1.0             # air above and below the structure [um]
substrate_h = 1.0 
# %% Generate GDS library
