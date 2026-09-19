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
import gdstk
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

# %% Pick an example gds file to replicate meep FDTD

