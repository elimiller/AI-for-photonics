# %% imports
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import gdsfactory as gf
from gdsfactory.technology import LayerLevel, LayerStack
from gplugins.gmeep.get_meep_geometry import get_meep_geometry_from_component
import os

from Unit_cell_generation import*
# %%
# %% Generate GDS library
def generate_unit_cell_gds_lib(pillar_geometry,period,more):
    return data_lib
