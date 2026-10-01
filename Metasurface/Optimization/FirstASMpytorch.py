
# %%  Readme
# This will be a worst case supercell sim
#Goal of this notebook is to get idea of error introduced by LPA

# %% imports
import os
import sys, time, random, platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
# PyTorch TensorBoard support, as in the introyt training tutorial
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pandas as pd
import gdsfactory as gf
# %% Construction of supercell to ASM
#We want worst case scenario, call this a 2pi phase gradient 
# in 12 cells
path = Path(__file__).parent.parent
print(path)
save_path = path / "Unit_cell_Libraries" /  'Ellipse Pillar SiN on SiO2 532 nm KAIST' 
print(save_path)
gds_lib_path = save_path / 'GDS Library'
data_lib_path = save_path / 'Library Data'
csv_path = data_lib_path / "ellipse_pillar_library_data.csv"
print(csv_path)
master_library = pd.read_csv(csv_path)
library = master_library[(master_library['pillar_height_um'] == 0.8) & (master_library['period_um'] == 0.35)]
print(library)
# %% Get phase array
ML2_phase_profile = np.load("ML2_phase_profile.npy")

# %%
