
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

# PyTorch TensorBoard support, as in the introyt training tutorial
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pandas as pd
import gdsfactory as gf

# %%
