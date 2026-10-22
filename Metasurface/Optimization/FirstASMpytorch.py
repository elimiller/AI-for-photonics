
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
from scipy.interpolate import interp1d, UnivariateSpline
# %% Platform and hardware info

LOG = []
def log(msg=""):
    LOG.append(str(msg)); print(msg)

if torch.cuda.is_available():
    DEVICE, BACKEND = torch.device("cuda"), "CUDA"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE, BACKEND = torch.device("mps"), "MPS (Apple Silicon)"
else:
    DEVICE, BACKEND = torch.device("cpu"), "CPU"

SEED = 0
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything()

log(f"python      : {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
log(f"torch       : {torch.__version__}   numpy: {np.__version__}   matplotlib: {matplotlib.__version__}")
log(f"backend     : {BACKEND}  ({DEVICE})")
if DEVICE.type == "cuda":
    log(f"gpu         : {torch.cuda.get_device_name(0)}  |  cuda {torch.version.cuda}")
log(f"seed        : {SEED}")
# %% Bring in library data and create array of unit cell responses
path = Path(__file__).parent.parent
print(path)
save_path = path / "Unit_cell_Libraries" /  'Ellipse Pillar SiN on SiO2 532 nm KAIST' 
print(save_path)
gds_lib_path = save_path / 'GDS Library'
data_lib_path = save_path / 'Library Data'
csv_path = data_lib_path / "ellipse_pillar_library_data.csv"
print(csv_path)
master_library = pd.read_csv(csv_path)
#Get correct subset of library and remove values where there is no tranmsission
library = master_library[(master_library['pillar_height_um'] == 0.8) & (master_library['period_um'] == 0.35) & (master_library['radius_x_um'] == master_library['radius_y_um']) & ((master_library['radius_x_um'] <= 0.108205128
))]
print(library)

# %% Get phase array from phase map
ML1_phase_profile = np.load(path / "Optimization" / "ML1_phase_profile.npy")
np.shape(ML1_phase_profile)
print(ML1_phase_profile[0][0])
# %% Check how to index
test = np.zeros((2, 2))
test[0][0] = 1
print(test)
# %% Adjust grid size to match lattice constant function
def grid_matching(x,y,new_period,old_phase_map):
    #this function will take the old phase map and match it to the new one
    #Only valid for square grids hence only x as input. Although this is easy to resolve
    #Only valid if old_period < new_period < 2*old_period
    #Just an approximation still, will need further optimizatoin I'm sure
    x = x*10**6
    y = y*10**6
    old_period = x[1] - x[0]
    new_x = np.arange(x[0], x[-1], new_period)
    new_y = np.arange(y[0], y[-1], new_period)
    new_phase_map = np.zeros((len(new_x), len(new_y)))
    for i in range(len(new_x)-1):
        for j in range(len(new_y)-1):
            my = int(np.floor((new_y[j+1]-y[0])/old_period))
            mx = int(np.floor((new_x[i+1]-x[0])/old_period))
            #Case 1: simplest, 1 point in interval for x and y, so 4 areas to consider 
            if (np.floor((new_x[i+1])/old_period) - np.floor((new_x[i])/old_period) == 1) & (np.floor((new_y[j+1])/old_period) - np.floor((new_y[j])/old_period) == 1):
                new_phase_map[i][j] = (old_phase_map[mx][my]*(x[mx] - new_x[i])*(y[my] - new_y[j]) + old_phase_map[mx+1][my+1]*(new_x[i+1] - x[mx])*(new_y[j+1]-y[my]) + old_phase_map[mx][my+1]*(x[mx] - new_x[i])*(new_y[j+1] - y[my]) + old_phase_map[mx+1][my]*(new_x[i+1] - x[mx])*(y[my] - new_y[j]))/(new_period**2)
            #Case 2: 2 points in x and y interval, so 9 areas to consider
            elif ((np.floor((new_x[i+1])/old_period) - np.floor((new_x[i])/old_period) == 2) & (np.floor((new_y[j+1])/old_period) - np.floor((new_y[j])/old_period) == 2)):
                new_phase_map[i][j] = (old_phase_map[mx-1][my-1]*(x[mx-1] - new_x[i])*(y[my-1]-new_y[j]) + old_phase_map[mx+1][my+1]*(new_x[i+1] - x[mx])*(new_y[j+1]-y[my]) + old_phase_map[mx][my]*(x[mx]-x[mx-1])*(y[my]-y[my-1])+ old_phase_map[mx][my-1]*(x[mx]-x[mx-1])*(y[my-1]-new_y[j]) + old_phase_map[mx-1][my]*(x[mx-1]-new_x[i])*(y[my]-y[my-1])+old_phase_map[mx+1][my-1]*(new_x[i+1] - x[mx])*(y[my-1]-new_y[j])+old_phase_map[mx-1][my+1]*(x[mx-1] - new_x[i])*(new_y[j+1] - y[my])+old_phase_map[mx+1][my]*(new_x[i+1] - x[mx])*(y[my] - y[my-1]) + old_phase_map[mx][my+1]*(x[mx]-x[mx-1])*(new_y[j+1] - y[my]))/(new_period**2)
            #Case 3: 1 point in x and 2 points in y interval, so 6 areas to consider
            elif ((np.floor((new_x[i+1])/old_period) - np.floor((new_x[i])/old_period) == 1) & (np.floor((new_y[j+1])/old_period) - np.floor((new_y[j])/old_period) == 2)):
                new_phase_map[i][j] = (old_phase_map[mx][my-1]*(x[mx] - new_x[i])*(y[my-1]-new_y[j]) + old_phase_map[mx+1][my+1]*(new_x[i+1] - x[mx])*(new_y[j+1]-y[my]) + old_phase_map[mx][my]*(x[mx]-new_x[i])*(y[my]-y[my-1])+ old_phase_map[mx+1][my]*(new_x[i+1]-x[mx])*(y[my]-y[my-1]) + old_phase_map[mx+1][my-1]*(new_x[i+1] - x[mx])*(y[my-1]-new_y[j])+old_phase_map[mx][my+1]*(x[mx]-new_x[i])*(new_y[j+1] - y[my]))/(new_period**2)
            #Case 4: 2 points in x and 1 point in y interval, so 6 areas to consider
            elif ((np.floor((new_x[i+1])/old_period) - np.floor((new_x[i])/old_period) == 2) & (np.floor((new_y[j+1])/old_period) - np.floor((new_y[j])/old_period) == 1)):
                new_phase_map[i][j] = (old_phase_map[mx-1][my]*(x[mx-1] - new_x[i])*(y[my]-new_y[j]) + old_phase_map[mx+1][my+1]*(new_x[i+1] - x[mx])*(new_y[j+1]-y[my]) + old_phase_map[mx][my]*(x[mx]-x[mx-1])*(y[my]-new_y[j])+ old_phase_map[mx][my+1]*(x[mx]-x[mx-1])*(new_y[j+1]-y[my]) + old_phase_map[mx-1][my+1]*(x[mx-1] - new_x[i])*(new_y[j+1]-y[my])+old_phase_map[mx+1][my]*(new_x[i+1] - x[mx])*(y[my]-new_y[j]))/(new_period**2)
            else:
                raise ValueError("This function is only valid for old_period < new_period < 2*old_period")
    return np.mod(new_phase_map, 2*np.pi)

# %%  grid match 

w0_out = np.array([10.58e-6, 42.34e-6])
w_ml = np.linalg.norm(w0_out)
aperture_radius = 2 * w_ml
N = 801
x = np.linspace(-aperture_radius, aperture_radius, N)
y = np.linspace(-aperture_radius, aperture_radius, N)
new_phase_map = grid_matching(x,y,0.35,ML1_phase_profile)
# %% Initialize radii tensor debugging
np.shape(new_phase_map)
# %% Plot grid matched phase map
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
im = ax.imshow(
    new_phase_map,
    extent=[x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6],
    origin="lower",
    cmap="twilight",  # Cyclic colormap for wrapped phase
    vmin=0,
    vmax=2 * np.pi,
    interpolation="nearest",
    aspect="equal",
)

cbar = fig.colorbar(im, ax=ax, label="Phase (rad)")
cbar.set_ticks(
    [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
    labels=["0", "π/2", "π", "3π/2", "2π"],
)

ax.set(
    xlabel="x (μm)",
    ylabel="y (μm)",
    title="Metalens 1 — wrapped phase profile",
)
plt.show()
# %% phase response vs radii unwrap 
#first, plot phase response from library to reconfirm what we have
fit_library = library.sort_values("radius_x_um").copy()

radius_response = fit_library["radius_x_um"].to_numpy()
phase_response_unwrapped = np.unwrap(
    np.deg2rad(fit_library["zero_order_tm_phase_deg"].to_numpy())
)
transmission_response = fit_library["zero_order_tm_magnitude"].to_numpy()
plt.plot(radius_response, phase_response_unwrapped, "o")
print(radius_response.max())
# %% Initialize pytorch tensor. Construct apreture i.e., apply data library to phase map
def initialize_radii(phase_map,library):
    #This function will take the phase map and the library and return a tensor of radii
    #The phase map is assumed to be in radians and wrapped to [0, 2pi]
    #The library is assumed to be a pandas dataframe with columns 'phase' and 'radius_x_um' and 'radius_y_um'
    #The output is a tensor of shape (N, N, 2) where N is the size of the phase map and the last dimension is (radius_x_um, radius_y_um)
    #The output is also a torch tensor on the device specified by DEVICE
    radii = np.zeros((phase_map.shape[0], phase_map.shape[1]))
    for i in range(phase_map.shape[0]):
        for j in range(phase_map.shape[1]):
            phase = phase_map[i, j]
            # Find the closest phase in the library
            closest_phase_idx = (library['zero_order_tm_phase_deg']*np.pi/180 - phase).abs().idxmin()
            radii[i, j] = library.loc[closest_phase_idx, 'radius_x_um']
    return torch.tensor(radii, device=DEVICE)
def intialize_radii_tensor(phase_map, phase_response,radius_response):
    radii_tensor = torch.zeros(phase_map.shape, device=DEVICE)
    for i in range(phase_map.shape[0]):
        for j in range(phase_map.shape[1]):
            phase = phase_map[i, j]
            # Find the closest phase in the library
            phase_error = np.abs(np.angle(np.exp(1j*(phase_response - phase))))
            closest_phase_idx = phase_error.argmin()
            radii_tensor[i, j] = radius_response[closest_phase_idx]
    return radii_tensor
radii_tensor = intialize_radii_tensor(new_phase_map, phase_response_unwrapped, radius_response)
print(radii_tensor.max())
# %% Convert initial radii to leaf parameter u for continous differentiability

def smooth_radius(u,radius_response,delta = 0):
    r_min = radius_response.min()
    r_max = radius_response.max()
    
    return r_min + delta + (r_max - r_min - 2*delta)*torch.sigmoid(u)

def radii_to_u(radii_tensor,radius_response,delta = 0,epsilon = 1e-6):
    #Add clamp (epsilon) value to avoid infinite u values, so it is never actually at the boundary
    logarg = ((radii_tensor - radius_response.min() - delta) / (radius_response.max() - radius_response.min() - 2*delta))
    logarg = logarg.clamp(epsilon,1-epsilon)
    return torch.logit(logarg)
u_tensor = radii_to_u(radii_tensor,radius_response,delta = 0).detach().clone().requires_grad_(True)
print(torch.isfinite(u_tensor).all())
print(u_tensor.max())
print(u_tensor.min())
print(u_tensor)
smooth_radius_tensor = smooth_radius(u_tensor,radius_response,delta = 0)
print(torch.abs(smooth_radius_tensor - radii_tensor).max())
# %% Connect the u tensor/radius to the phase response via torch peacewise intterpolant
# Fixed library tables for the differentiable forward model.
radius_table = torch.as_tensor(
    radius_response, dtype=torch.float32, device=DEVICE
)
phase_table = torch.as_tensor(
    phase_response_unwrapped, dtype=torch.float32, device=DEVICE
)
amplitude_table = torch.as_tensor(
    transmission_response, dtype=torch.float32, device=DEVICE
)
def interpolate_response(query_radius, radius_table, response_table):
    """
    Interpolate y_table(x_table) at every value in x.

    x:       arbitrary-shaped tensor, e.g. the 2-D radius map
    x_table: sorted 1-D library radii
    y_table: library response at those radii
    """
    idx = torch.searchsorted(radius_table, query_radius, right=True)
    idx = idx.clamp(1, radius_table.numel() - 1)

    x0 = radius_table[idx - 1]
    x1 = radius_table[idx]
    y0 = response_table[idx - 1]
    y1 = response_table[idx]

    fraction = (query_radius - x0) / (x1 - x0)
    return y0 + fraction * (y1 - y0)
# %% 1. Define Target Functions (each plane) and Derivative
X_MIN, X_MAX = -aperture_radius, aperture_radius
Y_MIN, Y_MAX = -aperture_radius, aperture_radius
# N_TOTAL = 2000
# BATCH_SIZE = 140
w0x = 10.58e-6  # meters, matching the aperture coordinates
w0y = 42.34e-6
# TODO 1: Define the target function 
# Hint: use PyTorch operations so the function works directly on tensors.
def target_amplitude(x,y,w0x,w0y,A):
    return A*torch.exp(-(x**2/(w0x**2) + y**2/(w0y**2)))
target_x = torch.linspace(X_MIN, X_MAX, 401, device=DEVICE)
target_y = torch.linspace(Y_MIN, Y_MAX, 401, device=DEVICE)
target_X, target_Y = torch.meshgrid(target_x, target_y, indexing="xy")
target_field = target_amplitude(target_X, target_Y, w0x, w0y, 1.0)

fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
im = ax.imshow(
    target_field.detach().cpu().numpy(),
    extent=[X_MIN * 1e6, X_MAX * 1e6, Y_MIN * 1e6, Y_MAX * 1e6],
    origin="lower",
    cmap="viridis",
    aspect="equal",
    vmin=0,
    vmax=1,
)
fig.colorbar(im, ax=ax, label="Target amplitude")
ax.set(xlabel="x (μm)", ylabel="y (μm)", title="Target amplitude")
plt.show()
# TODO 2: Create N_TOTAL evenly spaced x values on [X_MIN, X_MAX].[Y_MIN, Y_MAX]
# The network expects shape [N, 1], so add a singleton second dimension.
# x_all = torch.linspace(X_MIN, X_MAX, N_TOTAL, device=DEVICE).unsqueeze(1)
# y_all = torch.linspace(Y_MIN, Y_MAX, N_TOTAL, device=DEVICE).unsqueeze(1)

# TODO 3: Evaluate the target function at every x value.
# y_all = target_amplitude(x_all, y_all, w0x, w0y, 1.0)

# N_TRAIN = int(0.70 * N_TOTAL)
# N_VAL   = int(0.15 * N_TOTAL)
# N_TEST  = N_TOTAL - N_TRAIN - N_VAL

# Random permutation
# g = torch.Generator().manual_seed(SEED)
# indices = torch.randperm(N_TOTAL, generator=g)

# train_idx = indices[:N_TRAIN]
# val_idx   = indices[N_TRAIN:N_TRAIN + N_VAL]
# test_idx  = indices[N_TRAIN + N_VAL:]

# Get tensors
# x_train, y_train = x_all[train_idx], y_all[train_idx]
# x_val,   y_val   = x_all[val_idx],   y_all[val_idx]
# x_test,  y_test  = x_all[test_idx],  y_all[test_idx]

# Construct the datasets
# training_set = torch.utils.data.TensorDataset(x_train, y_train)
# validation_set = torch.utils.data.TensorDataset(x_val, y_val)
# test_set = torch.utils.data.TensorDataset(x_test, y_test)

# training_loader = torch.utils.data.DataLoader(
#     training_set, batch_size=BATCH_SIZE, shuffle=True
# )
# validation_loader = torch.utils.data.DataLoader(
#     validation_set, batch_size=BATCH_SIZE, shuffle=False
# )
# test_loader = torch.utils.data.DataLoader(
#     test_set, batch_size=BATCH_SIZE, shuffle=False
# )

# Report
# log(f"Training set has {len(training_set)} samples")
# log(f"Validation set has {len(validation_set)} samples")
# log(f"Test set has {len(test_set)} samples")

# plt.figure(figsize=(9, 3))
# plt.plot(x_all.cpu(), y_all.cpu(), lw=1.5, color="0.75", label="$f(x) = x\\,\\sin(x)$")
# plt.xlabel("x"); plt.ylabel("f(x)"); plt.grid(alpha=0.3)
# plt.legend()
# plt.show()

# %% 2. Model definition (Forward ASM and loss functions)
class FunctionFitter(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(1, 64)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, 64)
        self.linear3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        # TODO 4: Implement the forward pass.
        # Apply linear1 -> activation -> linear2 -> activation -> linear3.
        # YOUR CODE HERE
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)

        return x

seed_everything()
model = FunctionFitter().to(DEVICE)

log(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, "
    f"{sum(isinstance(m, torch.nn.Linear) for m in model.modules())} Linear layers")
print(model)

# %% 3. Training loop
EPOCHS, LR, REPORT_EVERY = 2000, 2e-3, len(training_loader)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # TODO 5: Clear gradients from the previous batch.
        optimizer.zero_grad()

        # TODO 6: Run the model on this batch.
        outputs = model(inputs)

        # TODO 7: Compute mean-squared error between predictions and labels.
        loss = loss_fn(outputs, labels)

        # TODO 8: Backpropagate the loss, then update the model weights.
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % REPORT_EVERY == REPORT_EVERY - 1:
            last_loss = running_loss / REPORT_EVERY
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# %% 4. Training and output logging perhaps executable file

# %% Validate worst case supercell near field for FDTD vs ASM
# %% Stitch together final gds 
