
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
metasurface_path = Path(__file__).resolve().parent.parent
if str(metasurface_path) not in sys.path:
    sys.path.insert(0, str(metasurface_path))
from Unit_cell_generation import *
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

def radii_to_u(radii_tensor,radius_response,delta = 0,epsilon = 1e-2):
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
# %% 1. Define Target Functions and incident field
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
target_x = torch.arange(X_MIN, X_MAX, 0.35e-6, device=DEVICE)
target_y = torch.arange(Y_MIN, Y_MAX, 0.35e-6, device=DEVICE)
target_X, target_Y = torch.meshgrid(target_x, target_y, indexing="xy")
target_field = target_amplitude(target_X, target_Y, w0x, w0y, 1.0)
def incident_field(
    x, y, w0=2.008e-6, z=None, wavelength=532e-9, A=1.0, n=1.0
):
    """Circular Gaussian complex field; all lengths are in meters.

    w0 is the minimum 1/e field (1/e^2 intensity) radius, and A is
    the on-axis field amplitude at the waist. Positive z places this
    plane after the waist (diverging beam); negative z gives convergence.
    With z omitted, match the phase-profile design's beam radius w_ml
    at the metalens (z = 516.946 um for the current design).
    Phase follows the ASM convention exp(+1j * kz * z).
    """
    z_R = torch.as_tensor(
        torch.pi * n * w0**2 / wavelength, dtype=x.dtype, device=x.device
    )
    if z is None:
        design_radius = torch.as_tensor(w_ml, dtype=x.dtype, device=x.device)
        z = z_R * torch.sqrt((design_radius / w0)**2 - 1)
    z = torch.as_tensor(z, dtype=x.dtype, device=x.device)
    k = 2 * torch.pi * n / wavelength
    r_squared = x**2 + y**2
    w = w0 * torch.sqrt(1 + (z / z_R)**2)
    inverse_R = z / (z**2 + z_R**2)  # Finite at the waist, where 1/R = 0.
    gouy_phase = torch.atan(z / z_R)

    amplitude = A * (w0 / w) * torch.exp(-r_squared / w**2)
    phase = k * z + 0.5 * k * r_squared * inverse_R - gouy_phase
    return amplitude * torch.exp(1j * phase)

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

# %% 2. Model definition (Forward ASM and loss functions)
class Angular_Spectral_Method(torch.nn.Module):

    def __init__(self,shape,dx,dy,wavelength,z,n = 1.0,dtype = torch.float32):
        super().__init__()

        self.shape = tuple(shape)
        nx,ny = self.shape
        fx = torch.fft.fftfreq(nx, dx)
        fy = torch.fft.fftfreq(ny, dy)
        FX,FY = torch.meshgrid(fx,fy,indexing = 'ij')
        k = 2*torch.pi/wavelength
        kx = 2*torch.pi*FX
        ky = 2*torch.pi*FY
        kz_squared = k**2 - kx**2 - ky**2
        kz = torch.sqrt(torch.complex(kz_squared, torch.zeros_like(kz_squared)))
        H = torch.exp(1j*kz*z)
        self.register_buffer('H', H)
    def forward(self, field):
        spectrum = torch.fft.fft2(field)
        popagated_spectrum = spectrum * self.H
        return torch.fft.ifft2(popagated_spectrum)

seed_everything()
model = Angular_Spectral_Method(target_X.shape, dx=0.35e-6, dy=0.35e-6, wavelength=532e-9, z=2.64529e-3).to(DEVICE)

log(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, "
    f"{sum(isinstance(m, torch.nn.Linear) for m in model.modules())} Linear layers")
print(model)
# %% Define loss function
def overlap_loss(propagated, target):
    """Normalized coherent overlap for two 2D scalar fields."""
    if propagated.shape != target.shape:
        raise ValueError("Fields must have identical shapes and coordinates.")

    overlap = torch.sum(target.conj() * propagated)

    target_norm = torch.sum(target.abs().square())
    propagated_norm = torch.sum(propagated.abs().square())

    denominator = target_norm * propagated_norm
    tiny = torch.finfo(denominator.dtype).tiny

    efficiency = overlap.abs().square() / denominator.clamp_min(tiny)

    return 1.0 - efficiency
# %% 3. One optimization step
EPOCHS, LR, REPORT_EVERY = 5000, 2e-3, 5

loss_fn = overlap_loss
optimizer = torch.optim.Adam([u_tensor], lr=LR)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

def run_one_epoch(epoch_index, tb_writer=None):
    if target_X.shape != u_tensor.shape:
        raise ValueError('Shapes dont match')
    radii = smooth_radius(u_tensor,radius_response)
    phase = interpolate_response(radii, radius_table, phase_table)
    amplitude = interpolate_response(radii, radius_table, amplitude_table)
    transmission = torch.polar(amplitude, phase)
    incident_field_tensor = incident_field(target_X, target_Y)
    E0 = incident_field_tensor * transmission
    Ez = model(E0)
    target_field = target_amplitude(
    target_X, target_Y, w0x=10.58e-6, w0y=42.34e-6, A=1.0)

    loss = overlap_loss(Ez, target_field)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()

# %% 4. Optimization loop
start_time = time.time()
for epoch in range(EPOCHS):
    last_loss = run_one_epoch(epoch)
    # scheduler.step()

    if epoch % REPORT_EVERY == 0:
        elapsed = time.time() - start_time
        log(f"Epoch {epoch:>4d} | loss {last_loss:.6f} | "
            f"elapsed {elapsed:.1f} sec")
        start_time = time.time()

# %% Plot optimized field (Written completely by codex)
with torch.no_grad():
    optimized_radii = smooth_radius(u_tensor, radius_response)
    optimized_phase = interpolate_response(
        optimized_radii, radius_table, phase_table
    )
    optimized_amplitude = interpolate_response(
        optimized_radii, radius_table, amplitude_table
    )
    optimized_transmission = torch.polar(optimized_amplitude, optimized_phase)
    optimized_field = model(
        incident_field(target_X, target_Y) * optimized_transmission
    )
    optimized_overlap = 1.0 - overlap_loss(optimized_field, target_field).item()
    # Remove the constant phase offset, which the overlap loss ignores.
    global_phase = torch.angle(torch.sum(target_field.conj() * optimized_field))
    field_amplitude = optimized_field.abs().cpu().numpy()
    field_phase = torch.angle(
        optimized_field * torch.exp(-1j * global_phase)
    ).cpu().numpy()

extent_um = [
    target_x[0].item() * 1e6, target_x[-1].item() * 1e6,
    target_y[0].item() * 1e6, target_y[-1].item() * 1e6,
]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
amplitude_image = axes[0].imshow(
    field_amplitude, extent=extent_um, origin="lower",
    cmap="viridis", vmin=0, aspect="equal",
)
fig.colorbar(amplitude_image, ax=axes[0], label="Field amplitude |E|")
axes[0].set(title="Optimized amplitude", xlabel="x (μm)", ylabel="y (μm)")

phase_image = axes[1].imshow(
    field_phase, extent=extent_um, origin="lower",
    cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal",
)
phase_colorbar = fig.colorbar(
    phase_image, ax=axes[1], label="Phase relative to target (rad)"
)
phase_colorbar.set_ticks([-np.pi, 0, np.pi], labels=["−π", "0", "π"])
axes[1].set(title="Optimized phase", xlabel="x (μm)", ylabel="y (μm)")
fig.suptitle(f"Optimized output field — coherent overlap {optimized_overlap:.4f}")
plt.show()
# %% Final overlap


# %% Validate worst case supercell near field for FDTD vs ASM
# %% Stitch together final gds and save
metasurface_path = Path(__file__).resolve().parent.parent
lens_gds_path = metasurface_path / "Metalens GDS"

def gds_stitching(
    optimized_radii,
    output_path=None,
    pitch_um=0.35,
    x_um=None,
    y_um=None,
    circle_points=128,
):
    """Export a [y, x] radius map as circular pillars on GDS layer (1, 0).

    Radii, pitch, and optional 1-D placement coordinates are in micrometers.
    Without placement coordinates, center the array at the origin. All pixels
    are exported, matching the square aperture used by the forward model.
    Return the written GDS path.
    """
    if isinstance(optimized_radii, torch.Tensor):
        optimized_radii = optimized_radii.detach().cpu().numpy()
    radii_um = np.asarray(optimized_radii, dtype=float)
    if radii_um.ndim != 2 or radii_um.size == 0:
        raise ValueError("optimized_radii must be a nonempty 2-D radius map.")
    if not np.isfinite(radii_um).all() or (radii_um <= 0).any():
        raise ValueError("All pillar radii must be finite and positive.")
    if not np.isfinite(pitch_um) or pitch_um <= 0:
        raise ValueError("pitch_um must be finite and positive.")
    if not isinstance(circle_points, int) or circle_points < 8:
        raise ValueError("circle_points must be an integer >= 8.")

    ny, nx = radii_um.shape
    coordinates = []
    for values, size in ((x_um, nx), (y_um, ny)):
        if values is None:
            values = (np.arange(size) - (size - 1) / 2) * pitch_um
        elif isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=float)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError("Placement coordinates must match the radius-map axes.")
        if size > 1 and not np.allclose(
            np.diff(values), pitch_um, rtol=1e-4, atol=1e-6
        ):
            raise ValueError("Placement coordinates must use pitch_um spacing.")
        coordinates.append(values)
    x_positions, y_positions = coordinates

    # Reuse an active PDK, or activate the generic PDK for explicit GDS layers.
    gf.get_active_pdk("generic")
    lens = gf.Component()
    angles = np.linspace(0, 2 * np.pi, circle_points, endpoint=False)
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    for iy, y_position in enumerate(y_positions):
        for ix, x_position in enumerate(x_positions):
            points = radii_um[iy, ix] * unit_circle + (x_position, y_position)
            lens.add_polygon(points, layer=(1, 0))

    output_path = (
        lens_gds_path / "optimized_metalens.gds"
        if output_path is None else Path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return lens.write_gds(output_path, with_metadata=False)

with torch.no_grad():
    optimized_radii = smooth_radius(u_tensor, radius_response)
saved_lens_path = gds_stitching(
    optimized_radii,
    x_um=target_x * 1e6,
    y_um=target_y * 1e6,
)
log(f"Saved optimized lens GDS: {saved_lens_path}")

# %%
