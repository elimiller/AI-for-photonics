#Claude's Rudimentary Unit Cell. To be ran locally4
# %%READme
"""
Single unit-cell MEEP simulation: C-shaped (split-ring-like) dielectric pillar
on a substrate, illuminated by a normal-incidence broadband pulse.

Goal: time this single run to extrapolate compute cost for ~11,000 geometries.

Physics setup:
  - Periodic boundary conditions in x, y (infinite array of unit cells)
  - PML absorbing boundaries in z
  - Substrate: n = 1.45, semi-infinite (extends into -z PML)
  - Pillar: n = 2.0, C-shaped cross-section (square ring with a gap = "C"),
    extruded in z to form a finite-height pillar sitting on the substrate
  - Source: broadband Gaussian pulse, normal incidence, polarized along x
  - Monitors: flux planes above pillar (reflection) and below substrate
    (transmission), Fourier-transformed to give the FULL spectrum from
    ONE time-domain run (this is what makes MEEP efficient for sweeps)

Install (free, local, no license):
    conda create -n meep -c conda-forge pymeep pymeep-extras -y
    conda activate meep
    python c_pillar_unit_cell.py

Run time is what you're measuring here -- wrap the whole sim in the
wall-clock timer at the bottom and use that number to extrapolate to
11,000 geometries (see note at the end of this file).
"""
# %% Imports
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import gdsfactory as gf
import gplugins
import gplugins.gmeep as gm 
from gplugins.gmeep.get_simulation import get_simulation
import os
# %% Import necessary functions

directory = os.getcwd() # For data savepath for later perhaps
from Unit_cell_generation import*
# %% Input params
# ---------------------------------------------------------------------
# GEOMETRY PARAMETERS (these are exactly the knobs you'll sweep later)
# ---------------------------------------------------------------------
period      = 0.60      # unit cell period in x and y (um)
pillar_h    = 0.30       # pillar height (um)
sub_h       = 0.50       # substrate thickness modeled explicitly before PML (um)
arm_w       = 0.10       # width of the C's arms (um)
outer_side  = 0.40
cutout_horizontal_length = outer_side - arm_w
cutout_vertical_length = outer_side - 2 * arm_w
gap_w       = 0.08       # width of the gap that turns the ring into a "C" (um)

n_pillar    = 2.00        # pillar refractive index
n_sub       = 1.45        # substrate refractive index

resolution  = 40          # pixels per um -- lower this first to test speed,
                           # raise it once you trust convergence

# Wavelength range of interest -- set this to whatever band you actually
# care about (visible/NIR/MIR). Example below spans 1.0-2.0 um.
wl_min, wl_max = 0.531, 0.533
fmin, fmax = 1 / wl_max, 1 / wl_min
fcen = 0.5 * (fmin + fmax)
df   = fmax - fmin
nfreq = 100    # number of points across the spectrum (from ONE run)

# ---------------------------------------------------------------------
# CELL / BOUNDARY SETUP
# ---------------------------------------------------------------------
dpml = 0.5                     # PML thickness (um) -- roughly 1 wavelength
space_buffer = 1.0
z_span = sub_h + pillar_h + 2*space_buffer  # air gap above pillar for source/monitor
cell_z = z_span + 2 * dpml
# Geometry Creaation
cell = mp.Vector3(period, period, cell_z)

pml_layers = [mp.PML(dpml, direction=mp.Z)]
#To be changed in a minute
k_point = mp.Vector3(0, 0, 0)   # normal incidence -> use Bloch-periodic BCs
                                 # (k_point=0 here; MEEP handles x/y periodicity
                                 # automatically when boundary layers are only in z)
# %% Geometries
# ---------------------------------------------------------------------
# GEOMETRY: build the C-shape as a square ring (4 boxes) minus a gap
# ---------------------------------------------------------------------
# z-center of the pillar and substrate for placement
sub_center_z  = -0.5 * cell_z + dpml + sub_h / 2 + space_buffer
pillar_center_z = -0.5 * cell_z + dpml + sub_h + pillar_h / 2 + space_buffer

substrate = mp.Block(
    size=mp.Vector3(period, period, sub_h),
    center=mp.Vector3(0, 0, sub_center_z),
    material=mp.Medium(index=n_sub),
)

# Build the ring as an outer square minus an inner square (leaves a frame
# of width arm_w), then cut a gap out of one side to make it a "C".
outer = mp.Block(
    size=mp.Vector3(outer_side, outer_side, pillar_h),
    center=mp.Vector3(0.5 * period - outer_side, 0.5*period - outer_side, pillar_center_z),
    material=mp.Medium(index=n_pillar),
)
inner_side = outer_side - 2 * arm_w
inner_void = mp.Block(
    size=mp.Vector3(cutout_horizontal_length, cutout_vertical_length, pillar_h + 0.01),
    center=mp.Vector3(0.5 * (period - outer_side) - cutout_horizontal_length / 2, 0.5*(period - outer_side) - arm_w - cutout_vertical_length / 2, pillar_center_z),
    material=mp.Medium(index=1.0),  # air, carves out the ring's interior
)
gap_void = mp.Block(
    size=mp.Vector3(gap_w, arm_w + 0.02, pillar_h + 0.01),
    center=mp.Vector3(0, outer_side / 2 - arm_w / 2, pillar_center_z),
    material=mp.Medium(index=1.0),  # air, cuts the gap into the ring -> "C"
)

geometry = [substrate, outer, inner_void, gap_void]

# ---------------------------------------------------------------------
# SOURCE: broadband pulse, normal incidence, x-polarized plane wave
# ---------------------------------------------------------------------
src_z = -0.5 * cell_z + dpml + 0.3  # just inside the PML, below substrate... 
# NOTE: for a transmission/reflection setup you typically put the source
# ABOVE the pillar (or below the substrate) and monitors on both sides.
# Placing it below the substrate here so transmission = through both
# substrate and pillar.
# %% Sources
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ex,
        center=mp.Vector3(0, 0, src_z),
        size=mp.Vector3(period, period, 0),
    )
]
# %% Simulation setup
sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    k_point=k_point,
    resolution=resolution,
    default_material=mp.Medium(index=1.0),  # air everywhere else
)
sim.plot2D(
    output_plane=mp.Volume(center=mp.Vector3(), size=cell),
    fields=mp.Ex,
    output_directory="plots",
    plot_boundaries=True,
)
# %% Monitors
# ---------------------------------------------------------------------
# FLUX MONITORS: one run -> full spectrum via DFT
# ---------------------------------------------------------------------
refl_z = src_z + 0.2
tran_z = 0.5 * cell_z - dpml - 0.3

refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, refl_z),
                         size=mp.Vector3(period, period, 0))
tran_fr = mp.FluxRegion(center=mp.Vector3(0, 0, tran_z),
                         size=mp.Vector3(period, period, 0))

refl = sim.add_flux(fcen, df, nfreq, refl_fr)
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# ---------------------------------------------------------------------
# RUN + TIME IT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()

    sim.run(until_after_sources=mp.stop_when_dft_decayed())

    t1 = time.time()
    elapsed = t1 - t0
    print(f"\n=== Single unit-cell run time: {elapsed:.1f} seconds ===")
    print(f"Estimated time for 11,000 geometries (serial, same machine): "
          f"{elapsed * 11000 / 3600:.1f} hours")

    freqs = np.array(mp.get_flux_freqs(tran))
    wavelengths = 1 / freqs
    tran_flux = np.array(mp.get_fluxes(tran))
    refl_flux = np.array(mp.get_fluxes(refl))

    np.savez("spectrum_output.npz",
             wavelength_um=wavelengths,
             transmission=tran_flux,
             reflection=refl_flux)

    print("Saved spectrum_output.npz -- plot transmission/reflection vs "
          "wavelength_um to sanity-check the resonance.")
# %% Plotting the spectrum 
data = np.load("spectrum_output.npz")
wavelengths = data["wavelength_um"]
transmission = data["transmission"]
reflection = data["reflection"]

plt.figure(figsize=(7, 5))
# plt.plot(wavelengths, transmission, label="Transmission")
plt.plot(wavelengths, reflection, label="Reflection")
plt.xlabel("Wavelength (μm)")
plt.ylabel("Flux (normalized)")
plt.title("C-pillar unit cell spectrum")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# %%
