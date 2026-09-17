# %% [markdown]
"""FDTD unit cell for one a-Si cylindrical nanopillar at 1550 nm.

This implements one entry from the cylindrical phase library in
Targholizadeh, Nikulin, and Jha, *Nanophotonics* (2026),
doi:10.1002/nap2.70172. The selected radius is 0.180 um, whose published
target phase is pi/8. The paper specifies a 0.85 um tall a-Si pillar on a
SiO2 substrate in a square 0.60 um-period lattice.

The simulation calculates the complex transmitted field relative to an
otherwise identical bare-SiO2 reference cell. Consequently:

    transmission = |E_pillar / E_reference|^2
    phase        = angle(E_pillar / E_reference) modulo 2*pi

All lengths are in um. Run this file from the Metasurface directory using
the `metasurface_project_0` conda environment.
"""

# %% Imports -- retained from the previous notebook/script
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


# %% Manual unit-cell parameters
wavelength = 1.55             # design wavelength [um]
period = 0.60                 # square-lattice pitch [um]
pillar_h = 0.85               # a-Si cylinder height [um]
radius_list = [0.102,0.140,0.157,0.168,0.180,0.193,0.214,0.242]                # selected Figure 2a library radius [um]
example_radius = 0.180     # radius to display before launching a sweep [um]
RUN_SIMULATION = True     # set True only when ready to run the FDTD sweep

# The paper cites cryogenic Si optical data. These lossless indices are a
# practical first model; replace n_a_si with the cited 5 K value/data model
# when a quantitative match to the authors' Tidy3D result is required.
n_a_si = 3.48
n_sio2 = 1.444

resolution = 50               # pixels / um; increase after convergence test
dpml = 0.8                    # z-only absorbing boundary thickness [um]
air_padding = 1.0             # air above and below the structure [um]
substrate_h = 1.0             # explicit SiO2 thickness before the lower PML [um]


# %% Convert the gdsfactory circle to 3D Meep geometry with gplugins
# A PDK is required to resolve gdsfactory's (layer, datatype) tuple.
gf.gpdk.PDK.activate()

PILLAR_LAYER = (1, 0)  # circle_unit_cell() draws its polygon on this layer
unit_cell = circle_unit_cell(example_radius)
layer_stack = LayerStack(
    layers={
        "a_si_pillar": LayerLevel(
            layer=PILLAR_LAYER,
            thickness=pillar_h,
            zmin=0.0,          # cylinder sits on the substrate top at z = 0
            material="a_si",
        )
    }
)
pillar_geometry = get_meep_geometry_from_component(
    component=unit_cell,
    layer_stack=layer_stack,
    material_name_to_meep={"a_si": n_a_si},
    wavelength=wavelength,
)


# %% Meep cell, source, and monitor locations
cell_z = substrate_h + pillar_h + 2 * air_padding + 2 * dpml
cell = mp.Vector3(period, period, cell_z)
pml_layers = [mp.PML(dpml, direction=mp.Z)]

substrate = mp.Block(
    size=mp.Vector3(period, period, substrate_h),
    center=mp.Vector3(0, 0, -substrate_h / 2),
    material=mp.Medium(index=n_sio2),
)

# Launch from the air side (+z) and sample transmitted field in the substrate
# side (-z). x/y are periodic because PML is applied only along z.
source_z = 0.5 * cell_z - dpml - 0.35
monitor_z = -0.5 * cell_z + dpml + 0.35
fcen = 1 / wavelength


def run_unit_cell(extra_geometry: list[mp.GeometricObject]) -> complex:
    """Returns the area-averaged complex Ex at the transmission plane."""
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=[substrate, *extra_geometry],
        sources=[
            mp.Source(
                mp.GaussianSource(fcen, fwidth=0.05 * fcen),
                component=mp.Ex,
                center=mp.Vector3(0, 0, source_z),
                size=mp.Vector3(period, period, 0),
            )
        ],
        k_point=mp.Vector3(),  # normal incidence / periodic unit cell
        resolution=resolution,
        default_material=mp.air,
    )
    transmission_plane = mp.Volume(
        center=mp.Vector3(0, 0, monitor_z),
        size=mp.Vector3(period, period, 0),
    )
    dft = sim.add_dft_fields([mp.Ex], fcen, 0, 1, where=transmission_plane)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ex, mp.Vector3(0, 0, monitor_z), 1e-3
        )
    )
    field = sim.get_dft_array(dft, mp.Ex, 0)
    return np.mean(field)


def plot_unit_cell_geometry(radius: float) -> None:
    """Show a top-down layout of one periodic unit cell without running Meep."""
    example_cell = circle_unit_cell(radius)
    example_cell.plot()
    ax = plt.gca()
    ax.add_patch(
        Rectangle(
            (-period / 2, -period / 2), period, period,
            fill=False, linestyle="--", linewidth=1.2, edgecolor="black",
        )
    )
    ax.set_aspect("equal")
    ax.set_title(f"a-Si cylinder unit cell: radius = {radius:.3f} um")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    plt.show()


if __name__ == "__main__":
    plot_unit_cell_geometry(example_radius)

# %% Loop through radii
sim_dict = {}
if RUN_SIMULATION:
    sweep_t0 = time.perf_counter()
    for idx, radius in enumerate(radius_list):
        if idx == 0:
            PILLAR_LAYER = (1, 0)  # circle_unit_cell() draws its polygon on this layer
            unit_cell = circle_unit_cell(radius)
            layer_stack = LayerStack(
                layers={
                    "a_si_pillar": LayerLevel(
                        layer=PILLAR_LAYER,
                        thickness=pillar_h,
                        zmin=0.0,          # cylinder sits on the substrate top at z = 0
                        material="a_si",
                    )
                }
            )
            pillar_geometry = get_meep_geometry_from_component(
                component=unit_cell,
                layer_stack=layer_stack,
                material_name_to_meep={"a_si": n_a_si},
                wavelength=wavelength,
            )
            t0 = time.perf_counter()
            print(
                f"[{idx + 1}/{len(radius_list)}] radius={radius:.3f} um: "
                "running bare reference...",
                flush=True,
            )
            reference_field = run_unit_cell([])
            print(
                f"[{idx + 1}/{len(radius_list)}] radius={radius:.3f} um: "
                "running pillar...",
                flush=True,
            )
            pillar_field = run_unit_cell(pillar_geometry)
            sim_dict[radius] = (reference_field,pillar_field)
            print(
                f"Elapsed time for radius={radius:.3f} um: "
                f"{time.perf_counter() - t0:.1f} s",
                flush=True,
            )
        else:
            PILLAR_LAYER = (1, 0)  # circle_unit_cell() draws its polygon on this layer
            unit_cell = circle_unit_cell(radius)
            layer_stack = LayerStack(
                layers={
                    "a_si_pillar": LayerLevel(
                        layer=PILLAR_LAYER,
                        thickness=pillar_h,
                        zmin=0.0,          # cylinder sits on the substrate top at z = 0
                        material="a_si",
                    )
                }
            )
            pillar_geometry = get_meep_geometry_from_component(
                component=unit_cell,
                layer_stack=layer_stack,
                material_name_to_meep={"a_si": n_a_si},
                wavelength=wavelength,
            )
            t0 = time.perf_counter()
            print(
                f"[{idx + 1}/{len(radius_list)}] radius={radius:.3f} um: "
                "running pillar...",
                flush=True,
            )
            pillar_field = run_unit_cell(pillar_geometry)
            sim_dict[radius] = (reference_field, pillar_field)
            print(
                f"Elapsed time for radius={radius:.3f} um: "
                f"{time.perf_counter() - t0:.1f} s",
                flush=True,
            )
    print(
        f"Total sweep time: {time.perf_counter() - sweep_t0:.1f} s",
        flush=True,
    )

    # %% Plot Transmission and Phase vs Radius
    radii = []
    transmissions = []
    phases = []
    for radius, (reference_field, pillar_field) in sim_dict.items():
        transmission_coefficient = pillar_field / reference_field
        power_transmission = abs(transmission_coefficient) ** 2
        phase = np.degrees(np.angle(transmission_coefficient) % (2 * np.pi))
        radii.append(radius)
        transmissions.append(power_transmission)
        phases.append(phase)

    fig, ax_phase = plt.subplots()
    phase_points = ax_phase.scatter(
        radii, phases, marker="o", color="tab:orange", label="Phase"
    )
    ax_phase.set_xlabel("Pillar radius (um)")
    ax_phase.set_ylabel("Phase (degrees)", color="tab:orange")
    ax_phase.tick_params(axis="y", labelcolor="tab:orange")
    ax_phase.set_ylim(0, 360)

    ax_transmission = ax_phase.twinx()
    transmission_points = ax_transmission.scatter(
        radii, transmissions, marker="x", color="tab:blue", label="Power transmission"
    )
    ax_transmission.set_ylabel("Power transmission", color="tab:blue")
    ax_transmission.tick_params(axis="y", labelcolor="tab:blue")

    ax_transmission.set_title("Transmission and phase versus pillar radius")
    ax_phase.legend(handles=[phase_points, transmission_points])
    fig.tight_layout()
    plt.show()

# %% Doucmentation for future work
source_width_x = 1
source_width_y = 1
def elliptical_gaussian(p):
    return np.exp(-(p.x / wx)**2 - (p.y / wy)**2)

mp.Source(
    mp.GaussianSource(fcen, fwidth=0.05 * fcen),
    component=mp.Ex,
    center=mp.Vector3(0, 0, source_z),
    size=mp.Vector3(source_width_x, source_width_y, 0),
    amp_func=elliptical_gaussian,
)
#Also change kpoint in sim, and 
\(\hat{k}=(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ -\cos\theta)\)
k = mp.vector3(np.sin(theta)*np.cos)
# %%
