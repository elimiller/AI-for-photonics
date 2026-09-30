# %% Imports and legacy-library location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATA_PATH = (
    PROJECT_ROOT
    / "Metasurface"
    / "Unit_cell_Libraries"
    / "Ellipse Pillar SiN on SiO2 532 nm KAIST"
    / "Library Data"
    / "Copy of ellipse_pillar_library_legacy_data.csv"
)


# %% Load and plot legacy rotation sweep
def load_legacy_ellipse_data(csv_path: Path = LEGACY_DATA_PATH) -> pd.DataFrame:
    """Load the copied legacy ellipse-pillar library with numeric columns."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Legacy library data was not found: {csv_path}")

    data = pd.read_csv(csv_path)
    numeric_columns = (
        "radius_x_um",
        "radius_y_um",
        "rotation_deg",
        "pillar_height_um",
        "period_um",
        "incident_angle_deg",
        "zero_order_tm_phase_deg",
        "total_transmitted_power_ratio",
    )
    data.loc[:, list(numeric_columns)] = data.loc[:, list(numeric_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    return data.dropna(subset=list(numeric_columns))


def plot_rotation_response(
    radius_x_um: float = 0.11,
    radius_y_um: float = 0.06,
    pillar_height_um: float = 0.7,
    period_um: float = 0.3,
    incident_angle_deg: float = 0.0,
    csv_path: Path = LEGACY_DATA_PATH,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot TM phase and total transmission against ellipse rotation.

    The defaults select a complete 0°–150° legacy rotation sweep.  Change the
    geometry and condition arguments to plot another group in the same CSV.
    """
    data = load_legacy_ellipse_data(csv_path)
    selected = data.loc[
        np.isclose(data["radius_x_um"], radius_x_um)
        & np.isclose(data["radius_y_um"], radius_y_um)
        & np.isclose(data["pillar_height_um"], pillar_height_um)
        & np.isclose(data["period_um"], period_um)
        & np.isclose(data["incident_angle_deg"], incident_angle_deg)
    ].sort_values("rotation_deg")

    if selected.empty:
        raise ValueError(
            "No legacy records match "
            f"r_x={radius_x_um:g} µm, r_y={radius_y_um:g} µm, "
            f"h={pillar_height_um:g} µm, p={period_um:g} µm, and "
            f"incidence={incident_angle_deg:g}°"
        )

    rotations = selected["rotation_deg"].to_numpy()
    phase = selected["zero_order_tm_phase_deg"].to_numpy() % 360.0
    transmission = selected["total_transmitted_power_ratio"].to_numpy()

    fig, ax_phase = plt.subplots(figsize=(7.5, 4.8))
    phase_points = ax_phase.plot(
        rotations, phase, "o-", color="tab:orange", label="TM phase"
    )[0]
    ax_phase.set(
        xlabel="Ellipse rotation (degrees)",
        ylabel="TM phase (degrees)",
        ylim=(0, 360),
        xticks=rotations,
        title=(
            "Legacy ellipse-pillar rotation sweep: "
            f"$r_x$={radius_x_um:g} µm, $r_y$={radius_y_um:g} µm, "
            f"h={pillar_height_um:g} µm, p={period_um:g} µm"
        ),
    )
    ax_phase.tick_params(axis="y", labelcolor="tab:orange")
    ax_phase.grid(axis="x", alpha=0.3)

    ax_transmission = ax_phase.twinx()
    transmission_points = ax_transmission.plot(
        rotations, transmission, "x-", color="tab:blue", label="Transmission"
    )[0]
    ax_transmission.set_ylabel("Total transmitted power", color="tab:blue")
    ax_transmission.tick_params(axis="y", labelcolor="tab:blue")
    ax_phase.legend(
        [phase_points, transmission_points],
        ["TM phase", "Total transmission"],
        loc="best",
    )
    fig.tight_layout()
    plt.show()
    return fig, (ax_phase, ax_transmission)


def plot_phase_vs_aspect_ratio(
    pillar_height_um: float = 0.7,
    period_um: float = 0.3,
    incident_angle_deg: float = 0.0,
    csv_path: Path = LEGACY_DATA_PATH,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot TM phase against ``r_x / r_y``, colored by ellipse rotation."""
    data = load_legacy_ellipse_data(csv_path)
    selected = data.loc[
        np.isclose(data["pillar_height_um"], pillar_height_um)
        & np.isclose(data["period_um"], period_um)
        & np.isclose(data["incident_angle_deg"], incident_angle_deg)
    ].copy()
    if selected.empty:
        raise ValueError(
            "No legacy records match "
            f"h={pillar_height_um:g} µm, p={period_um:g} µm, and "
            f"incidence={incident_angle_deg:g}°."
        )

    selected["aspect_ratio"] = (
        selected["radius_x_um"] / selected["radius_y_um"]
    )
    selected["phase_deg"] = selected["zero_order_tm_phase_deg"] % 360.0

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    rotations = np.sort(selected["rotation_deg"].unique())
    colors = plt.get_cmap("tab10", len(rotations))
    for color_index, rotation in enumerate(rotations):
        rotation_data = selected.loc[
            np.isclose(selected["rotation_deg"], rotation)
        ].sort_values("aspect_ratio")
        ax.scatter(
            rotation_data["aspect_ratio"],
            rotation_data["phase_deg"],
            color=colors(color_index),
            label=f"{rotation:g}°",
        )

    ax.set(
        xlabel=r"Aspect ratio $r_x / r_y$",
        ylabel="TM phase (degrees)",
        ylim=(0, 360),
        title=(
            "Legacy ellipse-pillar phase response: "
            f"h={pillar_height_um:g} µm, p={period_um:g} µm"
        ),
    )
    ax.grid(alpha=0.3)
    ax.legend(title="Rotation", ncols=2)
    fig.tight_layout()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    plot_phase_vs_aspect_ratio()

# %%
