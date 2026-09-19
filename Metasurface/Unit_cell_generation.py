# %% Imports
import gdsfactory as gf
import os
from pathlib import Path
from collections.abc import Callable, Iterable
from inspect import Parameter, signature
from itertools import product
# ## Structures
# %% Different Geometries

def c_shape_unit_cell(outer_length,arm_w,rotation):
    c = gf.Component()
    p1 = c.add_polygon([(-outer_length/2, -outer_length/2), (-outer_length/2, outer_length/2), (outer_length/2, outer_length/2), (outer_length/2, -outer_length/2)], layer=(1, 0))
    p2 = c.add_polygon([(outer_length/2,outer_length/2 -arm_w), (outer_length/2,-outer_length/2 + arm_w), (-outer_length/2 + arm_w, -outer_length/2 + arm_w), (-outer_length/2 + arm_w,outer_length/2 -arm_w)], layer=(2, 0))
    r1 = c.get_region(layer=(1, 0))
    r2 = c.get_region(layer=(2, 0))
    r3 = r1 - r2
    c = gf.Component()
    c.add_polygon(r3, layer=(2, 0))
    c.rotate(rotation)
    return c


def circle_unit_cell(radius):
    return gf.components.circle(radius=radius, layer=(1, 0))
def nano_brick_cell(lx,ly):
    c = gf.Component()
    c.add_polygon([(-lx/2, -ly/2), (-lx/2, ly/2), (lx/2, ly/2), (lx/2, -ly/2)], layer=(1, 0))
    return c

if __name__ == "__main__":
    # Optional standalone layout export; keep imports side-effect free so these
    # factories can be used directly by simulation scripts.
    gf.gpdk.PDK.activate()
    example_cell = c_shape_unit_cell(20, 2, 45)
    example_cell.write_gds("demo_unit_cell.gds")
# %% Make gds library for actual project
@gf.cell(overwrite_existing=True)
def elliptical_pillar_component(
    r_x: float, r_y: float, theta: float
) -> gf.Component:
    """Create one named elliptical-pillar layout cell.

    Recreating an identical parameter set replaces the existing live cell rather
    than retaining another cell with the same name.
    """
    pillar = gf.Component()
    ellipse = pillar.add_ref(
        gf.components.ellipse(radii=(r_x, r_y), layer=(1, 0))
    )
    ellipse.drotate(theta)
    return pillar


def elliptical_pillar_gds(r_x: float, r_y: float, theta: float, directory: Path) -> Path:
    """Write one rotated elliptical pillar GDS to ``directory``.

    Args:
        r_x: Ellipse radius along its local x axis, in layout units.
        r_y: Ellipse radius along its local y axis, in layout units.
        theta: Counter-clockwise rotation in degrees.
        directory: Existing directory that will receive the generated ``.gds`` file.

    Returns:
        Path to the written GDS file (not the in-memory gdsfactory component).
    """
    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path object")
    if not directory.is_dir():
        raise NotADirectoryError(f"GDS output directory does not exist: {directory}")

    pillar = elliptical_pillar_component(r_x=r_x, r_y=r_y, theta=theta)

    gds_path = directory / f"elliptical_pillar_rx_{r_x}_ry_{r_y}_theta_{theta}.gds"
    # The filename is the geometry identifier for this library.  Re-generating
    # the same geometry intentionally replaces its previous GDS export.
    if gds_path.exists():
        gds_path.unlink()
    pillar.write_gds(gds_path)
    return gds_path
# %% Generic gds library sweep saver function
def generate_unit_cell_gds_lib(
    pillar_geometry: Callable[..., Path],
    directory: Path,
    **parameter_values: Iterable[object],
) -> list[Path]:
    """Write a GDS for every combination of a pillar exporter's parameters.

    ``pillar_geometry`` must have a keyword parameter named ``directory``.
    Supply an iterable for each geometry parameter that should be swept.

    Example:
        generate_unit_cell_gds_lib(
            elliptical_pillar_gds,
            gds_directory,
            r_x=[0.10, 0.15, 0.20],
            r_y=[0.10, 0.15],
            theta=[0, 45, 90],
        )
    """
    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path object")

    function_parameters = signature(pillar_geometry).parameters
    if "directory" not in function_parameters:
        raise ValueError("pillar_geometry must define a 'directory' parameter")

    geometry_parameter_names = [
        name
        for name, parameter in function_parameters.items()
        if name != "directory"
        and parameter.kind
        in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    ]
    unknown_parameters = set(parameter_values) - set(geometry_parameter_names)
    if unknown_parameters:
        names = ", ".join(sorted(unknown_parameters))
        raise ValueError(f"Unknown parameter sweep(s) for {pillar_geometry.__name__}: {names}")

    missing_parameters = [
        name
        for name in geometry_parameter_names
        if function_parameters[name].default is Parameter.empty
        and name not in parameter_values
    ]
    if missing_parameters:
        names = ", ".join(missing_parameters)
        raise ValueError(f"Missing parameter sweep(s): {names}")

    parameter_names = list(parameter_values)
    parameter_sweeps = [list(parameter_values[name]) for name in parameter_names]
    gds_files = []
    seen_parameter_combinations: set[tuple[object, ...]] = set()
    for values in product(*parameter_sweeps):
        # Repeated parameter combinations identify the same output filename.
        # Generate it once; that first export replaces any existing file on disk.
        if values in seen_parameter_combinations:
            continue
        seen_parameter_combinations.add(values)
        geometry_kwargs = dict(zip(parameter_names, values, strict=True))
        gds_files.append(pillar_geometry(directory=directory, **geometry_kwargs))

    return gds_files
# %%
