# %% Imports
import gdsfactory as gf
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
# %%
