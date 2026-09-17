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
    c = gf.Component()
    c.add_circle(radius, layer=(1, 0))
    return c
# %% Klayout visualize
gf.gpdk.PDK.activate()
Example_cell = c_shape_unit_cell(20, 2, 45)
Example_cell.write_gds("demo_unit_cell.gds")  # Write it to a GDS file. You can open it in klayout.
# %%
