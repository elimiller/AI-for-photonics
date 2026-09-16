# %% Imports
import gdsfactory as gf
# ## Structures
# %% Create a simple polygon

# gf.gpdk.PDK.activate()
# # Create a blank component (essentially an empty GDS cell with some special features).
# c = gf.Component()
# p1 = c.add_polygon([(-8, -6), (6, 8), (7, 17), (9, 5)], layer=(1, 0))
# c.write_gds("demo.gds")  # Write it to a GDS file. You can open it in klayout.
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

# %% Klayout visualize
gf.gpdk.PDK.activate()
Example_cell = c_shape_unit_cell(20, 2, 45)
Example_cell.write_gds("demo_unit_cell.gds")  # Write it to a GDS file. You can open it in klayout.
# %%
