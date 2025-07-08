import xarray as xr
from namesm import *

sat = "GFO"
passn = "008"
file1 = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{passn}.nc"
ds = xr.open_dataset(file1,decode_times=False)
print(ds)

# Search for "QC" in all attributes
search_term = "Pass"

print("-------------------------------------------------")
# Global attributes
for k, v in ds.attrs.items():
    if search_term.lower() in str(v).lower() or search_term.lower() in k.lower():
        print(f"Global: {k} = {v}")

# Variable attributes and names
for var_name, var in ds.data_vars.items():
    if search_term.lower() in var_name.lower():
        print(f"Variable name: {var_name}")
    for k, v in var.attrs.items():
        if search_term.lower() in str(v).lower() or search_term.lower() in k.lower():
            print(f"Variable {var_name}.{k} = {v}")

# Coordinate attributes and names
for coord_name, coord in ds.coords.items():
    if search_term.lower() in coord_name.lower():
        print(f"Coordinate name: {coord_name}")
    for k, v in coord.attrs.items():
        if search_term.lower() in str(v).lower() or search_term.lower() in k.lower():
            print(f"Coordinate {coord_name}.{k} = {v}")
