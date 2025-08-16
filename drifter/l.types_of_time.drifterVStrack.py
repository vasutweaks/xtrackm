import os
import sys

import numpy as np
import xarray as xr
from tools_xtrackm import *
import sys

import xarray as xr

def grep_netcdf_meta(file: str, query: str):
    """
    Search for a string in variable names, variable attributes,
    coordinate names, and global attributes of an xarray dataset.
    Parameters
    ----------
    ds : xr.Dataset
        The opened xarray dataset.
    query : str
        The search string (case-insensitive).
    """
    ds = xr.open_dataset(file, decode_cf=False)
    q = query.lower()
    # Search variable names
    print("variable names --------------")
    for var in ds.data_vars:
        if q in var.lower():
            print(var)
    # Search variable attributes
    print("variable attributes --------------")
    for var in ds.data_vars:
        for attr, value in ds[var].attrs.items():
            if q in attr.lower() or q in str(value).lower():
                print(var, attr, value)
    # Search coordinate names
    print("coordinate names and attributes --------------")
    for coord in ds.coords:
        if q in coord.lower():
            print(coord)
        for attr, value in ds[coord].attrs.items():
            if q in attr.lower() or q in str(value).lower():
                print(coord, attr, value)
    # Search global attributes
    print("global attributes --------------")
    for attr, value in ds.attrs.items():
        if q in attr.lower() or q in str(value).lower():
            print(attr, value)

# drifter_data_loc = f"/home/srinivasu/allData/drifter1/"
# track_data_loc = f"/home/srinivasu/xtrackm/data/"
# sat = "GFO"
# fd = f"{drifter_data_loc}/netcdf_15001_current/track_reg/drifter_6h_133666.nc"
# ft = f"{track_data_loc}/{sat}/ctoh.sla.ref.GFO.nindian.238.nc"

fd = "drifter_6h_133666.nc"
ft = "ctoh.sla.ref.GFO.nindian.238.nc"

# decode_cd=False is the key here as xarray interfers with raw meta data of the file
# ds_d = xr.open_dataset(fd, drop_variables=["WMO"], decode_cf=False)
# ds_t = xr.open_dataset(ft, decode_cf=False)
ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
ds_t = xr.open_dataset(ft, decode_times=False)
print(ds_d)
print(ds_t)

type_d = type(ds_d.time.values[0,0])
type_t = type(ds_t.time.values[0,0])

print(f"type of drifter time is {type_d}")
print(f"type of track time is {type_t}")

print(f"time units of drifter *******************")
grep_netcdf_meta(fd, "since")
print(f"time units of track *******************")
grep_netcdf_meta(ft, "since")

drift_tsta_o1, drift_tend_o1 = (
    ds_d.start_date.values[0],
    ds_d.end_date.values[0],
)
print(f"{type(drift_tsta_o1)}")
drift_tsta_o, drift_tend_o = n64todatetime1(drift_tsta_o1), n64todatetime1(drift_tend_o1)
# print(type(drift_tsta_o))

sla = track_dist_time_asn(ds_t, var_str="sla", units_in="m")
print(type(sla.time.values[0]))

times_drift = ds_d.time.isel(traj=0).values
ve_da = drifter_time_asn(ds_d, var_str="ve")
vn_da = drifter_time_asn(ds_d, var_str="vn")
lons_da = drifter_time_asn(ds_d, var_str="longitude")
lats_da = drifter_time_asn(ds_d, var_str="latitude")
sys.exit(0)
