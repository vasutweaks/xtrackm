import math

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tools_xtrackm import *

oscar_dir = "/home/srinivasu/allData/oscar_0.25/"
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
out_dir = f"/home/srinivasu/xtrackm/guv/oscar_at_intersections/"

ds_oscar_u = xr.open_dataset(f"{oscar_dir}/oscar_0.25_u_full_xr_concat.nc")
print(ds_oscar_u)
u_oscar = ds_oscar_u.u

ds_oscar_v = xr.open_dataset(f"{oscar_dir}/oscar_0.25_v_full_xr_concat.nc")
print(ds_oscar_v)
v_oscar = ds_oscar_v.v

df_all_list = []
for sat in sats_new:
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    df_all_list.append(df)
df_all = pd.concat(df_all_list, ignore_index=True)

track_self = df_all["track_self"].apply(lambda x: str(x)).values
track_other = df_all["track_other"].apply(lambda x: str(x)).values
lons_inter = df_all["lons_inter"].values
lats_inter = df_all["lats_inter"].values
x_from_coast_self = df_all["x_from_coast_self"].values
x_from_coast_other = df_all["x_from_coast_other"].values
angle_acute = df_all["angle_acute"].values
angle_obtuse = df_all["angle_obtuse"].values
sat = df_all["sat"].values

ln = len(track_self)

k = 0
for i in range(ln):
    track_number_self = track_self[i]
    track_number_other = track_other[i]
    lon1 = lons_inter[i]
    lat1 = lats_inter[i]
    sat1 = sat[i]
    if sat1 != "HY2B":
    # if sat1 != "S3B":
        continue
    k = k + 1
    print(f"{k} {track_number_self} {track_number_other}")
    # extrack oscar at lon1, lat1
    u_at = u_oscar.sel(lon=lon1, lat=lat1, method="nearest")
    v_at = v_oscar.sel(lon=lon1, lat=lat1, method="nearest")
    ds_out = xr.Dataset({"u": u_at, "v": v_at})
    filename = f"{out_dir}/oscar_at_intersection_{sat1}_{track_number_self}_{track_number_other}.nc"
    print(filename)
    ds_out.to_netcdf(filename)
