import math

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tools_xtrackm import *

dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
ekman_dir = "/home/srinivasu/slnew/xtrackd/computed/"
out_dir = f"/home/srinivasu/xtrackm/ekman/ekman_at_intersections/"

ds_ekman = xr.open_dataset(f"{ekman_dir}/full_series_ekman_uv_ccmp3.0_cdo.nc", chunks={"time": 365, "lat": 50, "lon": 50})
print(ds_ekman)
u_ekman = ds_ekman.ekman_u
v_ekman = ds_ekman.ekman_v

df_all_list = [pd.read_csv(f"tracks_intersections_{sat}_1.csv") for sat in sats_new]
df_all = pd.concat(df_all_list, ignore_index=True)

track_self = df_all["track_self"].apply(lambda x: str(x)).values
track_other = df_all["track_other"].apply(lambda x: str(x)).values
lons_inter = df_all["lons_inter"].values
lats_inter = df_all["lats_inter"].values
x_from_coast_self = df_all["x_from_coast_self"].values
x_from_coast_other = df_all["x_from_coast_other"].values
angle_acute = df_all["angle_acute"].values
angle_obtuse = df_all["angle_obtuse"].values
sats = df_all["sat"].values

ln = len(track_self)

sat_here = "TP+J1+J2+J3+S6A"
k = 0
for i in range(ln):
    track_number_self = track_self[i]
    track_number_other = track_other[i]
    lon1 = lons_inter[i]
    lat1 = lats_inter[i]
    sat1 = sats[i]
    if sat1 != sat_here:
    # if sat1 != "S3B":
        continue
    k = k + 1
    print(f"{k} {track_number_self} {track_number_other}")
    # extrack ekman at lon1, lat1
    u_at = u_ekman.sel(lon=lon1, lat=lat1, method="nearest")
    v_at = v_ekman.sel(lon=lon1, lat=lat1, method="nearest")
    ds_out = xr.Dataset({"u": u_at, "v": v_at})
    filename = f"{out_dir}/ekman_at_intersection_{sat1}_{track_number_self}_{track_number_other}.nc"
    print(filename)
    ds_out.to_netcdf(filename)
