import glob
import math

import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.ticker import FuncFormatter
from tools_xtrackm import *
import pickle

L2P_PATH = "/home/srinivasu/allData/aviso_l2p/s3a/"
sat = "S3A"
EAST_COAST_REG = (78.0, 95, 0, 25.4)

x_common = xr.DataArray(np.arange(0, 1_300_001, 6_000), dims=["x"],
  attrs={"units": "m"})  # 0..1300 km, 1 km step

with open("common_passes_l2pVSxtrack_selected.pkl", "rb") as f:
    common = pickle.load(f)

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
std_arrays = []
track_numbers = []
for pass1 in common:
    pass2 = pass1[1:]
    pass3 = pass1.lstrip("0")
    f = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{pass2}.nc"
    ds = xr.open_dataset(f, decode_times=False)
    lon_eq = ds.lon.values[0]
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lon_equat = lons_track[0]
    lon_coast = lons_track[-1]
    lat_equat = lats_track[0]
    lat_coast = lats_track[-1]
    a1 = math.atan2(lat_coast - lat_equat, lon_coast - lon_equat)
    a1_d = math.degrees(a1)
    track_number = int(ds.Pass.values if hasattr(ds.Pass, "values") else ds.Pass)
    lonm = lons_track.mean()
    latm = lats_track.mean()
    sla_track = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
    sla_track = sla_track.interp(x=x_common)
    std = sla_track.count(dim="time")
    print(f"{sat} {track_number} -------------------")
    ax1.scatter(std.x.values, std.values, color='cyan', marker='.', s=1)
    std_arrays.append(std)
    track_numbers.append(pass3)

track_coord = xr.DataArray(track_numbers, dims=["track"], name="track")
std_all = xr.concat(std_arrays, dim=track_coord)
std_mean = std_all.mean(dim="track", skipna=True)

ax1.plot(
    std_mean.x.values,
    std_mean.values,
    color="b",
    linewidth=3,
    label="XTRACK",
)

ax1.set_xlabel("Distance (km)")
ax1.set_ylabel("STD (m)")
ax1.axvline(x=0, color="k", linewidth=0.5, linestyle="--")
ax1.text(
    0.02,
    0.5,
    "coast",
    transform=ax1.transAxes,
    fontsize=10,
    fontweight="bold",
    color="r",
    rotation=90,
)
ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)
ax1.legend(title="Track", ncol=4, fontsize=7, title_fontsize=8)

std_arrays = []
track_numbers = []
for pass1 in common:
    f = f"{L2P_PATH}/aviso_l2p_s3a_dist_time_{pass1}_asn.nc"
    with xr.open_dataset(f, decode_times=False) as ds:
        track_number = ds.attrs["pass"]
        lons_track = ds.lons.values
        lats_track = ds.lats.values
        ln = len(lons_track)
        sla_track = ds.sla
        std = sla_track.count(dim="time")
        print(f"{sat} {track_number} -------------------")
        ax1.scatter(std.x.values, std.values, color='gray', marker='.', s=1)
        std_arrays.append(std)
        track_numbers.append(pass3)

track_coord = xr.DataArray(track_numbers, dims=["track"], name="track")
std_all = xr.concat(std_arrays, dim=track_coord)
std_mean = std_all.mean(dim="track", skipna=True)
ax1.plot(
    std_mean.x.values,
    std_mean.values,
    color="dimgray",
    linewidth=3,
    label="L2P",
)

ax1.set_xlabel("Distance (km)")
ax1.set_ylabel("STD (m)")
ax1.axvline(x=0, color="k", linewidth=0.5, linestyle="--")
ax1.text(
    0.02,
    0.5,
    "coast",
    transform=ax1.transAxes,
    fontsize=10,
    fontweight="bold",
    color="r",
    rotation=90,
)
ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)
ax1.legend(title="Track", ncol=4, fontsize=7, title_fontsize=8)

plt.tight_layout()
plt.show()
plt.close("all")

