import glob
import math

import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.ticker import FuncFormatter
from tools_xtrackm import *
import pickle

sat = "S3A"
cmap = "twilight_shifted"
EAST_COAST_REG = (78.0, 95, 0, 25.4)

ds_xtr = xr.open_dataset("count1_mean_S3A_xtrack.nc")
ds_l2p = xr.open_dataset("/home/srinivasu/allData/aviso_l2p/s3a/count1_mean_S3A_l2p.nc")

count1_xtr = ds_xtr["count1"]
count1_l2p = ds_l2p["count1"]

dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo5.cdf")
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(2, 1, 2)

dse.ROSE.sel(ETOPO05_X=slice(*NIO[:2])).sel(ETOPO05_Y=slice(*NIO[2:])).plot(
    ax=ax1, add_colorbar=False, add_labels=False, cmap=cmap
)
decorate_axis(ax1, "", *NIO, step=8)
ax1.text(
    0.45,
    0.80,
    f"MISSION: {sat}",
    transform=ax1.transAxes,
    fontsize=10,
    fontweight="bold",
    color="b",
)

with open("common_passes_l2pVSxtrack.pkl", "rb") as f:
    common = pickle.load(f)

colors = colormaps["tab20"](np.linspace(0, 1, len(common)))
i = 0
for pass1 in common:
    pass2 = pass1[1:]
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

    color = colors[i]
    print(f"{sat} {track_number} -------------------")
    ax1.scatter(lons_track, lats_track, s=2, color=color)
    if is_within_region(lonm, latm, *NIO):
        ax1.text(
            lonm,
            latm,
            s=str(track_number),
            fontsize=7,
            rotation=a1_d,
            color='k',
            ha='center',
            va='center',
            transform=ccrs.PlateCarree(),
        )
    i += 1

# ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))

ax2.plot(count1_xtr, label="xtrack count1", color="red")
ax2.plot(count1_l2p, label="l2p count1", color="blue")
ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)
ax2.set_xlabel("distance along track (km)")
ax2.set_xlim(-10, 125)
ax2.set_ylabel("count1")
ax2.set_title("valid points Xtrack VS l2p")
ax2.axvline(x=0, color="k", linewidth=0.5, linestyle="--")
ax2.text(
    0.02,
    0.5,
    "coast",
    transform=ax2.transAxes,
    fontsize=10,
    fontweight="bold",
    color="k",
    rotation=90,
)

plt.legend()
plt.tight_layout()
plt.show()
plt.close("all")

dse.close()
