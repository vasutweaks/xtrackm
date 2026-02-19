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

ds_xtr = xr.open_dataset("../computed/std_mean_S3A_xtrack.nc")
ds_l2p = xr.open_dataset("/home/srinivasu/allData/aviso_l2p/s3a/std_mean_S3A_l2p.nc")

std_xtr = ds_xtr["std"]
std_l2p = ds_l2p["std"]

dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo5.cdf")
files = sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc"))

track_data = []
x_common = xr.DataArray(np.arange(0, 1_300_001, 6_000), dims=["x"],
  attrs={"units": "m"})  # 0..1300 km, 1 km step

# open common pickle
with open("common_passes_l2pVSxtrack.pkl", "rb") as f:
    common = pickle.load(f)

for pass1 in common:
    pass2 = pass1[1:]
    f = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{pass2}.nc"
    with xr.open_dataset(f, decode_times=False) as ds:
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
        std = sla_track.std(dim="time")
        track_data.append((track_number, lons_track, lats_track, lonm, latm, a1_d, std))

# We need to know the number of tracks that matched the filter criteria
# i.e east coast region and ascending tracks beforehand so that we can
# set colormap

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

colors = colormaps["tab20"](np.linspace(0, 1, len(track_data)))

for i, (track_number, lons_track, lats_track, lonm, latm, a1_d, std) in enumerate(track_data):
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
        )

# ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))

ax2.plot(std_xtr, label="xtrack std", color="red")
ax2.plot(std_l2p, label="l2p std", color="blue")
ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)
ax2.set_xlabel("distance along track (km)")
ax2.set_ylabel("std")
ax2.set_title("std_mean_xtrack vs std_mean_l2p")
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
plt.savefig(f"compare_std_mean_l2pVSxtrack_map_{sat}.png", dpi=300)
ln_sf(f"compare_std_mean_l2pVSxtrack_map_{sat}.png", "fig01a.png")
plt.show()
plt.close("all")

dse.close()
