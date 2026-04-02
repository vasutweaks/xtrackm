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

with open("common_passes_l2pVSxtrack_selected.pkl", "rb") as f:
    common = pickle.load(f)

colors = colormaps["tab20"](np.linspace(0, 1, len(common)))
std_arrays = []
i=0
for pass1 in common:
    f = f"aviso_l2p_s3a_dist_time_{pass1}_asn.nc"
    with xr.open_dataset(f, decode_times=False) as ds:
        track_number = ds.attrs["pass"]
        lons_track = ds.lons.values
        lats_track = ds.lats.values
        ln = len(lons_track)
        sla_track = ds.sla
        middle_lon = lons_track[ln // 2]
        middle_lat = lats_track[ln // 2]
        mid_lon_i = np.argmin(np.abs(lons_track - middle_lon))
        mid_lat_i = np.argmin(np.abs(lats_track - middle_lat))
        lonm = middle_lon
        latm = middle_lat
        a1 = local_angle(lons_track, lats_track, mid_lat_i, ioff=3, degrees=False)
        a1_d = math.degrees(a1)
        std = sla_track.std(dim="time")
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
        ax2.plot(std.x.values, std.values, color=color, linewidth=1, label=str(track_number))
        std_arrays.append(std)
        i+=1

# std_mean = xr.concat(std_arrays, dim="track").mean(dim="track", skipna=True)
# save std mean
# ds_out = xr.Dataset({"std": std_mean})
# ds_out.to_netcdf(f"std_mean_{sat}_l2p.nc")
std_all = xr.concat(std_arrays, dim="track")
std_mean = std_all.mean(dim="track", skipna=True)

ds_out2 = xr.Dataset({"std_all": std_all})
ds_out2.to_netcdf(f"std_all_{sat}_l2p.nc")
ds_out1 = xr.Dataset({"std": std_mean})
ds_out1.to_netcdf(f"std_mean_{sat}_l2p.nc")
ax2.plot(
    std_mean.x.values,
    std_mean.values,
    color="dimgray",
    linewidth=3,
    label="Mean",
)

ax2.set_xlabel("Distance (km)")
ax2.set_ylabel("STD (m)")
ax2.axvline(x=0, color="k", linewidth=0.5, linestyle="--")
ax2.text(
    0.02,
    0.5,
    "coast",
    transform=ax2.transAxes,
    fontsize=10,
    fontweight="bold",
    color="r",
    rotation=90,
)
ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)
ax2.legend(title="Track", ncol=4, fontsize=7, title_fontsize=8)

plt.tight_layout()
plt.savefig(f"std_mean_{sat}_l2p.png", dpi=300)
plt.show()
plt.close("all")

dse.close()
