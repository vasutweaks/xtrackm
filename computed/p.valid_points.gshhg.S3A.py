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

x_common = xr.DataArray(np.arange(0, 1_300_001, 6_675), dims=["x"],
  attrs={"units": "m"})  # 0..1300 km, 1 km step

# open common pickle
with open("common_passes_l2pVSxtrack.pkl", "rb") as f:
    common = pickle.load(f)

colors = colormaps["tab20"](np.linspace(0, 1, len(common)))
counts_arrays = []
i=0
# exclude_tracks=["679"]
exclude_tracks = ["451", "565", "679", "23", "621"]
for pass1 in common:
    pass2 = pass1[1:]
    pass3 = pass1.lstrip('0')
    if pass3 in exclude_tracks:
        continue
    f = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{pass2}.nc"
    ds = xr.open_dataset(f, decode_times=False)
    lon_eq = ds.lon.values[0]
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    ln = len(lons_track)
    lon_equat = lons_track[0]
    lon_coast = lons_track[-1]
    lat_equat = lats_track[0]
    lat_coast = lats_track[-1]
    a1 = math.atan2(lat_coast - lat_equat, lon_coast - lon_equat)
    a1_d = math.degrees(a1)
    track_number = ds.Pass
    lonm = lons_track.mean()
    latm = lats_track.mean()
    color = colors[i]
    ax1.scatter(lons_track, lats_track, s=2, color=colors[i])
    if is_within_region(lonm, latm, *NIO):
        ax1.text(
            lonm,
            latm,
            s=str(track_number),
            fontsize=7,
            rotation=a1_d,
            color='k',
            ha="center",
            va="center",
            transform=ccrs.PlateCarree(),
        )
    # sla_track = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
    sla_asn = track_dist_time_asn_gshhg(ds, var_str="sla", units_in="m")
    # gshhg = ds.dist_to_coast_gshhg.values
    gshhg_da = dist_asn_gshhg(ds, var_str="dist_to_coast_gshhg")
    gshhg_da_interp = gshhg_da.interp(x=x_common)
    counts = sla_asn.count(dim="time")
    count_perc = 100 * counts / len(sla_asn.time)
    count_perc1 = xr.where(count_perc == 0, np.nan, count_perc)
    count_perc_interp = count_perc1.interp(x=x_common)
    count_da = xr.DataArray(count_perc_interp.values, coords={"x": gshhg_da_interp.x.values}, dims=["x"])
    ax2.scatter(gshhg_da_interp.values, count_perc_interp.values, s=1, label=pass2, color=color)
    print(f"{sat} {track_number} -------------------")
    # ax2.scatter(gshhg, count2.values, color=color, s=1, label=str(track_number))
    # ax2.scatter(
    #     count_perc_interp.x.values,
    #     count_perc_interp.values,
    #     marker=".",
    #     s=1.0,
    # )
    counts_arrays.append(count_perc_interp)
    i+=1

counts_mean = xr.concat(counts_arrays, dim="track").mean(dim="track", skipna=True)
ds_out = xr.Dataset({"counts": counts_mean})
ds_out.to_netcdf(f"counts_mean_S3A_xtrack.nc")
ax2.plot(
    counts_mean.x.values,
    counts_mean.values,
    color="dimgray",
    linewidth=3,
    label="Mean",
)

ax2.set_xlabel("Distance (km)")
ax2.set_ylabel("counts (m)")
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
ax2.set_xlim(-10e3, 300e3)
ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)
ax2.legend(title="Track", ncol=4, fontsize=7, title_fontsize=8)

plt.tight_layout()
plt.show()
plt.close("all")

dse.close()
