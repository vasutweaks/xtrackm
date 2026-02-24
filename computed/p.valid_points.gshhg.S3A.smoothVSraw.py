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
COAST_NEAR_M = 50_000

frac1 = "0.05"
# smooth_dir = f"sla_loess_{frac1}a/"
dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo5.cdf")
fig = plt.figure(figsize=(10, 9))
ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

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


def valid_percent(da):
    counts = da.count(dim="time")
    count_perc = 100 * counts / da.sizes["time"]
    return xr.where(count_perc == 0, np.nan, count_perc)

# open common pickle
with open("common_passes_l2pVSxtrack.pkl", "rb") as f:
    common = pickle.load(f)

colors = colormaps["tab20"](np.linspace(0, 1, len(common)))
counts_raw_arrays = []
counts_smooth_arrays = []
coast_raw_track = []
coast_smooth_track = []
i=0
for pass1 in common:
    pass2 = pass1[1:]
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
    sla_asn = track_dist_time_asn_gshhg(ds, var_str="sla", units_in="m")
    sla_track = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
    sla_smooth = track_smooth_loess_statsmodels_x(sla_track, frac=0.05)
    gshhg_da = dist_asn_gshhg(ds, var_str="dist_to_coast_gshhg")
    gshhg_da_interp = gshhg_da.interp(x=x_common)
    count_raw_interp = valid_percent(sla_asn).interp(x=x_common)
    count_smooth_interp = valid_percent(sla_smooth).interp(x=x_common)

    coast_mask = np.abs(gshhg_da_interp.values) <= COAST_NEAR_M
    raw_near = np.nanmean(count_raw_interp.values[coast_mask]) if coast_mask.any() else np.nan
    smooth_near = np.nanmean(count_smooth_interp.values[coast_mask]) if coast_mask.any() else np.nan

    ax2.scatter(
        gshhg_da_interp.values,
        count_raw_interp.values,
        s=3,
        color=color,
        alpha=0.5,
    )
    ax2.scatter(
        gshhg_da_interp.values,
        count_smooth_interp.values,
        s=3,
        color=color,
        alpha=0.5,
        marker="x",
    )
    print(
        f"{sat} {track_number}: near-coast valid% raw={raw_near:.2f}, "
        f"smooth={smooth_near:.2f}, delta={smooth_near - raw_near:.2f}"
    )

    counts_raw_arrays.append(count_raw_interp)
    counts_smooth_arrays.append(count_smooth_interp)
    coast_raw_track.append(raw_near)
    coast_smooth_track.append(smooth_near)
    i+=1

counts_raw_mean = xr.concat(counts_raw_arrays, dim="track").mean(dim="track", skipna=True)
counts_smooth_mean = xr.concat(counts_smooth_arrays, dim="track").mean(dim="track", skipna=True)
counts_delta_mean = counts_smooth_mean - counts_raw_mean

ds_out = xr.Dataset({
    "counts_raw": counts_raw_mean,
    "counts_smooth": counts_smooth_mean,
    "counts_delta": counts_delta_mean,
})
ds_out.to_netcdf(f"counts_mean_S3A_xtrack_smoothVSraw.nc")

ax2.plot(
    counts_raw_mean.x.values,
    counts_raw_mean.values,
    color="dimgray",
    linewidth=3,
    label="Mean raw",
)
ax2.plot(
    counts_smooth_mean.x.values,
    counts_smooth_mean.values,
    color="tab:blue",
    linewidth=3,
    label=f"Mean smooth (frac={frac1})",
)

ax2.set_xlabel("Distance (km)")
ax2.set_ylabel("% valid points")
ax2.axvline(x=0, color="k", linewidth=0.5, linestyle="--")
ax2.axvspan(-COAST_NEAR_M, COAST_NEAR_M, color="tab:red", alpha=0.08)
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
ax2.set_xlim(-10e3, 150e3)
ax2.set_ylim(0, 105)
ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)
ax2.legend(fontsize=8, ncol=2)

ax3.plot(
    counts_delta_mean.x.values,
    counts_delta_mean.values,
    color="tab:green",
    linewidth=2,
)
ax3.axhline(y=0, color="k", linewidth=0.8, linestyle="--")
ax3.axvline(x=0, color="k", linewidth=0.5, linestyle="--")
ax3.axvspan(-COAST_NEAR_M, COAST_NEAR_M, color="tab:red", alpha=0.08)
ax3.set_xlabel("Distance (km)")
ax3.set_ylabel("Smooth - Raw\n(% points)")
ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
ax3.set_xlim(-10e3, 150e3)
ax3.grid(alpha=0.3, linestyle="--", linewidth=0.5)

coast_raw_mean = float(np.nanmean(coast_raw_track))
coast_smooth_mean = float(np.nanmean(coast_smooth_track))
coast_delta_mean = coast_smooth_mean - coast_raw_mean
ax3.text(
    0.01,
    0.95,
    (
        f"|dist_to_coast| <= {COAST_NEAR_M/1000:.0f} km: "
        f"raw={coast_raw_mean:.2f}%, smooth={coast_smooth_mean:.2f}%, "
        f"delta={coast_delta_mean:.2f}%"
    ),
    transform=ax3.transAxes,
    fontsize=9,
    va="top",
)

print(
    f"ALL TRACKS near-coast (|dist|<={COAST_NEAR_M/1000:.0f} km): "
    f"raw={coast_raw_mean:.2f}%, smooth={coast_smooth_mean:.2f}%, "
    f"delta={coast_delta_mean:.2f}%"
)

plt.tight_layout()
plt.show()
plt.close("all")

dse.close()
