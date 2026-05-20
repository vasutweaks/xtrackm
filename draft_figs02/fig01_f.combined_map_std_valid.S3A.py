import math
import pickle

import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.ticker import FuncFormatter
from tools_xtrackm import *
from pathlib import Path
import sys

SAT = "S3A"
L2P_PATH = "/home/srinivasu/allData/aviso_l2p/s3a/"
EAST_COAST_REG = (78.0, 95, 0, 25.4)
cmap = "twilight_shifted"

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")

with open("common_passes_l2pVSxtrack_selected.pkl", "rb") as f:
    common = pickle.load(f)

trim_lim = 300
x_common = xr.DataArray(
    np.arange(trim_lim) * 6_675,
    dims=["x"],
    attrs={"units": "m"},
)

# ── figure layout: map left (2 rows), two line plots right ──
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.3], wspace=0.05, hspace=0.3)

ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
ax_std = fig.add_subplot(gs[1, 1])
ax_vp = fig.add_subplot(gs[0, 1])

# ════════════════════════════════════════════════════════════
# LEFT PANEL — track map
# ════════════════════════════════════════════════════════════
dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo5.cdf")
rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*EAST_COAST_REG[:2])).sel(
    ETOPO05_Y=slice(*EAST_COAST_REG[2:])
)
rose_cut.plot(ax=ax_map, add_colorbar=False, add_labels=False, cmap=cmap)
decorate_axis(ax_map, "", *EAST_COAST_REG, step=4)
ax_map.text(
    0.25,
    0.90,
    f"MISSION: {SAT}",
    transform=ax_map.transAxes,
    fontsize=10,
    fontweight="bold",
    color="b",
)

colors = colormaps["tab20"](np.linspace(0, 1, len(common)))
for i, pass1 in enumerate(common):
    pass2 = pass1[1:]
    fn = f"../data/{SAT}/ctoh.sla.ref.{SAT}.nindian.{pass2}.nc"
    with xr.open_dataset(fn, decode_times=False) as ds:
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        ln = len(lons_track)
        mid_i = ln // 2 + ln // 4
        lonm, latm = lons_track[mid_i], lats_track[mid_i]
        a1_d = math.degrees(
            local_angle(lons_track, lats_track, mid_i, ioff=3, degrees=False)
        )
        # ax_map.scatter(lons_track, lats_track, s=48, color="w")
        ax_map.scatter(lons_track, lats_track, s=8, color="k")
        # if is_within_region(lonm, latm, *NIO):
        #     ax_map.text(
        #         lonm,
        #         latm,
        #         s=str(track_number),
        #         fontsize=12,
        #         rotation=a1_d,
        #         color="r",
        #         ha="center",
        #         va="center",
        #     )
        #
        if is_within_region(lonm, latm, *NIO):
            ax_map.text(
                lonm,
                latm,
                s=str(track_number),
                fontsize=14,
                rotation=a1_d,
                color="w",
                ha="center",
                va="center",
            )
dse.close()

# ════════════════════════════════════════════════════════════
# RIGHT-TOP — SLA standard deviation vs distance from coast
# ════════════════════════════════════════════════════════════
std_arrays_xt, std_arrays_l2p = [], []
track_numbers_xt, track_numbers_l2p = [], []

for pass1 in common:
    pass2 = pass1[1:]
    pass3 = pass1.lstrip("0")
    fn = f"../data/{SAT}/ctoh.sla.ref.{SAT}.nindian.{pass2}.nc"
    ds = xr.open_dataset(fn, decode_times=False)
    sla_track = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
    sla_track = sla_track.interp(x=x_common)
    std = sla_track.std(dim="time")
    ax_std.scatter(std.x.values, std.values, color="cyan", marker=".", s=1)
    std_arrays_xt.append(std)
    track_numbers_xt.append(pass3)

tc = xr.DataArray(track_numbers_xt, dims=["track"], name="track")
std_mean_xt = xr.concat(std_arrays_xt, dim=tc).mean(dim="track", skipna=True)
ax_std.plot(
    std_mean_xt.x.values,
    std_mean_xt.values,
    color="b",
    linewidth=3,
    label="XTRACK",
)

for pass1 in common:
    fn = f"{L2P_PATH}/aviso_l2p_s3a_dist_time_{pass1}_asn.nc"
    with xr.open_dataset(fn, decode_times=False) as ds:
        std = ds.sla.std(dim="time")
        ax_std.scatter(std.x.values, std.values, color="lightcoral", marker=".", s=1)
        std_arrays_l2p.append(std)
        track_numbers_l2p.append(ds.attrs["pass"])

tc = xr.DataArray(track_numbers_l2p, dims=["track"], name="track")
std_mean_l2p = xr.concat(std_arrays_l2p, dim=tc).mean(dim="track", skipna=True)
ax_std.plot(
    std_mean_l2p.x.values,
    std_mean_l2p.values,
    color="r",
    linewidth=3,
    label="L2P",
)

ax_std.set_xlabel("Distance (km)")
ax_std.set_ylabel("STD (m)")
ax_std.axvline(x=0, color="k", linewidth=0.5, linestyle="--")
ax_std.axvline(x=50e3, color="gray", linewidth=0.5, linestyle="--")
ax_std.axvline(x=100e3, color="gray", linewidth=0.5, linestyle="--")
ax_std.text(
    0.02,
    0.5,
    "coast",
    transform=ax_std.transAxes,
    fontsize=10,
    fontweight="bold",
    color="k",
    rotation=90,
)
ax_std.set_xlim(-50e3,1000e3)
ax_std.set_ylim(0.05, 0.25)
ax_std.set_xticks([0, 50e3, 100e3, 200e3, 400e3, 600e3, 800e3, 1000e3])
ax_std.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
ax_std.grid(alpha=0.3, linestyle="--", linewidth=0.5)
ax_std.legend(ncol=2, fontsize=8)

# ════════════════════════════════════════════════════════════
# RIGHT-BOTTOM — valid point count vs GSHHG distance from coast
# ════════════════════════════════════════════════════════════
cnt_arrays_xt, cnt_arrays_l2p = [], []
track_numbers_xt2, track_numbers_l2p2 = [], []

for pass1 in common:
    pass2 = pass1[1:]
    pass3 = pass1.lstrip("0")
    fn = f"../data/{SAT}/ctoh.sla.ref.{SAT}.nindian.{pass2}.nc"
    ds = xr.open_dataset(fn, decode_times=False)
    sla_track = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
    sla_track = sla_track.interp(x=x_common)
    gshhg_track = dist_asn_gshhg(
        ds, var_str="dist_to_coast_gshhg", units_in="m"
    )
    gshhg_interp = gshhg_track.interp(x=x_common)
    cnt = sla_track.count(dim="time")
    ax_vp.scatter(
        gshhg_interp.values, cnt.values, color="lightblue", marker=".", s=1
    )
    cnt_arrays_xt.append(cnt)
    track_numbers_xt2.append(pass3)

tc = xr.DataArray(track_numbers_xt2, dims=["track"], name="track")
cnt_mean_xt = xr.concat(cnt_arrays_xt, dim=tc).mean(dim="track", skipna=True)
ax_vp.plot(
    cnt_mean_xt.x.values,
    cnt_mean_xt.values,
    color="b",
    linewidth=3,
    label="XTRACK",
)

for pass1 in common:
    fn = f"{L2P_PATH}/aviso_l2p_s3a_dist_time_{pass1}_asn.nc"
    with xr.open_dataset(fn, decode_times=False) as ds:
        cnt = ds.sla.count(dim="time")
        ax_vp.scatter(
            ds.gshhg_da_trim.values, cnt.values, color="lightcoral", marker=".", s=1
        )
        cnt_arrays_l2p.append(cnt)
        track_numbers_l2p2.append(ds.attrs["pass"])

tc = xr.DataArray(track_numbers_l2p2, dims=["track"], name="track")
cnt_mean_l2p = xr.concat(cnt_arrays_l2p, dim=tc).mean(dim="track", skipna=True)
ax_vp.plot(
    cnt_mean_l2p.x.values,
    cnt_mean_l2p.values,
    color="r",
    linewidth=3,
    label="L2P",
)

ax_vp.set_xlabel("GSHHG Distance from coast (km)")
ax_vp.set_ylabel("Percentage Valid Points")
ax_vp.axvline(x=0, color="k", linewidth=0.5, linestyle="--")
ax_vp.axvline(x=50e3, color="gray", linewidth=0.5, linestyle="--")
ax_vp.axvline(x=100e3, color="gray", linewidth=0.5, linestyle="--")
ax_vp.text(
    0.02,
    0.5,
    "coast",
    transform=ax_vp.transAxes,
    fontsize=10,
    fontweight="bold",
    color="k",
    rotation=90,
)
ax_vp.set_xlim(-50e3, 1000e3)
ax_vp.set_xticks([0, 50e3, 100e3, 200e3, 400e3, 600e3, 800e3, 1000e3])
ax_vp.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
ax_vp.grid(alpha=0.3, linestyle="--", linewidth=0.5)
ax_vp.legend(ncol=2, fontsize=8)

# ── panel labels ──
ax_map.set_title("(a)", loc="left", fontweight="bold")
ax_std.set_title("(c)", loc="left", fontweight="bold")
ax_vp.set_title("(b)", loc="left", fontweight="bold")

plt.tight_layout()
# plt.savefig("f.combined_map_std_valid.S3A.png", dpi=300)
plt.savefig(f"{png_file}", dpi=300, bbox_inches="tight")
plt.savefig(f"{png_file}.jpeg", dpi=300, bbox_inches="tight")
plt.show()
plt.close("all")
