"""Figure 1 -- S3A east-coast passes: track map, valid-point fraction and SLA std
versus distance from the coast, X-TRACK against the standard AVISO L2P product.

Rewrite of draft_figs02/fig01_f.combined_map_std_valid.S3A.py against
tools_xtrackm_c.  Changes that affect the figure, not just the code:

  1. Panel (b) had the scatter on GSHHG distance and the bold mean line on
     geodesic along-track distance, in one axis labelled "GSHHG Distance from
     coast".  The two disagree by a median 10 km at 50-100 km and 18 km at
     100-200 km (and by up to 1744 km far offshore, since GSHHG measures to the
     nearest coast in any direction).  Everything here is binned in GSHHG.
  2. Panel (b) plotted raw cycle counts under the label "Percentage Valid
     Points".  X-TRACK has 115 cycles against L2P's 110-112, so the counts gave
     X-TRACK a 3-5% head start.  Both are now a percentage of their own record.
  3. Panel (c) y limits were fixed at (0.05, 0.25) and clipped one of the 24
     individual track curves (max std 0.294 m).  Limits now follow the data.
  4. Datasets are closed, and the curves are written to a .nc so the panels can
     be redrawn without re-reading the tracks.

Checked and deliberately kept: interpolating X-TRACK onto the 6.675 km
x_common grid is what puts it on the same axis as the precomputed L2P files,
and for S3A it is benign -- valid points 0.99x, mean std -0.8%.  Applying the
corrected mad_filter(3.5) before the std moves every band by <= 0.003 m, so the
result does not depend on outlier rejection; MAD_THRESH is left None.
"""

import math
import pickle
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.ticker import FuncFormatter

from tools_xtrackm import *  # decorate_axis, is_within_region, NIO, plt, ccrs
from tools_xtrackm_c import mad_filter, track_dist_time_asn_midx, track_angle_at

SAT = "S3A"
L2P_PATH = "/home/srinivasu/allData/aviso_l2p/s3a"
PASSES_PKL = "../draft_figs02/common_passes_l2pVSxtrack_selected.pkl"
TOPO = "/home/srinivasu/allData/topo/etopo5.cdf"
EAST_COAST_REG = (78.0, 95, 0, 25.4)
CMAP = "twilight_shifted"

TRIM_LIM = 300
DX = 6_675.0  # m, the S3A 1 Hz along-track spacing the L2P files were built on
GSHHG_BINS = np.arange(0.0, 1000e3 + 1, 10e3)  # 10 km bins for panel (b)
MAD_THRESH = None  # set to e.g. 3.5 to reject outliers before the std

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")
nc_file = script.with_name(script.stem + ".nc")

with open(PASSES_PKL, "rb") as fh:
    common = pickle.load(fh)

x_common = xr.DataArray(np.arange(TRIM_LIM) * DX, dims=["x"], attrs={"units": "m"})


def bin_in_gshhg(gshhg, values):
    """Mean of `values` in GSHHG-distance bins; NaN where a bin is empty."""
    gshhg = np.asarray(gshhg, dtype=float)
    values = np.asarray(values, dtype=float)
    good = np.isfinite(gshhg) & np.isfinite(values)
    idx = np.digitize(gshhg[good], GSHHG_BINS) - 1
    out = np.full(len(GSHHG_BINS) - 1, np.nan)
    inside = (idx >= 0) & (idx < out.size)
    if inside.any():
        total = np.bincount(idx[inside], weights=values[good][inside], minlength=out.size)
        count = np.bincount(idx[inside], minlength=out.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(count > 0, total / count, np.nan)
    return out


gshhg_centres = 0.5 * (GSHHG_BINS[:-1] + GSHHG_BINS[1:])


def read_xtrack(pass1):
    """SLA on (x, time) interpolated to x_common, plus GSHHG distance per point."""
    fn = f"../data/{SAT}/ctoh.sla.ref.{SAT}.nindian.{pass1[1:]}.nc"
    with xr.open_dataset(fn, decode_times=False) as ds:
        sla = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
        gshhg = xr.DataArray(
            np.asarray(ds.dist_to_coast_gshhg.values, dtype=float)[::-1],
            dims=["x"],
            coords={"x": sla["x"].values},
        )
        track_number = ds.Pass
        lons, lats = ds.lon.values.copy(), ds.lat.values.copy()
    sla = sla.reset_coords(drop=True).interp(x=x_common)
    gshhg = gshhg.interp(x=x_common)
    if MAD_THRESH is not None:
        sla = mad_filter(sla, MAD_THRESH, dim="time")
    return sla, gshhg, track_number, lons, lats


def read_l2p(pass1):
    with xr.open_dataset(f"{L2P_PATH}/aviso_l2p_s3a_dist_time_{pass1}_asn.nc",
                         decode_times=False) as ds:
        sla = ds.sla.transpose("x", "time").load()
        gshhg = ds.gshhg_da_trim.load()
        track_number = ds.attrs["pass"]
    if MAD_THRESH is not None:
        sla = mad_filter(sla, MAD_THRESH, dim="time")
    return sla, gshhg, track_number


# ── collect every curve first, then draw ────────────────────────────────────
std_xt, std_l2p, pct_xt, pct_l2p = [], [], [], []
scatter_xt, scatter_l2p = [], []  # (gshhg, percent) clouds for panel (b)
tracks_xt, tracks_l2p, geometry = [], [], []

for pass1 in common:
    sla, gshhg, number, lons, lats = read_xtrack(pass1)
    std_xt.append(sla.std(dim="time").values)
    pct = 100.0 * sla.count(dim="time").values / sla.sizes["time"]
    pct_xt.append(bin_in_gshhg(gshhg.values, pct))
    scatter_xt.append((gshhg.values, pct))
    tracks_xt.append(number)
    geometry.append((number, lons, lats))

    sla, gshhg, number = read_l2p(pass1)
    std_l2p.append(sla.std(dim="time").values)
    pct = 100.0 * sla.count(dim="time").values / sla.sizes["time"]
    pct_l2p.append(bin_in_gshhg(gshhg.values, pct))
    scatter_l2p.append((gshhg.values, pct))
    tracks_l2p.append(number)

std_xt, std_l2p = np.array(std_xt), np.array(std_l2p)
pct_xt, pct_l2p = np.array(pct_xt), np.array(pct_l2p)

ds_out = xr.Dataset(
    {
        "std_xtrack": (("track", "x"), std_xt),
        "std_l2p": (("track", "x"), std_l2p),
        "pct_valid_xtrack": (("track", "gshhg"), pct_xt),
        "pct_valid_l2p": (("track", "gshhg"), pct_l2p),
    },
    coords={"track": tracks_xt, "x": x_common.values, "gshhg": gshhg_centres},
    attrs={"mission": SAT, "mad_thresh": "none" if MAD_THRESH is None else MAD_THRESH,
           "source": script.name},
)
ds_out["x"].attrs.update(units="m", long_name="geodesic along-track distance from coast")
ds_out["gshhg"].attrs.update(units="m", long_name="GSHHG distance to nearest coast")
ds_out["std_xtrack"].attrs.update(units="m")
ds_out["std_l2p"].attrs.update(units="m")
ds_out["pct_valid_xtrack"].attrs.update(units="%")
ds_out["pct_valid_l2p"].attrs.update(units="%")
ds_out.to_netcdf(nc_file)
print(f"wrote {nc_file}")

# ── figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.3], wspace=0.05, hspace=0.3)
ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
ax_vp = fig.add_subplot(gs[0, 1])
ax_std = fig.add_subplot(gs[1, 1])

# (a) track map
with xr.open_dataset(TOPO) as dse:
    dse.ROSE.sel(ETOPO05_X=slice(*EAST_COAST_REG[:2]),
                 ETOPO05_Y=slice(*EAST_COAST_REG[2:])).plot(
        ax=ax_map, add_colorbar=False, add_labels=False, cmap=CMAP)
decorate_axis(ax_map, "", *EAST_COAST_REG, step=4)
ax_map.text(0.25, 0.90, f"MISSION: {SAT}", transform=ax_map.transAxes,
            fontsize=10, fontweight="bold", color="b")

for number, lons, lats in geometry:
    ax_map.scatter(lons, lats, s=8, color="k")
    mid = len(lons) // 2 + len(lons) // 4
    lonm, latm = lons[mid], lats[mid]
    if is_within_region(lonm, latm, *NIO):
        # geodesic heading, index-matched to the forward-ordered arrays
        angle = local_angle(lons, lats, mid, ioff=3, degrees=True)
        ax_map.text(lonm, latm, s=str(number), fontsize=14, rotation=angle,
                    color="w", ha="center", va="center")

# (b) valid-point percentage vs GSHHG distance
for gshhg, pct in scatter_xt:
    ax_vp.scatter(gshhg, pct, color="lightblue", marker=".", s=1)
for gshhg, pct in scatter_l2p:
    ax_vp.scatter(gshhg, pct, color="lightcoral", marker=".", s=1)
ax_vp.plot(gshhg_centres, np.nanmean(pct_xt, axis=0), color="b", lw=3, label="XTRACK")
ax_vp.plot(gshhg_centres, np.nanmean(pct_l2p, axis=0), color="r", lw=3, label="L2P")
ax_vp.set_xlabel("GSHHG distance from coast (km)")
ax_vp.set_ylabel("Valid points (% of cycles)")
ax_vp.set_ylim(0, 105)

# (c) SLA standard deviation vs along-track distance
for row in std_xt:
    ax_std.scatter(x_common.values, row, color="cyan", marker=".", s=1)
for row in std_l2p:
    ax_std.scatter(x_common.values, row, color="lightcoral", marker=".", s=1)
ax_std.plot(x_common.values, np.nanmean(std_xt, axis=0), color="b", lw=3, label="XTRACK")
ax_std.plot(x_common.values, np.nanmean(std_l2p, axis=0), color="r", lw=3, label="L2P")
ax_std.set_xlabel("Along-track distance from coast (km)")
ax_std.set_ylabel("STD (m)")
finite = np.concatenate([std_xt[np.isfinite(std_xt)], std_l2p[np.isfinite(std_l2p)]])
ax_std.set_ylim(0, np.nanmax(finite) * 1.05)

for ax in (ax_vp, ax_std):
    ax.axvline(x=0, color="k", lw=0.5, ls="--")
    for xv in (50e3, 100e3):
        ax.axvline(x=xv, color="gray", lw=0.5, ls="--")
    ax.text(0.02, 0.5, "coast", transform=ax.transAxes, fontsize=10,
            fontweight="bold", color="k", rotation=90)
    ax.set_xlim(-50e3, 1000e3)
    ax.set_xticks([0, 50e3, 100e3, 200e3, 400e3, 600e3, 800e3, 1000e3])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}"))
    ax.grid(alpha=0.3, ls="--", lw=0.5)
    ax.legend(ncol=2, fontsize=8)

ax_map.set_title("(a)", loc="left", fontweight="bold")
ax_vp.set_title("(b)", loc="left", fontweight="bold")
ax_std.set_title("(c)", loc="left", fontweight="bold")

fig.savefig(png_file, dpi=300, bbox_inches="tight")
fig.savefig(f"{png_file}.jpeg", dpi=300, bbox_inches="tight")
plt.show()
plt.close("all")
print(f"wrote {png_file}")
