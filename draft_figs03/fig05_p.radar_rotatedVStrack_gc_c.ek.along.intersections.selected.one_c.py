"""Figure 5 -- HF radar cross-track current against altimetry gc (+ Ekman) along
one satellite track crossing the ICORN radar footprint.

Rewrite of
draft_figs02/fig05_p.radar_rotatedVStrack_gc_c.ek.along.intersections.selected.one.py

WHAT WAS ALREADY RIGHT, and is kept unchanged:
  The rotation convention.  rotate_to_cross builds
      theta_cross = local_angle(lon_track, lat_track, ii) + pi/2
      cross       = cos(theta_cross)*u + sin(theta_cross)*v  =  -u sin(a) + v cos(a)
  and radar/c.extract_radar.along.intersections.py stores lon_track COAST-FIRST
  (ds_track.lon.values[::-1], named lons_coast_to_off), so `a` is the OFFSHORE
  heading.  That is exactly the convention compute_geostrophy_gc produces
  (summary section 8.1).  Verified with a synthetic uniform current on S3A track
  165: compute_geostrophy_gc +0.4231 m/s against rotate_to_cross +0.4232 m/s,
  ratio 1.0002.  The index is consistent too -- ii runs over the same array it
  indexes -- so the section 7(b) bug does not apply here either.  Unlike
  fig02-fig04 this figure compares like with like.

WHAT IS FIXED:

  1. THE gc + EKMAN SUM WAS ALIGNED BY LUCK.  `gc_ek_cut = gc_cut + ekman_cut`
     left ekman_cut on its own daily axis, so xarray aligned on the time
     INTERSECTION.  That works today only because the old track_dist_time_asn
     truncates cycle times to whole days (section 3), which happens to land all
     115 gc stamps exactly on daily Ekman stamps.  With the corrected sub-second
     times from track_dist_time_asn_midx the overlap is 0 of 115, the sum
     collapses to an empty time axis and the following .reindex() returns an
     all-NaN field -- silently.  The Ekman is now reindexed onto the gc times
     explicitly, with a tolerance, exactly as the radar already was.

  2. n IS REPORTED, AND THE SIGNIFICANCE LINE IS REAL.  The radar record
     (2009-12 to 2019-01) barely overlaps this gc record (2016-06 to 2024-11):
     after the 24 h match there are a median of 25 paired cycles per x, out of
     115.  The original showed no n anywhere and drew an arbitrary r = 0.7
     guide line.  Panel (e) now carries n(x) and a p = 0.05 threshold computed
     from the actual sample size.

  3. PANEL (c) IS LABELLED HONESTLY.  It plots gc + Ekman, not gc; neither the
     axis nor the caption said so.

  4. Geodesic headings (tools_xtrackm_c.local_angle) at the project-standard
     ioff = 3.  The original used the old flat-earth angle at
     ioff = max(1, min(5, ln//10)), a different baseline from every other script.

  5. bias is computed and now used; the corr / rmse / bias / n profiles are
     saved to a .nc; tight_layout no longer overrides the gridspec; the
     duplicated imports and the unused f_plot argument are gone.
"""

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import xskillscore as xs
from scipy.stats import t as student_t

from tools_xtrackm import *          # decorate_axis, gpd, ccrs, plt, radar_times, ...
from tools_xtrackm_c import local_angle   # geodesic; must win over the old one

METHOD, FRAC1 = "loess", "0.05"
RADAR_STR, SAT, TNO = "OR", "S3A", "165"
IOFF = 3                              # project standard
MATCH_TOL = np.timedelta64(24, "h")   # radar (daily) onto altimeter cycles
EKMAN_TOL = np.timedelta64(2, "D")    # ekman (daily) onto altimeter cycles
PVAL = 0.05
MIN_N = 10                            # a p-value alone lets 3-sample columns through

GC_DIR = f"../guv/gc_along_{METHOD}"
EXTRACTED_DIR = "../radar/radar_along_intersections"
EKMAN_DIR = "../radar/ekman_along_intersections"
SHAPE_DIR = "../draft_figs02"
BOB_REG = (78.0, 95.0, 4.0, 24.0)
ETOPO = "/home/srinivasu/allData/topo/etopo_60s_io.nc"

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")
nc_file = script.with_name(script.stem + ".nc")


def open_etopo():
    ds = xr.open_dataset(ETOPO)
    lon_name = next((n for n in ("lon", "longitude", "x", "ETOPO05_X")
                     if n in ds.coords or n in ds.variables), None)
    lat_name = next((n for n in ("lat", "latitude", "y", "ETOPO05_Y")
                     if n in ds.coords or n in ds.variables), None)
    if lon_name is None or lat_name is None:
        raise ValueError("could not detect lon/lat names in the ETOPO file")
    var = next((n for n in ("z", "elevation", "topo", "ROSE", "Band1")
                if n in ds.data_vars and lon_name in ds[n].dims and lat_name in ds[n].dims),
               None)
    if var is None:
        var = next(n for n, v in ds.data_vars.items()
                   if v.ndim == 2 and lon_name in v.dims and lat_name in v.dims)
    topo = ds[var].load()
    ds.close()
    return topo, lon_name, lat_name


def topo_along_track(topo, lon_name, lat_name, lons, lats, xvals, x_lo, x_hi):
    lon_vals = np.asarray(lons, dtype=float).copy()
    if np.nanmax(np.asarray(topo[lon_name].values, dtype=float)) > 180.0:
        lon_vals = np.mod(lon_vals, 360.0)
    out = topo.interp({lon_name: xr.DataArray(lon_vals, coords={"x": xvals}, dims=["x"]),
                       lat_name: xr.DataArray(np.asarray(lats, dtype=float),
                                              coords={"x": xvals}, dims=["x"])},
                      method="linear")
    if "x" not in out.dims:
        out = out.rename({out.dims[0]: "x"})
    return out.sel(x=slice(x_lo, x_hi))


def heading_offshore(lons, lats, xvals, degrees=False):
    """Track heading at every point, in the direction of increasing x (offshore).

    lon_track/lat_track are stored coast-first, so indexing them directly gives
    the offshore heading -- the convention gc uses.
    """
    ang = [local_angle(lons, lats, i, ioff=IOFF, degrees=degrees) for i in range(len(lons))]
    return xr.DataArray(np.array(ang), coords={"x": xvals}, dims=["x"])


def rotate_to_cross(u_field, v_field, lons, lats):
    """Component 90 deg counter-clockwise from the offshore heading: -u sin a + v cos a."""
    theta = heading_offshore(lons, lats, u_field.x.values) + np.pi / 2.0
    cross = np.cos(theta) * u_field + np.sin(theta) * v_field
    cross.name = "cross"
    return cross


def r_critical(n, p=PVAL):
    """Two-tailed correlation significant at p, for sample size n."""
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tc = student_t.ppf(1.0 - p / 2.0, np.maximum(n - 2, 1))
        return np.where(n > 2, tc / np.sqrt(np.maximum(n - 2, 1) + tc ** 2), np.nan)


def skill(obs, pred):
    good = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs.where(good), pred.where(good)
    n = good.sum(dim="time")
    return dict(
        corr=xs.pearson_r(o, p, dim="time", skipna=True),
        pval=xs.pearson_r_p_value(o, p, dim="time", skipna=True),
        rmse=xs.rmse(o, p, dim="time", skipna=True),
        bias=(o - p).mean(dim="time", skipna=True),
        n=n,
    )


# ── inputs ──────────────────────────────────────────────────────────────────
f_ex = f"{EXTRACTED_DIR}/radar_along_{RADAR_STR}_{SAT}_{TNO}.nc"
f_ek = f"{EKMAN_DIR}/ekman_along_{RADAR_STR}_{SAT}_{TNO}.nc"
f_gc = f"{GC_DIR}/gc_along_{SAT}_{TNO}_{METHOD}_{FRAC1}.nc"
for f_in in (f_ex, f_ek, f_gc):
    if not os.path.exists(f_in):
        raise FileNotFoundError(f"missing input: {f_in}")
print(f"radar={RADAR_STR} sat={SAT} track={TNO}")

with xr.open_dataset(f_ex) as ds_ex:
    radar_u, radar_v = ds_ex.radar_u.load(), ds_ex.radar_v.load()
    dist_to_coast_track = ds_ex.dist_to_coast_gshhg_track.load()
    lons_track, lats_track = ds_ex.lon_track.values, ds_ex.lat_track.values
    lon_x1, lat_x1 = float(ds_ex.attrs["lon_x1"]), float(ds_ex.attrs["lat_x1"])
    lon_x2, lat_x2 = float(ds_ex.attrs["lon_x2"]), float(ds_ex.attrs["lat_x2"])
    x1_m, x2_m = float(ds_ex.attrs["x1_m"]), float(ds_ex.attrs["x2_m"])
with xr.open_dataset(f_ek) as ds_ek:
    ek_u, ek_v = ds_ek.ekman_u.load(), ds_ek.ekman_v.load()
with xr.open_dataset(f_gc) as ds_gc:
    gc = ds_gc.gc.load()

x_lo, x_hi = min(x1_m, x2_m), max(x1_m, x2_m)
gc_cut = gc.sel(x=slice(x_lo, x_hi))

radar_cross = rotate_to_cross(radar_u, radar_v, lons_track, lats_track)
ekman_cross = rotate_to_cross(ek_u, ek_v, lons_track, lats_track)

radar_cut = radar_cross.interp(x=gc_cut.x, method="nearest").reindex(
    time=gc_cut.time, method="nearest", tolerance=MATCH_TOL)
# fix (1): put the Ekman on the gc time axis BEFORE adding, so the sum does not
# depend on the two axes happening to share exact timestamps
ekman_cut = ekman_cross.interp(x=gc_cut.x, method="nearest").reindex(
    time=gc_cut.time, method="nearest", tolerance=EKMAN_TOL)
gc_ek_cut = gc_cut + ekman_cut
print(f"  gc {int(np.isfinite(gc_cut.values).sum())} finite | "
      f"gc+ekman {int(np.isfinite(gc_ek_cut.values).sum())} finite "
      f"(an empty result here would mean the time alignment broke)")

topo_cut = topo_along_track(*open_etopo(), lons_track, lats_track,
                            radar_u.x.values, x_lo, x_hi).interp(
    x=gc_cut.x, method="nearest")
dist_cut = dist_to_coast_track.interp(x=gc_cut.x, method="nearest")
angle_cut = heading_offshore(lons_track, lats_track, radar_u.x.values,
                             degrees=True).interp(x=gc_cut.x, method="nearest")

s_no = skill(radar_cut, gc_cut)
s_ek = skill(radar_cut, gc_ek_cut)
n_med = int(np.nanmedian(s_ek["n"].values))
print(f"  paired cycles per x: min {int(s_ek['n'].min())} median {n_med} "
      f"max {int(s_ek['n'].max())} of {gc.time.size}")
print(f"  r significant at p={PVAL} needs |r| >= {r_critical(n_med):.3f} at n={n_med}")

# significance AND a minimum sample: p < 0.05 on its own admits columns with a
# handful of pairs, which showed up as isolated spikes near the far edge
keep_no = (s_no["pval"] < PVAL) & (s_no["n"] >= MIN_N)
keep_ek = (s_ek["pval"] < PVAL) & (s_ek["n"] >= MIN_N)
corr_no = s_no["corr"].where(keep_no)
corr_ek = s_ek["corr"].where(keep_ek)
rmse_no = s_no["rmse"].where(keep_no)
rmse_ek = s_ek["rmse"].where(keep_ek)
print(f"  x columns kept (p<{PVAL} and n>={MIN_N}): "
      f"{int(keep_ek.sum())} of {keep_ek.size}")

xr.Dataset(
    {"corr_noek": s_no["corr"], "corr_ek": s_ek["corr"],
     "rmse_noek": s_no["rmse"], "rmse_ek": s_ek["rmse"],
     "bias_noek": s_no["bias"], "bias_ek": s_ek["bias"],
     "pval_noek": s_no["pval"], "pval_ek": s_ek["pval"],
     "n_pairs": s_ek["n"], "topo": topo_cut,
     "dist_to_coast_gshhg": dist_cut, "heading_offshore_deg": angle_cut},
    attrs={"radar": RADAR_STR, "sat": SAT, "track": TNO, "method": METHOD,
           "frac": FRAC1, "ioff": IOFF, "pval": PVAL, "min_n": MIN_N,
           "n_median": n_med, "r_crit_at_n_median": float(r_critical(n_med)),
           "source": script.name},
).to_netcdf(nc_file)
print(f"wrote {nc_file}")

# ── figure ──────────────────────────────────────────────────────────────────
km = lambda d: d.assign_coords(x=d.x / 1000.0)
sat_tsta, sat_tend = get_time_limits_o(SAT)
radar_tsta, radar_tend = radar_times[RADAR_STR]
ov_start, ov_end = overlap_dates(radar_tsta, radar_tend, sat_tsta, sat_tend)
polygon = gpd.read_file(
    f"{SHAPE_DIR}/polygon_shapefile_around_{RADAR_STR}.shp").geometry.iloc[0]

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.8], hspace=0.28, wspace=0.22)
ax0 = fig.add_subplot(gs[0:2, 0], projection=ccrs.PlateCarree())
ax0b = fig.add_subplot(gs[2, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)

decorate_axis(ax0, "", *BOB_REG, step=4)
ax0.grid(alpha=0.3, linestyle="--")
ax0.add_geometries([polygon], ccrs.PlateCarree(), facecolor="tab:orange",
                   edgecolor="tab:red", alpha=0.25, linewidth=1.5)
ax0.plot(lons_track, lats_track, color="tab:blue", lw=1.2, transform=ccrs.PlateCarree())
pt = polygon.representative_point()
ax0.text(pt.x, pt.y, RADAR_STR, fontsize=11, fontweight="bold", color="tab:red",
         transform=ccrs.PlateCarree(), ha="center", va="center")
i_mid = len(lons_track) // 2
ax0.text(lons_track[i_mid], lats_track[i_mid], str(TNO), fontsize=10, color="k",
         rotation=local_angle(lons_track, lats_track, i_mid, ioff=IOFF, degrees=True),
         transform=ccrs.PlateCarree(), ha="center", va="center",
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2))
ax0.plot([lon_x1, lon_x2], [lat_x1, lat_x2], "x", color="k", markersize=5,
         transform=ccrs.PlateCarree())

km(topo_cut).plot(ax=ax0b, color="tab:brown", lw=1.4)
ax0b.axhline(0.0, color="gray", ls="--", lw=0.8)
ax0b.set_title("ETOPO along track (x1 to x2)")
ax0b.set_xlabel("x from coast (km)")
ax0b.set_ylabel("topo (m)")
ax0b.grid(alpha=0.3, ls="--")

levels, colors = (-0.3, 0.3), ("b", "r")
cf1 = km(gc_ek_cut).plot.contourf(ax=ax1, x="x", y="time", cmap="RdBu_r",
                                  vmin=-1.0, vmax=1.0, add_labels=False,
                                  add_colorbar=False)
km(gc_ek_cut).plot.contour(ax=ax1, x="x", y="time", levels=levels, colors=colors,
                           linestyles="dotted", linewidths=0.5)
km(radar_cut).plot.contourf(ax=ax2, x="x", y="time", cmap="RdBu_r", vmin=-1.0,
                            vmax=1.0, add_labels=False, add_colorbar=False)
km(radar_cut).plot.contour(ax=ax2, x="x", y="time", levels=levels, colors=colors,
                           linestyles="dashed", linewidths=0.5)
for ax, lab in ((ax1, "altimetry gc + Ekman (cross-track)"),
                (ax2, "HF radar (rotated to cross-track)")):
    ax.set_ylabel("time")
    ax.set_xlabel("")
    ax.grid(alpha=0.3, ls="--")
    ax.text(0.02, 0.97, lab, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                   alpha=0.8, ec="none"))
    if ov_start is not None and ov_end is not None:
        ax.set_ylim(ov_start, ov_end)

cax = ax1.inset_axes([0.0, 1.06, 1.0, 0.05])
cb = fig.colorbar(cf1, cax=cax, orientation="horizontal")
cb.set_label("m/s")
cax.xaxis.set_ticks_position("top")
cax.xaxis.set_label_position("top")

km(corr_no).plot(ax=ax3, color="c", lw=1.5, marker="x", label="corr, no Ekman")
km(corr_ek).plot(ax=ax3, color="b", lw=1.5, label="corr, with Ekman")
rc = r_critical(s_ek["n"].values)
ax3.plot(gc_cut.x.values / 1000.0, rc, color="k", ls=":", lw=1.2,
         label=f"p={PVAL} threshold (n median {n_med})")
ax3.axhline(0.0, color="gray", ls="--", lw=1.0)
ax3.set_ylim(-1.0, 1.0)
ax3.set_xlabel("x from coast (km)")
ax3.set_ylabel("corr(time)")
ax3.grid(alpha=0.3, ls="--")

ax3_r = ax3.twinx()
km(rmse_no).plot(ax=ax3_r, color="orange", lw=1.3, marker="x", label="rmse, no Ekman")
km(rmse_ek).plot(ax=ax3_r, color="red", lw=1.3, label="rmse, with Ekman")
ax3_r.set_ylabel("rmse (m/s)")
ax3_r.axhline(0.3, color="tab:red", ls="--", lw=0.5)
for ax, side, col in ((ax3, "left", "tab:blue"), (ax3_r, "right", "tab:red")):
    ax.spines[side].set_color(col)
    ax.spines[side].set_linewidth(2.5)
    ax.yaxis.label.set_color(col)
    ax.tick_params(axis="y", colors=col, width=2.0)
ll, la_ = ax3.get_legend_handles_labels()
lr, lb = ax3_r.get_legend_handles_labels()
ax3.legend(ll + lr, la_ + lb, loc="lower right", fontsize=8)
ax3.set_title(f"n = {int(s_ek['n'].min())}-{int(s_ek['n'].max())} paired cycles per x "
              f"(median {n_med} of {gc.time.size})", fontsize=9, loc="left")

for ax, lab in ((ax0, "a)"), (ax0b, "b)"), (ax1, "c)"), (ax2, "d)"), (ax3, "e)")):
    ax.text(0.95, 0.98, lab, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold")

fig.savefig(png_file, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"wrote {png_file}")
