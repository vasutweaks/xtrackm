"""Figure 6 (rewrite) -- CROSS-TRACK geostrophic current at EICC coastal
intersection points, with honest trend uncertainty.

Replaces draft_figs02/fig06_p.speed_at_coastal_points.eicc.trends.all_in_one.ERS.py,
whose three claims did not hold up (summary section 17):

  1. IT PLOTTED A QUANTITY THE DRAFT CALLS UNUSABLE.  gs = sqrt(gu^2 + gv^2),
     but at these seven points rms(gv)/rms(gu) is 1.34 to 5.41 and rms(gv)
     reaches 1.16 m/s -- far above any plausible EICC.  The tracks are near
     meridional, so v is the ill-conditioned half of the inversion, and the
     draft says so itself: "Meridional currents are poorly resolved with
     correlations <0.4 ... most of the analysis is limited to well resolved
     zonal component."  Recomputed on the resolved component the trends shrink
     by 3-12x and one changes sign.
     -> This script uses the CROSS-TRACK current gc, which is what a single
        track actually measures.  No inversion, no ill-conditioned v.

  2. THE SIGNIFICANCE WAS AN ARTEFACT OF THE FILTER.  The trend was fitted with
     scipy linregress TO THE 500-DAY LOW-PASS SERIES and its p-value taken at
     face value.  The filter makes residuals autocorrelated by construction
     (lag-1 r 0.89-0.95), so the effective sample is 5-11, not the 165-186 the
     p-value assumed.  6 of 7 points were labelled significant; NONE survive.
     -> Here the trend is fitted to the UNFILTERED series, the low pass is for
        display only, and the standard error is inflated by the lag-1
        autocorrelation of the residuals (Nychka/Santer effective DOF).

  3. |a| + |b| IS NOT |a + b|.  gs = |geostrophic| + |Ekman| overestimated the
     speed of the total current by 10-19% and injected a spurious seasonal
     signal, since the bias depends on the angle between the two vectors.
     -> The Ekman is projected onto the same cross-track direction and ADDED AS
        A COMPONENT.

  4. FOUR INSTRUMENTS, ONE RECORD.  ERS1+ERS2+ENV+SRL concatenates ERS-1/2,
     Envisat and Saral across a 2.6 year gap, and most of the old "trend" was a
     step down from ERS to Envisat.  Because speed is positive definite, any
     drop in noise between instruments produces exactly that signature.
     -> The trend is reported BOTH ways: a plain fit, and a fit that allows a
        free offset per instrument era, so a step masquerading as a trend is
        visible rather than hidden.

  5. The raw series is now drawn.  The original plotted only the low pass and
     the fit, with the y axis tightened to the low-pass range, so the scatter
     the p-value depends on was invisible.

Sign convention: gc is the cross-track component 90 deg counter-clockwise from
the OFFSHORE heading, gc = -u sin(a) + v cos(a) = +(g/f) dh/dx (section 8.1).
Positive gc is north-eastward along this coast; a negative trend means the
cross-track flow is weakening in that sense, not that "the current is slowing".
"""

import os
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt
from scipy.stats import t as tdist

sys.path.insert(0, "/home/srinivasu/xtrackm/lib")
from tools_xtrackm_c import track_angle_at

SAT = "ERS1+ERS2+ENV+SRL"
METHOD, FRAC = "loess", "0.05"
CSV = f"../draft_figs02/intersections_near_east_coast_{SAT}.csv"
GC_DIR = f"../guv/gc_along_{METHOD}"
EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd"
TRACK_DIR = f"../data/{SAT}"
ETOPO = os.path.expanduser("~/allData/topo/etopo5.cdf")

EKMAN_DEPTH = "0"
LOWPASS_DAYS = 500.0          # display only -- the trend is fitted to raw data
USE = "self"                  # which track of the pair supplies gc
MAP_EXTENT = (78.5, 88.0, 10.0, 22.0)
ERAS = [("ERS1/2", "1992-01-01", "2002-12-31"),
        ("Envisat", "2003-01-01", "2012-12-31"),
        ("Saral", "2013-01-01", "2017-12-31")]

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")
nc_file = script.with_name(script.stem + ".nc")


def lowpass_display(values, dt_days):
    """500-day low pass for the eye only; NaNs restored afterwards."""
    nan = ~np.isfinite(values)
    if nan.all():
        return values
    filled = pd.Series(values).interpolate(limit_direction="both").values
    b, a = butter(4, (1.0 / LOWPASS_DAYS) / (0.5 / dt_days), btype="low")
    if filled.size > 3 * max(len(a), len(b)):
        out = filtfilt(b, a, filled)
    else:
        w = max(3, int(round(LOWPASS_DAYS / dt_days)))
        out = pd.Series(filled).rolling(w, center=True, min_periods=1).mean().values
    out[nan] = np.nan
    return out


def trend_with_error(t_days, y, eras_index=None):
    """Least-squares trend with an autocorrelation-inflated standard error.

    Fitted to the RAW series.  The effective sample size follows the lag-1
    autocorrelation of the residuals, n_eff = n (1 - r1) / (1 + r1), which is
    the standard correction for a serially correlated trend estimate.
    `eras_index` adds a free offset per instrument era, so an inter-mission step
    cannot be absorbed into the slope.
    """
    m = np.isfinite(y)
    n = int(m.sum())
    if n < 10:
        return None
    t, yy = t_days[m], y[m]
    cols = [np.ones(n), t]
    if eras_index is not None:
        e = eras_index[m]
        for k in np.unique(e)[1:]:            # first era is the reference level
            cols.append((e == k).astype(float))
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ beta
    dof = n - A.shape[1]
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if n > 2 else 0.0
    r1 = min(max(r1, 0.0), 0.99)
    n_eff = max(A.shape[1] + 1.0, n * (1.0 - r1) / (1.0 + r1))
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(cov[1, 1])) * np.sqrt(n / n_eff)   # inflate for serial corr
    dof_eff = max(n_eff - A.shape[1], 1.0)
    tcrit = float(tdist.ppf(0.975, dof_eff))
    slope_yr, se_yr = beta[1] * 365.25, se * 365.25
    return dict(slope=slope_yr, se=se_yr, ci=tcrit * se_yr,
                p=float(2 * (1 - tdist.cdf(abs(slope_yr / se_yr), dof_eff))),
                n=n, n_eff=float(n_eff), r1=r1,
                intercept=float(beta[0]), beta=beta, A_cols=A.shape[1])


def load_point(row):
    """Cross-track gc at one intersection, plus the cross-track Ekman."""
    tno = int(row["track_self"] if USE == "self" else row["track_other"])
    x_at = float(row["x_from_coast_self" if USE == "self" else "x_from_coast_other"])
    with xr.open_dataset(f"{GC_DIR}/gc_along_{SAT}_{tno}_{METHOD}_{FRAC}.nc") as ds:
        gc = ds["gc"].sel(x=x_at, method="nearest").load()
    # project the Ekman onto the same cross-track direction as gc
    with xr.open_dataset(f"{TRACK_DIR}/ctoh.sla.ref.{SAT}.nindian.{tno:03d}.nc",
                         decode_times=False) as dst:
        alpha, _ = track_angle_at(dst, lat=float(row["lats_inter"]), order="coast")
    us = f"{SAT}_{int(row['track_self'])}_{int(row['track_other'])}"
    f_ek = f"{EKMAN_DIR}/{us}_ekman_{EKMAN_DEPTH}m_at_intersections_cd.nc"
    with xr.open_dataset(f_ek) as dse:
        ue = dse["u_at"].reindex(time=gc.time, method="nearest",
                                 tolerance=np.timedelta64(2, "D")).values.astype(float)
        ve = dse["v_at"].reindex(time=gc.time, method="nearest",
                                 tolerance=np.timedelta64(2, "D")).values.astype(float)
    ek_cross = -ue * np.sin(alpha) + ve * np.cos(alpha)
    return gc, ek_cross, np.degrees(alpha), tno, float(gc.x)


df = pd.read_csv(CSV).sort_values("lats_inter").iloc[1:].reset_index(drop=True)
N_PTS = len(df)
LABELS = [chr(ord("A") + i) for i in range(N_PTS)]
COLORS = [plt.cm.tab10(i % 10) for i in range(N_PTS)]

points = []
print(f"{'pt':<3}{'lat':>6}{'track':>7}{'x km':>7}{'alpha':>8} | "
      f"{'trend':>9}{'+-95%':>9}{'p':>7}{'n':>5}{'n_eff':>7} | {'trend, era-adj':>15}{'p':>7}")
for i, row in df.iterrows():
    gc, ek_cross, alpha_deg, tno, x_used = load_point(row)
    t = pd.to_datetime(gc.time.values)
    t_days = (t - t[0]).days.values.astype(float)
    era_idx = np.zeros(len(t), dtype=int)
    for k, (_, a, b) in enumerate(ERAS):
        era_idx[(t >= pd.Timestamp(a)) & (t <= pd.Timestamp(b))] = k

    total = gc.values.astype(float) + ek_cross
    tr = trend_with_error(t_days, total)
    tr_era = trend_with_error(t_days, total, eras_index=era_idx)
    dt = float(np.median(np.diff(t_days)))

    points.append(dict(
        label=LABELS[i], color=COLORS[i], lon=float(row["lons_inter"]),
        lat=float(row["lats_inter"]), track=tno, x_km=x_used / 1000.0,
        alpha=alpha_deg, times=t, raw=total,
        lp=lowpass_display(total, dt), t_days=t_days,
        gc_only=trend_with_error(t_days, gc.values.astype(float)),
        tr=tr, tr_era=tr_era))
    print(f"{LABELS[i]:<3}{row['lats_inter']:6.2f}{tno:7d}{x_used/1000:7.1f}{alpha_deg:8.1f} | "
          f"{tr['slope']:+9.4f}{tr['ci']:9.4f}{tr['p']:7.2f}{tr['n']:5d}{tr['n_eff']:7.1f} | "
          f"{tr_era['slope']:+15.4f}{tr_era['p']:7.2f}")

n_sig = sum(p["tr"]["p"] < 0.05 for p in points)
n_sig_era = sum(p["tr_era"]["p"] < 0.05 for p in points)
print(f"\nsignificant at p<0.05 (autocorrelation-corrected): {n_sig}/{N_PTS} plain, "
      f"{n_sig_era}/{N_PTS} allowing per-instrument offsets")

xr.Dataset(
    {k: (("point",), np.array([p[src][k] for p in points]))
     for src, keys in (("tr", ("slope", "se", "ci", "p", "n", "n_eff", "r1")),)
     for k in keys}
    | {f"era_{k}": (("point",), np.array([p["tr_era"][k] for p in points]))
       for k in ("slope", "ci", "p")}
    | {f"gconly_{k}": (("point",), np.array([p["gc_only"][k] for p in points]))
       for k in ("slope", "ci", "p")}
    | {"lat": (("point",), np.array([p["lat"] for p in points])),
       "lon": (("point",), np.array([p["lon"] for p in points])),
       "track": (("point",), np.array([p["track"] for p in points])),
       "x_km": (("point",), np.array([p["x_km"] for p in points])),
       "heading_offshore_deg": (("point",), np.array([p["alpha"] for p in points]))},
    coords={"point": LABELS},
    attrs={"sat": SAT, "quantity": "cross-track gc + cross-track Ekman",
           "ekman_depth_m": EKMAN_DEPTH, "lowpass_days": LOWPASS_DAYS,
           "trend_fitted_to": "raw (unfiltered) series",
           "error": "autocorrelation-inflated, n_eff = n(1-r1)/(1+r1)",
           "source": script.name},
).to_netcdf(nc_file)
print(f"wrote {nc_file}")

# ── figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 15))
gsl = gridspec.GridSpec(N_PTS, 2, figure=fig, width_ratios=[1, 2.5],
                        hspace=0.0, wspace=0.16)

# (top left) trend against latitude, with 95% intervals
ax_tr = fig.add_subplot(gsl[0, 0])
lats = [p["lat"] for p in points]
for key, col, lab, off in (("gc_only", "0.55", "gc only", -0.10),
                           ("tr", "tab:blue", "gc + Ekman", 0.0),
                           ("tr_era", "tab:red", "gc + Ekman, era-adjusted", 0.10)):
    ax_tr.errorbar([p[key]["slope"] for p in points], np.array(lats) + off,
                   xerr=[p[key]["ci"] for p in points], fmt="o", ms=4,
                   color=col, ecolor=col, elinewidth=1.2, capsize=2.5, label=lab)
ax_tr.axvline(0.0, color="k", lw=1.0)
ax_tr.set_xlabel("trend (m s$^{-1}$ yr$^{-1}$)", fontsize=9)
ax_tr.set_ylabel("latitude (N)", fontsize=9)
ax_tr.tick_params(labelsize=8)
ax_tr.grid(alpha=0.3, ls="--")
ax_tr.legend(fontsize=7, loc="lower left")
ax_tr.set_title("trend with 95% interval\n(error bars corrected for serial correlation)",
                fontsize=8.5, loc="left")

# (left, remaining rows) map
ax_map = fig.add_subplot(gsl[1:, 0], projection=ccrs.PlateCarree())
ax_map.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
ax_map.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
ax_map.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
ax_map.coastlines(resolution="50m", linewidth=1.0, zorder=2)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.6,
                      linestyle="--")
gl.top_labels = gl.right_labels = False
with xr.open_dataset(ETOPO) as dse:
    rs = dse.ROSE.sel(ETOPO05_X=slice(MAP_EXTENT[0], MAP_EXTENT[1]),
                      ETOPO05_Y=slice(MAP_EXTENT[2], MAP_EXTENT[3]))
    for lev, col in ((-200, "steelblue"), (-1000, "r")):
        ax_map.contour(rs.ETOPO05_X.values, rs.ETOPO05_Y.values, rs.values,
                       levels=[lev], colors=col, linewidths=0.7, linestyles="--",
                       transform=ccrs.PlateCarree(), zorder=4)
ax_map.plot([p["lon"] for p in points], lats, color="gray", lw=0.8, ls="--",
            transform=ccrs.PlateCarree(), zorder=3)
for p in points:
    ax_map.plot(p["lon"], p["lat"], "o", ms=5, color=p["color"],
                markeredgecolor="black", markeredgewidth=0.8,
                transform=ccrs.PlateCarree(), zorder=5)
    ax_map.text(p["lon"] + 0.18, p["lat"] + 0.10, p["label"], fontsize=10,
                fontweight="bold", color=p["color"], transform=ccrs.PlateCarree(),
                zorder=6)
    sig = p["tr"]["p"] < 0.05
    ax_map.text(p["lon"] + 0.18, p["lat"] - 0.50,
                f"{p['tr']['slope']:+.3f} $\\pm$ {p['tr']['ci']:.3f}"
                + ("" if sig else "  (n.s.)"),
                fontsize=6.5, color=p["color"] if sig else "0.55",
                transform=ccrs.PlateCarree(), zorder=6)

# (right) one row per point
for idx, p in enumerate(points):
    ax = fig.add_subplot(gsl[N_PTS - 1 - idx, 1])
    for yr in range(p["times"].year.min(), p["times"].year.max() + 1):
        ax.axvspan(pd.Timestamp(yr, 2, 1), pd.Timestamp(yr, 5, 31),
                   color="yellow", alpha=0.13, zorder=0)
        ax.axvspan(pd.Timestamp(yr, 6, 1), pd.Timestamp(yr, 9, 30),
                   color="cyan", alpha=0.13, zorder=0)
        ax.axvspan(pd.Timestamp(yr, 10, 1), pd.Timestamp(yr, 12, 31),
                   color="gray", alpha=0.13, zorder=0)
    # fix (5): the raw series the trend is actually fitted to
    ax.plot(p["times"], p["raw"], ".", ms=2.2, color="0.45", alpha=0.55, zorder=1)
    ax.plot(p["times"], p["lp"], color=p["color"], lw=2.0, zorder=3)
    tr = p["tr"]
    fit = tr["intercept"] + tr["beta"][1] * p["t_days"]
    band = tr["ci"] / 365.25 * np.abs(p["t_days"] - p["t_days"].mean())
    ax.plot(p["times"], fit, "k--", lw=1.5, zorder=4)
    ax.fill_between(p["times"], fit - band, fit + band, color="k", alpha=0.13,
                    zorder=2, lw=0)
    ax.axhline(0.0, color="k", lw=0.4, alpha=0.5)
    ax.text(0.005, 0.5, p["label"], transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=p["color"], va="center", ha="left")
    ax.text(0.985, 0.90,
            f"T{p['track']}  x={p['x_km']:.0f} km   "
            f"trend {tr['slope']:+.3f}$\\pm${tr['ci']:.3f} m/s/yr, p={tr['p']:.2f}"
            f"  (n={tr['n']}, n$_{{eff}}$={tr['n_eff']:.0f})",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, ec="none"))
    lo, hi = np.nanpercentile(p["raw"], [1, 99])
    pad = 0.10 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("m/s", fontsize=12)
    ax.tick_params(labelsize=11)
    if idx > 0:
        ax.set_xticklabels([])
    if idx == N_PTS - 1:
        ax.legend(handles=[
            Line2D([0], [0], ls="none", marker=".", color="0.45", label="gc + Ekman (raw)"),
            Line2D([0], [0], color=p["color"], lw=2.0, label=f"{LOWPASS_DAYS:.0f}-day low-pass"),
            Line2D([0], [0], color="k", lw=1.5, ls="--", label="linear trend (fitted to raw)"),
            Patch(facecolor="k", alpha=0.13, label="95% band"),
            Patch(facecolor="yellow", alpha=0.4, label="Feb-May"),
            Patch(facecolor="cyan", alpha=0.4, label="Jun-Sep"),
            Patch(facecolor="gray", alpha=0.4, label="Oct-Dec"),
        ], loc="upper left", fontsize=9, ncol=4)

fig.suptitle(f"Cross-track geostrophic + Ekman current at EICC coastal "
             f"intersections -- {SAT}", fontsize=13, fontweight="bold", y=0.995)
fig.savefig(png_file, dpi=300, bbox_inches="tight")
plt.show()
plt.close("all")
print(f"wrote {png_file}")
