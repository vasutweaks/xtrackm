"""Figure 4 -- GDP drifter ve against altimetry gu, with and without Ekman.

Rewrite of draft_figs02/fig04_p.scatter_drifter_ve_vs_gu_ww_ek.py, reading the
corrected CSV from
    drifter_track_reg/c.extract.drifter_veANDgu_ek_at_intersections_c.py
which fixes two things the old CSV could not express:

  1. DROGUE STATE.  48.5% of the points the published figure plots come from
     drifters that had already lost their drogue -- surface followers, biased
     downwind by roughly 1% of wind speed.  drogue_lost_date was in the drifter
     files all along and the old producer never read it.
  2. THE "72 HOUR" DRIFTER FILTER WAS AN 18 HOUR FILTER.  butter_lowpass_1d
     hardcodes fs = 1 sample/day but the drifter series is 6-hourly, so the
     -3 dB cutoff was 0.75 d rather than 3 d (summary section 9.2).  Refiltering
     with lowpass_time() drops drifter_ve rms from 0.4638 to 0.4471 m/s.

Changes in this script:
  3. THE TIME MATCH IS FILTERED, not merely printed.  The old script set
     MAX_ABS_TIME_DIFF_DAYS = None, computed the offset statistics and did
     nothing with them: median |dt| 2.50 d, p90 4.50 d, max 126 d, only 20.8%
     within a day.
  4. Reduced major axis instead of ordinary least squares, rmse about the 1:1
     line instead of the residual about the fit, and bias -- as in fig03
     (section 13.1).  Here OLS reads 0.833 and RMA 1.145, straddling 1, so the
     sign of the conclusion depended on the estimator.
  5. The coastal subset is stated on the figure.  The old suptitle saying
     "coast <= 100 km" was commented out, so a reader could not tell that 328 of
     1622 rows and 18 of 68 intersections were being shown.
  6. Both panels read ONE dataframe.  The old script loaded
     drifter_ve_vs_gu_{SAT}.csv and drifter_ve_vs_gu_ek_{SAT}.csv, which are
     byte-identical files (cmp confirms, 314589 bytes each).
  7. Axis limits follow the data (the old ones were asymmetric: vmin to 1.5);
     the 1:1 line is dark, not white, since the axes extend past the hexbins.

EKMAN DEPTH: kept at 0 m, deliberately.  fig02 and fig03 needed the 15 m Ekman
to match their 15 m buoy bin, but drifters are the exception -- with drogues at
15 m, 0 m still verifies marginally better (r 0.726 vs 0.705, rmse 0.343 vs
0.348).  The CSV carries u_ek_15m alongside; set EKMAN_COL to test it.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

SAT = "TP+J1+J2+J3+S6A"
CSV = f"../drifter_track_reg/drifter_ve_vs_gu_ek_{SAT}_c.csv"
COAST_DIST_PATH = "~/allData/topo/GMT_intermediate_coast_distance_01d_track_reg.nc"

MAX_COAST_KM = 100.0        # the coastal subset the figure is about
DROGUED_ONLY = True         # fix (1)
MAX_ABS_TIME_DIFF_DAYS = 2.0  # fix (3); None to disable
MAX_CLOSEST_KM = None       # the candidate boxes already bound this at 46 km
EKMAN_COL = "u_ek"          # or "u_ek_15m"
HEXBIN_MIN_N = 300          # below this a scatter reads better than a hexbin

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")
nc_file = script.with_name(script.stem + ".nc")


def coast_distance_km(lons, lats):
    with xr.open_dataset(os.path.expanduser(COAST_DIST_PATH)) as ds:
        return ds["coast_distance"].sel(
            lon=xr.DataArray(np.asarray(lons, dtype=float), dims="point"),
            lat=xr.DataArray(np.asarray(lats, dtype=float), dims="point"),
            method="nearest").values.astype(float)


def stats(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return None
    ols_slope, ols_int = np.polyfit(x, y, 1)
    rma_slope = np.sign(np.cov(x, y)[0, 1]) * np.sqrt(y.var() / x.var())
    return dict(
        n=x.size, r=float(np.corrcoef(x, y)[0, 1]),
        ols_slope=float(ols_slope), ols_int=float(ols_int),
        rma_slope=float(rma_slope), rma_int=float(y.mean() - rma_slope * x.mean()),
        rmse_11=float(np.sqrt(np.mean((y - x) ** 2))),
        resid_fit=float(np.sqrt(np.sum((y - (ols_slope * x + ols_int)) ** 2)
                                / (x.size - 2))),
        bias=float(np.mean(y - x)),
    )


def add_panel(ax, x, y, title, xlabel, ylabel, panel_label, extent, groups):
    s = stats(x, y)
    m = np.isfinite(x) & np.isfinite(y)
    # hexbin only pays off once the cloud is dense; the corrected filters leave
    # a modest sample, where a hexbin is nearly all single-count cells
    if m.sum() >= HEXBIN_MIN_N:
        hb = ax.hexbin(x[m], y[m], gridsize=45, cmap="viridis", mincnt=1, extent=extent)
    else:
        hb = None
        ax.scatter(x[m], y[m], s=34, c="tab:blue", alpha=0.65,
                   edgecolors="white", linewidths=0.5, zorder=3)
    xs = np.linspace(extent[0], extent[1], 200)
    ax.plot(xs, s["rma_slope"] * xs + s["rma_int"], color="crimson", lw=2.0,
            label="Reduced major axis")
    ax.plot(xs, xs, color="0.25", ls="--", lw=1.2, label="1:1 line")
    ax.text(0.03, 0.97,
            f"N = {s['n']}  ({groups[0]} intersections, {groups[1]} drifters)\n"
            f"r = {s['r']:.3f}\n"
            f"RMA slope = {s['rma_slope']:.3f}   (OLS {s['ols_slope']:.3f})\n"
            f"rmse vs 1:1 = {s['rmse_11']:.3f} m/s\n"
            f"resid about fit = {s['resid_fit']:.3f} m/s\n"
            f"bias = {s['bias']:+.3f} m/s",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.6, ec="gray"))
    ax.text(0.62, 0.98, title, transform=ax.transAxes, va="top", ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.6, ec="white"))
    ax.text(0.03, 0.03, panel_label, transform=ax.transAxes, va="bottom",
            ha="left", fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return hb, s


def main():
    df = pd.read_csv(CSV, parse_dates=["drifter_time", "gu_time"])
    print(f"{len(df)} rows in {Path(CSV).name}")

    # --- the filter cascade, printed so every cut is visible ---------------
    df["coast_km"] = coast_distance_km(df["lon_inter"], df["lat_inter"])
    df["abs_dt"] = df["gu_dt_days"].abs()
    cascade = [("all rows", np.ones(len(df), bool))]
    keep = np.isfinite(df["coast_km"]) & (df["coast_km"].abs() <= MAX_COAST_KM)
    cascade.append((f"coast <= {MAX_COAST_KM:.0f} km", keep.to_numpy()))
    if DROGUED_ONLY:
        keep = keep & df["drogued"]
        cascade.append(("+ drogued only", keep.to_numpy()))
    if MAX_ABS_TIME_DIFF_DAYS is not None:
        keep = keep & (df["abs_dt"] <= MAX_ABS_TIME_DIFF_DAYS)
        cascade.append((f"+ |dt| <= {MAX_ABS_TIME_DIFF_DAYS:g} d", keep.to_numpy()))
    if MAX_CLOSEST_KM is not None:
        keep = keep & (df["closest_km"] <= MAX_CLOSEST_KM)
        cascade.append((f"+ closest <= {MAX_CLOSEST_KM:g} km", keep.to_numpy()))

    yv = df["drifter_ve"].to_numpy(float)
    xv = df["gu"].to_numpy(float) + df[EKMAN_COL].to_numpy(float)
    print(f"\n  {'filter':<26} {'n':>5} {'r':>7} {'RMA':>7} {'rmse(1:1)':>10}")
    for label, mask in cascade:
        s = stats(xv[mask], yv[mask])
        if s:
            print(f"  {label:<26} {s['n']:>5} {s['r']:>7.3f} {s['rma_slope']:>7.3f} "
                  f"{s['rmse_11']:>10.3f}")

    sub = df.loc[keep].reset_index(drop=True)
    groups = (len(sub.groupby(["track_self", "track_other"])), sub["drifter_id"].nunique())
    y = sub["drifter_ve"].to_numpy(float)
    x_no = sub["gu"].to_numpy(float)
    x_ek = x_no + sub[EKMAN_COL].to_numpy(float)

    vals = np.concatenate([a[np.isfinite(a)] for a in (x_no, x_ek, y)])
    lim = 1.05 * np.abs(vals).max()
    extent = (-lim, lim, -lim, lim)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharex=True, sharey=True,
                             constrained_layout=True)
    hb0, s0 = add_panel(axes[0], x_no, y, "Without Ekman", "Altimetry gu (m/s)",
                        "Drifter ve (m/s)", "a)", extent, groups)
    hb1, s1 = add_panel(axes[1], x_ek, y, f"With Ekman ({EKMAN_COL})",
                        f"Altimetry gu + {EKMAN_COL} (m/s)", None, "b)", extent, groups)
    if hb0 is not None and hb1 is not None:
        cmax = max(hb0.get_array().max(), hb1.get_array().max())
        hb0.set_clim(1, cmax)
        hb1.set_clim(1, cmax)
        fig.colorbar(hb1, ax=axes, location="right", fraction=0.03,
                     pad=0.02).set_label("Count")

    cuts = [f"coast <= {MAX_COAST_KM:.0f} km"]
    if DROGUED_ONLY:
        cuts.append("drogued only")
    if MAX_ABS_TIME_DIFF_DAYS is not None:
        cuts.append(f"|dt| <= {MAX_ABS_TIME_DIFF_DAYS:g} d")
    fig.suptitle(f"{SAT}: drifter ve vs altimetry   |   " + ",  ".join(cuts))

    xr.Dataset(
        {f"{k}_{f}": ((), v) for k, s in (("noek", s0), ("ek", s1)) for f, v in s.items()},
        attrs={"sat": SAT, "ekman_col": EKMAN_COL, "max_coast_km": MAX_COAST_KM,
               "drogued_only": int(DROGUED_ONLY),
               "max_abs_dt_days": (-1.0 if MAX_ABS_TIME_DIFF_DAYS is None
                                   else MAX_ABS_TIME_DIFF_DAYS),
               "n_intersections": groups[0], "n_drifters": groups[1],
               "rows_total": len(df), "rows_used": len(sub), "source": script.name},
    ).to_netcdf(nc_file)
    print(f"\nwrote {nc_file}")

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"wrote {png_file}")


if __name__ == "__main__":
    main()
