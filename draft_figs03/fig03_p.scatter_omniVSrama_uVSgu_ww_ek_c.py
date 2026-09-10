"""Figure 3 -- RAMA and OMNI zonal current against altimetry gu, with and without
the Ekman contribution, as hexbin scatters with a fit.

Rewrite of draft_figs02/fig03_p.scatter_omniVSrama_uVSgu_ww_ek.py.
Changes that affect the numbers on the figure, not just the code:

  1. EKMAN DEPTH MATCHED TO THE BUOY.  Both CSV producers
     (guv/p.{omni,rama}_uvVStrack_guv_c.ek.nearest.all.py) take the buoy current
     at dpth = 15.0 m but set EKMAN_DEPTH = "0", so the published u_ek column is
     the SURFACE Ekman.  The geostrophic part is near depth-independent in the
     upper layer; the Ekman spiral is not.  Measured on these same rows:

        RAMA  gu alone          r 0.483  RMA 1.128  rmse(1:1) 0.177
              gu + ek  0 m      r 0.662  RMA 0.797  rmse(1:1) 0.175
              gu + ek 15 m      r 0.622  RMA 1.030  rmse(1:1) 0.157
        OMNI  gu alone          r 0.516  RMA 0.795  rmse(1:1) 0.198
              gu + ek  0 m      r 0.584  RMA 0.686  rmse(1:1) 0.209   <- WORSE
              gu + ek 15 m      r 0.573  RMA 0.765  rmse(1:1) 0.192

     At 0 m the Ekman is too strong for a 15 m measurement: it fixes the phase
     (r rises) but overshoots the amplitude, and for OMNI it makes the rmse
     about the 1:1 line WORSE than gu alone.  At the matching depth the rmse
     improves instead and RAMA's amplitude match becomes RMA slope 1.03.
     u_ek is therefore re-attached here from
     ekman/ekman_at_intersections_cd/*_ekman_{EKMAN_DEPTH}m_*.nc, matched on
     calendar date.  Set EKMAN_DEPTH = 0 to reproduce the published panels.

  2. THE SLOPE IS NO LONGER ORDINARY LEAST SQUARES.  np.polyfit(x, y, 1)
     regresses the buoy on the altimetry; both are noisy, so OLS is biased low
     (regression dilution) -- OMNI OLS 0.400 against a reduced-major-axis 0.687.
     Printing "y = 0.400x" reads as a 2.5x amplitude error that is mostly the
     estimator.  The drawn fit is now reduced major axis; the OLS slope is still
     reported in the box so the difference stays visible.

  3. THE ERROR IS NOW MEASURED ABOUT THE 1:1 LINE.  The original reported
     sqrt(sum((y - yhat)^2)/(n-2)), the residual about the FITTED line
     (0.137 / 0.142), while drawing a 1:1 line.  The disagreement about 1:1 is
     0.175 / 0.209 -- 22-47% larger.  Both are now shown, labelled.

  4. N NO LONGER OVERSTATES THE SAMPLE.  205 RAMA rows come from 10
     intersections and 5 buoys; 1148 OMNI rows from 26 intersections and 12
     buoys.  The points are clustered and serially correlated, so the panel now
     reports intersections and buoys alongside N.

  5. Axis limits follow the data (2 of 1148 OMNI gu values reached 1.28 and were
     binned off-screen by the fixed +-1 limits, while still driving the colour
     scale).  The title reports the distance range actually present rather than
     the nominal MAX_DISTANCE_KM, which is inert here: RAMA reaches 79.5 km and
     OMNI 68.7 km, so a 250 km cut removes nothing.

Checked and left alone: u_ek has no NaN in either CSV, so the no-Ekman and
with-Ekman panels share the same rows (the 25% of intersections with an all-NaN
Ekman file, summary section 11, are all outside this selection).  Both CSV
producers read the stored guv_at_intersection_*.nc, so the section 8.6
local-copy sign problem does not apply here.  Both apply the 72 h buoy filter.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

CSV_DIR = "../guv"
EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd"
RAMA_CSV = f"{CSV_DIR}/rama_guv_c_nearest_global_matches.ek.csv"
OMNI_CSV = f"{CSV_DIR}/omni_guv_c_nearest_global_matches.ek.csv"

EKMAN_DEPTH = 15          # m, matched to the buoy bin; 0 reproduces the original
MIN_DISTANCE_KM = None
MAX_DISTANCE_KM = 250.0

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")
nc_file = script.with_name(script.stem + ".nc")


def reattach_ekman(df, depth):
    """u_ek at `depth` for every row, matched on calendar date.

    The CSV's own u_ek went through the producer's 72 h filter and daily
    resample; re-attaching the 0 m field this way reproduces it to corr 0.998
    (median |diff| 0.005 m/s), which the script prints as a check.
    """
    out = np.full(len(df), np.nan)
    for (sat, self_, other), idx in df.groupby(
            ["sat", "track_self", "track_other"]).groups.items():
        fn = f"{EKMAN_DIR}/{sat}_{self_}_{other}_ekman_{depth}m_at_intersections_cd.nc"
        if not os.path.exists(fn):
            continue
        with xr.open_dataset(fn) as ds:
            series = pd.Series(np.asarray(ds["u_at"].values, dtype=float),
                               index=pd.DatetimeIndex(ds["time"].values).normalize())
        series = series[~series.index.duplicated()]
        want = pd.DatetimeIndex(df.loc[idx, "timestamp"]).normalize()
        out[df.index.get_indexer(idx)] = series.reindex(want).to_numpy()
    return out


def load(csv_path, y_col):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    dist = df["distance_km"].to_numpy(dtype=float)
    keep = np.isfinite(dist)
    if MIN_DISTANCE_KM is not None:
        keep &= dist >= MIN_DISTANCE_KM
    if MAX_DISTANCE_KM is not None:
        keep &= dist <= MAX_DISTANCE_KM
    df = df.loc[keep].reset_index(drop=True)

    df["u_ek_matched"] = reattach_ekman(df, EKMAN_DEPTH)
    check = reattach_ekman(df, 0)
    ok = np.isfinite(check) & np.isfinite(df["u_ek"].to_numpy(dtype=float))
    if ok.sum():
        print(f"  {Path(csv_path).name}: re-attached 0 m vs the CSV u_ek -> "
              f"corr {np.corrcoef(check[ok], df['u_ek'].to_numpy(float)[ok])[0, 1]:.4f}, "
              f"median |diff| "
              f"{np.median(np.abs(check[ok] - df['u_ek'].to_numpy(float)[ok])):.4f} m/s")
    df["y"] = df[y_col]
    return df


def stats(x, y):
    """r, reduced-major-axis and OLS slopes, and both error measures."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    ols_slope, ols_int = np.polyfit(x, y, 1)
    # reduced major axis: symmetric in x and y, so noise in the altimetry does
    # not drag the slope toward zero the way it does in OLS
    rma_slope = np.sign(np.cov(x, y)[0, 1]) * np.sqrt(y.var() / x.var())
    rma_int = y.mean() - rma_slope * x.mean()
    return dict(
        n=x.size, r=r,
        ols_slope=float(ols_slope), ols_int=float(ols_int),
        rma_slope=float(rma_slope), rma_int=float(rma_int),
        rmse_11=float(np.sqrt(np.mean((y - x) ** 2))),
        resid_fit=float(np.sqrt(np.sum((y - (ols_slope * x + ols_int)) ** 2)
                                / (x.size - 2))),
        bias=float(np.mean(y - x)),
    )


def add_panel(ax, x, y, title, xlabel, ylabel, panel_label, extent, groups):
    s = stats(x, y)
    m = np.isfinite(x) & np.isfinite(y)
    hb = ax.hexbin(x[m], y[m], gridsize=55, cmap="viridis", mincnt=1, extent=extent)
    xs = np.linspace(extent[0], extent[1], 200)
    ax.plot(xs, s["rma_slope"] * xs + s["rma_int"], color="crimson", lw=2.0,
            label="Reduced major axis")
    # black, not white: the axes now extend past the hexbins onto white ground
    ax.plot(xs, xs, color="0.25", ls="--", lw=1.2, label="1:1 line")
    ax.text(0.03, 0.97,
            f"N = {s['n']}  ({groups[0]} intersections, {groups[1]} buoys)\n"
            f"r = {s['r']:.3f}\n"
            f"RMA slope = {s['rma_slope']:.3f}   (OLS {s['ols_slope']:.3f})\n"
            f"rmse vs 1:1 = {s['rmse_11']:.3f} m/s\n"
            f"resid about fit = {s['resid_fit']:.3f} m/s\n"
            f"bias = {s['bias']:+.3f} m/s",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.6, ec="gray"))
    ax.text(0.62, 0.98, title, transform=ax.transAxes, va="top", ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.6, ec="white"))
    ax.text(0.03, 0.03, panel_label, transform=ax.transAxes, va="bottom",
            ha="left", fontsize=14, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return hb, s


def main():
    print("re-attachment check:")
    rama = load(RAMA_CSV, "rama_u")
    omni = load(OMNI_CSV, "omni_u")

    cases = []
    for name, df, id_col in (("RAMA", rama, "rama_id"), ("OMNI", omni, "omni_id")):
        groups = (len(df.groupby(["sat", "track_self", "track_other"])),
                  df[id_col].nunique())
        dmin, dmax = df["distance_km"].min(), df["distance_km"].max()
        for ek in (False, True):
            x = df["gu"].to_numpy(float) + (df["u_ek_matched"].to_numpy(float)
                                            if ek else 0.0)
            cases.append(dict(name=name, ek=ek, x=x, y=df["y"].to_numpy(float),
                              groups=groups, dmin=dmin, dmax=dmax))

    allx = np.concatenate([c["x"][np.isfinite(c["x"])] for c in cases])
    ally = np.concatenate([c["y"][np.isfinite(c["y"])] for c in cases])
    lim = 1.05 * max(np.abs(allx).max(), np.abs(ally).max())
    extent = (-lim, lim, -lim, lim)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True,
                            constrained_layout=True)
    hbs, results = {}, {}
    for ax, c, lab in zip(axs.ravel(), cases, "abcd"):
        ek_lbl = f"gu + u_ek ({EKMAN_DEPTH} m)" if c["ek"] else "gu (no Ekman)"
        title = (f"{c['name']} u vs {ek_lbl}\n"
                 f"{c['dmin']:.0f}-{c['dmax']:.0f} km from the buoy")
        xlab = (f"Altimetry gu + u_ek ({EKMAN_DEPTH} m) (m/s)" if c["ek"]
                else "Altimetry gu (m/s)") if lab in "cd" else None
        ylab = f"{c['name']} u (m/s)" if lab in "ac" else None
        hb, s = add_panel(ax, c["x"], c["y"], title, xlab, ylab, f"{lab})",
                          extent, c["groups"])
        key = f"{c['name']}_{'ek' if c['ek'] else 'noek'}"
        hbs[key], results[key] = hb, s
        print(f"  {key:<10} n={s['n']:4d} r={s['r']:.3f} RMA={s['rma_slope']:.3f} "
              f"OLS={s['ols_slope']:.3f} rmse(1:1)={s['rmse_11']:.3f} "
              f"bias={s['bias']:+.3f}")

    for row, keys in ((0, ("RAMA_noek", "RAMA_ek")), (1, ("OMNI_noek", "OMNI_ek"))):
        vmax = max(float(np.nanmax(hbs[k].get_array())) for k in keys)
        for k in keys:
            hbs[k].set_clim(1.0, vmax)
        fig.colorbar(hbs[keys[0]], ax=axs[row, :], location="right",
                     fraction=0.03, pad=0.02).set_label("Count")

    xr.Dataset(
        {f"{k}_{f}": ((), v) for k, s in results.items() for f, v in s.items()},
        attrs={"ekman_depth_m": EKMAN_DEPTH,
               "max_distance_km": MAX_DISTANCE_KM,
               "rama_rows": len(rama), "omni_rows": len(omni),
               "source": script.name},
    ).to_netcdf(nc_file)
    print(f"wrote {nc_file}")

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"wrote {png_file}")


if __name__ == "__main__":
    main()
