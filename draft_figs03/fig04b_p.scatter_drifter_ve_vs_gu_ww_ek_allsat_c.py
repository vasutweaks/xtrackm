"""Figure 4b -- GDP drifter ve against altimetry gu (+ Ekman), ALL EIGHT MISSIONS
pooled, with the pooled result decomposed by mission.

Companion to fig04_..._c.py, which shows TP+J1+J2+J3+S6A alone.  The candidate
pickles previously existed for only three missions, so the drifter comparison
could not be pooled; c.candidate_buoys.for.intersection_points_c.py now
generates them for all eight and the extraction produces one CSV per mission --
15042 rows against the 1622 of the single-mission figure.

The same two corrections apply as in fig04 (summary section 14): the drogue
state is honoured, and drifter ve is low-passed at a real 3 days rather than the
18 hours the hardcoded fs = 1 sample/day produced on a 6-hourly series.

THE TIME CUT IS SCALED BY THE REPEAT PERIOD.  |drifter_time - gu_time| is the
offset to the NEAREST altimeter cycle, so it is bounded by half the repeat and
its median tracks the mission:
    TP/TPN 10 d repeat -> median 2.50 / 2.75 d      HY2A/HY2B 14 d -> 4.00 d
    GFO    17 d        -> 4.50 d                    S3A/S3B   27 d -> 6.75 / 7.00 d
    ERS    35 d        -> 8.75 d
A fixed 2-day cut would therefore keep 69 of 169 TP points but only 34 of 239
S3A and 78 of 657 ERS -- it silently selects for the short-repeat missions.
TIME_CUT_REPEAT_FRACTION cuts at a fraction of each mission's own repeat, which
removes a comparable share everywhere.  Set MAX_ABS_TIME_DIFF_DAYS instead for
an absolute cut.

HY2B is retained but is not usable on its own: 91 rows, 5 surviving the quality
cuts.  It contributes ~0.4% of the pooled sample; panel (c)/(d) show why it
should not be quoted separately.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "/home/srinivasu/xtrackm/lib")
from namesm import sats_new, sats_tfreq

CSV_DIR = "../drifter_track_reg"
COAST_DIST_PATH = "~/allData/topo/GMT_intermediate_coast_distance_01d_track_reg.nc"

MAX_COAST_KM = None            # None = every distance; 100.0 reproduces fig04's cut
DROGUED_ONLY = True
TIME_CUT_REPEAT_FRACTION = 0.25   # mission-fair; None to disable
MAX_ABS_TIME_DIFF_DAYS = None     # absolute alternative, applied if set
EKMAN_COL = "u_ek"
MIN_N_PER_SAT = 50                # below this a mission is flagged as unreliable

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
    if x.size < 5:
        return None
    ols_slope, ols_int = np.polyfit(x, y, 1)
    rma_slope = np.sign(np.cov(x, y)[0, 1]) * np.sqrt(y.var() / x.var())
    return dict(
        n=x.size, r=float(np.corrcoef(x, y)[0, 1]),
        ols_slope=float(ols_slope), ols_int=float(ols_int),
        rma_slope=float(rma_slope), rma_int=float(y.mean() - rma_slope * x.mean()),
        rmse_11=float(np.sqrt(np.mean((y - x) ** 2))),
        resid_fit=float(np.sqrt(np.sum((y - (ols_slope * x + ols_int)) ** 2) / (x.size - 2))),
        bias=float(np.mean(y - x)),
    )


def load_all():
    frames = []
    for sat in sats_new:
        fn = f"{CSV_DIR}/drifter_ve_vs_gu_ek_{sat}_c.csv"
        if not os.path.exists(fn):
            print(f"  {sat:<22} missing {Path(fn).name}")
            continue
        d = pd.read_csv(fn)
        if not len(d):
            continue
        d["sat"] = sat
        d["repeat_days"] = float(sats_tfreq[sat])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def scatter_panel(ax, df, x, y, title, xlabel, ylabel, panel_label, lim, colors):
    s = stats(x, y)
    for sat in [a for a in sats_new if a in set(df["sat"])]:
        m = (df["sat"] == sat).to_numpy() & np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=14, alpha=0.55, color=colors[sat],
                   edgecolors="none", label=f"{sat} ({m.sum()})", zorder=3)
    xs = np.linspace(-lim, lim, 200)
    ax.plot(xs, s["rma_slope"] * xs + s["rma_int"], color="crimson", lw=2.2,
            zorder=5, label="Reduced major axis")
    ax.plot(xs, xs, color="0.25", ls="--", lw=1.2, zorder=4, label="1:1 line")
    ax.text(0.03, 0.97,
            f"N = {s['n']}  ({df['sat'].nunique()} missions, "
            f"{df['drifter_id'].nunique()} drifters)\n"
            f"r = {s['r']:.3f}\n"
            f"RMA slope = {s['rma_slope']:.3f}   (OLS {s['ols_slope']:.3f})\n"
            f"rmse vs 1:1 = {s['rmse_11']:.3f} m/s\n"
            f"bias = {s['bias']:+.3f} m/s",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.75, ec="gray"))
    ax.text(0.5, 0.985, title, transform=ax.transAxes, va="top", ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.6, ec="white"))
    ax.text(0.03, 0.03, panel_label, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return s


def main():
    print("loading per-mission CSVs ...")
    df = load_all()
    df["coast_km"] = coast_distance_km(df["lon_inter"], df["lat_inter"])
    df["abs_dt"] = df["gu_dt_days"].abs()
    x_all = (df["gu"] + df[EKMAN_COL]).to_numpy(float)
    y_all = df["drifter_ve"].to_numpy(float)

    # both panels must use identical rows, so require a finite Ekman value even
    # in the without-Ekman panel (3 rows differed otherwise)
    keep = np.isfinite(df[EKMAN_COL].to_numpy(float)) & np.isfinite(y_all) & np.isfinite(
        df["gu"].to_numpy(float))
    cascade = [("all rows, all missions", keep.copy())]
    if MAX_COAST_KM is not None:
        keep &= (df["coast_km"].abs() <= MAX_COAST_KM).to_numpy()
        cascade.append((f"coast <= {MAX_COAST_KM:g} km", keep.copy()))
    if DROGUED_ONLY:
        keep &= df["drogued"].to_numpy()
        cascade.append(("drogued only", keep.copy()))
    if MAX_ABS_TIME_DIFF_DAYS is not None:
        keep &= (df["abs_dt"] <= MAX_ABS_TIME_DIFF_DAYS).to_numpy()
        cascade.append((f"|dt| <= {MAX_ABS_TIME_DIFF_DAYS:g} d", keep.copy()))
    elif TIME_CUT_REPEAT_FRACTION is not None:
        keep &= (df["abs_dt"] <= TIME_CUT_REPEAT_FRACTION * df["repeat_days"]).to_numpy()
        cascade.append((f"|dt| <= {TIME_CUT_REPEAT_FRACTION:g} x repeat", keep.copy()))

    print(f"\n  {'filter':<26} {'n':>6} {'r':>7} {'RMA':>7} {'rmse(1:1)':>10}")
    for label, mask in cascade:
        s = stats(x_all[mask], y_all[mask])
        if s:
            print(f"  {label:<26} {s['n']:>6} {s['r']:>7.3f} {s['rma_slope']:>7.3f} "
                  f"{s['rmse_11']:>10.3f}")

    sub = df.loc[keep].reset_index(drop=True)
    x_ek = (sub["gu"] + sub[EKMAN_COL]).to_numpy(float)
    x_no = sub["gu"].to_numpy(float)
    y = sub["drifter_ve"].to_numpy(float)
    lim = 1.05 * np.nanmax(np.abs(np.concatenate([x_no, x_ek, y])))

    present = [s for s in sats_new if s in set(sub["sat"])]
    colors = {s: c for s, c in zip(present, plt.cm.tab10(np.linspace(0, 1, 10)))}

    fig, axs = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    s_no = scatter_panel(axs[0, 0], sub, x_no, y, "Without Ekman",
                         "Altimetry gu (m/s)", "Drifter ve (m/s)", "a)", lim, colors)
    s_ek = scatter_panel(axs[0, 1], sub, x_ek, y, f"With Ekman ({EKMAN_COL})",
                         f"Altimetry gu + {EKMAN_COL} (m/s)", None, "b)", lim, colors)
    axs[0, 1].legend(loc="lower right", fontsize=7, ncol=1, framealpha=0.85)

    # --- (c) and (d): the pooled number decomposed by mission ---------------
    per = {}
    for sat in present:
        m = (sub["sat"] == sat).to_numpy()
        per[sat] = (stats(x_no[m], y[m]), stats(x_ek[m], y[m]), int(m.sum()))
    labels = [s for s in present if per[s][1] is not None]
    idx = np.arange(len(labels))

    # dots rather than bars: the spread in r is narrow (0.62-0.79) and truncating
    # a bar axis to show it would misread
    r_no = [per[s][0]["r"] for s in labels]
    r_ek = [per[s][1]["r"] for s in labels]
    axs[1, 0].vlines(idx, r_no, r_ek, color="0.7", lw=2, zorder=1)
    axs[1, 0].scatter(idx, r_no, s=70, color="0.55", zorder=3, label="without Ekman")
    axs[1, 0].scatter(idx, r_ek, s=70, color="tab:blue", zorder=3, label="with Ekman")
    axs[1, 0].axhline(s_ek["r"], color="crimson", lw=1.6, ls="--",
                      label=f"pooled, with Ekman ({s_ek['r']:.3f})")
    axs[1, 0].set_ylabel("correlation r")
    axs[1, 0].set_title("(c) the pooled r, decomposed by mission", loc="left",
                        fontweight="bold")
    axs[1, 0].legend(fontsize=8, loc="lower left")

    rma = [per[s][1]["rma_slope"] for s in labels]
    small = [per[s][2] < MIN_N_PER_SAT for s in labels]
    axs[1, 1].vlines(idx, 1.0, rma, color="0.7", lw=2, zorder=1)
    axs[1, 1].scatter(idx, rma, s=80, zorder=3,
                      color=["tab:red" if k else "tab:green" for k in small])
    axs[1, 1].axhline(1.0, color="0.25", ls="--", lw=1.2, label="perfect amplitude")
    axs[1, 1].axhline(s_ek["rma_slope"], color="crimson", lw=1.6, ls="--",
                      label=f"pooled ({s_ek['rma_slope']:.3f})")
    for i, s in enumerate(labels):
        axs[1, 1].annotate(f"n={per[s][2]}", (i, rma[i]), ha="center",
                           va="bottom", fontsize=8, xytext=(0, 6),
                           textcoords="offset points")
    axs[1, 1].scatter([], [], s=80, color="tab:red", label=f"n < {MIN_N_PER_SAT}, not reliable")
    axs[1, 1].set_ylabel("RMA slope (with Ekman)")
    axs[1, 1].set_title("(d) amplitude match by mission", loc="left", fontweight="bold")
    axs[1, 1].legend(fontsize=8, loc="lower left")

    for ax in (axs[1, 0], axs[1, 1]):
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(alpha=0.25, axis="y")

    cuts = ["all coast distances" if MAX_COAST_KM is None else f"coast <= {MAX_COAST_KM:g} km"]
    if DROGUED_ONLY:
        cuts.append("drogued only")
    if MAX_ABS_TIME_DIFF_DAYS is not None:
        cuts.append(f"|dt| <= {MAX_ABS_TIME_DIFF_DAYS:g} d")
    elif TIME_CUT_REPEAT_FRACTION is not None:
        cuts.append(f"|dt| <= {TIME_CUT_REPEAT_FRACTION:g} x repeat period")
    fig.suptitle("Drifter ve vs altimetry, all missions pooled   |   " + ",  ".join(cuts),
                 fontsize=13)

    out = {f"pooled_{k}_{f}": ((), v) for k, s in (("noek", s_no), ("ek", s_ek))
           for f, v in s.items()}
    for sat in labels:
        for k, s in (("noek", per[sat][0]), ("ek", per[sat][1])):
            out.update({f"{sat}_{k}_{f}": ((), v) for f, v in s.items()})
    xr.Dataset(out, attrs={
        "missions": ", ".join(present), "ekman_col": EKMAN_COL,
        "max_coast_km": -1.0 if MAX_COAST_KM is None else MAX_COAST_KM,
        "drogued_only": int(DROGUED_ONLY),
        "time_cut_repeat_fraction": (-1.0 if TIME_CUT_REPEAT_FRACTION is None
                                     else TIME_CUT_REPEAT_FRACTION),
        "rows_total": len(df), "rows_used": len(sub), "source": script.name,
    }).to_netcdf(nc_file)
    print(f"\nwrote {nc_file}")

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"wrote {png_file}")


if __name__ == "__main__":
    main()
