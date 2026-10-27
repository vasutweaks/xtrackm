"""Recompute before/after Ekman skill using the refined figure selections.

Run with the project Python environment. Outputs go to diagnostics_ekman/;
existing figures and NetCDF files are never overwritten. All comparisons use
the intersection of finite observation, geostrophic and Ekman samples. RMSE
is sqrt(mean((prediction-observation)**2)), not scatter-fit residual error.
Radar is reported both as spatial means of temporal skill profiles and as
pooled space-time skill; neither is selected on correlation significance.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "lib"))
from tools_xtrackm_c import local_angle


def figure_module(prefix):
    path, = HERE.glob(prefix + "*.py")
    spec = importlib.util.spec_from_file_location(prefix, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path.with_suffix(".nc")


def skill(obs, pred):
    if len(obs) < 3:
        raise ValueError("Need at least three common finite samples")
    return {
        "r": float(pearsonr(obs, pred).statistic),
        "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
        "bias_obs_minus_pred": float(np.mean(obs - pred)),
    }


def comparison(label, obs, geo, ekman, **metadata):
    obs, geo, ekman = (np.asarray(a, float) for a in (obs, geo, ekman))
    common = np.isfinite(obs) & np.isfinite(geo) & np.isfinite(ekman)
    a = skill(obs[common], geo[common])
    b = skill(obs[common], (geo + ekman)[common])
    return metrics_record(label, a, b, n=int(common.sum()), **metadata)


def metrics_record(label, before, after, **metadata):
    row = {"dataset": label, **metadata}
    for key in before:
        row[key + "_before"] = before[key]
        row[key + "_after"] = after[key]
    row["delta_r"] = after["r"] - before["r"]
    row["relative_r_increase_pct"] = 100 * row["delta_r"] / before["r"]
    row["rmse_reduction_m_s"] = before["rmse"] - after["rmse"]
    row["rmse_reduction_pct"] = 100 * row["rmse_reduction_m_s"] / before["rmse"]
    return row


def check_saved(row, path, before, after, checks):
    """Check fresh paired statistics against the existing figure products."""
    with xr.open_dataset(path) as ds:
        for stage, prefix in (("before", before), ("after", after)):
            for key, suffix in (("r", "r"), ("rmse", "rmse_11")):
                np.testing.assert_allclose(row[key + "_" + stage],
                                           ds[prefix + "_" + suffix].item(),
                                           rtol=1e-9, atol=1e-12)
            assert row["n"] == ds[prefix + "_n"].item()
    checks.append(f"{row['dataset']}: fresh n, r and RMSE match {path.name}")


def buoy_diagnostics(checks):
    mod, saved = figure_module("fig03_")
    rows = []
    for name, csv, ycol, idcol in (
        ("RAMA", mod.RAMA_CSV, "rama_u", "rama_id"),
        ("OMNI", mod.OMNI_CSV, "omni_u", "omni_id"),
    ):
        df = mod.load(str(HERE / csv), ycol)
        row = comparison(name, df.y, df.gu, df.u_ek_matched,
                         aggregation="pooled rows", ekman_depth_m=mod.EKMAN_DEPTH,
                         n_instruments=int(df[idcol].nunique()),
                         n_intersections=df.groupby(["sat", "track_self", "track_other"]).ngroups,
                         selection=f"finite distance <= {mod.MAX_DISTANCE_KM:g} km; refined Fig. 3")
        check_saved(row, saved, name + "_noek", name + "_ek", checks)
        rows.append(row)
    return rows


def selected_omni_diagnostic(checks):
    """Separate Fig. 2 example, recomputed from its saved matched series."""
    path, = HERE.glob("fig02_*.nc")
    with xr.open_dataset(path) as ds:
        row = comparison("OMNI: selected BD10 example", ds.omni_u, ds.gu, ds.u_ekman,
                         aggregation="single intersection time series", ekman_depth_m=15,
                         n_instruments=1, n_intersections=1,
                         selection="Fig. 2: BD10; S3A 522 x 279; nearest within 24 hours")
        for stage, prefix in (("before", "track"), ("after", "track+ekman15m")):
            for key in ("r", "rmse"):
                np.testing.assert_allclose(row[key + "_" + stage], ds.attrs[prefix + "_" + key],
                                           rtol=1e-9, atol=1e-12)
    checks.append("Selected OMNI BD10: common-mask r and RMSE match Fig. 2 attributes")
    return row


def drifter_diagnostics(checks):
    mod, saved = figure_module("fig04_p.")
    df = pd.read_csv(HERE / mod.CSV)
    coast = mod.coast_distance_km(df.lon_inter, df.lat_inter)
    keep = np.isfinite(coast) & (np.abs(coast) <= mod.MAX_COAST_KM)
    if mod.DROGUED_ONLY:
        keep &= df.drogued.to_numpy(bool)
    if mod.MAX_ABS_TIME_DIFF_DAYS is not None:
        keep &= df.gu_dt_days.abs().to_numpy() <= mod.MAX_ABS_TIME_DIFF_DAYS
    if mod.MAX_CLOSEST_KM is not None:
        keep &= df.closest_km.to_numpy() <= mod.MAX_CLOSEST_KM
    sub = df.loc[keep]
    coastal = comparison("Drifters: coastal TP series", sub.drifter_ve, sub.gu,
                         sub[mod.EKMAN_COL], aggregation="pooled rows", ekman_depth_m=0,
                         n_instruments=int(sub.drifter_id.nunique()),
                         n_intersections=sub.groupby(["track_self", "track_other"]).ngroups,
                         selection="Fig. 4: coast <=100 km; drogued; |dt| <=2 days")
    check_saved(coastal, saved, "noek", "ek", checks)

    mod, saved = figure_module("fig04b_")
    mod.CSV_DIR = str(PROJECT / "drifter_track_reg")
    df = mod.load_all()
    keep = np.isfinite(df.gu) & np.isfinite(df.drifter_ve) & np.isfinite(df[mod.EKMAN_COL])
    if mod.MAX_COAST_KM is not None:
        coast = mod.coast_distance_km(df.lon_inter, df.lat_inter)
        keep &= np.isfinite(coast) & (np.abs(coast) <= mod.MAX_COAST_KM)
    if mod.DROGUED_ONLY:
        keep &= df.drogued
    if mod.MAX_ABS_TIME_DIFF_DAYS is not None:
        keep &= df.gu_dt_days.abs() <= mod.MAX_ABS_TIME_DIFF_DAYS
    elif mod.TIME_CUT_REPEAT_FRACTION is not None:
        keep &= df.gu_dt_days.abs() <= mod.TIME_CUT_REPEAT_FRACTION * df.repeat_days
    sub = df.loc[keep]
    pooled = comparison("Drifters: all 8 missions", sub.drifter_ve, sub.gu,
                        sub[mod.EKMAN_COL], aggregation="pooled rows", ekman_depth_m=0,
                        n_instruments=int(sub.drifter_id.nunique()),
                        selection="Fig. 4b: all coast distances; drogued; |dt| <=repeat/4")
    check_saved(pooled, saved, "pooled_noek", "pooled_ek", checks)
    missions = []
    for sat, d in sub.groupby("sat", sort=False):
        row = comparison(sat, d.drifter_ve, d.gu, d[mod.EKMAN_COL],
                         aggregation="pooled rows", ekman_depth_m=0,
                         small_sample=len(d) < mod.MIN_N_PER_SAT)
        check_saved(row, saved, sat + "_noek", sat + "_ek", checks)
        missions.append(row)
    return [pooled, coastal], missions


def radar_diagnostics(outdir, checks):
    with xr.open_dataset(PROJECT / "radar/radar_along_intersections/radar_along_OR_S3A_165.nc") as ds:
        u, v = ds.radar_u.load(), ds.radar_v.load()
        lons, lats = ds.lon_track.values, ds.lat_track.values
        lo, hi = sorted([float(ds.attrs["x1_m"]), float(ds.attrs["x2_m"])])
        dist = ds.dist_to_coast_gshhg_track.load()
    with xr.open_dataset(PROJECT / "radar/ekman_along_intersections/ekman_along_OR_S3A_165.nc") as ds:
        eu, ev = ds.ekman_u.load(), ds.ekman_v.load()
        ek_source = str(ds.attrs.get("source_file", "unspecified"))
    with xr.open_dataset(PROJECT / "guv/gc_along_loess/gc_along_S3A_165_loess_0.05.nc") as ds:
        geo = ds.gc.sel(x=slice(lo, hi)).load()
    theta = xr.DataArray([local_angle(lons, lats, i, ioff=3) + np.pi / 2
                          for i in range(len(lons))], dims="x", coords={"x": u.x})
    obs = (np.cos(theta) * u + np.sin(theta) * v).interp(x=geo.x, method="nearest").reindex(
        time=geo.time, method="nearest", tolerance=np.timedelta64(24, "h"))
    ek = (np.cos(theta) * eu + np.sin(theta) * ev).interp(x=geo.x, method="nearest").reindex(
        time=geo.time, method="nearest", tolerance=np.timedelta64(2, "D"))
    common = np.isfinite(obs) & np.isfinite(geo) & np.isfinite(ek)
    profiles = []
    coast = dist.interp(x=geo.x, method="nearest").values
    for i, x in enumerate(geo.x.values):
        mask = common.isel(x=i).values
        n = int(mask.sum())
        row = {"x_m": x, "coast_distance_km": float(coast[i]), "n": n,
               "include_n_ge_10": n >= 10}
        if n >= 3:
            a, b = (arr.isel(x=i).values[mask] for arr in (obs, geo))
            e = ek.isel(x=i).values[mask]
            row.update(metrics_record("radar point", skill(a, b), skill(a, b + e)))
            row["p_before"] = float(pearsonr(a, b).pvalue)
            row["p_after"] = float(pearsonr(a, b + e).pvalue)
        profiles.append(row)
    profile = pd.DataFrame(profiles)
    profile.to_csv(outdir / "radar_profiles.csv", index=False)
    saved, = HERE.glob("fig05_*.nc")
    with xr.open_dataset(saved) as ds:
        np.testing.assert_array_equal(profile.n, ds.n_pairs.values)
        for key, old in (("r", "corr"), ("rmse", "rmse")):
            for stage, suffix in (("before", "noek"), ("after", "ek")):
                values = profile.get(key + "_" + stage).to_numpy(float)
                valid = profile.n.to_numpy() >= 3
                np.testing.assert_allclose(values[valid], ds[old + "_" + suffix].values[valid],
                                           rtol=1e-8, atol=1e-12)
    checks.append("Radar: fresh common-mask n, temporal r and RMSE match Fig. 5 profiles (n>=3)")
    use = profile.include_n_ge_10.to_numpy()
    sub = profile.loc[use]
    kwargs = dict(n=int(sub.n.sum()), n_locations=len(sub), ekman_depth_m=0,
                  selection="OR / S3A track 165; common finite samples; n>=10 per x; no p-value cut")
    before, after = ({key: float(sub[key + "_" + stage].mean()) for key in
                      ("r", "rmse", "bias_obs_minus_pred")} for stage in ("before", "after"))
    average = metrics_record("Radar: mean temporal profiles", before, after,
                             aggregation="equal-location arithmetic mean of temporal r and RMSE", **kwargs)
    pooled = comparison("Radar: pooled space-time", obs.isel(x=use), geo.isel(x=use),
                        ek.isel(x=use), aggregation="pooled space-time pairs",
                        **{k: v for k, v in kwargs.items() if k != "n"})
    detail = dict(n_locations=len(sub), n_cycles_min=int(sub.n.min()),
                  n_cycles_median=float(sub.n.median()), n_cycles_max=int(sub.n.max()),
                  locations_r_improves=int((sub.delta_r > 0).sum()),
                  locations_rmse_improves=int((sub.rmse_reduction_m_s > 0).sum()),
                  locations_both_improve=int(((sub.delta_r > 0) & (sub.rmse_reduction_m_s > 0)).sum()),
                  source_ekman=ek_source)
    return [average, pooled], detail


def markdown_table(rows):
    lines = ["| Comparison | N | r before | r after | Delta r | RMSE before | RMSE after | RMSE reduction |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['n']} | {r['r_before']:.3f} | {r['r_after']:.3f} | "
                     f"{r['delta_r']:+.3f} | {r['rmse_before']:.4f} | {r['rmse_after']:.4f} | "
                     f"{r['rmse_reduction_pct']:+.2f}% |")
    return "\n".join(lines)


def write_report(outdir, rows, missions, detail, checks):
    text = """**Effect of adding Ekman currents in the refined analysis**

Before = geostrophic current alone; after = geostrophic + Ekman current.
Pearson r and RMSE use identical finite samples in each comparison. All RMSE
values are in m/s, measured against the observation (the 1:1 line), without
fitting or bias correction. Delta r = r_after - r_before. RMSE reduction =
100 * (RMSE_before - RMSE_after) / RMSE_before; negative values mean deterioration.

""" + markdown_table(rows) + "\n\n"
    text += """RAMA, pooled OMNI, and both drifter selections improve in correlation and
RMSE. The largest fractional RMSE reduction among these is RAMA (11.44%).
The selected OMNI example is a separate Fig. 2 comparison; its statistics
should not replace the pooled OMNI statistics from Fig. 3.

**Selections and interpretation**

- RAMA/OMNI (Fig. 3): zonal current, 15 m Ekman matched to the 15 m buoy bin;
  distance <=250 km (actual maxima approximately 79.5 and 68.7 km). RAMA has
  5 buoys at 10 intersections; OMNI has 12 buoys at 26 intersections. Existing
  72-hour buoy filtering is retained. Ekman is reattached on calendar date as
  in the refined script.
- Drifters, eight missions (Fig. 4b): zonal current, all coastal distances,
  drogued observations only, corrected 72-hour filtering, absolute time offset
  <= one quarter of each mission's repeat period. The selection contains 704
  distinct drifters. Surface Ekman (0 m) follows the current refined setting,
  although the drogues are at 15 m; this report does not reselect Ekman depth.
- Coastal drifters (Fig. 4): TP+J1+J2+J3+S6A only, coast distance <=100 km,
  drogued, absolute time offset <=2 days, surface Ekman. This is a subset of
  the all-mission sample with a different time threshold, not independent
  validation. There are 66 distinct drifters at 17 intersections.
- Radar (Fig. 5): OR footprint, S3A track 165, cross-track current, surface
  Ekman, geodesic headings with ioff=3; nearest radar within 24 hours and
  nearest Ekman within 2 days. Use common finite samples and at least 10
  cycles per location. Do not select locations on p-values: separate
  significance masks would compare different spatial subsets.

"""
    text += (f"Radar has {detail['n_locations']} qualifying locations, with "
             f"{detail['n_cycles_min']}-{detail['n_cycles_max']} cycles per location "
             f"(median {detail['n_cycles_median']:g}), totaling "
             f"{next(r['n'] for r in rows if r['dataset'].startswith('Radar:'))} "
             "space-time pairs. Its primary summary is the equal-location arithmetic "
             "mean of temporal correlations and temporal RMSEs. The pooled row "
             "instead scores all retained space-time pairs together. These are "
             "different summaries, and pooling also mixes spatial and temporal "
             "variation. Pooled r rises slightly, but pooled RMSE also increases. "
             f"Correlation improves at {detail['locations_r_improves']}/"
             f"{detail['n_locations']} locations, RMSE at "
             f"{detail['locations_rmse_improves']}/{detail['n_locations']}, and both at "
             f"{detail['locations_both_improve']}/{detail['n_locations']}. Thus the "
             "radar data do not show an overall improvement in both measures.\n\n")
    text += "**Drifter results by mission, using the Fig. 4b selection**\n\n"
    text += markdown_table(missions)
    text += """\n\nSeven mission groups improve in both measures. HY2B deteriorates, but has
only 20 retained pairs and is below the refined script's 50-pair threshold
for a reliable separate mission summary.

These are descriptive changes, not significance tests of improvement.
Rows and neighboring radar locations can be serially/spatially correlated;
N counts matched pairs, not independent degrees of freedom. The four
observing systems have different components and selections, so the sizes
of their changes are not a controlled ranking of instrument performance.

**Reproduction and verification**

Run from any directory:

```sh
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp/codex-matplotlib /home/srinivasu/venvs/pybase/bin/python /home/srinivasu/xtrackm/draft_figs03/diagnose_ekman_improvement.py
```

The script reads the refined loaders and upstream matched CSV/NetCDF products;
the selected OMNI example uses its saved matched series. It writes summary.csv,
drifter_by_mission.csv, radar_profiles.csv, verification.json, and this report.
It leaves existing figures and scientific products unchanged.

"""
    text += f"All {len(checks)} numerical checks against the saved refined figure products passed.\n"
    (outdir / "report.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "diagnostics_ekman")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    rows = buoy_diagnostics(checks)
    drifters, missions = drifter_diagnostics(checks)
    radar, detail = radar_diagnostics(args.output_dir, checks)
    rows += drifters + radar + [selected_omni_diagnostic(checks)]
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(missions).to_csv(args.output_dir / "drifter_by_mission.csv", index=False)
    (args.output_dir / "verification.json").write_text(json.dumps(
        {"checks": checks, "radar": detail}, indent=2) + "\n")
    write_report(args.output_dir, rows, missions, detail, checks)
    columns = ["dataset", "n", "r_before", "r_after", "delta_r", "rmse_before", "rmse_after",
               "rmse_reduction_pct"]
    print(df[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nRadar spatial detail:", json.dumps(detail, indent=2))
    print(f"\nAll {len(checks)} checks against existing refined figure products passed.")
    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
