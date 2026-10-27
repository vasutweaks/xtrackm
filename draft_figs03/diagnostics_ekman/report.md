**Effect of adding Ekman currents in the refined analysis**

Before = geostrophic current alone; after = geostrophic + Ekman current.
Pearson r and RMSE use identical finite samples in each comparison. All RMSE
values are in m/s, measured against the observation (the 1:1 line), without
fitting or bias correction. Delta r = r_after - r_before. RMSE reduction =
100 * (RMSE_before - RMSE_after) / RMSE_before; negative values mean deterioration.

| Comparison | N | r before | r after | Delta r | RMSE before | RMSE after | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAMA | 205 | 0.483 | 0.622 | +0.139 | 0.1767 | 0.1565 | +11.44% |
| OMNI | 1148 | 0.516 | 0.573 | +0.057 | 0.1982 | 0.1921 | +3.06% |
| Drifters: all 8 missions | 4771 | 0.685 | 0.727 | +0.042 | 0.3016 | 0.2800 | +7.16% |
| Drifters: coastal TP series | 69 | 0.791 | 0.829 | +0.037 | 0.2915 | 0.2724 | +6.53% |
| Radar: mean temporal profiles | 629 | 0.396 | 0.368 | -0.028 | 0.3118 | 0.3267 | -4.76% |
| Radar: pooled space-time | 629 | 0.454 | 0.462 | +0.008 | 0.3116 | 0.3233 | -3.76% |
| OMNI: selected BD10 example | 62 | 0.665 | 0.735 | +0.070 | 0.1736 | 0.1612 | +7.10% |

RAMA, pooled OMNI, and both drifter selections improve in correlation and
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

Radar has 27 qualifying locations, with 13-27 cycles per location (median 25), totaling 629 space-time pairs. Its primary summary is the equal-location arithmetic mean of temporal correlations and temporal RMSEs. The pooled row instead scores all retained space-time pairs together. These are different summaries, and pooling also mixes spatial and temporal variation. Pooled r rises slightly, but pooled RMSE also increases. Correlation improves at 16/27 locations, RMSE at 10/27, and both at 9/27. Thus the radar data do not show an overall improvement in both measures.

**Drifter results by mission, using the Fig. 4b selection**

| Comparison | N | r before | r after | Delta r | RMSE before | RMSE after | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| GFO | 281 | 0.744 | 0.788 | +0.044 | 0.2667 | 0.2406 | +9.78% |
| TP+J1+J2+J3+S6A | 511 | 0.749 | 0.787 | +0.038 | 0.2890 | 0.2679 | +7.31% |
| S3A | 963 | 0.740 | 0.757 | +0.017 | 0.3176 | 0.2958 | +6.86% |
| S3B | 530 | 0.677 | 0.753 | +0.076 | 0.3235 | 0.2777 | +14.14% |
| TPN+J1N+J2N+J3N | 78 | 0.710 | 0.750 | +0.040 | 0.3389 | 0.3207 | +5.37% |
| ERS1+ERS2+ENV+SRL | 2292 | 0.617 | 0.668 | +0.051 | 0.2974 | 0.2816 | +5.33% |
| HY2A | 96 | 0.674 | 0.781 | +0.107 | 0.2508 | 0.2282 | +9.04% |
| HY2B | 20 | 0.773 | 0.753 | -0.020 | 0.2272 | 0.2555 | -12.44% |

Seven mission groups improve in both measures. HY2B deteriorates, but has
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

All 14 numerical checks against the saved refined figure products passed.
