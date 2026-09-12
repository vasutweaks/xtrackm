"""Read-only diagnostic reanalysis of Figure 6 inputs; does not revise the figure.

Run from this directory: MPLCONFIGDIR=/tmp/eicc-mpl python eicc_trend_audit.py
Output: eicc_trend_audit.csv and eicc_trend_sensitivity.csv.
HAC inference is exploratory: lags count retained observations, not calendar days.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
SAT = "ERS1+ERS2+ENV+SRL"
points = pd.read_csv(ROOT / f"intersections_near_east_coast_{SAT}.csv").sort_values("lats_inter").iloc[1:]
rows, sensitivity = [], []
for label, (_, point) in zip("ABCDEFG", points.iterrows()):
    pair = f"{SAT}_{int(point.track_self)}_{int(point.track_other)}"
    with xr.open_dataset(ROOT.parent / "guv/guv_at_intersections_loess" / f"guv_at_intersection_{pair}_loess_0.05.nc") as ds:
        gu, gv = ds.gu.load(), ds.gv.load()
    with xr.open_dataset(ROOT.parent / "ekman/ekman_at_intersections_cd" / f"{pair}_ekman_0m_at_intersections_cd.nc") as ds:
        eu, ev = ds.u_at.load(), ds.v_at.load()
    original_n = gu.sizes["time"]
    gu, gv, eu, ev = xr.align(gu, gv, eu, ev, join="inner")
    t = pd.DatetimeIndex(gu.time.values)
    days = np.asarray((t - t[0]).days, dtype=float)
    years = days / 365.25
    dt = np.median(np.diff(days))
    b, a = butter(4, (1 / 500) / (0.5 / dt), btype="low")
    quantities = {
        "scalar_sum": np.hypot(gu, gv) + np.hypot(eu, ev),
        "vector_speed": np.hypot(gu + eu, gv + ev),
        "geostrophic_speed": np.hypot(gu, gv),
        "ekman_speed": np.hypot(eu, ev),
        "abs_gu": abs(gu),
        "abs_gv": abs(gv),
    }
    common_valid = np.isfinite(quantities["vector_speed"])
    quantities = {name: da.where(common_valid) for name, da in quantities.items()}
    for name, da in quantities.items():
        y = np.asarray(da, dtype=float)
        valid = np.isfinite(y)
        yi = pd.Series(y).interpolate(limit_direction="both").to_numpy()
        lp = filtfilt(b, a, yi)
        lp[~valid] = np.nan
        fit = linregress(years[valid], lp[valid])
        resid = lp[valid] - (fit.intercept + fit.slope * years[valid])
        row = dict(point=label, quantity=name, first=str(t[valid][0].date()), last=str(t[valid][-1].date()),
                   original_n=original_n, aligned_n=len(t), valid_n=int(valid.sum()),
                   median_dt_days=dt, max_timestamp_gap_days=np.diff(days).max(),
                   max_valid_gap_days=np.diff(days[valid]).max(), mean=float(np.mean(y[valid])),
                   lp_slope_per_year=fit.slope, lp_naive_p=fit.pvalue,
                   lp_residual_lag1=np.corrcoef(resid[:-1], resid[1:])[0, 1],
                   fitted_decline_pct=-100 * fit.slope * (years[valid][-1]-years[valid][0]) / (fit.intercept+fit.slope*years[valid][0]))
        # Unfiltered data: fit trend together with annual/semiannual harmonics.
        X = np.column_stack([np.ones(len(t)), years,
                             np.sin(2*np.pi*years), np.cos(2*np.pi*years),
                             np.sin(4*np.pi*years), np.cos(4*np.pi*years)])
        for lag in (6, 12, 24):
            model = sm.OLS(y[valid], X[valid]).fit(cov_type="HAC", cov_kwds={"maxlags":lag, "use_correction":True}, use_t=True)
            row["raw_seasonal_slope_per_year"] = model.params[1]
            row[f"hac{lag}_p"] = model.pvalues[1]
            row[f"hac{lag}_ci_low"], row[f"hac{lag}_ci_high"] = model.conf_int()[1]
        rows.append(row)
        if name in ("scalar_sum", "vector_speed"):
            for start, end in ((1992,2016),(1995,2016),(2000,2016),(1992,2010),(1995,2010)):
                mask = valid & (t.year >= start) & (t.year <= end)
                model = sm.OLS(y[mask], X[mask]).fit(cov_type="HAC", cov_kwds={"maxlags":12, "use_correction":True}, use_t=True)
                sensitivity.append(dict(point=label, quantity=name, start=start, end=end, n=int(mask.sum()), slope=model.params[1], hac12_p=model.pvalues[1]))
    print(label, "gaps > 100 days:", [(str(t[i].date()), str(t[i+1].date()), int(days[i+1]-days[i])) for i in np.where(np.diff(days)>100)[0]], flush=True)

table = pd.DataFrame(rows)
table.to_csv(ROOT / "eicc_trend_audit.csv", index=False)
pd.DataFrame(sensitivity).to_csv(ROOT / "eicc_trend_sensitivity.csv", index=False)
print(table[table.quantity.isin(["scalar_sum","vector_speed"])][["point","quantity","first","last","valid_n","mean","lp_slope_per_year","lp_naive_p","lp_residual_lag1","raw_seasonal_slope_per_year","hac12_p","hac12_ci_low","hac12_ci_high"]].to_string(index=False))
