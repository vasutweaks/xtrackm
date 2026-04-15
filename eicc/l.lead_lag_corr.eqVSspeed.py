#!/usr/bin/env python3
"""
Lead-lag correlation: equatorial windstress (5S-5N, 40-100E) vs
geostrophic current speed at EICC coastal points A-G (ERS1+ERS2+ENV+SRL).

Positive lag  → equatorial stress LEADS speed.
Negative lag  → speed leads equatorial stress.

Usage:
    python l.lead_lag_corr.eqVSspeed.py            # filtered (500-day LP)
    python l.lead_lag_corr.eqVSspeed.py --raw       # unfiltered timeseries
"""

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import regionmask
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from tools_xtrackm import *

# ── configuration ────────────────────────────────────────────────────────────
CCMP_U_FILE  = "/home/srinivasu/allData/ccmp31/ccmp_v31_daily_iod_u_mon.nc"
CCMP_V_FILE  = "/home/srinivasu/allData/ccmp31/ccmp_v31_daily_iod_v_mon.nc"

SAT_VAL   = "ERS1+ERS2+ENV+SRL"
METHOD    = "loess"
FRAC1     = "0.05"
GUV_DIR   = "../guv/guv_at_intersections_loess/"
EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd/"
EKMAN_DEPTH = "0"

# Set USE_FILTERED=False to skip the 500-day low-pass on both series
USE_FILTERED = "--raw" not in sys.argv

EQ_LAT_BOUNDS = (-5.0, 5.0)
EQ_LON_BOUNDS = (40.0, 100.0)

LOWPASS_CUTOFF_DAYS = 500.0
LAG_MAX_DAYS        = 1095   # ±3 years
POINT_LABELS        = list("ABCDEFG")

RHO_AIR    = 1.22
DRAG_COEFF = 1.3e-3

# ── helpers ───────────────────────────────────────────────────────────────────

def track_to_str(track):
    track_f = float(track)
    return str(int(track_f)) if track_f.is_integer() else str(track)


def build_unique_str(sat, row):
    return f"{sat}_{track_to_str(row['track_self'])}_{track_to_str(row['track_other'])}"


def compute_windstress_magnitude(u, v):
    speed = np.hypot(u, v)
    tau_x = RHO_AIR * DRAG_COEFF * speed * u
    tau_y = RHO_AIR * DRAG_COEFF * speed * v
    return np.hypot(tau_x, tau_y)


def subset_lat_lon(da, lat_bounds, lon_bounds):
    return da.sel(
        latitude=slice(lat_bounds[0], lat_bounds[1]),
        longitude=slice(lon_bounds[0], lon_bounds[1]),
    )


def area_weighted_mean(da_3d, mask_2d=None):
    if mask_2d is not None:
        valid = np.isfinite(mask_2d) & (mask_2d > 0)
        da_3d = da_3d.where(valid)
    weights = xr.DataArray(
        np.cos(np.deg2rad(da_3d.latitude)),
        coords={"latitude": da_3d.latitude},
        dims=("latitude",),
    )
    return da_3d.weighted(weights).mean(("latitude", "longitude"), skipna=True)


def remove_monthly_climatology(ts):
    clim = ts.groupby("time.month").mean("time")
    return ts.groupby("time.month") - clim


def lowpass_filter_days(ts_vals, times_pd, cutoff_days=500.0, order=4):
    """Low-pass filter on a numpy array; returns filtered array."""
    dt_days = float(
        np.median(np.diff(times_pd).astype("timedelta64[s]").astype(float) / 86400.0)
    )
    nyq = 0.5 / dt_days
    wn  = (1.0 / cutoff_days) / nyq
    if not (0.0 < wn < 1.0):
        samples = max(3, int(round(cutoff_days / dt_days)))
        return pd.Series(ts_vals).rolling(samples, center=True, min_periods=1).mean().values
    b, a = butter(order, wn, btype="low")
    nan_mask = np.isnan(ts_vals)
    ts_interp = pd.Series(ts_vals).interpolate(limit_direction="both").values
    if len(ts_interp) > 3 * max(len(a), len(b)):
        out = filtfilt(b, a, ts_interp)
    else:
        samples = max(3, int(round(cutoff_days / dt_days)))
        out = pd.Series(ts_interp).rolling(samples, center=True, min_periods=1).mean().values
    out[nan_mask] = np.nan
    return out


def detrend_and_normalise(arr):
    """Remove linear trend then z-score; NaNs preserved."""
    valid = np.isfinite(arr)
    if valid.sum() < 3:
        return arr
    x = np.where(valid)[0].astype(float)
    y = arr[valid]
    fit = linregress(x, y)
    detrended = arr.copy()
    detrended[valid] -= fit.intercept + fit.slope * np.where(valid)[0]
    std = np.nanstd(detrended)
    if std > 0:
        detrended /= std
    return detrended


def lead_lag_correlation(x, y, lags_samples):
    """
    Pearson cross-correlation at each integer lag.
    Positive lag → x leads y (y lags behind x by `lag` samples).
    """
    corrs = []
    for lag in lags_samples:
        if lag == 0:
            a, b = x, y
        elif lag > 0:   # x leads y
            a = x[lag:]
            b = y[:-lag]
        else:            # y leads x
            a = x[:lag]
            b = y[-lag:]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 20:
            corrs.append(np.nan)
        else:
            corrs.append(float(np.corrcoef(a[valid], b[valid])[0, 1]))
    return np.array(corrs)


# ── 1. Load equatorial windstress timeseries ─────────────────────────────────
print("Loading equatorial windstress …")
ds_u   = xr.open_dataset(CCMP_U_FILE)
ds_v   = xr.open_dataset(CCMP_V_FILE)
u_eq   = subset_lat_lon(ds_u["uwnd"], EQ_LAT_BOUNDS, EQ_LON_BOUNDS)
v_eq   = subset_lat_lon(ds_v["vwnd"], EQ_LAT_BOUNDS, EQ_LON_BOUNDS)
tau_eq = compute_windstress_magnitude(u_eq, v_eq)

land       = regionmask.defined_regions.natural_earth_v5_0_0.land_110
land_mask  = land.mask(tau_eq.longitude, tau_eq.latitude, wrap_lon=True)
ocean_mask = xr.where(land_mask.isnull(), 1.0, np.nan)
ts_eq_raw  = area_weighted_mean(tau_eq, mask_2d=ocean_mask)
ts_eq_anom = remove_monthly_climatology(ts_eq_raw)

eq_times_pd = pd.to_datetime(ts_eq_anom.time.values)
eq_vals_raw = ts_eq_anom.values.astype(float)
if USE_FILTERED:
    eq_vals_lp = lowpass_filter_days(eq_vals_raw, eq_times_pd, LOWPASS_CUTOFF_DAYS)
else:
    eq_vals_lp = eq_vals_raw.copy()
eq_vals_dn  = detrend_and_normalise(eq_vals_lp)

# ── 2. Load speed at A-G coastal points ──────────────────────────────────────
print("Loading coastal speed timeseries …")
df     = pd.read_csv(f"intersections_near_east_coast_{SAT_VAL}.csv")
df_out = df.sort_values("lats_inter").iloc[1:].reset_index(drop=True)
assert len(df_out) == len(POINT_LABELS)

speed_series = []   # list of (times_pd, gs_lp_vals)

for i, row in df_out.iterrows():
    sat1  = row["sat"]
    tno_s = str(row["track_self"])
    tno_o = str(row["track_other"])
    f_guv = (
        f"{GUV_DIR}guv_at_intersection_{sat1}_{tno_s}_{tno_o}_{METHOD}_{FRAC1}.nc"
    )
    ds_guv = xr.open_dataset(f_guv)
    gu = ds_guv.gu
    gv = ds_guv.gv
    gs = np.sqrt(gu**2 + gv**2)

    unique_str = build_unique_str(sat1, row)
    f_ek = (
        f"{EKMAN_DIR}/{unique_str}_ekman_{EKMAN_DEPTH}m_at_intersections_cd.nc"
    )
    ds_ek  = xr.open_dataset(f_ek)
    u_ek   = ds_ek["u_at"]
    v_ek   = ds_ek["v_at"]
    s_ek   = np.sqrt(u_ek**2 + v_ek**2)
    gs     = gs + s_ek

    times   = pd.to_datetime(gs.time.values)
    gs_vals = gs.values.astype(float)
    gs_lp   = lowpass_filter_days(gs_vals, times, LOWPASS_CUTOFF_DAYS) if USE_FILTERED else gs_vals.copy()
    speed_series.append((times, gs_lp))

# ── 3. Resample speed to the equatorial stress monthly time axis ──────────────
print("Resampling speed to monthly grid …")

speed_dn_monthly = []
for label, (sp_times, sp_vals) in zip(POINT_LABELS, speed_series):
    s = pd.Series(sp_vals, index=sp_times)
    s_monthly = s.resample("MS").mean()
    # Align to the eq_times monthly index
    eq_monthly_idx = pd.DatetimeIndex(eq_times_pd).to_period("M").to_timestamp()
    s_reindexed = s_monthly.reindex(eq_monthly_idx)
    arr = detrend_and_normalise(s_reindexed.values.astype(float))
    speed_dn_monthly.append(arr)
    print(f"  {label}: {s_reindexed.notna().sum()} valid months")

# ── 4. Compute lead-lag correlation ──────────────────────────────────────────
# Monthly sampling → 1 sample ≈ 30.44 days
DT_MONTHLY = 30.4375

max_lag_months = int(np.round(LAG_MAX_DAYS / DT_MONTHLY))
lags_samples   = np.arange(-max_lag_months, max_lag_months + 1)
lags_days      = lags_samples * DT_MONTHLY
lags_months    = lags_samples

filter_label = "500-day LP, detrended" if USE_FILTERED else "unfiltered, detrended"
out_png      = "lead_lag_corr_eqVSspeed_raw.png" if not USE_FILTERED else "lead_lag_corr_eqVSspeed.png"

print(f"\nMode: {'filtered (500-day LP)' if USE_FILTERED else 'unfiltered (raw)'}")
print(f"\nLead-lag results  [+lag → equatorial stress leads speed]")
print(f"{'Pt':>2}  {'peak r':>8}  {'lag (months)':>13}  {'lag (days)':>10}")
print("-" * 42)

results = []
for label, sp_dn in zip(POINT_LABELS, speed_dn_monthly):
    corr = lead_lag_correlation(eq_vals_dn, sp_dn, lags_samples)
    valid_idx = np.where(np.isfinite(corr))[0]
    if len(valid_idx) == 0:
        peak_lag_days, peak_lag_m, peak_corr = np.nan, np.nan, np.nan
    else:
        pk            = valid_idx[np.argmax(np.abs(corr[valid_idx]))]
        peak_lag_days = lags_days[pk]
        peak_lag_m    = float(lags_months[pk])
        peak_corr     = corr[pk]
    results.append(
        dict(
            label=label,
            corr=corr,
            peak_lag_days=peak_lag_days,
            peak_lag_months=peak_lag_m,
            peak_corr=peak_corr,
        )
    )
    print(f"  {label}  {peak_corr:+8.3f}  {peak_lag_m:+13.0f}  {peak_lag_days:+10.0f}")

# ── 5. Figure ─────────────────────────────────────────────────────────────────
POINT_COLORS = [plt.cm.tab10(i) for i in range(len(POINT_LABELS))]

fig = plt.figure(figsize=(14, 8))
gs_layout = gridspec.GridSpec(
    1, 2, figure=fig, width_ratios=[2.5, 1], wspace=0.3
)
ax_corr = fig.add_subplot(gs_layout[0])
ax_bar  = fig.add_subplot(gs_layout[1])

# ── left: correlation curves ──────────────────────────────────────────────────
for entry, color in zip(results, POINT_COLORS):
    ax_corr.plot(
        lags_months, entry["corr"],
        color=color, lw=1.8, label=entry["label"]
    )
    if np.isfinite(entry["peak_lag_months"]):
        ax_corr.axvline(
            entry["peak_lag_months"],
            color=color, lw=0.6, linestyle=":",
        )
        ax_corr.plot(
            entry["peak_lag_months"], entry["peak_corr"],
            "o", color=color, markersize=6, zorder=5,
        )

ax_corr.axhline(0, color="k", lw=0.8)
ax_corr.axvline(0, color="k", lw=0.8, linestyle="--")
ax_corr.set_xlabel("Lag (months)  [+ve → eq stress leads speed]", fontsize=12)
ax_corr.set_ylabel("Pearson r", fontsize=12)
ax_corr.set_title(
    f"Lead-lag correlation: equatorial windstress vs EICC speed\n"
    f"(500-day LP, detrended; {SAT_VAL})",
    fontsize=11,
)
ax_corr.legend(loc="upper right", fontsize=11, title="Point")
ax_corr.grid(True, alpha=0.3, linestyle="--")
ax_corr.set_xlim(lags_months[0], lags_months[-1])

# ── right: bar chart of peak lag per point ────────────────────────────────────
peak_lags_m = [r["peak_lag_months"] for r in results]
peak_corrs  = [r["peak_corr"]       for r in results]

y_pos = np.arange(len(POINT_LABELS))
ax_bar.barh(
    y_pos,
    peak_lags_m,
    color=POINT_COLORS,
    edgecolor="k",
    linewidth=0.7,
    height=0.6,
)
for i, (pl, pc) in enumerate(zip(peak_lags_m, peak_corrs)):
    if np.isfinite(pl):
        offset = 1.5 if pl >= 0 else -1.5
        ax_bar.text(
            pl + offset, i,
            f"r={pc:+.2f}",
            va="center",
            ha="left" if pl >= 0 else "right",
            fontsize=9,
            color=POINT_COLORS[i],
        )

ax_bar.axvline(0, color="k", lw=0.8, linestyle="--")
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(POINT_LABELS, fontsize=12)
ax_bar.set_xlabel("Peak lag (months)\n[+ve → eq stress leads speed]", fontsize=10)
ax_bar.set_title("Peak lag per point", fontsize=11)
ax_bar.grid(True, axis="x", alpha=0.3, linestyle="--")

plt.savefig("lead_lag_corr_eqVSspeed.png", dpi=300, bbox_inches="tight")
print("\nSaved: lead_lag_corr_eqVSspeed.png")
plt.show()
plt.close("all")
