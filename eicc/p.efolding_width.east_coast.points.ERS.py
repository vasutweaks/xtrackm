import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from tools_shapely import *
from tools_xtrackm import *
import pickle


def build_unique_str(sat, row):
    track_self = track_to_str(row["track_self"])
    track_other = track_to_str(row["track_other"])
    return f"{sat}_{track_self}_{track_other}"


def track_to_str(track):
    track_f = float(track)
    if track_f.is_integer():
        return str(int(track_f))
    return str(track)


method = "loess"
frac1 = "0.05"

sat_val = "ERS1+ERS2+ENV+SRL"
# open east_coast_tracks_ERS1+ERS2+ENV+SRL.pkl
with open(f"east_coast_tracks_{sat_val}.pkl", "rb") as f:
    east_coast_tracks = pickle.load(f)

N_PTS = len(east_coast_tracks)
POINT_COLORS = [plt.cm.tab10(i) for i in range(N_PTS)]
POINT_LABELS = list("ABCDEFG")

# Map extent: 10N–22N, 78E–88E
MAP_EXTENT = (78.5, 88.0, 10.0, 22.0)

for track_number, lon_equat in east_coast_tracks.items():
    f_gc = f"../guv/gc_along_loess/gc_along_{sat_val}_{track_number}_{method}_{frac1}.nc"
    ds_gc = xr.open_dataset(f_gc)
    gc = ds_gc.gc
    gc.isel(time=0).plot()
    plt.show()
    # times = pd.to_datetime(gs.time.values)
    # gs_vals_raw = gs.values.astype(float)
    #
    # # 500-day low-pass filter
    # dt_days = np.median(
    #     np.diff(times).astype("timedelta64[s]").astype(float) / 86400.0
    # )
    # print(dt_days)
    # nyq = 0.5 / dt_days
    # cutoff = 1.0 / 500.0
    # b_filt, a_filt = butter(4, cutoff / nyq, btype="low")
    # nan_mask = np.isnan(gs_vals_raw)
    # gs_interp = (
    #     pd.Series(gs_vals_raw).interpolate(limit_direction="both").values
    # )
    # if len(gs_interp) > 3 * max(len(a_filt), len(b_filt)):
    #     gs_lp_vals = filtfilt(b_filt, a_filt, gs_interp)
    # else:
    #     samples_500d = max(3, int(round(500.0 / dt_days)))
    #     gs_lp_vals = (
    #         pd.Series(gs_interp)
    #         .rolling(window=samples_500d, center=True, min_periods=1)
    #         .mean()
    #         .values
    #     )
    # gs_lp_vals[nan_mask] = np.nan
    # gs_lowpass = xr.DataArray(
    #     gs_lp_vals, coords=gs.coords, dims=gs.dims, name="gs_lowpass"
    # )
    #
    # # Linear trend on low-pass series
    # t_days = (times - times[0]).days.astype(float)
    # valid = np.isfinite(gs_lp_vals)
    # trend_res = linregress(t_days[valid], gs_lp_vals[valid])
    # fit_vals = trend_res.intercept + trend_res.slope * t_days
    # trend_fit = xr.DataArray(
    #     fit_vals, coords=gs.coords, dims=gs.dims, name="trend_fit"
    # )
    # baseline_speed = float(fit_vals[0])
    # delta_30y = float((trend_res.slope * 365.25) * 30.0)
    # pct_change_30y = (
    #     (delta_30y / baseline_speed) * 100.0
    #     if baseline_speed != 0.0
    #     else np.nan
    # )
