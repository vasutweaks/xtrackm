import os

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
from scipy.stats import linregress
from tools_shapely import *
from tools_xtrackm import *
import sys

# copilot --resume=d419a1e5-878f-477e-813b-3b06f3d9a50b

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
EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd/"
EKMAN_DEPTH = "0"

sat_val = "S3A"

df = pd.read_csv(f"intersections_near_east_coast_{sat_val}.csv")

# Skip the first point (lowest latitude)
df_out = df.sort_values("lats_inter").iloc[1:].reset_index(drop=True)

# print(len(df_out))
N_PTS = len(df_out)
assert len(df_out) == N_PTS, (
    f"Expected {N_PTS} points after skipping first, got {len(df_out)}"
)
# sys.exit()

POINT_COLORS = [plt.cm.tab10(i) for i in range(N_PTS)]
POINT_LABELS = list("ABCDEFG")

# Map extent: 10N–22N, 78E–88E
MAP_EXTENT = (78.5, 88.0, 10.0, 22.0)

# --- Pre-collect all timeseries data ---
point_data = []
for i, row in df_out.iterrows():
    sat1 = row["sat"]
    tno_s = str(row["track_self"])
    tno_o = str(row["track_other"])
    lon1 = float(row["lons_inter"])
    lat1 = float(row["lats_inter"])

    f_guv = (
        f"../guv/guv_at_intersections_{method}/"
        f"guv_at_intersection_{sat1}_{tno_s}_{tno_o}_{method}_{frac1}.nc"
    )
    ds_guv = xr.open_dataset(f_guv)
    gu = ds_guv.gu
    gv = ds_guv.gv
    gs = np.sqrt(gu**2 + gv**2)
    unique_str = build_unique_str(sat1, row)
    f_ek = f"{EKMAN_DIR}/{unique_str}_ekman_{EKMAN_DEPTH}m_at_intersections_cd.nc"
    ds_ek = xr.open_dataset(f_ek)
    u_ek = ds_ek['u_at']
    v_ek = ds_ek['v_at']
    s_ek = np.sqrt(u_ek**2 + v_ek**2)
    gs = gs + s_ek
    times = pd.to_datetime(gs.time.values)
    gs_vals_raw = gs.values.astype(float)

    # 500-day low-pass filter
    dt_days = np.median(
        np.diff(times).astype("timedelta64[s]").astype(float) / 86400.0
    )
    print(dt_days)
    nyq = 0.5 / dt_days
    cutoff = 1.0 / 500.0
    b_filt, a_filt = butter(4, cutoff / nyq, btype="low")
    nan_mask = np.isnan(gs_vals_raw)
    gs_interp = (
        pd.Series(gs_vals_raw).interpolate(limit_direction="both").values
    )
    if len(gs_interp) > 3 * max(len(a_filt), len(b_filt)):
        gs_lp_vals = filtfilt(b_filt, a_filt, gs_interp)
    else:
        samples_500d = max(3, int(round(500.0 / dt_days)))
        gs_lp_vals = (
            pd.Series(gs_interp)
            .rolling(window=samples_500d, center=True, min_periods=1)
            .mean()
            .values
        )
    gs_lp_vals[nan_mask] = np.nan
    gs_lowpass = xr.DataArray(
        gs_lp_vals, coords=gs.coords, dims=gs.dims, name="gs_lowpass"
    )

    # Linear trend on low-pass series
    t_days = (times - times[0]).days.astype(float)
    valid = np.isfinite(gs_lp_vals)
    trend_res = linregress(t_days[valid], gs_lp_vals[valid])
    fit_vals = trend_res.intercept + trend_res.slope * t_days
    trend_fit = xr.DataArray(
        fit_vals, coords=gs.coords, dims=gs.dims, name="trend_fit"
    )

    point_data.append(
        dict(
            sat1=sat1,
            tno_s=tno_s,
            tno_o=tno_o,
            lon1=lon1,
            lat1=lat1,
            gs=gs,
            gs_lowpass=gs_lowpass,
            trend_fit=trend_fit,
            times=times,
            slope_yr=trend_res.slope * 365.25,
            stderr_yr=trend_res.stderr * 365.25,
            pvalue=trend_res.pvalue,
            significant=trend_res.pvalue < 0.05,
        )
    )

# --- Build figure: left = tall map, right = 7 timeseries rows ---
fig = plt.figure(figsize=(18, 14))
gs_layout = gridspec.GridSpec(
    N_PTS,
    2,
    figure=fig,
    width_ratios=[1, 2.5],
    hspace=0.0,
    wspace=0.15,
)

# --- Left: vertically elongated cartopy map spanning all rows ---
ax_map = fig.add_subplot(gs_layout[1:, 0], projection=ccrs.PlateCarree())
ax_map.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
ax_map.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
ax_map.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
ax_map.coastlines(resolution="50m", linewidth=1.0, zorder=2)
gl = ax_map.gridlines(
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.6,
    linestyle="--",
)
gl.top_labels = False
gl.right_labels = False

# --- Isobaths: 200, 1000, 2000 m ---
dse = xr.open_dataset(os.path.expanduser("~/allData/topo/etopo5.cdf"))
rose_slice = dse.ROSE.sel(
    ETOPO05_X=slice(MAP_EXTENT[0], MAP_EXTENT[1]),
    ETOPO05_Y=slice(MAP_EXTENT[2], MAP_EXTENT[3]),
)
lons_topo = rose_slice.ETOPO05_X.values
lats_topo = rose_slice.ETOPO05_Y.values
topo_vals = rose_slice.values
iso_levels = [-2000, -1000, -200]
iso_level1 = [-200]
cs_iso = ax_map.contour(
    lons_topo, lats_topo, topo_vals,
    levels=iso_level1,
    colors="steelblue",
    linewidths=0.7,
    linestyles="--",
    transform=ccrs.PlateCarree(),
    zorder=4,
)
iso_level2 = [-1000]
cs_iso = ax_map.contour(
    lons_topo, lats_topo, topo_vals,
    levels=iso_level2,
    colors="r",
    linewidths=0.7,
    linestyles="--",
    transform=ccrs.PlateCarree(),
    zorder=4,
)
# ax_map.clabel(
#     cs_iso,
#     fmt={-200: "200 m", -1000: "1000 m", -2000: "2000 m"},
#     fontsize=7,
#     inline=True,
# )

# Connect all 7 points with a dashed gray line
lons_pts = [e["lon1"] for e in point_data]
lats_pts = [e["lat1"] for e in point_data]
ax_map.plot(
    lons_pts,
    lats_pts,
    color="gray",
    linewidth=0.8,
    linestyle="--",
    transform=ccrs.PlateCarree(),
    zorder=3,
)

# Plot each point with its assigned color and letter label
for idx, entry in enumerate(point_data):
    color = POINT_COLORS[idx]
    label = POINT_LABELS[idx]
    ax_map.plot(
        entry["lon1"],
        entry["lat1"],
        marker="o",
        markersize=5,
        color=color,
        markeredgecolor="black",
        markeredgewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax_map.text(
        entry["lon1"] + 0.18,
        entry["lat1"] + 0.1,
        label,
        fontsize=10,
        fontweight="bold",
        color=color,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    # Slope annotation: gray if not significant, colored if significant
    slope_color = color if entry["significant"] else "0.6"
    ax_map.text(
        entry["lon1"] + 0.18,
        entry["lat1"] - 0.45,
        f"{entry['slope_yr']:+.4f} m/s/yr",
        fontsize=7,
        color=slope_color,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

# --- Description box above the map ---
desc_lines = ["Pt   Longitude   Latitude    Tracks"]
desc_lines.append("─" * 38)
for idx, entry in enumerate(point_data):
    desc_lines.append(
        f" {POINT_LABELS[idx]}   {entry['lon1']:6.2f}°E  "
        f"{entry['lat1']:5.2f}°N   T{entry['tno_s']}×T{entry['tno_o']}"
    )
ax_map.text(
    0.15, 1.05,
    "\n".join(desc_lines),
    transform=ax_map.transAxes,
    fontsize=7.5,
    fontfamily="monospace",
    va="bottom", ha="left",
    clip_on=False,
    bbox=dict(
        boxstyle="round,pad=0.45",
        facecolor="white",
        edgecolor="0.35",
        alpha=0.95,
    ),
    zorder=10,
)

# --- Right: 7 timeseries subplots, A at bottom (row 6) → G at top (row 0) ---
for idx, entry in enumerate(point_data):
    row = N_PTS - 1 - idx  # A → row 6, G → row 0
    ax_ts = fig.add_subplot(gs_layout[row, 1])
    color = POINT_COLORS[idx]
    label = POINT_LABELS[idx]
    times = entry["times"]
    gs_da = entry["gs"]
    gs_lowpass = entry["gs_lowpass"]
    trend_fit = entry["trend_fit"]

    # 500-day low-pass (same color, bold)
    ax_ts.plot(
        times,
        gs_lowpass.values,
        color=color,
        linewidth=2.0,
        label="500-day LP",
    )
    # Linear trend (black dashed)
    ax_ts.plot(
        times,
        trend_fit.values,
        color="k",
        linewidth=1.5,
        linestyle="--",
        label="trend",
    )

    # Seasonal background shading
    year_min, year_max = times.year.min(), times.year.max()
    for yr in range(year_min, year_max + 1):
        ax_ts.axvspan(
            pd.Timestamp(yr, 2, 1),
            pd.Timestamp(yr, 5, 31),
            color="yellow",
            alpha=0.15,
            zorder=0,
        )
        ax_ts.axvspan(
            pd.Timestamp(yr, 6, 1),
            pd.Timestamp(yr, 9, 30),
            color="cyan",
            alpha=0.15,
            zorder=0,
        )
        ax_ts.axvspan(
            pd.Timestamp(yr, 10, 1),
            pd.Timestamp(yr, 12, 31),
            color="gray",
            alpha=0.15,
            zorder=0,
        )
        ax_ts.axvline(
            pd.Timestamp(yr, 1, 1),
            color="k",
            linewidth=0.3,
            linestyle="--",
            alpha=0.4,
        )

    # Point label inside left edge of plot
    ax_ts.text(
        0.005, 0.5, label,
        transform=ax_ts.transAxes, fontsize=12,
        fontweight="bold", color=color,
        va="center", ha="left",
    )
    # Tighten y-axis to the filtered (low-pass) series range
    all_vals = gs_lowpass.values[np.isfinite(gs_lowpass.values)]
    ypad = (all_vals.max() - all_vals.min()) * 0.08
    ax_ts.set_ylim(all_vals.min() - ypad, all_vals.max() + ypad)
    ax_ts.set_ylabel("m/s", fontsize=14)
    ax_ts.tick_params(labelsize=12)
    # Only bottom subplot (A, idx=0) shows x-tick labels
    if idx > 0:
        ax_ts.set_xticklabels([])

    # Legend only on the top panel (G, idx=N_PTS-1)
    if idx == N_PTS - 1:
        handles = [
            Line2D([0], [0], color=color, lw=2.0, label="500-day low-pass"),
            Line2D(
                [0],
                [0],
                color="k",
                lw=1.5,
                linestyle="--",
                label="linear trend",
            ),
            Patch(
                facecolor="yellow",
                alpha=0.4,
                edgecolor="none",
                label="Feb–May",
            ),
            Patch(
                facecolor="cyan", alpha=0.4, edgecolor="none", label="Jun–Sep"
            ),
            Patch(
                facecolor="gray", alpha=0.4, edgecolor="none", label="Oct–Dec"
            ),
        ]
        ax_ts.legend(handles=handles, loc="upper right", fontsize=12, ncol=3)

# fig.suptitle(
#     f"Current speed at EICC coastal intersection points — {sat_val}",
#     fontsize=13,
#     fontweight="bold",
#     y=1.002,
# )
out_png = f"speed_timeseries_allinone_{sat_val}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.show()
plt.close("all")
print(f"Saved: {out_png}")
