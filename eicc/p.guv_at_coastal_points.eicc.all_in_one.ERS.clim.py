import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch
from tools_shapely import *
from tools_xtrackm import *

# copilot --resume=d419a1e5-878f-477e-813b-3b06f3d9a50b

method = "loess"
frac1 = "0.05"

sat_val = "TP+J1+J2+J3+S6A"
sat_val = "S3A"
sat_val = "ERS1+ERS2+ENV+SRL"

df = pd.read_csv(f"intersections_near_east_coast_{sat_val}.csv")

# Skip the first point (lowest latitude)
df_out = df.sort_values("lats_inter").iloc[1:].reset_index(drop=True)

N_PTS = 7
assert len(df_out) == N_PTS, (
    f"Expected {N_PTS} points after skipping first, got {len(df_out)}"
)

POINT_COLORS = [plt.cm.tab10(i) for i in range(N_PTS)]
POINT_LABELS = list("ABCDEFG")

# Map extent: 10N–22N, 78E–88E
MAP_EXTENT = (78.0, 88.0, 10.0, 22.0)

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
    times = pd.to_datetime(gs.time.values)
    # Monthly climatology (mean of each calendar month across all years)
    gs_clim = gs.groupby("time.month").mean("time")
    gu_clim = gu.groupby("time.month").mean("time")

    point_data.append(
        dict(
            sat1=sat1,
            tno_s=tno_s,
            tno_o=tno_o,
            lon1=lon1,
            lat1=lat1,
            gs=gs,
            gu_clim=gu_clim,
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
    wspace=0.3,
)

# --- Left: vertically elongated cartopy map spanning all rows ---
ax_map = fig.add_subplot(gs_layout[:, 0], projection=ccrs.PlateCarree())
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
# ax_map.set_title(f"{sat_val}\nIntersection points", fontsize=9)

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
        markersize=9,
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

# --- Right: 7 monthly climatology subplots, A at bottom (row 6) → G at top (row 0) ---
MONTHS = np.arange(1, 13)
MONTH_NAMES = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

for idx, entry in enumerate(point_data):
    row = N_PTS - 1 - idx  # A → row 6, G → row 0
    ax_ts = fig.add_subplot(gs_layout[row, 1])
    color = POINT_COLORS[idx]
    label = POINT_LABELS[idx]
    gu_clim = entry["gu_clim"]

    # Seasonal background shading (month-based x-axis)
    ax_ts.axvspan(2 - 0.5, 5 + 0.5, color="yellow", alpha=0.15, zorder=0)   # Feb–May
    ax_ts.axvspan(6 - 0.5, 9 + 0.5, color="cyan",   alpha=0.15, zorder=0)   # Jun–Sep
    ax_ts.axvspan(10 - 0.5, 12 + 0.5, color="gray", alpha=0.15, zorder=0)   # Oct–Dec

    # Monthly climatology bar plot
    # ax_ts.bar(
    #     MONTHS,
    #     gs_clim.values,
    #     color=color,
    #     alpha=0.8,
    #     edgecolor="k",
    #     linewidth=0.5,
    #     zorder=2,
    # )

    ax_ts.plot(
        MONTHS,
        gu_clim.values,
        color=color,
        linewidth=3,
        zorder=2,
    )

    # Point label inside left edge of plot
    ax_ts.text(
        0.005, 0.5, label,
        transform=ax_ts.transAxes, fontsize=12,
        fontweight="bold", color=color,
        va="center", ha="left",
    )

    ax_ts.set_xlim(0.5, 12.5)
    all_vals = gs_clim.values[np.isfinite(gs_clim.values)]
    ypad = (all_vals.max() - all_vals.min()) * 0.08
    # ax_ts.set_ylim(-all_vals.max() - ypad, all_vals.max() + ypad)
    # ax_ts.set_ylim(-all_vals.max() - ypad, all_vals.max() + ypad)
    ax_ts.set_ylabel("m/s", fontsize=14)
    ax_ts.set_xticks(MONTHS)
    ax_ts.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax_ts.tick_params(labelsize=12)
    # Only bottom subplot (A, idx=0) shows month labels
    if idx == 0:
        ax_ts.set_xticklabels(MONTH_NAMES)
    else:
        ax_ts.set_xticklabels([])

    # Legend only on the top panel (G, idx=N_PTS-1)
    if idx == N_PTS - 1:
        handles = [
            Patch(facecolor="yellow", alpha=0.4, edgecolor="none", label="Feb–May"),
            Patch(facecolor="cyan",   alpha=0.4, edgecolor="none", label="Jun–Sep"),
            Patch(facecolor="gray",   alpha=0.4, edgecolor="none", label="Oct–Dec"),
        ]
        ax_ts.legend(handles=handles, loc="upper right", fontsize=12, ncol=3)
        ax_ts.set_title("Monthly climatology of |u| (gs)", fontsize=11)

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
