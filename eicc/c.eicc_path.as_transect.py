#!/usr/bin/env python3
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# codex resume 019d11d3-4f11-75a3-9cd3-897d94b9bb77

SAT = "HY2A"
SAT = "S3A"
SAT = "TPN+J1N+J2N+J3N"
SAT = "ERS1+ERS2+ENV+SRL"
SAT = "TP+J1+J2+J3+S6A"
INPUT_CSV = f"tracks_intersections_{SAT}_1.csv"
COAST_DIST_PATH = (
    "~/allData/topo/GMT_intermediate_coast_distance_01d_track_reg.nc"
)
THRESHOLD_KM = 100.0
BOB_EXTENT = (78.0, 100.0, 4.0, 24.0)  # lon_min, lon_max, lat_min, lat_max
OUT_PNG = f"intersections_near_east_coast_{SAT}.png"
OUT_CSV = f"intersections_near_east_coast_{SAT}.csv"


def get_coast_distance_km(lons, lats):
    with xr.open_dataset(os.path.expanduser(COAST_DIST_PATH)) as ds_coast:
        lon_da = xr.DataArray(np.asarray(lons, dtype=float), dims="point")
        lat_da = xr.DataArray(np.asarray(lats, dtype=float), dims="point")
        coast_km = ds_coast["coast_distance"].sel(
            lon=lon_da, lat=lat_da, method="nearest"
        )
    return coast_km.values.astype(float)


def plot_bob(df_all, df_sel, THRESHOLD_KM, OUT_PNG):
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=proj)

    ax.set_extent(BOB_EXTENT, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.coastlines(resolution="50m", linewidth=1.0, zorder=2)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.scatter(
        df_all["lons_inter"],
        df_all["lats_inter"],
        s=22,
        c="0.65",
        alpha=0.6,
        transform=proj,
        zorder=3,
        label="All intersections",
    )

    ax.scatter(
        df_sel["lons_inter"],
        df_sel["lats_inter"],
        s=36,
        c="crimson",
        edgecolors="black",
        linewidths=0.4,
        transform=proj,
        zorder=4,
        label=f"|coast distance| <= {THRESHOLD_KM:.1f} km",
    )

    ax.plot(
        df_sel["lons_inter"],
        df_sel["lats_inter"],
        color="tab:blue",
        linewidth=1.5,
        marker="o",
        markersize=4,
        transform=proj,
        zorder=3,
    )

    for i, row in df_sel.reset_index(drop=True).iterrows():
        lon = float(row["lons_inter"])
        lat = float(row["lats_inter"])
        txt = f"({lon:.2f}, {lat:.2f})"
        dx = 0.12 if i % 2 == 0 else -0.12
        dy = 0.10 if i % 3 == 0 else -0.10
        # ax.text(
        #     lon + dx,
        #     lat + dy,
        #     txt,
        #     fontsize=8,
        #     transform=proj,
        #     zorder=5,
        #     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        # )

    ax.set_title(
        f"Bay of Bengal intersections near coast\n"
        f"Selected points: {len(df_sel)} / {len(df_all)} (threshold = {THRESHOLD_KM:.1f} km)"
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.show()
    plt.close(fig)


df = pd.read_csv(INPUT_CSV)
coast_km = get_coast_distance_km(
    df["lons_inter"].values, df["lats_inter"].values
)
df["coast_distance_km_nearest"] = coast_km

# Distance filtering
keep_dist = np.isfinite(coast_km) & (
    np.abs(coast_km) <= float(THRESHOLD_KM)
)
lon_min, lon_max, lat_min, lat_max = BOB_EXTENT
# BOB extent filtering
keep_bob = (
    (df["lons_inter"] >= lon_min)
    & (df["lons_inter"] <= lon_max)
    & (df["lats_inter"] >= lat_min)
    & (df["lats_inter"] <= lat_max)
)
df_selected = df.loc[keep_dist & keep_bob].copy()

# Discard points with lat < 9N and lon > 90E
drop_mask = (df_selected["lats_inter"] < 9.0) | (
    df_selected["lons_inter"] > 90.0
)
df_selected = df_selected.loc[~drop_mask].copy()

# Sort by latitude
df_selected = df_selected.sort_values("lats_inter").copy()

# If multiple points share the same latitude at 2 decimal places, keep the one
# whose absolute coast distance is closest to half the threshold distance.
target_half = float(THRESHOLD_KM) / 2.0
df_selected["lat_2dp"] = df_selected["lats_inter"].round(2)
df_selected["dist_to_half"] = np.abs(
    np.abs(df_selected["coast_distance_km_nearest"]) - target_half
)
keep_idx = df_selected.groupby("lat_2dp")["dist_to_half"].idxmin()
df_selected = df_selected.loc[keep_idx].sort_values("lats_inter").copy()
df_selected = df_selected.drop(columns=["lat_2dp", "dist_to_half"])

selected_points = [
    (
        float(r["lons_inter"]),
        float(r["lats_inter"]),
        float(r["coast_distance_km_nearest"]),
    )
    for _, r in df_selected.iterrows()
]

df_selected.to_csv(OUT_CSV, index=False)
plot_bob(df, df_selected, THRESHOLD_KM, OUT_PNG)

print(f"Total points: {len(df)}")
print(
    f"Near-coast points in full domain (|distance| <= {THRESHOLD_KM:.1f} km): {int(np.sum(keep_dist))}"
)
print(f"Selected points in BOB extent after extra rules: {len(df_selected)}")
print(f"Saved selected points CSV: {OUT_CSV}")
print(f"Saved plot: {OUT_PNG}")
print("Selected points list as (lon, lat, coast_distance_km):")
for p in selected_points:
    print(p)
