#!/usr/bin/env python3
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

# codex resume 019d628e-7fbc-7580-a366-03b5eac03564
SATS = (
    "GFO",
    "TP+J1+J2+J3+S6A",
    "S3A",
    "S3B",
    "TPN+J1N+J2N+J3N",
    "ERS1+ERS2+ENV+SRL",
    "HY2A",
    "HY2B",
)
COAST_DIST_PATH = "~/allData/topo/GMT_intermediate_coast_distance_01d_track_reg.nc"
THRESHOLD_KM = 100.0
BOB_EXTENT = (78.0, 100.0, 4.0, 24.0)  # lon_min, lon_max, lat_min, lat_max
OUT_PNG = "intersections_near_east_coast_all_sats_4x2.png"
XTICKS = np.arange(80, 101, 5)
YTICKS = np.arange(5, 25, 5)


def get_coast_distance_km(ds_coast, lons, lats):
    lon_da = xr.DataArray(np.asarray(lons, dtype=float), dims="point")
    lat_da = xr.DataArray(np.asarray(lats, dtype=float), dims="point")
    coast_km = ds_coast["coast_distance"].sel(lon=lon_da, lat=lat_da, method="nearest")
    return coast_km.values.astype(float)


def select_points(df, threshold_km, bob_extent):
    coast_km = df["coast_distance_km_nearest"].values

    keep_dist = np.isfinite(coast_km) & (np.abs(coast_km) <= float(threshold_km))
    lon_min, lon_max, lat_min, lat_max = bob_extent
    keep_bob = (
        (df["lons_inter"] >= lon_min)
        & (df["lons_inter"] <= lon_max)
        & (df["lats_inter"] >= lat_min)
        & (df["lats_inter"] <= lat_max)
    )
    df_selected = df.loc[keep_dist & keep_bob].copy()

    # Discard points with lat < 9N and lon > 90E
    drop_mask = (df_selected["lats_inter"] < 9.0) | (df_selected["lons_inter"] > 90.0)
    df_selected = df_selected.loc[~drop_mask].copy()

    # Sort by latitude and de-duplicate close-latitude rows.
    df_selected = df_selected.sort_values("lats_inter").copy()
    target_half = float(threshold_km) / 2.0
    df_selected["lat_2dp"] = df_selected["lats_inter"].round(2)
    df_selected["dist_to_half"] = np.abs(np.abs(df_selected["coast_distance_km_nearest"]) - target_half)
    keep_idx = df_selected.groupby("lat_2dp")["dist_to_half"].idxmin()
    df_selected = df_selected.loc[keep_idx].sort_values("lats_inter").copy()
    df_selected = df_selected.drop(columns=["lat_2dp", "dist_to_half"])

    return df_selected, keep_dist


def draw_sat_subplot(ax, sat, df_all, df_sel, threshold_km, bob_extent):
    proj = ccrs.PlateCarree()

    ax.set_extent(bob_extent, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.coastlines(resolution="50m", linewidth=0.8, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")

    ax.scatter(
        df_all["lons_inter"],
        df_all["lats_inter"],
        s=10,
        c="0.65",
        alpha=0.5,
        transform=proj,
        zorder=3,
    )

    if not df_sel.empty:
        ax.scatter(
            df_sel["lons_inter"],
            df_sel["lats_inter"],
            s=20,
            c="crimson",
            edgecolors="black",
            linewidths=0.3,
            transform=proj,
            zorder=4,
        )

        ax.plot(
            df_sel["lons_inter"],
            df_sel["lats_inter"],
            color="tab:blue",
            linewidth=1.0,
            marker="o",
            markersize=2.5,
            transform=proj,
            zorder=3,
        )

    ax.set_title(f"{sat} ({len(df_sel)}/{len(df_all)})", fontsize=10)


def main():
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(20, 10),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": proj},
    )
    axes = axes.ravel()

    print(f"Threshold = {THRESHOLD_KM:.1f} km")

    with xr.open_dataset(os.path.expanduser(COAST_DIST_PATH)) as ds_coast:
        for i, sat in enumerate(SATS):
            input_csv = f"tracks_intersections_{sat}_1.csv"
            out_csv = f"intersections_near_east_coast_{sat}.csv"

            if not os.path.exists(input_csv):
                axes[i].set_title(f"{sat} (missing input)", fontsize=10)
                axes[i].set_global()
                print(f"{sat}: missing file {input_csv}")
                continue

            df = pd.read_csv(input_csv)
            coast_km = get_coast_distance_km(ds_coast, df["lons_inter"].values, df["lats_inter"].values)
            df["coast_distance_km_nearest"] = coast_km

            df_selected, keep_dist = select_points(df, THRESHOLD_KM, BOB_EXTENT)
            df_selected.to_csv(out_csv, index=False)

            draw_sat_subplot(axes[i], sat, df, df_selected, THRESHOLD_KM, BOB_EXTENT)

            print(
                f"{sat}: total={len(df)}, near_coast={int(np.sum(keep_dist))}, "
                f"selected_bob={len(df_selected)}, csv={out_csv}"
            )

    lon_formatter = LongitudeFormatter(number_format=".0f")
    lat_formatter = LatitudeFormatter(number_format=".0f")
    ncols = 4
    for idx, ax in enumerate(axes):
        row = idx // ncols
        col = idx % ncols
        ax.set_xticks(XTICKS, crs=proj)
        ax.set_yticks(YTICKS, crs=proj)
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(
            labelbottom=(row == 1),
            labelleft=(col == 0),
            labelsize=8,
        )

    fig.suptitle(
        f"Bay of Bengal intersections near east coast across satellites\n"
        f"Selected points satisfy |coast distance| <= {THRESHOLD_KM:.1f} km",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=180)
    plt.show()
    plt.close(fig)

    print(f"Saved multi-panel plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
