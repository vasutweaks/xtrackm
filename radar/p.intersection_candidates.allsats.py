import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tools_xtrackm import *

RADARS = ("TN", "AP", "OR")
BOB_REG = (78.0, 95.0, 4.0, 24.0)
OUT_DIR = "tracks_passing_maps"


def annotate_track(ax, lons, lats, label, color):
    if len(lons) < 2:
        return
    i_mid = len(lons) // 2
    ioff = max(1, min(5, len(lons) // 10))
    angle_deg = local_angle(lons, lats, i_mid, ioff=ioff, degrees=True)
    ax.text(
        lons[i_mid],
        lats[i_mid],
        label,
        fontsize=8,
        color=color,
        rotation=angle_deg,
        rotation_mode="anchor",
        ha="center",
        va="center",
        transform=ccrs.PlateCarree(),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.2),
    )


os.makedirs(OUT_DIR, exist_ok=True)

polygons = {}
for radar_str in RADARS:
    gdf = gpd.read_file(f"polygon_shapefile_around_{radar_str}.shp")
    polygons[radar_str] = gdf.geometry.iloc[0]

for radar_str in RADARS:
    polygon = polygons[radar_str]
    with open(f"tracks_passing_{radar_str}.pkl", "rb") as f:
        tracks_dict = pickle.load(f)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    decorate_axis(ax, f"{radar_str}: tracks passing radar polygon", *BOB_REG, step=2)
    ax.add_feature(cfeature.LAND, facecolor="0.92")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.grid(linestyle="--", alpha=0.35)

    for rr in RADARS:
        poly = polygons[rr]
        if rr == radar_str:
            facecolor = "tab:orange"
            edgecolor = "tab:red"
            alpha = 0.3
            linewidth = 1.8
        else:
            facecolor = "lightblue"
            edgecolor = "k"
            alpha = 0.18
            linewidth = 1.0
        ax.add_geometries(
            [poly],
            ccrs.PlateCarree(),
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
        centroid = poly.centroid
        ax.text(
            centroid.x,
            centroid.y,
            rr,
            transform=ccrs.PlateCarree(),
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
        )

    cmap = plt.cm.get_cmap("tab20", max(1, len(tracks_dict)))
    plotted_sats = set()
    n_tracks = 0
    for i_sat, (sat, files) in enumerate(sorted(tracks_dict.items())):
        color = cmap(i_sat)
        for f_track in files:
            with xr.open_dataset(f_track, decode_times=False) as ds:
                # track_number = track_id_str(ds.Pass)
                track_number = ds.Pass
                lons_track = ds.lon.values
                lats_track = ds.lat.values
            if len(lons_track) == 0:
                continue
            track_path = sg.LineString(list(zip(lons_track, lats_track)))
            if not polygon.intersects(track_path):
                continue

            label_sat = sat if sat not in plotted_sats else None
            ax.plot(
                lons_track,
                lats_track,
                color=color,
                linewidth=1.2,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                label=label_sat,
            )
            annotate_track(ax, lons_track, lats_track, track_number, color)
            plotted_sats.add(sat)
            n_tracks += 1
            print(f"{sat} track {track_number} is passing {radar_str}")

    if plotted_sats:
        ax.legend(loc="lower left", fontsize=8, ncols=2)

    ax.set_title(f"Bay of Bengal map: {radar_str} region ({n_tracks} tracks)")
    f_out = f"{OUT_DIR}/tracks_passing_{radar_str}_bob_map.png"
    plt.savefig(f_out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"saved {f_out}")
    plt.close(fig)
