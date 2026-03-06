import os
import pickle
import sys
from glob import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry.polygon import Polygon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.namesm import TRACKS_REG

SAT = "S3A"
DRIFTER_DIR = "/home/srinivasu/allData/drifter_track_reg/netcdf_all/"
CANDIDATE_PICKLE_DIR = "candidate_pickles"
OUTPUT_DIR = "candidate_plots"
THRESHOLD_DEG = 0.30


def decode_id(raw_id):
    if isinstance(raw_id, (bytes, np.bytes_)):
        return raw_id.decode("utf-8").strip()
    return str(raw_id).strip()


def build_drifter_index(drifter_files):
    drifter_index = {}
    for fd in drifter_files:
        with xr.open_dataset(fd, drop_variables=["WMO"]) as ds_d:
            drifter_id = decode_id(ds_d.ID.values[0])
        drifter_index[drifter_id] = fd
    return drifter_index


def closest_obs_index(ds_track, lon0, lat0):
    dx = (ds_track["longitude"] - lon0) * np.cos(np.deg2rad(lat0))
    dy = ds_track["latitude"] - lat0
    dist2 = (dx * dx + dy * dy).where(
        np.isfinite(ds_track["longitude"]) & np.isfinite(ds_track["latitude"])
    )
    if int(dist2.count().item()) == 0:
        return None
    return int(dist2.argmin(dim="obs").item())


def load_drifter_dataset(fd):
    with xr.open_dataset(fd, drop_variables=["WMO"]) as ds_d:
        ds_drifter = ds_d[["longitude", "latitude", "time", "ve"]].isel(traj=0).load()
    return ds_drifter


def plot_one_intersection(
    sat,
    row,
    threshold_deg,
    extent,
    candidate_pickle_dir,
    drifter_index,
    out_dir,
):
    lon_inter = float(row["lons_inter"])
    lat_inter = float(row["lats_inter"])
    track_number_self = str(row["track_self"])
    track_number_other = str(row["track_other"])
    unique_str = f"{sat}_{track_number_self}_{track_number_other}"
    pickle_path = os.path.join(candidate_pickle_dir, f"{unique_str}.pkl")

    candidate_ids = []
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            candidate_ids = pickle.load(f)
    if len(candidate_ids) == 0:
        print(f"No candidates for {unique_str}")
        return
    lon_inter_box_min = lon_inter - THRESHOLD_DEG
    lon_inter_box_max = lon_inter + THRESHOLD_DEG
    lat_inter_box_min = lat_inter - THRESHOLD_DEG 
    lat_inter_box_max = lat_inter + THRESHOLD_DEG 
    box = Polygon([
        (lon_inter_box_min, lat_inter_box_min),
        (lon_inter_box_max, lat_inter_box_min),
        (lon_inter_box_max, lat_inter_box_max),
        (lon_inter_box_min, lat_inter_box_max),
    ])

    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)

    ax.plot(
        lon_inter,
        lat_inter,
        marker="o",
        color="red",
        markersize=2,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    for drifter_id in candidate_ids[:]:
        fd = drifter_index.get(drifter_id)
        if fd is None:
            continue
        ax.add_geometries([box],
                           ccrs.PlateCarree(),
                           edgecolor="r",
                           facecolor="none",
                           linewidth=1)
        with xr.open_dataset(fd, drop_variables=["WMO"]) as ds_d:
            ds_drifter = ds_d[["longitude", "latitude", "time", "ve"]].isel(traj=0).load()
        ax.scatter(ds_drifter.longitude.values, ds_drifter.latitude.values, s=1, marker=".")
        idx = closest_obs_index(ds_drifter, lon_inter, lat_inter)
        lon_closest = ds_drifter.longitude.values[idx]
        lat_closest = ds_drifter.latitude.values[idx]
        if np.isfinite(lon_closest) and np.isfinite(lat_closest):
            ax.plot(
                lon_closest,
                lat_closest,
                marker="x",
                color="black",
                markersize=6,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
        break

    title = f"{unique_str} | candidates={len(candidate_ids)}"
    ax.set_title(title)

    out_path = os.path.join(out_dir, f"{unique_str}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    inter_csv = f"tracks_intersections_{SAT}_1.csv"
    inter_df = pd.read_csv(inter_csv)
    extent = list(TRACKS_REG)

    drifter_files = sorted(glob(os.path.join(DRIFTER_DIR, "drifter_6h_*.nc")))
    drifter_index = build_drifter_index(drifter_files)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for _, row in inter_df.iterrows():
        plot_one_intersection(
            sat=SAT,
            row=row,
            threshold_deg=THRESHOLD_DEG,
            extent=extent,
            candidate_pickle_dir=CANDIDATE_PICKLE_DIR,
            drifter_index=drifter_index,
            out_dir=OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()
