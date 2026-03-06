import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

SAT = "S3A"
DRIFTER_DIR = "/home/srinivasu/allData/drifter_track_reg/netcdf_all/"
INTERSECTION_CSV = f"tracks_intersections_{SAT}_1.csv"
CANDIDATE_PICKLE_DIR = "candidate_pickles"
GUV_DIR = "/home/srinivasu/xtrackm/guv/guv_at_intersections"
OUTPUT_CSV = f"drifter_ve_vs_gu_{SAT}.csv"
GUV_SUFFIX = "0.05"


def decode_id(raw_id):
    if isinstance(raw_id, (bytes, np.bytes_)):
        return raw_id.decode("utf-8").strip()
    return str(raw_id).strip()


def track_to_str(track):
    track_f = float(track)
    if track_f.is_integer():
        return str(int(track_f))
    return str(track)


def build_unique_str(sat, row):
    track_self = track_to_str(row["track_self"])
    track_other = track_to_str(row["track_other"])
    return f"{sat}_{track_self}_{track_other}"


def build_drifter_index(drifter_files):
    drifter_index = {}
    for fd in drifter_files:
        with xr.open_dataset(fd, drop_variables=["WMO"]) as ds_d:
            drifter_id = decode_id(ds_d.ID.values[0])
        drifter_index[drifter_id] = fd
    return drifter_index


def load_drifter_dataset(fd):
    with xr.open_dataset(fd, drop_variables=["WMO"]) as ds_d:
        ds_drifter = ds_d[["longitude", "latitude", "time", "ve"]].isel(traj=0).load()
    return ds_drifter


def closest_obs_index(ds_drifter, lon0, lat0):
    dx = (ds_drifter["longitude"] - lon0) * np.cos(np.deg2rad(lat0))
    dy = ds_drifter["latitude"] - lat0
    dist2 = (dx * dx + dy * dy).where(
        np.isfinite(ds_drifter["longitude"]) & np.isfinite(ds_drifter["latitude"])
    )
    if int(dist2.count().item()) == 0:
        return None
    return int(dist2.argmin(dim="obs").item())


def load_guv_series(unique_str):
    guv_path = os.path.join(GUV_DIR, f"guv_at_intersection_{unique_str}_{GUV_SUFFIX}.nc")
    if not os.path.exists(guv_path):
        return None
    with xr.open_dataset(guv_path) as ds_g:
        ds_guv = ds_g[["gu"]].sortby("time").load()
    return ds_guv


def main():
    inter_df = pd.read_csv(INTERSECTION_CSV)
    drifter_files = sorted(glob(os.path.join(DRIFTER_DIR, "drifter_6h_*.nc")))
    drifter_index = build_drifter_index(drifter_files)
    drifter_cache = {}

    rows_out = []

    for _, row in inter_df.iterrows():
        unique_str = build_unique_str(SAT, row)
        lon_inter = float(row["lons_inter"])
        lat_inter = float(row["lats_inter"])

        pickle_path = os.path.join(CANDIDATE_PICKLE_DIR, f"{unique_str}.pkl")
        if not os.path.exists(pickle_path):
            continue
        with open(pickle_path, "rb") as f:
            candidate_ids = pickle.load(f)
        if len(candidate_ids) == 0:
            continue

        ds_guv = load_guv_series(unique_str)
        if ds_guv is None:
            continue

        for drifter_id in candidate_ids:
            fd = drifter_index.get(drifter_id)
            if fd is None:
                continue

            if drifter_id not in drifter_cache:
                drifter_cache[drifter_id] = load_drifter_dataset(fd)
            ds_drifter = drifter_cache[drifter_id]

            close_idx = closest_obs_index(ds_drifter, lon_inter, lat_inter)
            if close_idx is None:
                continue

            drifter_time = ds_drifter["time"].isel(obs=close_idx).values
            drifter_ve = ds_drifter["ve"].isel(obs=close_idx).values
            if np.isnat(drifter_time) or not np.isfinite(drifter_ve):
                continue

            gu_interp = ds_guv["gu"].interp(time=drifter_time)
            gu_at_time = gu_interp.values
            if not np.isfinite(gu_at_time):
                gu_nearest = ds_guv["gu"].sel(time=drifter_time, method="nearest")
                gu_at_time = gu_nearest.values
                gu_time = gu_nearest["time"].values
            else:
                gu_time = drifter_time

            if not np.isfinite(gu_at_time):
                continue

            rows_out.append(
                {
                    "unique_str": unique_str,
                    "drifter_id": drifter_id,
                    "gu": float(gu_at_time),
                    "drifter_ve": float(drifter_ve),
                    "drifter_time": str(drifter_time),
                    "gu_time": str(gu_time),
                    "close_index": int(close_idx),
                }
            )

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
