import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import butter_lowpass_1d

method = "loess"
frac1 = "0.05"
SAT = "TPN+J1N+J2N+J3N"
SAT = "TP+J1+J2+J3+S6A"
DRIFTER_DIR = "/home/srinivasu/allData/drifter_track_reg/netcdf_all/"
INTERSECTION_CSV = f"tracks_intersections_{SAT}_1.csv"
CANDIDATE_PICKLE_DIR = "candidate_pickles"
GUV_DIR = f"/home/srinivasu/xtrackm/guv/guv_at_intersections_{method}"
OUTPUT_CSV = f"drifter_ve_vs_gu_ek_{SAT}_dy.csv"
EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd"
EKMAN_DEPTH = "0"

def filter_rama_uv_72h(rama_u, rama_v):
    rama_u_filt = butter_lowpass_1d(rama_u, cutoff_days=3, order=4)
    rama_v_filt = butter_lowpass_1d(rama_v, cutoff_days=3, order=4)
    return rama_u_filt, rama_v_filt


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
        ds_drifter = ds_d[["longitude", "latitude", "time", "ve", "vn"]].isel(traj=0).load()
        # get four point average of drifter
        ds_drifter = ds_drifter.coarsen(obs=4, boundary="pad").mean()
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
    guv_path = os.path.join(GUV_DIR, f"guv_at_intersection_{unique_str}_{method}_{frac1}.nc")
    if not os.path.exists(guv_path):
        return None
    with xr.open_dataset(guv_path) as ds_g:
        ds_guv = ds_g[["gu"]].sortby("time").load()
    return ds_guv


def main():
    inter_df = pd.read_csv(INTERSECTION_CSV)
    drifter_files = sorted(glob(os.path.join(DRIFTER_DIR, "drifter_6h_*.nc")))
    if len(drifter_files) == 0:
        raise("No glob files")
    drifter_index = build_drifter_index(drifter_files)
    drifter_cache = {}

    rows_out = []

    for _, row in inter_df.iterrows():

        unique_str = build_unique_str(SAT, row)
        lon_inter = float(row["lons_inter"])
        lat_inter = float(row["lats_inter"])
        if np.abs(lat_inter) < 5:
            continue

        pickle_path = os.path.join(CANDIDATE_PICKLE_DIR, f"{unique_str}.pkl")

        f_ek = f"{EKMAN_DIR}/{unique_str}_ekman_{EKMAN_DEPTH}m_at_intersections_cd.nc"
        ds_ek = xr.open_dataset(f_ek)
        u_ek = ds_ek['u_at']
        v_ek = ds_ek['v_at']
        u_ek, v_ek = filter_rama_uv_72h(u_ek, v_ek)
        if not os.path.exists(pickle_path):
            print(f"No pickle for {unique_str}")
            continue
        with open(pickle_path, "rb") as f:
            candidate_ids = pickle.load(f)
        if len(candidate_ids) == 0:
            print(f"No candidates in pickle for {unique_str}")
            continue

        ds_guv = load_guv_series(unique_str)
        if ds_guv is None:
            print(f"No GUV data for {unique_str}")
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
            ve = ds_drifter["ve"]
            vn = ds_drifter["vn"]
            ve, vn = filter_rama_uv_72h(ve, vn)
            drifter_ve = ve.isel(obs=close_idx).values
            if np.isnat(drifter_time) or not np.isfinite(drifter_ve):
                continue

            gu_interp = ds_guv["gu"].interp(time=drifter_time)
            u_ek1 = u_ek.sel(time=drifter_time, method="nearest").values
            v_ek1 = v_ek.sel(time=drifter_time, method="nearest").values
            gu_at_time = gu_interp.values

            # Always retain the nearest native GUV timestamp for temporal offset checks.
            gu_nearest = ds_guv["gu"].sel(time=drifter_time, method="nearest")
            gu_time = gu_nearest["time"].values

            # Keep interpolated gu when valid; otherwise fall back to nearest-sample gu.
            # np.isfinite(gu_at_time) is true if interpolation is successful
            if not np.isfinite(gu_at_time):
                gu_at_time = gu_nearest.values

            if not np.isfinite(gu_at_time):
                continue

            rows_out.append(
                {
                    "sat": SAT,
                    "track_self": track_to_str(row["track_self"]),
                    "track_other": track_to_str(row["track_other"]),
                    "lon_inter": lon_inter,
                    "lat_inter": lat_inter,
                    "drifter_id": drifter_id,
                    "gu": float(gu_at_time),
                    "u_ek": float(u_ek1),
                    "drifter_ve": float(drifter_ve),
                    "drifter_time": str(drifter_time),
                    "gu_time": str(gu_time),
                    "close_index": int(close_idx),
                }
            )

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out_df)} rows to {OUTPUT_CSV}")
    globals().update(locals())

if __name__ == "__main__":
    main()
