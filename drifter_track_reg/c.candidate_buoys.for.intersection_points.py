import os
# codex resume 019cc341-b33b-7b01-98bf-259b7a400a25
import numpy as np
import xarray as xr
from tools_xtrackm import *
import pickle


def is_around(lons, lats, lon1, lat1, dist_threshold=0.30):
    """
    Return True if any (lon, lat) point falls inside the rectangular region.
    include_boundary=True counts points on the edges/corners as inside.
    """
    lon_min = lon1 - dist_threshold
    lon_max = lon1 + dist_threshold
    lat_min = lat1 - dist_threshold
    lat_max = lat1 + dist_threshold
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.shape != lats.shape:
        raise ValueError("lons and lats must have the same shape")
    lons_in = (lons >= lon_min) & (lons <= lon_max)
    lats_in = (lats >= lat_min) & (lats <= lat_max)
    inside_mask = lons_in & lats_in
    return bool(np.any(inside_mask))


def build_drifter_meta(drifter_files, sat_tsta_o, sat_tend_o):
    drifter_meta = []
    """
    non time overlapping drifter_files are excluded here
    """
    for fd in drifter_files:
        with xr.open_dataset(fd, drop_variables=["WMO"]) as ds_d:
            drift_tsta_o1, drift_tend_o1 = (
                ds_d.start_date.values[0],
                ds_d.end_date.values[0],
            )
            drifter_tsta_o = np.datetime64(drift_tsta_o1, "s")
            drifter_tend_o = np.datetime64(drift_tend_o1, "s")

            overlap_tsta_o, overlap_tend_o = overlap_dates_np64(
                sat_tsta_o, sat_tend_o, drifter_tsta_o, drifter_tend_o
            )
            if np.isnat(overlap_tsta_o) or np.isnat(overlap_tend_o):
                continue

            lons_drift = np.asarray(ds_d.longitude.values, dtype=float)
            lats_drift = np.asarray(ds_d.latitude.values, dtype=float)

            byte_id = ds_d.ID.values[0]
            drifter_id = byte_id.decode("utf-8").strip()

            drifter_meta.append(
                {
                    "fd": fd,
                    "id": drifter_id,
                    "lon_min": np.nanmin(lons_drift),
                    "lon_max": np.nanmax(lons_drift),
                    "lat_min": np.nanmin(lats_drift),
                    "lat_max": np.nanmax(lats_drift),
                }
            )
    return drifter_meta


data_loc = "/home/srinivasu/allData/drifter_track_reg/netcdf_all/"
drifter_files = sorted(glob.glob(f"{data_loc}/drifter_6h_*.nc"))
sat = "TPN+J1N+J2N+J3N"
inter_df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
sat_tsta_o, sat_tend_o = get_time_limits_np64(sat)
dist_threshold = 0.30
drifter_meta = build_drifter_meta(drifter_files, sat_tsta_o, sat_tend_o)

for i, r in inter_df.iterrows():
    close_ones = []
    track_number_self = str(r["track_self"])
    track_number_other = str(r["track_other"])
    lons_inter1 = r["lons_inter"]
    lats_inter1 = r["lats_inter"]
    unique_str = f"{sat}_{track_number_self}_{track_number_other}"
    print(unique_str)

    lon_min = lons_inter1 - dist_threshold
    lon_max = lons_inter1 + dist_threshold
    lat_min = lats_inter1 - dist_threshold
    lat_max = lats_inter1 + dist_threshold

    for dmeta in drifter_meta:
        print(dmeta)
        if (
            dmeta["lon_max"] < lon_min
            or dmeta["lon_min"] > lon_max
            or dmeta["lat_max"] < lat_min
            or dmeta["lat_min"] > lat_max
        ):
            continue

        with xr.open_dataset(dmeta["fd"], drop_variables=["WMO"]) as ds_d:
            lons_drift = ds_d.longitude.values
            lats_drift = ds_d.latitude.values

        if is_around(
            lons_drift, lats_drift, lons_inter1, lats_inter1, dist_threshold
        ):
            close_ones.append(dmeta["id"])

    if close_ones:
        # dump to pickle
        print(f"{unique_str}, {len(close_ones)}")
        with open(f"candidate_pickles/{unique_str}.pkl", "wb") as f:
            pickle.dump(close_ones, f)
