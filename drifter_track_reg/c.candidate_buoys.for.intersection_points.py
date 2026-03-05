import os

import numpy as np
import xarray as xr
from tools_xtrackm import *
import pickle


def get_drifter_time_limits_np64(fd):
    ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
    drift_tsta_o1, drift_tend_o1 = (
        ds_d.start_date.values[0],
        ds_d.end_date.values[0],
    )
    drifter_tsta_o = np.datetime64(drift_tsta_o1, "s")
    drifter_tend_o = np.datetime64(drift_tend_o1, "s")
    ds_d.close()
    return drifter_tsta_o, drifter_tend_o


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
    lons_in = (lons > lon_min) & (lons < lon_max)
    lats_in = (lats > lat_min) & (lats < lat_max)
    inside_mask = lons_in & lats_in
    hit = np.any(inside_mask)
    if hit:
        print("At least one drifter point is around intersection point.")
    else:
        print("No points are inside the region.")
    return bool(hit)


data_loc = "/home/srinivasu/allData/drifter_track_reg/netcdf_all/"
drifter_files = sorted(glob.glob(f"{data_loc}/drifter_6h_*.nc"))
sat = "S3A"
inter_df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
sat_tsta_o, sat_tend_o = get_time_limits_np64(sat)
dist_threshold = 0.30

for i, r in inter_df.iterrows():
    close_ones = []
    track_number_self = str(r["track_self"])
    track_number_other = str(r["track_other"])
    lons_inter1 = r["lons_inter"]
    lats_inter1 = r["lats_inter"]
    unique_str = f"{sat}_{track_number_self}_{track_number_other}"
    for fd in drifter_files:
        # print(fd)
        drifter_tsta_o, drifter_tend_o = get_drifter_time_limits_np64(fd)
        overlap_tsta_o, overlap_tend_o = overlap_dates_np64(
            sat_tsta_o, sat_tend_o, drifter_tsta_o, drifter_tend_o
        )
        if np.isnat(overlap_tsta_o) or np.isnat(overlap_tend_o):
            print(f"No overlap for {fd}")
            continue
        print(f"{sat}, {overlap_tsta_o}, {overlap_tend_o}")
        basename1 = os.path.basename(fd)
        ds_d = xr.open_dataset(fd, drop_variables=["WMO"])

        byte_id = ds_d.ID.values[0]
        str_id = byte_id.decode("utf-8").strip()
        drifter_id = str_id

        lons_drift = ds_d.longitude.values
        lats_drift = ds_d.latitude.values

        if is_around(lons_drift, lats_drift, lons_inter1, lats_inter1, dist_threshold):
            close_ones.append(drifter_id)

    if close_ones:
        # dump to pickle
        with open(f"candidate_pickles/{unique_str}.pkl", "wb") as f:
            pickle.dump(close_ones, f)
