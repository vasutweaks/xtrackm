import os

import numpy as np
import xarray as xr
from tools_xtrackm import *
import sys


def is_within_region_any(lons,
                         lats,
                         lon_min,
                         lon_max,
                         lat_min,
                         lat_max,
                         include_boundary=True):
    """
    Return True if any (lon, lat) point falls inside the rectangular region.
    include_boundary=True counts points on the edges/corners as inside.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.shape != lats.shape:
        raise ValueError("lons and lats must have the same shape")
    if include_boundary:
        lons_in = (lons >= lon_min) & (lons <= lon_max)
        lats_in = (lats >= lat_min) & (lats <= lat_max)
    else:
        lons_in = (lons > lon_min) & (lons < lon_max)
        lats_in = (lats > lat_min) & (lats < lat_max)
    inside_mask = lons_in & lats_in
    hit = np.any(inside_mask)
    if hit:
        print("At least one point is inside the region.")
    else:
        print("No points are inside the region.")
    return bool(hit)


chunk_time_dict = {
    "1_5000": (
        np.datetime64("1985-10-30T00:00:00.000000000"),
        np.datetime64("1999-11-01T00:00:00.000000000"),
    ),
    "5001_10000": (
        np.datetime64("1997-03-20T00:00:00.000000000"),
        np.datetime64("2006-09-16T00:00:00.000000000"),
    ),
    "10001_15000": (
        np.datetime64("2006-05-24T00:00:00.000000000"),
        np.datetime64("2011-01-04T00:00:00.000000000"),
    ),
    "15001_current": (
        np.datetime64("2010-10-30T00:00:00.000000000"),
        np.datetime64("2024-12-31T00:00:00.000000000"),
    ),
}

d = 1.5
data_loc = f"/home/srinivasu/allData/drifter1/"
dist_threshold = 0.5  # in degrees
# chunks = ["5001_10000", "10001_15000"]
# take satellite from as first argument
sat_here = sys.argv[1]
df_all = pd.read_csv(f"tracks_intersections_{sat_here}_1.csv")
df_close = df_all.copy()

close_drifters_column = []

found = False
for i, r in df_all.iterrows():
    close_ones = []
    sat1 = r["sat"]
    # if sat1 != "S3A":
    #     continue
    track_tsta_o, track_tend_o = get_time_limits_o(sat1)
    track_number_self = str(r["track_self"])
    track_number_other = str(r["track_other"])
    lons_inter1 = r["lons_inter"]
    lats_inter1 = r["lats_inter"]
    lon1 = lons_inter1
    lat1 = lats_inter1
    if abs(lat1) < 2:
        close_drifters_column.append([])
        continue
    print(
        sat1,
        track_number_self,
        track_number_other,
        lons_inter1,
        lats_inter1,
    )
    lon_inter_box_min = lon1 - dist_threshold
    lon_inter_box_max = lon1 + dist_threshold
    lat_inter_box_min = lat1 - dist_threshold
    lat_inter_box_max = lat1 + dist_threshold
    for chunk in chunks:
        chunk_tsta_o1, chunk_tend_o1 = chunk_time_dict[chunk]
        chunk_tsta_o, chunk_tend_o = n64todatetime1(
            chunk_tsta_o1), n64todatetime1(chunk_tend_o1)
        chunk_overlap_tsta_o, chunk_overlap_tend_o = overlap_dates(
            track_tsta_o, track_tend_o, chunk_tsta_o, chunk_tend_o)
        if chunk_overlap_tsta_o is None or chunk_overlap_tend_o is None:
            continue
        for fd in sorted(
                glob.glob(
                    f"{data_loc}/netcdf_{chunk}/track_reg/drifter_6h_*.nc")):
            print(fd)
            basename1 = os.path.basename(fd)
            ds_d = xr.open_dataset(fd, drop_variables=["WMO"])

            byte_id = ds_d.ID.values[0]
            str_id = byte_id.decode("utf-8").strip()
            drifter_id = str_id

            drift_tsta_o1, drift_tend_o1 = (
                ds_d.start_date.values[0],
                ds_d.end_date.values[0],
            )
            drift_tsta_o, drift_tend_o = n64todatetime1(
                drift_tsta_o1), n64todatetime1(drift_tend_o1)
            overlap_tsta_o, overlap_tend_o = overlap_dates(
                track_tsta_o, track_tend_o, drift_tsta_o, drift_tend_o)
            print(f"overlap period {overlap_tsta_o} {overlap_tend_o}")
            if overlap_tsta_o is None or overlap_tend_o is None:
                continue
            lons_drift = ds_d.longitude.values
            lats_drift = ds_d.latitude.values
            if is_within_region_any(
                    lons_drift,
                    lats_drift,
                    lon_inter_box_min,
                    lon_inter_box_max,
                    lat_inter_box_min,
                    lat_inter_box_max,
            ):
                close_ones.append(drifter_id)
                found = True
    close_drifters_column.append(close_ones)
df_close["close_drifters_column"] = close_drifters_column
df_close.to_csv(f"close_drifters_at_intersection_point_{sat_here}.csv")
