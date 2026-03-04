import os

import numpy as np
import xarray as xr
from tools_xtrackm import *


def closest_index(lons_drift, lats_drift, lon1, lat1):
    dist = np.sqrt((lons_drift - lon1) ** 2 + (lats_drift - lat1) ** 2)
    return np.argmin(dist)


# all buoys' time range is within following ranges
chunk_time_dict = {
    "1_5000": (
        np.datetime64("1985-10-30T00:00:00.000000000"),
        np.datetime64("2001-09-26T00:00:00.000000000"),
    ),
    "5001_10000": (
        np.datetime64("1997-03-20T00:00:00.000000000"),
        np.datetime64("2009-04-25T00:00:00.000000000"),
    ),
    "10001_15000": (
        np.datetime64("2006-03-05T00:00:00.000000000"),
        np.datetime64("2012-11-04T00:00:00.000000000"),
    ),
    "15001_current": (
        np.datetime64("2010-09-30T00:00:00.000000000"),
        np.datetime64("2025-05-19T00:00:00.000000000"),
    ),
}


data_loc = "/home/srinivasu/allData/drifter_track_reg/"
sat = "S3A"

def time_excluded_chunk(sat):
    df_all = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    for chunk in chunks:
        chunk_tsta_o, chunk_tend_o = chunk_time_dict[chunk]
        for i, r in df_all.iterrows():
            close_ones = []
            sat1 = r["sat"]
            track_tsta_o, track_tend_o = get_time_limits_o_np64(sat1)
            track_number_self = str(r["track_self"])
            track_number_other = str(r["track_other"])
            chunk_overlap_tsta_o, chunk_overlap_tend_o = overlap_dates_np64(
                track_tsta_o, track_tend_o, chunk_tsta_o, chunk_tend_o
            )
            if np.isnat(chunk_overlap_tsta_o) or np.isnat(chunk_overlap_tend_o):
                continue
            print(f"{sat} {chunk}, {chunk_overlap_tsta_o}, {chunk_overlap_tend_o}")
    #         for fd in sorted(
    #             glob.glob(f"{data_loc}/netcdf_{chunk}/drifter_6h_*.nc")
    #         ):
    #             print(fd)
    #             basename1 = os.path.basename(fd)
    #             ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
    #
    #             byte_id = ds_d.ID.values[0]
    #             str_id = byte_id.decode("utf-8").strip()
    #             drifter_id = str_id
    #
    #             lons_da = drifter_time_asn(ds_d, var_str="longitude")
    #             lats_da = drifter_time_asn(ds_d, var_str="latitude")
    #             lons_da2 = lons_da.rolling(time=3).mean()
    #             lats_da2 = lats_da.rolling(time=3).mean()
    #             lons_da3 = lons_da2.ffill(dim="time").bfill(dim="time")
    #             lats_da3 = lats_da2.ffill(dim="time").bfill(dim="time")
    #
    #             lons_drift = lons_da3.values
    #             lats_drift = lats_da3.values
    #             if is_within_region_any(
    #                 lons_drift,
    #                 lats_drift,
    #                 lon_inter_box_min,
    #                 lon_inter_box_max,
    #                 lat_inter_box_min,
    #                 lat_inter_box_max,
    #             ):
    #                 close_ones.append(drifter_id)
    #                 nidx = closest_index(lons_drift, lats_drift, lon1, lat1)
    #                 close_drift_lon = lons_drift[nidx]
    #                 close_drift_lat = lats_drift[nidx]
    #                 found = True
    #     close_drifters_column.append(close_ones)
    # df_close["close_drifters_column"] = close_drifters_column
    # df_close.to_csv("close_drifters_at_intersection_point.csv")
    # dse.close()
