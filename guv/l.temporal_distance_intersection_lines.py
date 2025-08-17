import math

import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import sys


def get_time_limits_o(f):
    ds = xr.open_dataset(f, engine="h5netcdf")
    # ds = xr.open_dataset(f)
    time = ds.time.isel(points_numbers=0).values
    # these are cftime objects which are not very useful
    first_time = time[0]
    last_time = time[-1]
    # print(type(first_time), type(last_time))
    # Convert the first and last time stamps to string format
    first_time_str = first_time.strftime("%Y-%m-%d")
    last_time_str = last_time.strftime("%Y-%m-%d")
    # first_time_str = datetime.strptime(first_time, "%Y-%m-%d")
    # last_time_str = datetime.strptime(last_time, "%Y-%m-%d")
    # print(type(first_time_str), type(last_time_str))
    track_tsta_o = datetime.strptime(first_time_str, "%Y-%m-%d")
    track_tend_o = datetime.strptime(last_time_str, "%Y-%m-%d")
    return track_tsta_o, track_tend_o


for sat in sats_new:
    tfreq = sats_tfreq[sat]
    print(f"satellite name {sat} -------------- {tfreq} days")
    time_diff_list = []
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    for i, r in df.iterrows():
        sat1 = r["sat"]
        track_number_self = str(r["track_self"])
        track_number_other = str(r["track_other"])
        lons_inter1 = r["lons_inter"]
        lats_inter1 = r["lats_inter"]
        x_from_coast_self1 = r["x_from_coast_self"]
        x_from_coast_other1 = r["x_from_coast_other"]
        angle_acute = r["angle_acute"]
        angle_obtuse = r["angle_obtuse"]
        lon1 = lons_inter1
        lat1 = lats_inter1
        if abs(lat1) < 2:
            continue
        # print(
        #     sat,
        #     track_number_self,
        #     track_number_other,
        #     lons_inter1,
        #     lats_inter1,
        #     x_from_coast_self1,
        #     x_from_coast_other1,
        #     angle_acute,
        #     angle_obtuse,
        # )
        f_self = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
        f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
        tsta_o_self, tend_o_self = get_time_limits_o(f_self)
        tsta_o_other, tend_o_other = get_time_limits_o(f_other)
        time_diff = (tsta_o_self - tsta_o_other).days
        time_diff = (tend_o_self - tend_o_other).days
        time_diff_list.append(time_diff)
        # print(f"{track_number_self} {tsta_o_self} {tend_o_self}")
        # print(f"{track_number_other} {tsta_o_other} {tend_o_other}")
        # print(f"{track_number_other} {time_diff} in days")
    print(f"max inter track time diff {max(time_diff_list)} in days")
    print(f"min inter track time diff {min(time_diff_list)} in days")
