import glob

import numpy as np
import xarray as xr
from namesm import *


def is_odd(track_no):
    if int(track_no) % 2 == 1:
        return True
    else:
        return False


def orbit_type(track_no):
    if int(track_no) % 2 == 1:
        return "ascending"
    else:
        return "descending"


lst = []
# sats = ["GFO"]
for sat in sats_new:
    ascending_angles = []
    descending_angles = []
    for f in sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f, engine="h5netcdf", decode_times=False)
        if len(ds.points_numbers) < 12:
            continue
        track_number = ds.pass_number
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        lon_equat = lons_track[0]
        lat_equat = lats_track[0]
        lon_coast = lons_track[-1]
        lat_coast = lats_track[-1]
        orbit_str = orbit_type(track_number)

        slope_whole = (lat_coast - lat_equat) / (lon_coast - lon_equat)
        angle_whole_r = np.arctan(slope_whole)
        # angle_whole_d = angle_whole_r * (180 / math.pi)
        angle_whole_d = np.rad2deg(angle_whole_r)
        angle_whole_r2 = np.arctan2(lat_coast - lat_equat,
                                    lon_coast - lon_equat)
        # angle_whole_d2 = angle_whole_r2 * (180 / math.pi)
        angle_whole_d2 = np.rad2deg(angle_whole_r2)

        ll = len(lons_track)
        ll2 = ll // 2

        delta_lon = lons_track[ll2 + 5] - lons_track[ll2 - 5]
        delta_lat = lats_track[ll2 + 5] - lats_track[ll2 - 5]

        slope_point = delta_lat / delta_lon
        angle_point_r = np.arctan(slope_point)
        # angle_point_d = angle_point_r * (180 / math.pi)
        angle_point_d = np.rad2deg(angle_point_r)
        angle_point_r2 = np.arctan2(delta_lat, delta_lon)
        # angle_point_d2 = angle_point_r2 * (180 / math.pi)
        angle_point_d2 = np.rad2deg(angle_point_r2)
        # diff = angle_point_d - angle_whole_d
        # lst.append(diff)

        # print(f"whole angle: {angle_whole_d}, point angle: {angle_point_d}")
        # print(diff)
        # if not is_odd(track_number):
        #     print(f"{angle_point_d} {angle_point_d2} {orbit_str} {sat}")
        # if not is_odd(track_number):
        #     print(f"{sat} {angle_whole_d2} {orbit_str}")
        if is_odd(track_number):
            # print(f"{sat} {angle_whole_d2} {orbit_str}")
            ascending_angles.append(angle_whole_d2)
        if not is_odd(track_number):
            # print(f"{sat} {angle_whole_d2} {orbit_str}")
            descending_angles.append(angle_whole_d2)
    print(f"{sat} mean ascending: {np.mean(ascending_angles)}")
    print(f"{sat} mean descending: {np.mean(descending_angles)}")
    print(f"{sat} sum of mean: {np.mean(ascending_angles) + np.mean(descending_angles)}")
