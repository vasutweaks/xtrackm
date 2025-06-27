# picking up lons, lats at equator of tracks with negative slope
# finding successive distances between them
import glob
import math
import statistics

import xarray as xr
from geopy import distance
import numpy as np
from namesm import *

for sat in sats_new:
    lons_equat_n = []
    lats_equat_n = []
    lons_equat_p = []
    lats_equat_p = []
    for f in sorted(
        glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")
    ):
        # print(f)
        ds = xr.open_dataset(f, engine="h5netcdf", decode_times=False)
        if len(ds.points_numbers) == 0:
            continue
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        lon_equat = lons_track[0]
        lat_equat = lats_track[0]
        lon_coast = lons_track[-1]
        lat_coast = lats_track[-1]
        slope = (lat_coast - lat_equat) / (lon_coast - lon_equat)
        angle_r = math.atan(slope)
        angle_d = angle_r * (180 / math.pi)
        if abs(lat_equat) >= 0 and abs(lat_equat) < 0.1:
            if slope > 0:
                lons_equat_p.append(lon_equat)
                lats_equat_p.append(lat_equat)
            elif slope < 0:
                lons_equat_n.append(lon_equat)
                lats_equat_n.append(lat_equat)
    # print(f"{sat}", lons_equat_n)
    dists_p = []
    ln = len(lons_equat_p)
    for i in range(ln - 1):
        lons_equat1 = lons_equat_p[i]
        lats_equat1 = lats_equat_p[i]
        lons_equat2 = lons_equat_p[i + 1]
        lats_equat2 = lats_equat_p[i + 1]
        dist = distance.distance((lats_equat1, lons_equat1), (lats_equat2, lons_equat2)).km
        dists_p.append(dist)
    dists_n = []
    ln = len(lons_equat_n)
    for i in range(ln - 1):
        lons_equat1 = lons_equat_n[i]
        lats_equat1 = lats_equat_n[i]
        lons_equat2 = lons_equat_n[i + 1]
        lats_equat2 = lats_equat_n[i + 1]
        dist = distance.distance((lats_equat1, lons_equat1), (lats_equat2, lons_equat2)).km
        dists_n.append(dist)
    # print(dists)
    dist_mode_n = statistics.mode(dists_n)
    dist_mode_p = statistics.mode(dists_p)
    dn = dist_mode_n * np.cos(np.pi/2 - abs(angle_r))
    dp = dist_mode_p * np.cos(np.pi/2 - abs(angle_r))
    print(f"{sat}: {dist_mode_p}")
    # print(f"{sat}: {dist_mode_n}")
    # print(f"{sat}: {dp}")
    # x = input()
    # if x == "q":
    #     break
    # else:
    #     continue
