import glob
import math
import statistics

import xarray as xr
from geopy import distance
import numpy as np
from namesm import *

sats_ints = {}

for sat in sats_new:
    lons_equat = []
    lats_equat = []
    for f in sorted(
        glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")
    ):
        # print(f)
        ds = xr.open_dataset(f, engine="h5netcdf", decode_times=False)
        if len(ds.points_numbers) == 0:
            continue
        track_number = ds.pass_number
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        lon_equat = lons_track[0]
        lat_equat = lats_track[0]
        lon_coast = lons_track[-1]
        lat_coast = lats_track[-1]
        slope = (lat_coast - lat_equat) / (lon_coast - lon_equat)
        angle_r = math.atan(slope)
        angle_d = angle_r * (180 / math.pi)
        if slope < 0:
            lons_equat.append(lon_equat)
            lats_equat.append(lat_equat)
    lons_sorted = sorted(lons_equat)
    lats_sorted = sorted(lats_equat)
    ln = len(lons_equat)
    dists = []
    for i in range(ln - 1):
        lons_equat1 = lons_sorted[i]
        lats_equat1 = lats_sorted[i]
        lons_equat2 = lons_sorted[i + 1]
        lats_equat2 = lats_sorted[i + 1]
        dist = distance.distance((lats_equat1, lons_equat1), (lats_equat2, lons_equat2)).km
        dists.append(dist)
    # print(dists)
    dist_mode = statistics.mode(dists)
    d = dist_mode * np.cos(np.pi/2 - abs(angle_r))
    print(f"{sat}: {dist_mode}")
    print(f"{sat}: {d}")
