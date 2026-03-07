import math

import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import index_at_lat, local_angle

import math
from namesm import sats_new


ioff = 3
out_dir = "/home/srinivasu/xtrackm/guv/guv_at_intersections/"
print(sats_new)
for sat in sats_new[:]:
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
        f_self = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
        f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
        ds_self = xr.open_dataset(f_self,
                                  engine="netcdf4",
                                  decode_times=False)
        ds_other = xr.open_dataset(f_other,
                                   engine="netcdf4",
                                   decode_times=False)
        lons_track_self = ds_self.lon.values
        lats_track_self = ds_self.lat.values
        lon_coast_self = lons_track_self[-1]  # this on coast
        lat_coast_self = lats_track_self[-1]  # this on coast
        lon_equat_self = lons_track_self[0]  # this on equator
        lat_equat_self = lats_track_self[0]  # this on equator
        slope_self = (lats_track_self[-1] - lats_track_self[0]) / (
            lons_track_self[-1] - lons_track_self[0])
        lons_track_other = ds_other.lon.values
        lats_track_other = ds_other.lat.values
        lon_equat_other = lons_track_other[0]
        lat_equat_other = lats_track_other[0]
        lon_coast_other = lons_track_other[-1]
        lat_coast_other = lats_track_other[-1]
        slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                             lon_equat_other)
        a1l = math.atan2(lat_coast_self - lat_equat_self,
                        lon_coast_self - lon_equat_self)
        a1l_d = math.degrees(a1l)
        ii_self = index_at_lat(ds_self, lats_inter1)
        a1 = local_angle(lons_track_self, lats_track_self, ii_self, ioff=ioff)
        a1_d = math.degrees(a1)
        ii_other = index_at_lat(ds_other, lats_inter1)
        a2l = math.atan2(
            lat_coast_other - lat_equat_other,
            lon_coast_other - lon_equat_other,
        )
        a2l_d = math.degrees(a2l)
        a2 = local_angle(lons_track_other, lats_track_other, ii_other, ioff=ioff)
        a2_d = math.degrees(a2)
        # print(f"heading angle {sat}: a1 = {a1_d:.2f}, a2 = {a2_d:.2f}")
        diff1 = abs(a1 - a1l)
        diff2 = abs(a2 - a2l)
        if diff1 > 0.05 or diff2 > 0.05:
            print(f"{sat} | {track_number_self} diff1 = {diff1:.2f}, {track_number_other} diff2 = {diff2:.2f} Error xxxxxxxxxxxxxx")
        else:
            print(f"{sat} | {track_number_self} diff1 = {diff1:.2f}, {track_number_other} diff2 = {diff2:.2f}")

