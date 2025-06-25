import glob
import math

import shapely.geometry as sg
import xarray as xr
import numpy as np
import pandas as pd
from tools_xtrackm import *

dt0 = datetime.strptime("1950-01-01", "%Y-%m-%d")
var_str = "sla"

for sat in sats_new[:]:
    for f_self in sorted(
            glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds_self = xr.open_dataset(f_self,
                                  engine="h5netcdf",
                                  decode_times=False)
        if len(ds_self.points_numbers) < 12:
            continue
        # sla_self = track_dist_time_asn(ds_self, var_str="sla")
        ln = ds_self[var_str].sizes["points_numbers"]
        ln2 = ln // 2
        dvals_self = ds_self.time.isel(points_numbers=ln2).values
        dates_self = [dt0 + timedelta(days=int(d)) for d in dvals_self]
        time0_self = dates_self[0]
        track_number_self = ds_self.pass_number
        lons_track_self = ds_self.lon.values
        lats_track_self = ds_self.lat.values
        lon_equat_self = lons_track_self[0]
        lat_equat_self = lats_track_self[0]
        lon_coast_self = lons_track_self[-1]
        lat_coast_self = lats_track_self[-1]
        slope_self = (lat_coast_self - lat_equat_self) / (lon_coast_self -
                                                          lon_equat_self)
        if slope_self < 0:
            continue
        track_path_self = sg.LineString(zip(lons_track_self,
                                            lats_track_self))
        # track_path_self = sg.LineString([(lons_track_self[0], lats_track_self[0]),
        #                                  (lons_track_self[-1], lats_track_self[-1])])
        other_times = []
        for f_other in sorted(
                glob.glob(f"../data/ctoh.sla.ref.{sat}.nindian.*.nc")):
            # print(f)
            ds_other = xr.open_dataset(f_other,
                                       engine="h5netcdf",
                                       decode_times=False)
            if len(ds_other.points_numbers) < 12:
                continue
            # sla_other = track_dist_time_asn(ds_other, var_str="sla")
            times_other = ds_other.time
            track_number_other = ds_other.pass_number
            lons_track_other = ds_other.lon.values
            lats_track_other = ds_other.lat.values
            lon_equat_other = lons_track_other[0]
            lat_equat_other = lats_track_other[0]
            lon_coast_other = lons_track_other[-1]
            lat_coast_other = lats_track_other[-1]
            slope_other = (lat_coast_other - lat_equat_other) / (
                lon_coast_other - lon_equat_other)
            if slope_other > 0:
                continue
            # define a linestring for the track
            track_path_other = sg.LineString(
                    zip(lons_track_other, lats_track_other))
            # track_path_other = sg.LineString([(lons_track_other[0], lats_track_other[0]),
            #                                  (lons_track_other[-1], lats_track_other[-1])])
            if track_path_self.intersects(track_path_other):
                point = track_path_self.intersection(track_path_other)
                x_from_coast = distance.distance((lat_coast_self,
                                                  lon_coast_self),
                                                 (point.y, point.x)).km
                # x_idx = index_at_x(x, x_from_coast)
                lat_idx = index_at_lat(ds_other, point.y)
                ln = ds_other[var_str].sizes["points_numbers"]
                ln2 = ln // 2
                # dvals_other = ds_other.time.isel(points_numbers=ln2).values
                dvals_other = ds_other.time.isel(points_numbers=lat_idx).values
                dates_other = [dt0 + timedelta(days=int(d)) for d in dvals_other]
                time0_other = dates_other[0]
                dist_time = (time0_other - time0_self).days
                print(sat, track_number_self,
                      track_number_other,
                      time0_self, time0_other,
                      dist_time, lat_idx, x_from_coast)
                other_times.append((track_number_other, abs(dist_time)))
        if len(other_times) == 0:
            continue
        other_times.sort(key=lambda x: x[1], reverse=True)
        print(track_number_self, other_times[0], "-----------------------")
        break
