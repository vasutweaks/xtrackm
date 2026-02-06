import math

import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
from matplotlib import pyplot as plt
from scipy.stats import median_abs_deviation
import math


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
        f_self_smooth = f"../computed/sla_loess_0.05a/track_sla_along_{sat}_{track_number_self}_loess_0.05.nc"
        f_other_smooth = f"../computed/sla_loess_0.05a/track_sla_along_{sat}_{track_number_other}_loess_0.05.nc"
        ds_self = xr.open_dataset(f_self,
                                  engine="netcdf4",
                                  decode_times=False)
        ds_self_smooth = xr.open_dataset(f_self_smooth)
        sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
        sla_self_smooth = ds_self_smooth.sla_smooth
        ds_other = xr.open_dataset(f_other,
                                   engine="netcdf4",
                                   decode_times=False)
        ds_other_smooth = xr.open_dataset(f_other_smooth)
        sla_other = track_dist_time_asn(ds_other, var_str="sla", units_in="m")
        sla_other_smooth = ds_other_smooth.sla_smooth
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
        ii_self = index_at_lat(ds_self, lats_inter1)
        a1 = local_angle(lons_track_self, lats_track_self, ii_self, ioff=ioff)
        a1_d = math.degrees(a1)
        ii_other = index_at_lat(ds_other, lats_inter1)
        a2 = local_angle(lons_track_other, lats_track_other, ii_other, ioff=ioff)
        a2_d = math.degrees(a2)
        # print(f"heading angle {sat}: a1 = {a1l_d:.2f}, a2 = {a2l_d:.2f}")
        print(f"heading angle {sat}: a1 = {a1_d:.2f}, a2 = {a2_d:.2f}")
        gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
        gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
        gc_self_at = gc_self.sel(x=x_from_coast_self1,
                                 method="nearest",
                                 drop=True)
        gc_other_at = gc_other.sel(x=x_from_coast_other1,
                                   method="nearest",
                                   drop=True)
        gc_other_at = gc_other_at.interp(time=gc_self_at.time)
        gu, gv = geostrophic_components(gc_self_at, gc_other_at, a1, a2)
        # filter xarray dataarray gu if greater than 3.5 or less than -3.5
        gu_filt = gu.where((gu <= 3.5) & (gu >= -3.5))
        gu_filtered = mad_filter_1d_xr(gu_filt, 3.5)
        gv_filt = gv.where((gv <= 3.5) & (gv >= -3.5))
        gv_filtered = mad_filter_1d_xr(gv_filt, 3.5)
        # gu.plot(label="gu", marker=".")
        # gu_filtered.plot(label="gu_filtered", marker="x")
        # plt.legend()
        # plt.show()
        # create a ds_out with gu and gv
        ds_out = xr.Dataset({"gu": gu_filtered, "gv": gv_filtered})
        filename = f"{out_dir}/guv_at_intersection_{sat1}_{track_number_self}_{track_number_other}_0.05.nc"
        print(filename)
        ds_out.to_netcdf(filename)
