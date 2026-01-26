import math

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *


def geostrophic_components_from_a(gc1, gc2, a1, a2):
    """
    Compute zonal and meridional geostrophic current components from
    cross-track velocities and track slopes of two intersecting satellite tracks.
    Parameters:
    -----------
    gc1 : float or array-like
        Cross-track geostrophic velocity from track 1 (m/s)
    gc2 : float or array-like
        Cross-track geostrophic velocity from track 2 (m/s)
    slope1 : float
        Slope of track 1 (dy/dx = rise/run)
    slope2 : float
        Slope of track 2 (dy/dx = rise/run)
    Returns:
    --------
    u : float or array-like
        Zonal (eastward) velocity component (m/s)
    v : float or array-like
        Meridional (northward) velocity component (m/s)
    Raises:
    -------
    ValueError: If tracks are parallel (same slope)
    """
    # Calculate intersection angle
    theta = a2 - a1
    sin_theta = math.sin(theta)
    # Additional check using sine of intersection angle
    if abs(sin_theta) < 1e-6:
        raise ValueError(
            f"Tracks are nearly parallel (intersection angle = {math.degrees(theta):.2f}°). "
            "Cannot resolve both velocity components.")
    # Calculate trigonometric values
    cos_a1 = math.cos(a1)
    cos_a2 = math.cos(a2)
    sin_a1 = math.sin(a1)
    sin_a2 = math.sin(a2)
    # Solve for velocity components using the derived formulas:
    # u = (gc1 * cos(a2) - gc2 * cos(a1)) / sin(a2 - a1)
    # v = (gc2 * sin(a1) - gc1 * sin(a2)) / sin(a2 - a1)
    u = (gc1 * cos_a2 - gc2 * cos_a1) / sin_theta
    v = (gc2 * sin_a1 - gc1 * sin_a2) / sin_theta
    return u, v


for sat in sats_new:
    gu_all = []
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    for i, row in df.iterrows():
        track_number_self = str(row["track_self"])
        track_number_other = str(row["track_other"])
        lons_inter1 = row["lons_inter"]
        lats_inter1 = row["lats_inter"]
        lon1 = lons_inter1
        lat1 = lats_inter1
        x_from_coast_self1 = row["x_from_coast_self"]
        x_from_coast_other1 = row["x_from_coast_other"]
        angle_acute = row["angle_acute"]
        angle_obtuse = row["angle_obtuse"]
        if abs(lat1) < 2:
            continue
        print(
            sat,
            track_number_self,
            track_number_other,
            lons_inter1,
            lats_inter1,
            x_from_coast_self1,
            x_from_coast_other1,
            angle_acute,
            angle_obtuse,
        )
        f_self = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
        f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
        f_self_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat}_{track_number_self}_loess_0.2.nc"
        f_other_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat}_{track_number_other}_loess_0.2.nc"
        ds_self = xr.open_dataset(f_self, decode_times=False)
        ds_self_smooth = xr.open_dataset(f_self_smooth)
        sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
        sla_self_smooth = ds_self_smooth.sla_smooth
        ds_other = xr.open_dataset(f_other, engine="netcdf4", decode_times=False)
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
        angle_self = np.rad2deg(np.arctan(slope_self))
        track_path_self = sg.LineString(zip(lons_track_self, lats_track_self))
        lons_track_other = ds_other.lon.values
        lats_track_other = ds_other.lat.values
        lon_equat_other = lons_track_other[0]
        lat_equat_other = lats_track_other[0]
        lon_coast_other = lons_track_other[-1]
        lat_coast_other = lats_track_other[-1]
        slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                             lon_equat_other)
        angle_other = np.rad2deg(np.arctan(slope_other))
        track_path_other = sg.LineString(zip(lons_track_other, lats_track_other))
        gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
        gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
        # if track_path_self.intersects(track_path_other):
        gc_self_at = gc_self.sel(x=x_from_coast_self1, method="nearest", drop=True)
        gc_other_at = gc_other.sel(x=x_from_coast_other1,
                                   method="nearest",
                                   drop=True)
        gc_other_at = gc_other_at.interp(time=gc_self_at.time)
        # Better approach:
        a2 = math.atan2(lat_coast_other - lat_equat_other,
                        lon_coast_other - lon_equat_other)
        a1 = math.atan2(lat_coast_self - lat_equat_self,
                        lon_coast_self - lon_equat_self)

        gu, gv = geostrophic_components_from_a(gc_self_at, gc_other_at, a1, a2)
        # gu.plot()
        gu_all.extend(gu.values)
        # save as npz file
        # break
    # plot histogram of gu_all
    plt.hist(gu_all, bins=100, label=f"{sat}")
    np.savez(f"guv_at_intersections_{sat}_0.2.npz", gu_all=gu_all)
    # plt.show()
