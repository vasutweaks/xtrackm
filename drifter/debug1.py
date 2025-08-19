import ast
import os
import sys

import numpy as np
import xarray as xr
from geopy import distance

# import Polygon
from tools_xtrackm import *

# Three changes
# Index based selection
# local azimuths
# smoothing of ve and gc


def index_at_lat(ds, lat1):
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    i1 = np.abs(lats_track_rev - lat1).argmin()
    return i1


def convert_to_list(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError):
        return x  # Return as-is if conversion fails


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
    u = -1 * u
    return u, v


data_loc = f"/home/srinivasu/allData/drifter1/"
sat_here = sys.argv[1]
df_all = pd.read_csv(f"close_drifters_at_intersection_point_{sat_here}.csv")
df_all["close_drifters_column"] = df_all["close_drifters_column"].apply(
    convert_to_list)
dist_limit = 15  # km
gu_ats = []
ve_ats = []
df_out = pd.DataFrame()
idx_delta = 5
close_dists = []
tolerance = 0.05

for i, r in df_all.iterrows():
    sat1 = r["sat"]
    track_tsta_o, track_tend_o = get_time_limits_o(sat1)
    # print(type(track_tsta_o))
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
    close_ones = r["close_drifters_column"]
    # print(close_ones)
    # break
    if abs(lat1) < 2:
        continue
    print("----------------------------------------")
    print(
        sat1,
        track_number_self,
        track_number_other,
        lons_inter1,
        lats_inter1,
        x_from_coast_self1,
        x_from_coast_other1,
        angle_acute,
        angle_obtuse,
    )
    f_self = f"../data/{sat1}/ctoh.sla.ref.{sat1}.nindian.{track_number_self.zfill(3)}.nc"
    f_other = f"../data/{sat1}/ctoh.sla.ref.{sat1}.nindian.{track_number_other.zfill(3)}.nc"
    f_self_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat1}_{track_number_self}_loess_0.2.nc"
    f_other_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat1}_{track_number_other}_loess_0.2.nc"
    # opened with decode_times=False
    ds_self = xr.open_dataset(f_self, engine="h5netcdf", decode_times=False)
    ds_self_smooth = xr.open_dataset(f_self_smooth)
    ds_other = xr.open_dataset(f_other, engine="h5netcdf", decode_times=False)
    ds_other_smooth = xr.open_dataset(f_other_smooth)
    sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
    sla_self_smooth = ds_self_smooth.sla_smooth
    sla_other = track_dist_time_asn(ds_other, var_str="sla", units_in="m")
    sla_other_smooth = ds_other_smooth.sla_smooth
    lons_track_self = ds_self.lon.values
    lats_track_self = ds_self.lat.values
    lons_track_self_rev = ds_self.lon.values[::-1]
    lats_track_self_rev = ds_self.lat.values[::-1]
    # lons_track_self = sla_self.lon.values
    # lats_track_self = sla_self.lat.values
    lon_coast_self = lons_track_self[-1]  # this on coast
    lat_coast_self = lats_track_self[-1]  # this on coast
    lon_equat_self = lons_track_self[0]  # this on equator
    lat_equat_self = lats_track_self[0]  # this on equator
    slope_self = (lat_coast_self - lat_equat_self) / (lon_coast_self - lon_equat_self)
    angle_self = np.rad2deg(np.arctan(slope_self))
    lons_track_other = ds_other.lon.values
    lats_track_other = ds_other.lat.values
    lons_track_other_rev = ds_other.lon.values[::-1]
    lats_track_other_rev = ds_other.lat.values[::-1]
    # lons_track_other = sla_other.lon.values
    # lats_track_other = sla_other.lat.values
    lon_equat_other = lons_track_other[0]
    lat_equat_other = lats_track_other[0]
    lon_coast_other = lons_track_other[-1]
    lat_coast_other = lats_track_other[-1]
    slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other - lon_equat_other)
    angle_other = np.rad2deg(np.arctan(slope_other))
    idx_self = index_at_lat(ds_self, lat1)
    idx_other = index_at_lat(ds_other, lat1)
    lat1_test = lats_track_self_rev[idx_self]
    lon1_test = lons_track_self_rev[idx_self]
    lat2_test = lats_track_other_rev[idx_other]
    lon2_test = lons_track_other_rev[idx_other]
    print(f"idx_self {idx_self}, idx_other {idx_other}")
    print(f"lats_self {lat1_test}, lats_other {lat2_test}")
    assert math.isclose(lat1_test, lat2_test, abs_tol=tolerance), "intersection lats not close"
    print(f"lons_self {lon1_test}, lons_other {lon2_test}")
    assert math.isclose(lon1_test, lon2_test, abs_tol=tolerance), "intersection lons not close"
    gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
    gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
    gc_self_at = gc_self.isel(x=slice(idx_self-1, idx_self+1)).mean(dim="x")
    gc_other_at = gc_other.isel(x=slice(idx_other-1, idx_other+1)).mean(dim="x")
    # sys.exit(0)
    a1 = math.atan2(
        lats_track_self_rev[idx_self-2] - lats_track_self_rev[idx_self+2],
        lons_track_self_rev[idx_self-2] - lons_track_self_rev[idx_self+2],
    )
    a2 = math.atan2(
        lats_track_other_rev[idx_other-2] - lats_track_other_rev[idx_other+2],
        lons_track_other_rev[idx_other-2] - lons_track_other_rev[idx_other+2],
    )
    print(f"azimuths {a1:.2f} {a2:.2f}")
    a1a = math.atan2(lat_coast_self - lat_equat_self,
                    lon_coast_self - lon_equat_self)
    a2a = math.atan2(
        lat_coast_other - lat_equat_other,
        lon_coast_other - lon_equat_other,
    )
    print(f"azimuths {a1a:.2f} {a2a:.2f}")
    if i > 10:
        break
