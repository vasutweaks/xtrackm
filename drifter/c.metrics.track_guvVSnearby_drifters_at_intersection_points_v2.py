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


def closest_index(lons_drift, lats_drift, lon1, lat1):
    dist = [
        distance.distance((y, x), (lat1, lon1)).m
        for x, y in zip(lons_drift, lats_drift)
    ]
    return np.argmin(dist)


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

for i, r in df_all.iterrows():
    sat1 = r["sat"]
    track_tsta_o, track_tend_o = get_time_limits_o(sat1)
    print(type(track_tsta_o))
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
    print(close_ones)
    # break
    if abs(lat1) < 2:
        continue
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
    lon_coast_self = lons_track_self[-1]  # this on coast
    lat_coast_self = lats_track_self[-1]  # this on coast
    lon_equat_self = lons_track_self[0]  # this on equator
    lat_equat_self = lats_track_self[0]  # this on equator
    slope_self = (lat_coast_self - lat_equat_self) / (lon_coast_self - lon_equat_self)
    angle_self = np.rad2deg(np.arctan(slope_self))
    lons_track_other = ds_other.lon.values
    lats_track_other = ds_other.lat.values
    lon_equat_other = lons_track_other[0]
    lat_equat_other = lats_track_other[0]
    lon_coast_other = lons_track_other[-1]
    lat_coast_other = lats_track_other[-1]
    slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other - lon_equat_other)
    angle_other = np.rad2deg(np.arctan(slope_other))
    idx_self = index_at_lat(ds_self, lat1)
    idx_other = index_at_lat(ds_other, lat1)
    gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
    gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
    gc_self_at = gc_self.isel(x=slice(idx_self-1, idx_self+1)).mean(dim="x")
    gc_other_at = gc_other.isel(x=slice(idx_other-1, idx_other+1)).mean(dim="x")
    # sys.exit(0)
    a1 = math.atan2(
        lats_track_self[idx_self-2] - lats_track_self[idx_self+2],
        lons_track_self[idx_self-2] - lons_track_self[idx_self+2],
    )
    a2 = math.atan2(
        lats_track_other[idx_other-2] - lats_track_other[idx_other+2],
        lons_track_other[idx_other-2] - lons_track_other[idx_other+2],
    )
    print(f"azimuths {a1:.2f} {a2:.2f}")
    print(f"{angle_self} {angle_other} -------------------")
    # print(type(gu.time.values[0]), type(gv.time.values[0]))
    for id_drifter in close_ones:
        print(id_drifter)
        fd = f"{data_loc}/netcdf_all/track_reg/drifter_6h_{id_drifter}.nc"
        print(fd)
        basename1 = os.path.basename(fd)
        # opened without decode_times=False
        ds_d = xr.open_dataset(fd, drop_variables=["WMO"])

        byte_id = ds_d.ID.values[0]
        str_id = byte_id.decode("utf-8").strip()
        drifter_id = str_id

        drift_tsta_o1, drift_tend_o1 = (
            ds_d.start_date.values[0],
            ds_d.end_date.values[0],
        )
        drift_tsta_o, drift_tend_o = n64todatetime1(
            drift_tsta_o1), n64todatetime1(drift_tend_o1)
        # drift_tsta_o, drift_tend_o = drift_tsta_o1, drift_tend_o1
        print(type(track_tsta_o), type(track_tend_o))
        overlap_tsta_o, overlap_tend_o = overlap_dates(track_tsta_o,
                                                       track_tend_o,
                                                       drift_tsta_o,
                                                       drift_tend_o)
        print(f"satellite period {track_tsta_o} {track_tend_o}")
        print(f"drifter period {drift_tsta_o} {drift_tend_o}")
        print(f"overlap period {overlap_tsta_o} {overlap_tend_o}")
        if overlap_tsta_o is None or overlap_tend_o is None:
            continue
        ve_da = drifter_time_asn(ds_d, var_str="ve")
        vn_da = drifter_time_asn(ds_d, var_str="vn")
        lons_da = drifter_time_asn(ds_d, var_str="longitude")
        lats_da = drifter_time_asn(ds_d, var_str="latitude")
        lons_da2 = lons_da.rolling(time=3).mean()
        lats_da2 = lats_da.rolling(time=3).mean()
        lons_da3 = lons_da2.ffill(dim="time").bfill(dim="time")
        lats_da3 = lats_da2.ffill(dim="time").bfill(dim="time")
        ve_da = ve_da.rolling(time=3).mean()
        vn_da = vn_da.rolling(time=3).mean()

        lons_drift = lons_da3.values
        lats_drift = lats_da3.values
        driter_times = ve_da.time.values
        print(type(driter_times[0]))
        nidx = closest_index(lons_drift, lats_drift, lon1, lat1)
        match_time = driter_times[nidx]
        # ve_at = ve_da.isel(time=slice(nidx-1, nidx+1)).mean(dim="time").item()
        ve_at = ve_da.isel(time=nidx)
        try:
            gc_self_at = gc_self_at.interp(time=match_time)
        except:
            continue
        gc_other_at = gc_other_at.interp(time=match_time)
        gu_at, gv_at = geostrophic_components_from_a(gc_self_at, gc_other_at, a1, a2)
        print(type(gu_at))
        gu_at = gu_at.item()
        ve_at = ve_at.item()
        close_drift_lon = lons_drift[nidx]
        close_drift_lat = lats_drift[nidx]
        close_dist = distance.distance((lat1, lon1),
                                       (close_drift_lat, close_drift_lon)).km
        if close_dist < dist_limit:
            gu_ats.append(gu_at)
            ve_ats.append(ve_at)
            close_dists.append(close_dist)

df_out["gu"] = gu_ats
df_out["ve"] = ve_ats
df_out["close_dist"] = close_dists
df_out.to_csv(f"track_guvVSnearby_drifters_at_intersection_points_{sat1}_1.csv")
