import cmocean as cmo
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import sys

import math
import numpy as np


def geostrophic_components_from_slopes(gc1, gc2, slope1, slope2):
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
    # Convert slopes to azimuths (angles from east axis)
    a1 = math.atan(slope1)  # azimuth of track 1 in radians
    a2 = math.atan(slope2)  # azimuth of track 2 in radians
    # Calculate intersection angle
    theta = a2 - a1
    sin_theta = math.sin(theta)
    # Check for parallel tracks (same slope)
    if abs(slope1 - slope2) < 1e-10:
        raise ValueError(f"Tracks are parallel (slopes: {slope1:.6f}, {slope2:.6f}). "
                        "Cannot resolve both velocity components.")
    # Additional check using sine of intersection angle
    if abs(sin_theta) < 1e-6:
        raise ValueError(f"Tracks are nearly parallel (intersection angle = {math.degrees(theta):.2f}°). "
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


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate great-circle distance between (lon1, lat1)
    and arrays (lon2, lat2) in kilometers."""
    R = 6371  # Radius of Earth in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))


def find_n_closest_points(df, lon0, lat0, n=4):
    """Return the n rows in df closest to (lon0, lat0) based on
    lons_inter/lats_inter."""
    distances = haversine_distance(lon0, lat0, df["lons_inter"].values,
                                   df["lats_inter"].values)
    df = df.copy()
    df["distance_km"] = distances
    return df.nsmallest(n, "distance_km")


sat = "TP+J1+J2+J3+S6A"
d = 1.5
height, width = 9, 9

dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
omni_id = "BD09"
omni_data_dir = "/home/srinivasu/allData/MB_cleaned"
lon_omni, lat_omni = omni_d[omni_id]
dpth = 15.

ds_omni = xr.open_dataset(f"{omni_data_dir}/{omni_id}_uvs_z_xr_daily.nc", decode_times=False)
print(ds_omni)
ds_omni = change_time(ds_omni, "time")
omni_v = ds_omni.vd.interp(depth=dpth)
# make 10 average of omni_v
omni_v = omni_v.rolling(time=10, center=True).mean()
omni_v = 0.01 * omni_v
N = omni_v.count().item()
print(N)

df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
df_4 = find_n_closest_points(df, lon_omni, lat_omni, n=4)
print(omni_id)
print(df_4)

# sys.exit()
track_self = df_4["track_self"].apply(lambda x: str(x)).values
track_other = df_4["track_other"].apply(lambda x: str(x)).values
lons_inter = df_4["lons_inter"].values
lats_inter = df_4["lats_inter"].values
x_from_coast_self = df_4["x_from_coast_self"].values
x_from_coast_other = df_4["x_from_coast_other"].values
angle_acute = df_4["angle_acute"].values
angle_obtuse = df_4["angle_obtuse"].values

for i in range(len(df_4)):
    track_number_self = track_self[i]
    track_number_other = track_other[i]
    f_self = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
    f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
    f_self_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat}_{track_number_self}_loess_0.2.nc"
    f_other_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat}_{track_number_other}_loess_0.2.nc"
    lon1 = lons_inter[i]
    lat1 = lats_inter[i]
    x_from_coast_self1 = x_from_coast_self[i]
    x_from_coast_other1 = x_from_coast_other[i]
    angle_acute1 = angle_acute[i]
    angle_obtuse1 = angle_obtuse[i]
    ds_self = xr.open_dataset(f_self, decode_times=False)
    ds_self_smooth = xr.open_dataset(f_self_smooth)
    sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
    sla_self_smooth = ds_self_smooth.sla_smooth
    ds_other = xr.open_dataset(f_other, engine="h5netcdf", decode_times=False)
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
    slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other - lon_equat_other)
    angle_other = np.rad2deg(np.arctan(slope_other))
    track_path_other = sg.LineString(zip(lons_track_other, lats_track_other))
    gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
    gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
    # if track_path_self.intersects(track_path_other):
    gc_self_at = gc_self.sel(x=x_from_coast_self1,
                               method="nearest",
                               drop=True)
    gc_other_at = gc_other.sel(x=x_from_coast_other1,
                                 method="nearest",
                                 drop=True)
    gc_other_at = gc_other_at.interp(time=gc_self_at.time)
    gu, gv = geostrophic_components_from_slopes(gc_self_at, gc_other_at, slope_self, slope_other)
    # other_times.append((track_number_other, abs(dist_time)))
    fig1, ax1 = plt.subplots(1,
                            1,
                            figsize=(width, height),
                            layout="constrained")
    # gc_self_at.plot(ax=ax1, label=f"pass {track_number_self}")
    # gc_other_at.plot(ax=ax1, label=f"pass {track_number_other}")
    gv.plot(ax=ax1, label=f"pass {track_number_other}")
    omni_v.plot(ax=ax1, color="r", linewidth=2, label="omni")
    # text coordinate of intersection point
    plt.text(
        0.15,
        0.95,
        f"intersection point ({lon1:.2f}, {lat1:.2f})",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    plt.legend()
    # fig2, ax2 = plt.subplots(
    #     1,
    #     1,
    #     subplot_kw={"projection": ccrs.PlateCarree()},
    #     figsize=(width, height),
    #     layout="constrained",
    # )
    # dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    #     *TRACKS_REG[2:])).plot(ax=ax2,
    #                            add_colorbar=False,
    #                            add_labels=False,
    #                            cmap=cmap1)
    # decorate_axis(ax2, "", *TRACKS_REG)
    # ax2.grid()
    # ax2.plot(
    #     lon_omni,
    #     lat_omni,
    #     c="r",
    #     marker="o",
    #     markersize=6,
    #     markerfacecolor='red',
    #     markeredgecolor='black',
    #     markeredgewidth=2
    # )
    # # plot plat form code of omni at lon_omni, lat_omni
    # plt.text(lon_omni, lat_omni, f"{omni_id}", fontsize=10, color="k") 
    # ax2.scatter(
    #     lons_track_self,
    #     lats_track_self,
    #     marker=".",
    #     color="c",
    #     s=4,
    # )
    # ax2.scatter(
    #     lons_track_other,
    #     lats_track_other,
    #     marker=".",
    #     color="c",
    #     s=4,
    # )
    # lonm, latm = get_point_at_distance(lon_equat_self, lat_equat_self,
    #                                    lon_coast_self, lat_coast_self, d)
    # if is_within_region(lonm, latm, *TRACKS_REG):
    #     plt.text(
    #         lonm,
    #         latm,
    #         s=track_number_self,
    #         fontsize=14,
    #         rotation=angle_self,
    #         color="w",
    #     )
    # lonm, latm = get_point_at_distance(lon_equat_other, lat_equat_other, lon_coast_other, lat_coast_other, d)
    # if is_within_region(lonm, latm, *TRACKS_REG):
    #     plt.text(
    #         lonm,
    #         latm,
    #         s=track_number_other,
    #         fontsize=14,
    #         rotation=angle_other,
    #         color="w",
    #     )
    plt.show()
    plt.close("all")
    break

dse.close()
