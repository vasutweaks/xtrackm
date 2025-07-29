import math

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tools_xtrackm import *


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
        raise ValueError(
            f"Tracks are parallel (slopes: {slope1:.6f}, {slope2:.6f}). "
            "Cannot resolve both velocity components.")
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


cmap1 = cmo.cm.diff
sat = "TP+J1+J2+J3+S6A"
track_number_self, track_number_other = "53", "14"
d = 1.5
height, width = 9, 9
# oscar_dir = "/home/srinivasu/allData/oscar_0.25/"
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset

# ds_oscar_u = xr.open_dataset(f"{oscar_dir}/oscar_0.25_u_full_xr_concat.nc")
# print(ds_oscar_u)
# u_oscar = ds_oscar_u.u

# ds_oscar_v = xr.open_dataset(f"{oscar_dir}/oscar_0.25_v_full_xr_concat.nc")
# print(ds_oscar_v)
# v_oscar = ds_oscar_v.v

f_self = (
    f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
)
ds_self = xr.open_dataset(f_self, decode_times=False)
lons_track_self = ds_self.lon.values
lats_track_self = ds_self.lat.values
lon_equat_self = lons_track_self[0]  # this on coast
lat_equat_self = lats_track_self[0]  # this on coast
lon_coast_self = lons_track_self[-1]  # this on coast
lat_coast_self = lats_track_self[-1]  # this on coast

slope_self = (lats_track_self[-1] -
              lats_track_self[0]) / (lons_track_self[-1] - lons_track_self[0])
angle_self = np.rad2deg(np.arctan(slope_self))

f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
ds_other = xr.open_dataset(f_other, engine="h5netcdf", decode_times=False)
lons_track_other = ds_other.lon.values
lats_track_other = ds_other.lat.values
lon_equat_other = lons_track_other[0]
lat_equat_other = lats_track_other[0]
lon_coast_other = lons_track_other[-1]
lat_coast_other = lats_track_other[-1]

slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                     lon_equat_other)
angle_other = np.rad2deg(np.arctan(slope_other))

track_path_self = sg.LineString(zip(lons_track_self, lats_track_self))
track_path_other = sg.LineString(zip(lons_track_other, lats_track_other))

point = track_path_self.intersection(track_path_other)
lon1 = point.x
lat1 = point.y

sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
sla_other = track_dist_time_asn(ds_other, var_str="sla", units_in="m")

x_from_coast_self1 = distance.distance((lat_coast_self, lon_coast_self),
                                       (point.y, point.x)).m
x_from_coast_other1 = distance.distance((lat_coast_other, lon_coast_other),
                                        (point.y, point.x)).m
print(
    sat,
    track_number_self,
    track_number_other,
    point.x,
    point.y,
    x_from_coast_self1,
    x_from_coast_other1,
)

f_self_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat}_{track_number_self}_loess_0.2.nc"
f_other_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat}_{track_number_other}_loess_0.2.nc"

ds_self_smooth = xr.open_dataset(f_self_smooth)
ds_other_smooth = xr.open_dataset(f_other_smooth)
sla_self_smooth = ds_self_smooth.sla_smooth
sla_other_smooth = ds_other_smooth.sla_smooth

track_path_self = sg.LineString(zip(lons_track_self, lats_track_self))
track_path_other = sg.LineString(zip(lons_track_other, lats_track_other))

# u_oscar_at = u_oscar.sel(lon=lon1, lat=lat1, method="nearest")
# v_oscar_at = v_oscar.sel(lon=lon1, lat=lat1, method="nearest")
# ds_out = xr.Dataset({
#         "u_oscar_at": u_oscar_at,
#         "v_oscar_at": v_oscar_at,
#         })
#
# ds_out.to_netcdf(f"oscar_uv_at_{sat}_{track_number_self}_{track_number_other}.nc")

ds_oscar = xr.open_dataset(f"oscar_uv_at_{sat}_{track_number_self}_{track_number_other}.nc")
u_oscar_at = ds_oscar.u_oscar_at
v_oscar_at = ds_oscar.v_oscar_at

gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)

gc_self_at = gc_self.sel(x=x_from_coast_self1, method="nearest", drop=True)
gc_other_at = gc_other.sel(x=x_from_coast_other1, method="nearest", drop=True)
gc_other_at = gc_other_at.interp(time=gc_self_at.time)

# Better approach:
a2 = math.atan2(lat_coast_other - lat_equat_other, 
                lon_coast_other - lon_equat_other)
a1 = math.atan2(lat_coast_self - lat_equat_self, 
                lon_coast_self - lon_equat_self)

gu, gv = geostrophic_components_from_a(gc_self_at, gc_other_at, a1, a2)

fig1, ax1 = plt.subplots(1, 1, figsize=(width, height), layout="constrained")
gu.plot(ax=ax1, label=f"pass {track_number_other}", color="r", linewidth=2)
u_oscar_at.plot(ax=ax1, color="k", linewidth=2, label="oscar")

# gu.plot(ax=ax1, label=f"pass {track_number_other}", color="r", linewidth=2)
# u_oscar_at.plot(ax=ax1, color="k", linewidth=2, label="oscar")

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
fig2, ax2 = plt.subplots(
    1,
    1,
    subplot_kw={"projection": ccrs.PlateCarree()},
    figsize=(width, height),
    layout="constrained",
)
dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    *TRACKS_REG[2:])).plot(ax=ax2,
                           add_colorbar=False,
                           add_labels=False,
                           cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG)
ax2.grid()
ax2.scatter(
    lons_track_self,
    lats_track_self,
    marker=".",
    color="c",
    s=4,
)
ax2.scatter(
    lons_track_other,
    lats_track_other,
    marker=".",
    color="c",
    s=4,
)
lonm, latm = get_point_at_distance(lon_equat_self, lat_equat_self,
                                   lon_coast_self, lat_coast_self, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(
        lonm,
        latm,
        s=track_number_self,
        fontsize=14,
        rotation=angle_self,
        color="w",
    )
lonm, latm = get_point_at_distance(lon_equat_other, lat_equat_other,
                                   lon_coast_other, lat_coast_other, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(
        lonm,
        latm,
        s=track_number_other,
        fontsize=14,
        rotation=angle_other,
        color="w",
    )
plt.show()
plt.close("all")

