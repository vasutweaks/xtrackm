import math

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
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


sat = "TP+J1+J2+J3+S6A"
track_number_self, track_number_other = "53", "14"
ds_oscar = xr.open_dataset(f"oscar_uv_at_{sat}_{track_number_self}_{track_number_other}.nc")
u_oscar_at = ds_oscar.u_oscar_at
v_oscar_at = ds_oscar.v_oscar_at


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

slope_self = (lat_coast_self - lat_equat_self) / (lon_coast_self - lon_equat_self)

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

gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)

gc_self_at = gc_self.sel(x=x_from_coast_self1, method="nearest", drop=True)
gc_other_at = gc_other.sel(x=x_from_coast_other1, method="nearest", drop=True)
gc_other_at = gc_other_at.interp(time=gc_self_at.time)

# Better approach:
a1 = math.atan2(lat_coast_self - lat_equat_self, 
                lon_coast_self - lon_equat_self)
a2 = math.atan2(lat_coast_other - lat_equat_other, 
                lon_coast_other - lon_equat_other)

# gc_self_at = -1 * gc_self_at
# gc_other_at = -1 * gc_other_at
gu, gv = geostrophic_components_from_a(gc_self_at, gc_other_at, a1, a2)

# print correlation between gv and v_oscar_at

fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8), layout="constrained")

gu_cut = -1 * gu.interp(time=u_oscar_at.time)
gu_cut.plot(ax=ax1, label=f"track zonal", color="r", linewidth=2)
u_oscar_at.plot(ax=ax1, color="b", linewidth=2, label="oscar u")
corr = xs.pearson_r(gu_cut, u_oscar_at, dim="time", skipna=True)
corr1 = corr.item()
print(f"Correlation between gu and u_oscar_at: {corr1}")

# gv_cut = gv.interp(time=v_oscar_at.time)
# gv_cut.plot(ax=ax1, label=f"track meridional", color="r", linewidth=2)
# v_oscar_at.plot(ax=ax1, color="b", linewidth=2, label="oscar v")
# corr = xs.pearson_r(gv_cut, v_oscar_at, dim="time", skipna=True)
# corr1 = corr.item()
# print(f"Correlation between gv and v_oscar_at: {corr1}")

# text coordinate of intersection point
plt.text(
    0.15,
    0.95,
    f"intersection point ({lon1:.2f}, {lat1:.2f})",
    transform=ax1.transAxes,
    fontsize=14,
    fontweight="bold",
)
# set horizontal time limits in datetime objects
plt.xlim(datetime(2000, 1, 1), datetime(2010, 1, 1))
# plt.ylim(-1, 1)
plt.ylim(-0.8, 0.8)
plt.legend()
plt.show()
plt.close("all")

