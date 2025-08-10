import os

import numpy as np
import xarray as xr
from tools_xtrackm import *

sat, track_number_self, track_number_other, lons_inter, lats_inter = (
    "S3A",
    "8",
    "79",
    94.14047103085358,
    8.36326782978126,
)
d = 1.5
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
rose = dse.ROSE
cmap1 = "Greys_r"
lon1, lat1 = lons_inter, lats_inter
dist_threshold = 2
computed_loc = f"/home/srinivasu/slnew/xtrackm/computed/"
data_loc = f"/home/srinivasu/allData/drifter1/"
fd = f"{data_loc}/netcdf_15001_current/track_reg/drifter_6h_133666.nc"
box = create_box_patch(lons_inter, lats_inter, dist_threshold)
lon_max, lat_max = lons_inter + dist_threshold, lats_inter + dist_threshold
lat_min, lon_min = lats_inter - dist_threshold, lons_inter - dist_threshold
track_tsta_o, track_tend_o = get_time_limits_o(sat)
print(f"{sat} {track_tsta_o} {track_tend_o}")

basename1 = os.path.basename(fd)
ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
byte_id = ds_d.ID.values[0]
str_id = byte_id.decode("utf-8")
drifter_id = str_id
drift_tsta_o1, drift_tend_o1 = (
    ds_d.start_date.values[0],
    ds_d.end_date.values[0],
)
print(f"{drifter_id} {drift_tsta_o1} {drift_tend_o1}")
drift_tsta_o, drift_tend_o = n64todatetime(drift_tsta_o1), n64todatetime(
    drift_tend_o1)
overlap_tsta_o, overlap_tend_o = overlap_dates(track_tsta_o, track_tend_o,
                                               drift_tsta_o, drift_tend_o)
print(f"overlap period {overlap_tsta_o} {overlap_tend_o}")
times_drift = ds_d.time.isel(traj=0).values
lons_da = drifter_time_asn(ds_d, var_str="longitude")
lats_da = drifter_time_asn(ds_d, var_str="latitude")
lons_da1 = lons_da.resample(time="1D").mean()
lats_da1 = lats_da.resample(time="1D").mean()
lons_da2 = lons_da1.rolling(time=3).mean()
lats_da2 = lats_da1.rolling(time=3).mean()
lons_da3 = lons_da2.ffill(dim="time").bfill(dim="time")
lats_da3 = lats_da2.ffill(dim="time").bfill(dim="time")

lons_drift = lons_da3.values
lats_drift = lats_da3.values

f_self = (
    f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
)
f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
f_self_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat}_{track_number_self}_loess_0.2.nc"
f_other_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat}_{track_number_other}_loess_0.2.nc"
ds_self = xr.open_dataset(f_self, engine="h5netcdf", decode_times=False)
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
slope_self = (lats_track_self[-1] -
              lats_track_self[0]) / (lons_track_self[-1] - lons_track_self[0])
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

fig = plt.figure(figsize=(12, 10), layout="constrained")
ax2 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
rose.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    *TRACKS_REG[2:])).plot(ax=ax2,
                           add_colorbar=False,
                           add_labels=False,
                           cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG, step=5)
ax2.grid()
ax2.plot(
    lon1,
    lat1,
    c="r",
    marker="o",
    markersize=4,
    markerfacecolor="red",
)
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
ax2.scatter(
    lons_drift,
    lats_drift,
    marker=".",
    color="b",
    s=4,
)
# add polygon box as geometry
ax2.add_geometries([box],
                   ccrs.PlateCarree(),
                   edgecolor="g",
                   facecolor="none",
                   linewidth=2)
lonm, latm = get_point_at_distance(lon_equat_self, lat_equat_self,
                                   lon_coast_self, lat_coast_self, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(
        lonm,
        latm,
        s=track_number_self,
        fontsize=14,
        rotation=angle_self,
        color="k",
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
        color="k",
    )
plt.show()
