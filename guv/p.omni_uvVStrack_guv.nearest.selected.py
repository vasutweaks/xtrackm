import math

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *


def normalize_track_id(series):
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def filter_intersections(df, sat, track_self, track_other):
    sat = str(sat).strip()
    track_self = str(track_self).strip()
    track_other = str(track_other).strip()
    dff = df.copy()
    dff["track_self_norm"] = normalize_track_id(dff["track_self"])
    dff["track_other_norm"] = normalize_track_id(dff["track_other"])
    mask_sat = dff["sat"].astype(str).str.strip() == sat
    pair_forward = ((dff["track_self_norm"] == track_self) &
                    (dff["track_other_norm"] == track_other))
    pair_reverse = ((dff["track_self_norm"] == track_other) &
                    (dff["track_other_norm"] == track_self))
    return dff[mask_sat & (pair_forward | pair_reverse)].copy()


cmap1 = cmo.cm.diff
d = 1.5
dpth = 15.0
height, width = 9, 9
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
omni_data_dir = "/home/srinivasu/allData/MB_cleaned/"
frac1 = "0.05"

df_all_list = []
for sat in sats_new:
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    df_all_list.append(df)
df_all = pd.concat(df_all_list, ignore_index=True)

omni_id = "AD09"
sat = "S3B"
track_self = "580"
track_other = "651"

df_sel = filter_intersections(df_all, sat, track_self, track_other)
if df_sel.empty:
    raise ValueError(
        f"No intersections for sat={sat}, tracks=({track_self}, {track_other})"
    )

lon_omni, lat_omni = omni_d[omni_id]
ds_omni = xr.open_dataset(f"{omni_data_dir}/{omni_id}_uvs_z_xr_daily.nc",
                          decode_times=False)
print(ds_omni)
ds_omni = change_time(ds_omni, "time")
omni_u = ds_omni.ud.interp(depth=dpth)
omni_u = omni_u.rolling(time=10, center=True).mean()
omni_u = 0.01 * omni_u
omni_v = ds_omni.vd.interp(depth=dpth)
omni_v = omni_v.rolling(time=10, center=True).mean()
omni_v = 0.01 * omni_v
N = omni_v.count().item()
print(N)

df_1 = find_n_closest_points(df_sel, lon_omni, lat_omni, n=1)
row = df_1.iloc[0]
sat = row["sat"]
track_number_self = str(row["track_self_norm"])
track_number_other = str(row["track_other_norm"])
lons_inter1 = row["lons_inter"]
lats_inter1 = row["lats_inter"]
lon1 = lons_inter1
lat1 = lats_inter1
x_from_coast_self1 = row["x_from_coast_self"]
x_from_coast_other1 = row["x_from_coast_other"]
angle_acute = row["angle_acute"]
angle_obtuse = row["angle_obtuse"]
distance_km = row["distance_km"]
if abs(lat1) < 2:
    raise ValueError(f"Selected intersection near equator (lat={lat1:.2f}); skipped.")
print(
    sat,
    omni_id,
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
f_self_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_self}_loess_{frac1}.nc"
f_other_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_other}_loess_{frac1}.nc"
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
lon_coast_self = lons_track_self[-1]
lat_coast_self = lats_track_self[-1]
lon_equat_self = lons_track_self[0]
lat_equat_self = lats_track_self[0]
slope_self = (lats_track_self[-1] - lats_track_self[0]) / (
    lons_track_self[-1] - lons_track_self[0])
angle_self = np.rad2deg(np.arctan(slope_self))
lons_track_other = ds_other.lon.values
lats_track_other = ds_other.lat.values
lon_equat_other = lons_track_other[0]
lat_equat_other = lats_track_other[0]
lon_coast_other = lons_track_other[-1]
lat_coast_other = lats_track_other[-1]
slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                     lon_equat_other)
angle_other = np.rad2deg(np.arctan(slope_other))
gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
gc_self_at = gc_self.sel(x=x_from_coast_self1, method="nearest", drop=True)
gc_other_at = gc_other.sel(x=x_from_coast_other1, method="nearest", drop=True)
gc_other_at = gc_other_at.interp(time=gc_self_at.time)
a2 = math.atan2(lat_coast_other - lat_equat_other, lon_coast_other - lon_equat_other)
a1 = math.atan2(lat_coast_self - lat_equat_self, lon_coast_self - lon_equat_self)

gu, gv = geostrophic_components(gc_self_at, gc_other_at, a1, a2)
gu_cut = gu
gv_cut = gv

if distance_km > 30:
    raise ValueError(f"Nearest selected intersection too far from omni: {distance_km:.2f} km")

fig = plt.figure(figsize=(12, 8), layout="constrained")
ax1 = fig.add_subplot(2, 1, 1)
gu_cut.plot(ax=ax1, label="track zonal", color="r", linewidth=2)
tfreq = sats_tfreq[sat]
omni_v_rs = omni_v.resample(time=f"{tfreq}D").mean().interp(time=gv_cut.time)
omni_u.plot(ax=ax1, color="b", linewidth=2, label="omni u")
ax1.set_xlim(datetime(2019, 1, 1), datetime(2021, 3, 1))
corr = xs.pearson_r(gv_cut, omni_v_rs, dim="time", skipna=True)
corr1 = corr.item()
print(f"Correlation between gv and v_oscar_at: {corr1}")
info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat} {distance_km:.2f} km"
plt.text(0.15, 0.95, info, transform=ax1.transAxes, fontsize=14, fontweight="bold")
plt.legend()

ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    *TRACKS_REG[2:])).plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG, step=5)
ax2.grid()
ax2.plot(
    lon_omni,
    lat_omni,
    c="r",
    marker="o",
    markersize=6,
    markerfacecolor="red",
    markeredgecolor="black",
    markeredgewidth=2,
)
plt.text(lon_omni, lat_omni, f"{omni_id}", fontsize=10, color="k")
ax2.scatter(lons_track_self, lats_track_self, marker=".", color="c", s=4)
ax2.scatter(lons_track_other, lats_track_other, marker=".", color="c", s=4)
lonm, latm = get_point_at_distance(lon_equat_self, lat_equat_self, lon_coast_self,
                                   lat_coast_self, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(lonm, latm, s=track_number_self, fontsize=14, rotation=angle_self, color="w")
lonm, latm = get_point_at_distance(lon_equat_other, lat_equat_other, lon_coast_other,
                                   lat_coast_other, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(lonm, latm, s=track_number_other, fontsize=14, rotation=angle_other, color="w")

plt.show()
plt.savefig(
    f"omniVStrack_guv_pngs/{omni_id}_{sat}_{track_number_self}_{track_number_other}_trim.png"
)
plt.close("all")
ds_omni.close()
dse.close()
# plt.close("all")
#
# dse.close()
omni_id = "AD10"
sat = "S3B"
track_self = "466"
track_other = "651"

df_sel = filter_intersections(df_all, sat, track_self, track_other)
if df_sel.empty:
    raise ValueError(
        f"No intersections for sat={sat}, tracks=({track_self}, {track_other})"
    )

lon_omni, lat_omni = omni_d[omni_id]
ds_omni = xr.open_dataset(f"{omni_data_dir}/{omni_id}_uvs_z_xr_daily.nc",
                          decode_times=False)
print(ds_omni)
ds_omni = change_time(ds_omni, "time")
omni_u = ds_omni.ud.interp(depth=dpth)
omni_u = omni_u.rolling(time=10, center=True).mean()
omni_u = 0.01 * omni_u
omni_v = ds_omni.vd.interp(depth=dpth)
omni_v = omni_v.rolling(time=10, center=True).mean()
omni_v = 0.01 * omni_v
N = omni_v.count().item()
print(N)

df_1 = find_n_closest_points(df_sel, lon_omni, lat_omni, n=1)
row = df_1.iloc[0]
sat = row["sat"]
track_number_self = str(row["track_self_norm"])
track_number_other = str(row["track_other_norm"])
lons_inter1 = row["lons_inter"]
lats_inter1 = row["lats_inter"]
lon1 = lons_inter1
lat1 = lats_inter1
x_from_coast_self1 = row["x_from_coast_self"]
x_from_coast_other1 = row["x_from_coast_other"]
angle_acute = row["angle_acute"]
angle_obtuse = row["angle_obtuse"]
distance_km = row["distance_km"]
if abs(lat1) < 2:
    raise ValueError(f"Selected intersection near equator (lat={lat1:.2f}); skipped.")
print(
    sat,
    omni_id,
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
f_self_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_self}_loess_{frac1}.nc"
f_other_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_other}_loess_{frac1}.nc"
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
lon_coast_self = lons_track_self[-1]
lat_coast_self = lats_track_self[-1]
lon_equat_self = lons_track_self[0]
lat_equat_self = lats_track_self[0]
slope_self = (lats_track_self[-1] - lats_track_self[0]) / (
    lons_track_self[-1] - lons_track_self[0])
angle_self = np.rad2deg(np.arctan(slope_self))
lons_track_other = ds_other.lon.values
lats_track_other = ds_other.lat.values
lon_equat_other = lons_track_other[0]
lat_equat_other = lats_track_other[0]
lon_coast_other = lons_track_other[-1]
lat_coast_other = lats_track_other[-1]
slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                     lon_equat_other)
angle_other = np.rad2deg(np.arctan(slope_other))
gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
gc_self_at = gc_self.sel(x=x_from_coast_self1, method="nearest", drop=True)
gc_other_at = gc_other.sel(x=x_from_coast_other1, method="nearest", drop=True)
gc_other_at = gc_other_at.interp(time=gc_self_at.time)
a2 = math.atan2(lat_coast_other - lat_equat_other, lon_coast_other - lon_equat_other)
a1 = math.atan2(lat_coast_self - lat_equat_self, lon_coast_self - lon_equat_self)

gu, gv = geostrophic_components(gc_self_at, gc_other_at, a1, a2)
gu_cut = gu
gv_cut = gv

if distance_km > 30:
    raise ValueError(f"Nearest selected intersection too far from omni: {distance_km:.2f} km")

fig = plt.figure(figsize=(12, 8), layout="constrained")
ax1 = fig.add_subplot(2, 1, 1)
gu_cut.plot(ax=ax1, label="track zonal", color="r", linewidth=2)
tfreq = sats_tfreq[sat]
omni_v_rs = omni_v.resample(time=f"{tfreq}D").mean().interp(time=gv_cut.time)
omni_u.plot(ax=ax1, color="b", linewidth=2, label="omni u")
ax1.set_xlim(datetime(2019, 2, 1), datetime(2020, 6, 1))
corr = xs.pearson_r(gv_cut, omni_v_rs, dim="time", skipna=True)
corr1 = corr.item()
print(f"Correlation between gv and v_oscar_at: {corr1}")
info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat} {distance_km:.2f} km"
plt.text(0.15, 0.95, info, transform=ax1.transAxes, fontsize=14, fontweight="bold")
plt.legend()

ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    *TRACKS_REG[2:])).plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG, step=5)
ax2.grid()
ax2.plot(
    lon_omni,
    lat_omni,
    c="r",
    marker="o",
    markersize=6,
    markerfacecolor="red",
    markeredgecolor="black",
    markeredgewidth=2,
)
plt.text(lon_omni, lat_omni, f"{omni_id}", fontsize=10, color="k")
ax2.scatter(lons_track_self, lats_track_self, marker=".", color="c", s=4)
ax2.scatter(lons_track_other, lats_track_other, marker=".", color="c", s=4)
lonm, latm = get_point_at_distance(lon_equat_self, lat_equat_self, lon_coast_self,
                                   lat_coast_self, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(lonm, latm, s=track_number_self, fontsize=14, rotation=angle_self, color="w")
lonm, latm = get_point_at_distance(lon_equat_other, lat_equat_other, lon_coast_other,
                                   lat_coast_other, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(lonm, latm, s=track_number_other, fontsize=14, rotation=angle_other, color="w")

plt.show()
plt.savefig(
    f"omniVStrack_guv_pngs/{omni_id}_{sat}_{track_number_self}_{track_number_other}_trim.png"
)
plt.close("all")
ds_omni.close()
dse.close()
# plt.close("all")
#
# dse.close()
omni_id = "BD10"
sat = "S3B"
track_self = "636"
track_other = "393"

df_sel = filter_intersections(df_all, sat, track_self, track_other)
if df_sel.empty:
    raise ValueError(
        f"No intersections for sat={sat}, tracks=({track_self}, {track_other})"
    )

lon_omni, lat_omni = omni_d[omni_id]
ds_omni = xr.open_dataset(f"{omni_data_dir}/{omni_id}_uvs_z_xr_daily.nc",
                          decode_times=False)
print(ds_omni)
ds_omni = change_time(ds_omni, "time")
omni_u = ds_omni.ud.interp(depth=dpth)
omni_u = omni_u.rolling(time=10, center=True).mean()
omni_u = 0.01 * omni_u
omni_v = ds_omni.vd.interp(depth=dpth)
omni_v = omni_v.rolling(time=10, center=True).mean()
omni_v = 0.01 * omni_v
N = omni_v.count().item()
print(N)

df_1 = find_n_closest_points(df_sel, lon_omni, lat_omni, n=1)
row = df_1.iloc[0]
sat = row["sat"]
track_number_self = str(row["track_self_norm"])
track_number_other = str(row["track_other_norm"])
lons_inter1 = row["lons_inter"]
lats_inter1 = row["lats_inter"]
lon1 = lons_inter1
lat1 = lats_inter1
x_from_coast_self1 = row["x_from_coast_self"]
x_from_coast_other1 = row["x_from_coast_other"]
angle_acute = row["angle_acute"]
angle_obtuse = row["angle_obtuse"]
distance_km = row["distance_km"]
if abs(lat1) < 2:
    raise ValueError(f"Selected intersection near equator (lat={lat1:.2f}); skipped.")
print(
    sat,
    omni_id,
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
f_self_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_self}_loess_{frac1}.nc"
f_other_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_other}_loess_{frac1}.nc"
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
lon_coast_self = lons_track_self[-1]
lat_coast_self = lats_track_self[-1]
lon_equat_self = lons_track_self[0]
lat_equat_self = lats_track_self[0]
slope_self = (lats_track_self[-1] - lats_track_self[0]) / (
    lons_track_self[-1] - lons_track_self[0])
angle_self = np.rad2deg(np.arctan(slope_self))
lons_track_other = ds_other.lon.values
lats_track_other = ds_other.lat.values
lon_equat_other = lons_track_other[0]
lat_equat_other = lats_track_other[0]
lon_coast_other = lons_track_other[-1]
lat_coast_other = lats_track_other[-1]
slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                     lon_equat_other)
angle_other = np.rad2deg(np.arctan(slope_other))
gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
gc_self_at = gc_self.sel(x=x_from_coast_self1, method="nearest", drop=True)
gc_other_at = gc_other.sel(x=x_from_coast_other1, method="nearest", drop=True)
gc_other_at = gc_other_at.interp(time=gc_self_at.time)
a2 = math.atan2(lat_coast_other - lat_equat_other, lon_coast_other - lon_equat_other)
a1 = math.atan2(lat_coast_self - lat_equat_self, lon_coast_self - lon_equat_self)

gu, gv = geostrophic_components(gc_self_at, gc_other_at, a1, a2)
gu_cut = gu
gv_cut = gv

if distance_km > 30:
    raise ValueError(f"Nearest selected intersection too far from omni: {distance_km:.2f} km")

fig = plt.figure(figsize=(12, 8), layout="constrained")
ax1 = fig.add_subplot(2, 1, 1)
gu_cut.plot(ax=ax1, label="track zonal", color="r", linewidth=2)
tfreq = sats_tfreq[sat]
omni_v_rs = omni_v.resample(time=f"{tfreq}D").mean().interp(time=gv_cut.time)
omni_u.plot(ax=ax1, color="b", linewidth=2, label="omni u")
ax1.set_xlim(datetime(2019, 1, 1), datetime(2023, 1, 1))
corr = xs.pearson_r(gv_cut, omni_v_rs, dim="time", skipna=True)
corr1 = corr.item()
print(f"Correlation between gv and v_oscar_at: {corr1}")
info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat} {distance_km:.2f} km"
plt.text(0.15, 0.95, info, transform=ax1.transAxes, fontsize=14, fontweight="bold")
plt.legend()

ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    *TRACKS_REG[2:])).plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG, step=5)
ax2.grid()
ax2.plot(
    lon_omni,
    lat_omni,
    c="r",
    marker="o",
    markersize=6,
    markerfacecolor="red",
    markeredgecolor="black",
    markeredgewidth=2,
)
plt.text(lon_omni, lat_omni, f"{omni_id}", fontsize=10, color="k")
ax2.scatter(lons_track_self, lats_track_self, marker=".", color="c", s=4)
ax2.scatter(lons_track_other, lats_track_other, marker=".", color="c", s=4)
lonm, latm = get_point_at_distance(lon_equat_self, lat_equat_self, lon_coast_self,
                                   lat_coast_self, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(lonm, latm, s=track_number_self, fontsize=14, rotation=angle_self, color="w")
lonm, latm = get_point_at_distance(lon_equat_other, lat_equat_other, lon_coast_other,
                                   lat_coast_other, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(lonm, latm, s=track_number_other, fontsize=14, rotation=angle_other, color="w")

plt.show()
plt.savefig(
    f"omniVStrack_guv_pngs/{omni_id}_{sat}_{track_number_self}_{track_number_other}_trim.png"
)
plt.close("all")
ds_omni.close()
dse.close()
# plt.close("all")
#
# dse.close()
omni_id = "AD06"
sat = "S3A"
track_self = "324"
track_other = "195"

df_sel = filter_intersections(df_all, sat, track_self, track_other)
if df_sel.empty:
    raise ValueError(
        f"No intersections for sat={sat}, tracks=({track_self}, {track_other})"
    )

lon_omni, lat_omni = omni_d[omni_id]
ds_omni = xr.open_dataset(f"{omni_data_dir}/{omni_id}_uvs_z_xr_daily.nc",
                          decode_times=False)
print(ds_omni)
ds_omni = change_time(ds_omni, "time")
omni_u = ds_omni.ud.interp(depth=dpth)
omni_u = omni_u.rolling(time=10, center=True).mean()
omni_u = 0.01 * omni_u
omni_v = ds_omni.vd.interp(depth=dpth)
omni_v = omni_v.rolling(time=10, center=True).mean()
omni_v = 0.01 * omni_v
N = omni_v.count().item()
print(N)

df_1 = find_n_closest_points(df_sel, lon_omni, lat_omni, n=1)
row = df_1.iloc[0]
sat = row["sat"]
track_number_self = str(row["track_self_norm"])
track_number_other = str(row["track_other_norm"])
lons_inter1 = row["lons_inter"]
lats_inter1 = row["lats_inter"]
lon1 = lons_inter1
lat1 = lats_inter1
x_from_coast_self1 = row["x_from_coast_self"]
x_from_coast_other1 = row["x_from_coast_other"]
angle_acute = row["angle_acute"]
angle_obtuse = row["angle_obtuse"]
distance_km = row["distance_km"]
if abs(lat1) < 2:
    raise ValueError(f"Selected intersection near equator (lat={lat1:.2f}); skipped.")
print(
    sat,
    omni_id,
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
f_self_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_self}_loess_{frac1}.nc"
f_other_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_other}_loess_{frac1}.nc"
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
lon_coast_self = lons_track_self[-1]
lat_coast_self = lats_track_self[-1]
lon_equat_self = lons_track_self[0]
lat_equat_self = lats_track_self[0]
slope_self = (lats_track_self[-1] - lats_track_self[0]) / (
    lons_track_self[-1] - lons_track_self[0])
angle_self = np.rad2deg(np.arctan(slope_self))
lons_track_other = ds_other.lon.values
lats_track_other = ds_other.lat.values
lon_equat_other = lons_track_other[0]
lat_equat_other = lats_track_other[0]
lon_coast_other = lons_track_other[-1]
lat_coast_other = lats_track_other[-1]
slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                     lon_equat_other)
angle_other = np.rad2deg(np.arctan(slope_other))
gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
gc_self_at = gc_self.sel(x=x_from_coast_self1, method="nearest", drop=True)
gc_other_at = gc_other.sel(x=x_from_coast_other1, method="nearest", drop=True)
gc_other_at = gc_other_at.interp(time=gc_self_at.time)
a2 = math.atan2(lat_coast_other - lat_equat_other, lon_coast_other - lon_equat_other)
a1 = math.atan2(lat_coast_self - lat_equat_self, lon_coast_self - lon_equat_self)

gu, gv = geostrophic_components(gc_self_at, gc_other_at, a1, a2)
gu_cut = gu
gv_cut = gv

if distance_km > 30:
    raise ValueError(f"Nearest selected intersection too far from omni: {distance_km:.2f} km")

fig = plt.figure(figsize=(12, 8), layout="constrained")
ax1 = fig.add_subplot(2, 1, 1)
gu_cut.plot(ax=ax1, label="track zonal", color="r", linewidth=2)
tfreq = sats_tfreq[sat]
omni_v_rs = omni_v.resample(time=f"{tfreq}D").mean().interp(time=gv_cut.time)
omni_u.plot(ax=ax1, color="b", linewidth=2, label="omni u")
ax1.set_xlim(datetime(2016, 3, 1), datetime(2020, 9, 1))
corr = xs.pearson_r(gv_cut, omni_v_rs, dim="time", skipna=True)
corr1 = corr.item()
print(f"Correlation between gv and v_oscar_at: {corr1}")
info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat} {distance_km:.2f} km"
plt.text(0.15, 0.95, info, transform=ax1.transAxes, fontsize=14, fontweight="bold")
plt.legend()

ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    *TRACKS_REG[2:])).plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG, step=5)
ax2.grid()
ax2.plot(
    lon_omni,
    lat_omni,
    c="r",
    marker="o",
    markersize=6,
    markerfacecolor="red",
    markeredgecolor="black",
    markeredgewidth=2,
)
plt.text(lon_omni, lat_omni, f"{omni_id}", fontsize=10, color="k")
ax2.scatter(lons_track_self, lats_track_self, marker=".", color="c", s=4)
ax2.scatter(lons_track_other, lats_track_other, marker=".", color="c", s=4)
lonm, latm = get_point_at_distance(lon_equat_self, lat_equat_self, lon_coast_self,
                                   lat_coast_self, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(lonm, latm, s=track_number_self, fontsize=14, rotation=angle_self, color="w")
lonm, latm = get_point_at_distance(lon_equat_other, lat_equat_other, lon_coast_other,
                                   lat_coast_other, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(lonm, latm, s=track_number_other, fontsize=14, rotation=angle_other, color="w")

plt.show()
plt.savefig(
    f"omniVStrack_guv_pngs/{omni_id}_{sat}_{track_number_self}_{track_number_other}_trim.png"
)
plt.close("all")
ds_omni.close()
dse.close()
# plt.close("all")
#
# dse.close()
omni_id = "BD14"
sat = "S3B"
track_self = "94"
track_other = "51"

df_sel = filter_intersections(df_all, sat, track_self, track_other)
if df_sel.empty:
    raise ValueError(
        f"No intersections for sat={sat}, tracks=({track_self}, {track_other})"
    )

lon_omni, lat_omni = omni_d[omni_id]
ds_omni = xr.open_dataset(f"{omni_data_dir}/{omni_id}_uvs_z_xr_daily.nc",
                          decode_times=False)
print(ds_omni)
ds_omni = change_time(ds_omni, "time")
omni_u = ds_omni.ud.interp(depth=dpth)
omni_u = omni_u.rolling(time=10, center=True).mean()
omni_u = 0.01 * omni_u
omni_v = ds_omni.vd.interp(depth=dpth)
omni_v = omni_v.rolling(time=10, center=True).mean()
omni_v = 0.01 * omni_v
N = omni_v.count().item()
print(N)

df_1 = find_n_closest_points(df_sel, lon_omni, lat_omni, n=1)
row = df_1.iloc[0]
sat = row["sat"]
track_number_self = str(row["track_self_norm"])
track_number_other = str(row["track_other_norm"])
lons_inter1 = row["lons_inter"]
lats_inter1 = row["lats_inter"]
lon1 = lons_inter1
lat1 = lats_inter1
x_from_coast_self1 = row["x_from_coast_self"]
x_from_coast_other1 = row["x_from_coast_other"]
angle_acute = row["angle_acute"]
angle_obtuse = row["angle_obtuse"]
distance_km = row["distance_km"]
if abs(lat1) < 2:
    raise ValueError(f"Selected intersection near equator (lat={lat1:.2f}); skipped.")
print(
    sat,
    omni_id,
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
f_self_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_self}_loess_{frac1}.nc"
f_other_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_other}_loess_{frac1}.nc"
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
lon_coast_self = lons_track_self[-1]
lat_coast_self = lats_track_self[-1]
lon_equat_self = lons_track_self[0]
lat_equat_self = lats_track_self[0]
slope_self = (lats_track_self[-1] - lats_track_self[0]) / (
    lons_track_self[-1] - lons_track_self[0])
angle_self = np.rad2deg(np.arctan(slope_self))
lons_track_other = ds_other.lon.values
lats_track_other = ds_other.lat.values
lon_equat_other = lons_track_other[0]
lat_equat_other = lats_track_other[0]
lon_coast_other = lons_track_other[-1]
lat_coast_other = lats_track_other[-1]
slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                     lon_equat_other)
angle_other = np.rad2deg(np.arctan(slope_other))
gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
gc_self_at = gc_self.sel(x=x_from_coast_self1, method="nearest", drop=True)
gc_other_at = gc_other.sel(x=x_from_coast_other1, method="nearest", drop=True)
gc_other_at = gc_other_at.interp(time=gc_self_at.time)
a2 = math.atan2(lat_coast_other - lat_equat_other, lon_coast_other - lon_equat_other)
a1 = math.atan2(lat_coast_self - lat_equat_self, lon_coast_self - lon_equat_self)

gu, gv = geostrophic_components(gc_self_at, gc_other_at, a1, a2)
gu_cut = gu
gv_cut = gv

if distance_km > 30:
    raise ValueError(f"Nearest selected intersection too far from omni: {distance_km:.2f} km")

fig = plt.figure(figsize=(12, 8), layout="constrained")
ax1 = fig.add_subplot(2, 1, 1)
gu_cut.plot(ax=ax1, label="track zonal", color="r", linewidth=2)
tfreq = sats_tfreq[sat]
omni_v_rs = omni_v.resample(time=f"{tfreq}D").mean().interp(time=gv_cut.time)
omni_u.plot(ax=ax1, color="b", linewidth=2, label="omni u")
ax1.set_xlim(datetime(2016, 3, 1), datetime(2022, 4, 1))
corr = xs.pearson_r(gv_cut, omni_v_rs, dim="time", skipna=True)
corr1 = corr.item()
print(f"Correlation between gv and v_oscar_at: {corr1}")
info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat} {distance_km:.2f} km"
plt.text(0.15, 0.95, info, transform=ax1.transAxes, fontsize=14, fontweight="bold")
plt.legend()

ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    *TRACKS_REG[2:])).plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG, step=5)
ax2.grid()
ax2.plot(
    lon_omni,
    lat_omni,
    c="r",
    marker="o",
    markersize=6,
    markerfacecolor="red",
    markeredgecolor="black",
    markeredgewidth=2,
)
plt.text(lon_omni, lat_omni, f"{omni_id}", fontsize=10, color="k")
ax2.scatter(lons_track_self, lats_track_self, marker=".", color="c", s=4)
ax2.scatter(lons_track_other, lats_track_other, marker=".", color="c", s=4)
lonm, latm = get_point_at_distance(lon_equat_self, lat_equat_self, lon_coast_self,
                                   lat_coast_self, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(lonm, latm, s=track_number_self, fontsize=14, rotation=angle_self, color="w")
lonm, latm = get_point_at_distance(lon_equat_other, lat_equat_other, lon_coast_other,
                                   lat_coast_other, d)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(lonm, latm, s=track_number_other, fontsize=14, rotation=angle_other, color="w")

plt.show()
plt.savefig(
    f"omniVStrack_guv_pngs/{omni_id}_{sat}_{track_number_self}_{track_number_other}_trim.png"
)
plt.close("all")
ds_omni.close()
dse.close()
# plt.close("all")
#
# dse.close()
