import os
from functools import lru_cache

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from geopy import distance
from tools_xtrackm import *
# codex resume 019cc947-b006-7d82-8941-d68e330c388a

@lru_cache(maxsize=None)
def get_track_geometry(sat, track_number):
    f_track = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{str(track_number).zfill(3)}.nc"
    with xr.open_dataset(
        f_track, engine="netcdf4", decode_times=False
    ) as ds_track:
        lons = ds_track.lon.values
        lats = ds_track.lat.values
    lon_equat = lons[0]
    lat_equat = lats[0]
    lon_coast = lons[-1]
    lat_coast = lats[-1]
    slope = (lat_coast - lat_equat) / (lon_coast - lon_equat)
    angle_deg = np.rad2deg(np.arctan(slope))
    return lons, lats, lon_equat, lat_equat, lon_coast, lat_coast, angle_deg


def build_unique_str(sat, row):
    track_self = track_to_str(row["track_self"])
    track_other = track_to_str(row["track_other"])
    return f"{sat}_{track_self}_{track_other}"


def track_to_str(track):
    track_f = float(track)
    if track_f.is_integer():
        return str(int(track_f))
    return str(track)


cmap1 = cmo.cm.diff
d = 1.5
dpth = 15
height, width = 9, 9
frac1 = "0.05"


dse = xr.open_dataset("~/allData/topo/etopo5.cdf")
rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)
omni_data_dir = "/home/srinivasu/allData/MB_cleaned/raju_omni_clean"
guv_dir = "/home/srinivasu/xtrackm/guv/guv_at_intersections"
out_match_csv = "omni_guv_c_nearest_global_matches.csv"
matched_rows_all = []
# SELECT_SAT = "TP+J1+J2+J3+S6A"
# SELECT_OMNI_ID = "BD09"
# SELECT_TRACK_SELF = "53"
# SELECT_TRACK_OTHER = "14"
SELECT_SAT = 'S3A'
SELECT_OMNI_ID = 'BD10'
SELECT_TRACK_SELF = '522'
SELECT_TRACK_OTHER = '279'
SELECT_SAT = 'S3A'
SELECT_OMNI_ID = 'BD10'
SELECT_TRACK_SELF = '636'
SELECT_TRACK_OTHER = '393'

EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections"
# TP+J1+J2+J3+S6A BD09 53 14
# S3A BD10 522 279 

df_all_list = []
for sat in sats_new:
    df_all_list.append(pd.read_csv(f"tracks_intersections_{sat}_1.csv"))
df_all = pd.concat(df_all_list, ignore_index=True)


lon_omni, lat_omni = omni_d[SELECT_OMNI_ID]
omni_file = f"{omni_data_dir}/{SELECT_OMNI_ID}_current_sst.nc"
print(omni_file)
if not os.path.exists(omni_file):
    print(f"File not found: {omni_file}")
    raise(f"no omni file")
if abs(lat_omni) < 2:
    raise(f"lat less than 2")

ds_omni = xr.open_dataset(omni_file)
print(ds_omni)
# print(ds_omni)
ds_omni = xr.open_dataset(omni_file)
print(ds_omni)
omni_u = ds_omni.u_current.interp(current_depth=dpth)
# omni_u = omni_u.rolling(time=10, center=True).mean()
omni_u = omni_u
omni_v = ds_omni.v_current.interp(current_depth=dpth)
# omni_v = omni_v.rolling(time=10, center=True).mean()
omni_v = omni_v
N = omni_v.count().item()

df_selected = df_all[
    (df_all["sat"] == SELECT_SAT)
    & (df_all["track_self"].astype(str) == SELECT_TRACK_SELF)
    & (df_all["track_other"].astype(str) == SELECT_TRACK_OTHER)
].copy()
if df_selected.empty:
    raise("selected sat/track combination not found in intersections file")
lons = df_selected["lons_inter"].to_numpy(dtype=float)
lats = df_selected["lats_inter"].to_numpy(dtype=float)
distances = [distance.distance((lat, lon), (lat_omni, lon_omni)).km for lat, lon in zip(lats, lons)]
df_selected["distance_km"] = distances
row = df_selected.nsmallest(1, "distance_km").iloc[0]

sat = row["sat"]
track_number_self = str(row["track_self"])
track_number_other = str(row["track_other"])
lon1 = row["lons_inter"]
lat1 = row["lats_inter"]
distance_km = row["distance_km"]
angle_acute = row["angle_acute"]
angle_obtuse = row["angle_obtuse"]
unique_str = build_unique_str(sat, row)
f_ek = f"{EKMAN_DIR}/{unique_str}_ekman_at_intersections.nc"
ds_ek = xr.open_dataset(f_ek)
u_ek = ds_ek['u_at']
v_ek = ds_ek['v_at']

print(
    sat,
    SELECT_OMNI_ID,
    track_number_self,
    track_number_other,
    lon1,
    lat1,
    angle_acute,
    angle_obtuse,
)

f_guv = f"{guv_dir}/guv_at_intersection_{sat}_{track_number_self}_{track_number_other}_{frac1}.nc"
if not os.path.exists(f_guv):
    print(f"missing file: {f_guv}")
    raise("missing file")

with xr.open_dataset(f_guv) as ds_guv:
    gu = ds_guv.gu
    gv = ds_guv.gv

(
    lons_track_self,
    lats_track_self,
    lon_equat_self,
    lat_equat_self,
    lon_coast_self,
    lat_coast_self,
    angle_self,
) = get_track_geometry(sat, track_number_self)
(
    lons_track_other,
    lats_track_other,
    lon_equat_other,
    lat_equat_other,
    lon_coast_other,
    lat_coast_other,
    angle_other,
) = get_track_geometry(sat, track_number_other)

gu_cut = gu
gv_cut = gv

# Collocate omni at track-gu timestamps; avoid sat-dependent smoothing.
omni_u_match = omni_u.reindex(
    time=gu_cut.time,
    method="nearest",
    tolerance=np.timedelta64(24, "h"),
)
ek_u_match = u_ek.reindex(
    time=gu_cut.time,
    method="nearest",
    tolerance=np.timedelta64(24, "h"),
)
valid = np.isfinite(gu_cut) & np.isfinite(omni_u_match)
gu_pair = gu_cut.where(valid)
omni_u_pair = omni_u_match.where(valid)
ek_u_pair = ek_u_match.where(valid)

gu_ek_pair = gu_pair + ek_u_pair


df_pairs = xr.Dataset({"gu": gu_pair, "omni_u": omni_u_pair}).to_dataframe().reset_index()
df_pairs = df_pairs.dropna(subset=["gu", "omni_u"])

# check valid points
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax1 = fig.add_subplot(2, 1, 2)
omni_u_pair.plot(ax=ax1, color="r", linewidth=2, label="omni")
gu_pair.plot(ax=ax1, color="k", linewidth=2, label="track")
gu_ek_pair.plot(ax=ax1, color="b", linewidth=2, label="track+ekman")

corr = xs.pearson_r(gu_pair, omni_u_pair, dim="time", skipna=True)
corr1 = corr.item()

info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat} {distance_km:.2f} km"
print(f"Correlation between gu and omni u: {corr1}")
plt.text(
    0.25,
    0.90,
    info,
    transform=ax1.transAxes,
    fontsize=12,
    fontweight="bold",
)
plt.legend()

rose_cut.plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG, step=10)
ax2.grid()
ax2.plot(
    lon_omni,
    lat_omni,
    c="r",
    marker="o",
    markersize=4,
    markerfacecolor="red",
    markeredgecolor="red",
    markeredgewidth=2,
)

ax2.plot(
    lon1,
    lat1,
    c="k",
    marker=".",
    markersize=8,
    markerfacecolor="black",
    markeredgecolor="black",
    markeredgewidth=2,
)
plt.text(lon_omni, lat_omni, f"{SELECT_OMNI_ID}", fontsize=10, color="k")
ax2.scatter(
    lons_track_self, lats_track_self, marker=".", color="c", s=4
)
ax2.scatter(
    lons_track_other, lats_track_other, marker=".", color="c", s=4
)

lonm, latm = get_point_at_distance(
    lon_equat_self,
    lat_equat_self,
    lon_coast_self,
    lat_coast_self,
    d,
)
info = f"{sat} {track_number_self} {track_number_other} {distance_km:.2f} km"
plt.text(
    0.55,
    0.90,
    info,
    transform=ax2.transAxes,
    fontsize=10,
    fontweight="bold",
)
if is_within_region(lonm, latm, *TRACKS_REG):
    plt.text(
        lonm,
        latm,
        s=track_number_self,
        fontsize=14,
        rotation=angle_self,
        color="w",
    )

lonm, latm = get_point_at_distance(
    lon_equat_other,
    lat_equat_other,
    lon_coast_other,
    lat_coast_other,
    d,
)
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

dse.close()
