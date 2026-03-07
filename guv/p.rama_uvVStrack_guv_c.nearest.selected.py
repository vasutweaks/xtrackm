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


cmap1 = cmo.cm.diff
d = 1.5
height, width = 9, 9
frac1 = "0.05"


dse = xr.open_dataset("~/allData/topo/etopo5.cdf")
rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)
rama_data_dir = "/home/srinivasu/allData/rama/cur/"
guv_dir = "/home/srinivasu/xtrackm/guv/guv_at_intersections"
out_match_csv = "rama_guv_c_nearest_global_matches.csv"
matched_rows_all = []
SELECT_SAT = "ERS1+ERS2+ENV+SRL"
SELECT_RAMA_ID = "15n90e"
SELECT_TRACK_SELF = "810"
SELECT_TRACK_OTHER = "423"


df_all_list = []
for sat in sats_new:
    df_all_list.append(pd.read_csv(f"tracks_intersections_{sat}_1.csv"))
df_all = pd.concat(df_all_list, ignore_index=True)


lon_rama, lat_rama = rama_d[SELECT_RAMA_ID]
rama_file = f"{rama_data_dir}/cur{SELECT_RAMA_ID}_dy.cdf"
if not os.path.exists(rama_file):
    raise("No rama file")
print(SELECT_RAMA_ID, lon_rama, lat_rama)

with xr.open_dataset(f"{rama_data_dir}/cur{SELECT_RAMA_ID}_dy.cdf") as ds_rama:
    # print(ds_rama)
    rama_u = 0.01 * ds_rama.U_320.isel(depth=0, drop=True)
    rama_v = 0.01 * ds_rama.V_321.isel(depth=0, drop=True)

df_selected = df_all[
    (df_all["sat"] == SELECT_SAT)
    & (df_all["track_self"].astype(str) == SELECT_TRACK_SELF)
    & (df_all["track_other"].astype(str) == SELECT_TRACK_OTHER)
].copy()
if df_selected.empty:
    raise("selected sat/track combination not found in intersections file")
lons = df_selected["lons_inter"].to_numpy(dtype=float)
lats = df_selected["lats_inter"].to_numpy(dtype=float)
distances = [distance.distance((lat, lon), (lat_rama, lon_rama)).km for lon, lat in zip(lons, lats)]
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

print(
    sat,
    SELECT_RAMA_ID,
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

# Collocate RAMA at track-gu timestamps; avoid sat-dependent smoothing.
rama_u_match = rama_u.reindex(
    time=gu_cut.time,
    method="nearest",
    tolerance=np.timedelta64(24, "h"),
)
valid = np.isfinite(gu_cut) & np.isfinite(rama_u_match)
gu_pair = gu_cut.where(valid)
rama_u_pair = rama_u_match.where(valid)

df_pairs = xr.Dataset({"gu": gu_pair, "rama_u": rama_u_pair}).to_dataframe().reset_index()
df_pairs = df_pairs.dropna(subset=["gu", "rama_u"])

# check valid points
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax1 = fig.add_subplot(2, 1, 2)
rama_u_pair.plot(ax=ax1, color="r", linewidth=2, label="rama")
gu_pair.plot(ax=ax1, color="k", linewidth=2, label="track")

corr = xs.pearson_r(gu_pair, rama_u_pair, dim="time", skipna=True)
corr1 = corr.item()

info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat} {distance_km:.2f} km"
print(f"Correlation between gu and rama u: {corr1}")
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
    lon_rama,
    lat_rama,
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
plt.text(lon_rama, lat_rama, f"{SELECT_RAMA_ID}", fontsize=10, color="k")
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
