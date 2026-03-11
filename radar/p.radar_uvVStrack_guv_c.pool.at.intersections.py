import os
from functools import lru_cache

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
from matplotlib.colors import ListedColormap

# codex resume 019cce35-fb6a-7e52-9604-ec4d6b2361c9

cmap1 = cmo.cm.diff
cmap2 = ListedColormap(['white'])
d = 1.5
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
oscar_extracted_dir = f"/home/srinivasu/xtrackm/guv/oscar_at_intersections/"
guv_dir = "/home/srinivasu/xtrackm/guv/guv_at_intersections"
EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections"
frac1 = "0.05"

radar_str = "OR"
width, height = 13, 6
REG = radar_extents[radar_str + "_BLOCK"]

gdf = gpd.read_file(f"polygon_shapefile_around_{radar_str}.shp")
polygon = gdf.geometry.iloc[0]

@lru_cache(maxsize=None)
def get_track_geometry(sat, track_number):
    f_track = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{str(track_number).zfill(3)}.nc"
    with xr.open_dataset(f_track, engine="netcdf4", decode_times=False) as ds_track:
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


dsr = xr.open_dataset(
    f"/home/srinivasu/allData/radar/{radar_str}Codar_daily.nc")
# print(dsr)
dsr = dsr.rename({
    "XAXS": "longitude",
    "YAXS": "latitude",
    "ZAXS": "lev",
    "TAXIS1D": "time"
})
# print(dsr)
u_radar = 0.01 * dsr.U_RADAR.isel(lev=0, drop=True)
v_radar = 0.01 * dsr.V_RADAR.isel(lev=0, drop=True)
s_radar = 0.01 * dsr.S_RADAR.isel(lev=0, drop=True)
sizes = u_radar.sizes
ln = sizes["time"]

df_all_list = []
for sat in sats_new:
    df = pd.read_csv(f"../guv/tracks_intersections_{sat}_1.csv")
    df_all_list.append(df)
df_all = pd.concat(df_all_list, ignore_index=True)

ru_gu_records = []

ln = len(df)
for idx, row in df_all.iterrows():
    sat1 = row["sat"]
    track_number_self = str(row["track_self"])
    track_number_other = str(row["track_other"])
    lon1 = row["lons_inter"]
    lat1 = row["lats_inter"]
    lons_inter1 = row["lons_inter"]
    lats_inter1 = row["lats_inter"]
    angle_acute = row["angle_acute"]
    angle_obtuse = row["angle_obtuse"]
    f_ek = f"{EKMAN_DIR}/{unique_str}_ekman_at_intersections.nc"
    ds_ek = xr.open_dataset(f_ek)
    u_ek = ds_ek['u_at']
    v_ek = ds_ek['v_at']
    # check if radar region contains intersection
    if not polygon.contains(Point(lon1, lat1)):
        continue
    if abs(lat1) < 2:
        continue
    # print(
    #     sat1,
    #     track_number_self,
    #     track_number_other,
    #     lons_inter1,
    #     lats_inter1,
    #     x_from_coast_self1,
    #     x_from_coast_other1,
    #     angle_acute,
    #     angle_obtuse,
    # )
    ru_at = u_radar.sel(longitude=lon1, latitude=lat1, method="nearest", drop=True)
    rv_at = v_radar.sel(longitude=lon1, latitude=lat1, method="nearest", drop=True)
    f_oscar = f"{oscar_extracted_dir}/oscar_at_intersection_{sat1}_{track_number_self}_{track_number_other}.nc"
    if not os.path.exists(f_oscar):
        continue
    with xr.open_dataset(f_oscar) as ds_oscar:
        oscar_u = ds_oscar.u
        oscar_v = ds_oscar.v

    f_guv = f"{guv_dir}/guv_at_intersection_{sat1}_{track_number_self}_{track_number_other}_{frac1}.nc"
    if not os.path.exists(f_guv):
        print(f"missing file: {f_guv}")
        continue
    with xr.open_dataset(f_guv) as ds_guv:
        gu = ds_guv.gu.load()
        gv = ds_guv.gv.load()

    (
        lons_track_self,
        lats_track_self,
        lon_equat_self,
        lat_equat_self,
        lon_coast_self,
        lat_coast_self,
        angle_self,
    ) = get_track_geometry(sat1, track_number_self)
    (
        lons_track_other,
        lats_track_other,
        lon_equat_other,
        lat_equat_other,
        lon_coast_other,
        lat_coast_other,
        angle_other,
    ) = get_track_geometry(sat1, track_number_other)
    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(3, 1, 2)
    ax1 = fig.add_subplot(3, 1, 3)

    gu_cut = gu
    gv_cut = gv

    # Collocate radar at track-gu timestamps (same approach as RAMA workflow).
    ru_match = ru_at.reindex(
        time=gu_cut.time,
        method="nearest",
        tolerance=np.timedelta64(24, "h"),
    )
    ek_u_match = u_ek.reindex(
        time=gu_cut.time,
        method="nearest",
        tolerance=np.timedelta64(24, "h"),
    )
    valid_u = np.isfinite(gu_cut) & np.isfinite(ru_match)
    gu_cut = gu_cut.where(valid_u)
    ru_at = ru_match.where(valid_u)
    ek_u = ek_u_match.where(valid)
    gu_ek = gu_cut + ek_u
    oscar_u = oscar_u.reindex(time=gu_cut.time)
    oscar_max = oscar_u.max().item()
    gu_cut = gu_cut.where(gu_cut < oscar_max, np.nan)
    gu_cut = gu_cut.where(gu_cut > -oscar_max, np.nan)

    rv_match = rv_at.reindex(
        time=gv_cut.time,
        method="nearest",
        tolerance=np.timedelta64(24, "h"),
    )
    ek_v_match = v_ek.reindex(
        time=gv_cut.time,
        method="nearest",
        tolerance=np.timedelta64(24, "h"),
    )
    valid_v = np.isfinite(gv_cut) & np.isfinite(rv_match)
    gv_cut = gv_cut.where(valid_v)
    rv_at = rv_match.where(valid_v)
    ek_v = ek_v_match.where(valid)
    gv_ek = gv_cut + ek_v
    oscar_v = oscar_v.reindex(time=gv_cut.time)
    oscar_max = oscar_v.max().item()
    gv_cut = gv_cut.where(gv_cut < oscar_max, np.nan)
    gv_cut = gv_cut.where(gv_cut > -oscar_max, np.nan)

    oscar_u.plot(ax=ax0, color="b", linewidth=2, label="oscar u")
    gu_cut.plot(ax=ax0, label=f"track zonal", color="k", linewidth=2)
    ru_at.plot(ax=ax0, label=f"radar zonal", color="r", linewidth=2)
    # ax0.set_xlim(datetime(2010, 1, 1), datetime(2020, 1, 1))
    ax0.legend()
    ax0.set_ylabel("oscar u m/s")

    oscar_v.plot(ax=ax1, color="b", linewidth=2, label="oscar v")
    gv_cut.plot(ax=ax1, label=f"track meridional", color="k", linewidth=2)
    rv_at.plot(ax=ax1, label=f"radar meridional", color="r", linewidth=2)

    valid_ru_gu = np.isfinite(gu_cut.values) & np.isfinite(ru_at.values)
    if np.any(valid_ru_gu):
        ru_gu_records.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(gu_cut.time.values[valid_ru_gu]),
                    "sat": sat1,
                    "track_self": track_number_self,
                    "track_other": track_number_other,
                    "lon_intersection": lon1,
                    "lat_intersection": lat1,
                    "ru": ru_at.values[valid_ru_gu],
                    "gu": gu_cut.values[valid_ru_gu],
                }
            )
        )
    info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat1} {track_number_self} {track_number_other}"
    # ax1.set_xlim(datetime(2010, 1, 1), datetime(2020, 1, 1))
    plt.text(
        0.05,
        0.90,
        info,
        transform=ax0.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.set_ylabel("oscar v m/s")
    ax2 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
    dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
        *TRACKS_REG[2:])).plot(ax=ax2,
                               add_colorbar=False,
                               add_labels=False,
                               cmap=cmap2)
    decorate_axis(ax2, "", *TRACKS_REG, step=5)
    ax2.grid()
    ax2.add_geometries(
        [polygon],
        ccrs.PlateCarree(),
        facecolor="lightblue",
        edgecolor="black",
        alpha=0.2,
    )
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
    plt.savefig(
        f"track_guvVSoscar_at_intersection_points/guv_timeseries_{sat1}_{track_number_self}_{track_number_other}.png",
    )
    plt.show()
    plt.close("all")

dse.close()
if ru_gu_records:
    df_out = pd.concat(ru_gu_records, ignore_index=True)
else:
    df_out = pd.DataFrame(
        columns=[
            "time",
            "sat",
            "track_self",
            "track_other",
            "lon_intersection",
            "lat_intersection",
            "ru",
            "gu",
        ]
    )
df_out.to_csv(f"radar_valid_ru_gu_{radar_str}.csv", index=False)
