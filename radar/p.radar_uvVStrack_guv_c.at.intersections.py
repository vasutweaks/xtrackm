import os
from functools import lru_cache

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import xskillscore as xs
from matplotlib.colors import ListedColormap

# codex resume 019cce35-fb6a-7e52-9604-ec4d6b2361c9

method = "loess"
cmap1 = cmo.cm.diff
cmap2 = ListedColormap(['white'])
d = 1.5
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
oscar_extracted_dir = f"/home/srinivasu/xtrackm/guv/oscar_at_intersections/"
guv_dir = f"/home/srinivasu/xtrackm/guv/guv_at_intersections_{method}"
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

corrs_u = []
biass_u = []
rmses_u = []
unique_strs = []

corrs_v = []
biass_v = []
rmses_v = []
df_out = pd.DataFrame()

ln = len(df)
for idx, row in df_all.iterrows():
    sat1 = row["sat"]
    track_number_self = str(row["track_self"])
    track_number_other = str(row["track_other"])
    unique_str = f"{sat1}_{track_number_self}_{track_number_other}"
    lon1 = row["lons_inter"]
    lat1 = row["lats_inter"]
    lons_inter1 = row["lons_inter"]
    lats_inter1 = row["lats_inter"]
    angle_acute = row["angle_acute"]
    angle_obtuse = row["angle_obtuse"]
    # check if radar region contains intersection
    if not polygon.contains(Point(lon1, lat1)):
        continue
    if abs(lat1) < 5:
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

    f_guv = f"{guv_dir}/guv_at_intersection_{sat1}_{track_number_self}_{track_number_other}_{method}_{frac1}.nc"
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
    valid_u = np.isfinite(gu_cut) & np.isfinite(ru_match)
    gu_cut = gu_cut.where(valid_u)
    ru_at = ru_match.where(valid_u)
    oscar_u = oscar_u.reindex(time=gu_cut.time)
    oscar_max = oscar_u.max().item()
    gu_cut = gu_cut.where(gu_cut < oscar_max, np.nan)
    gu_cut = gu_cut.where(gu_cut > -oscar_max, np.nan)

    rv_match = rv_at.reindex(
        time=gv_cut.time,
        method="nearest",
        tolerance=np.timedelta64(24, "h"),
    )
    valid_v = np.isfinite(gv_cut) & np.isfinite(rv_match)
    gv_cut = gv_cut.where(valid_v)
    rv_at = rv_match.where(valid_v)
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

    # corr_u = xs.pearson_r(gu_cut, oscar_u, dim="time", skipna=True)
    # rmse_u = xs.rmse(gu_cut, oscar_u, dim="time", skipna=True)
    # bias_var_u = oscar_u - gu_cut
    corr_u = xs.pearson_r(gu_cut, ru_at, dim="time", skipna=True)
    rmse_u = xs.rmse(gu_cut, ru_at, dim="time", skipna=True)
    bias_var_u = ru_at - gu_cut
    bias_u = bias_var_u.mean(dim="time")
    # corr_v = xs.pearson_r(gv_cut, oscar_v, dim="time", skipna=True)
    # rmse_v = xs.rmse(gv_cut, oscar_v, dim="time", skipna=True)
    # bias_var_v = oscar_v - gv_cut
    corr_v = xs.pearson_r(gv_cut, rv_at, dim="time", skipna=True)
    rmse_v = xs.rmse(gv_cut, rv_at, dim="time", skipna=True)
    bias_var_v = rv_at - gv_cut
    bias_v = bias_var_v.mean(dim="time")

    corrs_u.append(corr_u.item())
    rmses_u.append(rmse_u.item())
    biass_u.append(bias_u.item())
    unique_strs.append(unique_str)

    corrs_v.append(corr_v.item())
    rmses_v.append(rmse_v.item())
    biass_v.append(bias_v.item())

    print(f"Correlation between gu and u_radar: {corr_u.item()}")
    print(f"Correlation between gv and v_radar: {corr_v.item()}")
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
# add corrs, rmses, biass to dataframe df_out
df_out = pd.DataFrame()
df_out["corrs"] = corrs_u
df_out["rmses"] = rmses_u
df_out["biass"] = biass_u
df_out["unique_str"] = unique_strs
# drop nans
df_out = df_out.dropna(subset=["corrs", "rmses", "biass"])
df_out.to_csv(f"radar_metrics_{radar_str}_u.csv", index=False)

df_out = pd.DataFrame()
df_out["corrs"] = corrs_v
df_out["rmses"] = rmses_v
df_out["biass"] = biass_v
df_out["unique_str"] = unique_strs

df_out = df_out.dropna(subset=["corrs", "rmses", "biass"])
df_out.to_csv(f"radar_metrics_{radar_str}_v.csv", index=False)
