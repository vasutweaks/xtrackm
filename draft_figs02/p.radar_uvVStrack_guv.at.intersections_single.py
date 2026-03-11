import math
import sys

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import xskillscore as xs
from matplotlib.colors import ListedColormap
from pathlib import Path

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")

radar_str = "TN"
track_self_here = 608
track_other_here = 137
sat_here = "S3A"
width, height = 13, 6
TRACKS_REG = (73, 99, 0, 26)
d = 1.5

cmap1 = cmo.cm.diff
cmap2 = ListedColormap(['white'])
oscar_dir = "/home/srinivasu/allData/oscar_0.25/"
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
oscar_extracted_dir = f"/home/srinivasu/xtrackm/guv/oscar_at_intersections/"

REG = radar_extents[radar_str + "_BLOCK"]
gdf = gpd.read_file(f"polygon_shapefile_around_{radar_str}.shp")
polygon = gdf.geometry.iloc[0]
radar_tsta_o, radar_tend_o = radar_times[radar_str]

OUT_PNG = f"guv_timeseries_{sat_here}_{track_self_here}_{track_other_here}_{radar_str}.png"

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

df = pd.read_csv(f"../guv/tracks_intersections_{sat_here}_1.csv")

df_filter = df[(df["track_self"] == track_self_here) & (df["track_other"] == track_other_here)]

ln = len(df)
for idx, row in df_filter.iterrows():
    sat1 = row["sat"]
    track_number_self = str(row["track_self"])
    track_number_other = str(row["track_other"])
    lon1 = row["lons_inter"]
    lat1 = row["lats_inter"]
    lons_inter1 = row["lons_inter"]
    lats_inter1 = row["lats_inter"]
    x_from_coast_self1 = row["x_from_coast_self"]
    x_from_coast_other1 = row["x_from_coast_other"]
    angle_acute = row["angle_acute"]
    angle_obtuse = row["angle_obtuse"]
    track_tsta_o, track_tend_o = get_time_limits_o(sat1)
    overlap_tsta, overlap_tend = overlap_dates(track_tsta_o, track_tend_o,
                                               radar_tsta_o, radar_tend_o)
    ru_at = u_radar.sel(longitude=lon1, latitude=lat1, method="nearest", drop=True)
    rv_at = v_radar.sel(longitude=lon1, latitude=lat1, method="nearest", drop=True)
    f_oscar = f"{oscar_extracted_dir}/oscar_at_intersection_{sat1}_{track_number_self}_{track_number_other}.nc"
    ds_oscar = xr.open_dataset(f_oscar)
    oscar_u = ds_oscar.u
    oscar_v = ds_oscar.v
    f_self = f"../data/{sat1}/ctoh.sla.ref.{sat1}.nindian.{track_number_self.zfill(3)}.nc"
    f_other = f"../data/{sat1}/ctoh.sla.ref.{sat1}.nindian.{track_number_other.zfill(3)}.nc"
    f_self_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat1}_{track_number_self}_loess_0.2.nc"
    f_other_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat1}_{track_number_other}_loess_0.2.nc"
    ds_self = xr.open_dataset(f_self, decode_times=False)
    ds_self_smooth = xr.open_dataset(f_self_smooth)
    sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
    sla_self_smooth = ds_self_smooth.sla_smooth
    ds_other = xr.open_dataset(f_other, decode_times=False)
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
    slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                         lon_equat_other)
    angle_other = np.rad2deg(np.arctan(slope_other))
    track_path_other = sg.LineString(zip(lons_track_other, lats_track_other))
    gc_self = compute_geostrophy_gc(ds_self, sla_self_smooth)
    gc_other = compute_geostrophy_gc(ds_other, sla_other_smooth)
    gc_self_at = gc_self.sel(x=x_from_coast_self1, method="nearest", drop=True)
    gc_other_at = gc_other.sel(x=x_from_coast_other1,
                               method="nearest",
                               drop=True)
    gc_other_at = gc_other_at.interp(time=gc_self_at.time)
    a2 = math.atan2(lat_coast_other - lat_equat_other,
                    lon_coast_other - lon_equat_other)
    a1 = math.atan2(lat_coast_self - lat_equat_self,
                    lon_coast_self - lon_equat_self)
    gu, gv = geostrophic_components(gc_self_at, gc_other_at, a1, a2)
    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(3, 1, 2)
    ax1 = fig.add_subplot(3, 1, 3)

    gu_cut = gu
    gv_cut = gv
    tfreq = sats_tfreq[sat1]

    oscar_u = oscar_u.resample(time=f"{tfreq}D").mean()
    ru_at = ru_at.resample(time=f"{tfreq}D").mean()
    oscar_u = oscar_u.interp(time=gu_cut.time)
    ru_at = ru_at.interp(time=gu_cut.time)
    oscar_max = oscar_u.max().item()
    gu_cut = gu_cut.where(gu_cut < oscar_max, np.nan)
    gu_cut = gu_cut.where(gu_cut > -oscar_max, np.nan)

    oscar_v = oscar_v.resample(time=f"{tfreq}D").mean()
    rv_at = rv_at.resample(time=f"{tfreq}D").mean()
    oscar_v = oscar_v.interp(time=gv_cut.time)
    rv_at = rv_at.interp(time=gu_cut.time)
    oscar_max = oscar_v.max().item()
    gv_cut = gv_cut.where(gv_cut < oscar_max, np.nan)
    gv_cut = gv_cut.where(gv_cut > -oscar_max, np.nan)

    oscar_u.plot(ax=ax0, color="b", linewidth=2, label="oscar u")
    gu_cut.plot(ax=ax0, label=f"track zonal", color="k", linewidth=2)
    ru_at.plot(ax=ax0, label=f"radar zonal", color="r", linewidth=2)
    ax0.legend()
    ax0.set_ylabel("oscar u m/s")
    oscar_v.plot(ax=ax1, color="b", linewidth=2, label="oscar v")
    gv_cut.plot(ax=ax1, label=f"track meridional", color="k", linewidth=2)
    rv_at.plot(ax=ax1, label=f"radar meridional", color="r", linewidth=2)
    corr_u = xs.pearson_r(gu_cut, ru_at, dim="time", skipna=True)
    rmse_u = xs.rmse(gu_cut, ru_at, dim="time", skipna=True)
    bias_var_u = ru_at - gu_cut
    bias_u = bias_var_u.mean(dim="time")
    corr_v = xs.pearson_r(gv_cut, rv_at, dim="time", skipna=True)
    rmse_v = xs.rmse(gv_cut, rv_at, dim="time", skipna=True)
    bias_var_v = rv_at - gv_cut
    bias_v = bias_var_v.mean(dim="time")
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
        facecolor="blue",
        edgecolor="red",
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
    # plot.text radar_str
    plt.text(
        0.15,
        0.90,
        f"radar region: {radar_str}",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
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
        f"guv_timeseries_{sat1}_{track_number_self}_{track_number_other}_{radar_str}.png",
    )
    ln_sf(
        f"guv_timeseries_{sat1}_{track_number_self}_{track_number_other}_{radar_str}.png", "fig06.png"
    )
    ln_sf(OUT_PNG, png_file)
    plt.show()
    plt.close("all")

dse.close()
