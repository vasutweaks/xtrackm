import cmocean as cmo
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import xskillscore as xs
# from rich.traceback import install
# install(show_locals=True, suppress=["site-packages"])

cmap1 = cmo.cm.diff
d = 1.5
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
aviso_msla_uv_dir = "/home/srinivasu/sealevel/data/msla_uv/"
aviso_msla_uv_dir = "/home/srinivasu/allData/aviso/msla_uv/"
ds_aviso_u = xr.open_dataset(f"{aviso_msla_uv_dir}/allyear_dt_global_allsat_msla_u.nc")
ds_aviso_v = xr.open_dataset(f"{aviso_msla_uv_dir}/allyear_dt_global_allsat_msla_v.nc")
print(ds_aviso_u)
aviso_u = ds_aviso_u.u
aviso_v = ds_aviso_v.v

rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(*TRACKS_REG[2:]))


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


def get_track_end_to_end_angle(sat, track_number):
    return get_track_geometry(sat, track_number)[-1]


def append_nan_metrics(corrs_u, rmses_u, biass_u, ngp_u, corrs_v, rmses_v, biass_v, ngp_v):
    corrs_u.append(np.nan)
    rmses_u.append(np.nan)
    biass_u.append(np.nan)
    ngp_u.append(np.nan)
    corrs_v.append(np.nan)
    rmses_v.append(np.nan)
    biass_v.append(np.nan)
    ngp_v.append(np.nan)


sat = "S3B"
sat = "TP+J1+J2+J3+S6A"
df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
df_out = df.copy()

corrs_u = []
biass_u = []
rmses_u = []
ngp_u = []

corrs_v = []
biass_v = []
rmses_v = []
ngp_v = []

ln = len(df)
frac1 = "0.05"
guv_dir = "/home/srinivasu/xtrackm/guv/guv_at_intersections"
for i in range(ln):
    row = df.iloc[i]
    sat1 = row["sat"]
    track_number_self = str(row["track_self"])
    track_number_other = str(row["track_other"])
    lons_inter1 = row["lons_inter"]
    lats_inter1 = row["lats_inter"]
    lon1 = lons_inter1
    lat1 = lats_inter1
    angle_acute = row["angle_acute"]
    angle_obtuse = row["angle_obtuse"]
    if abs(lat1) < 2:
        append_nan_metrics(corrs_u, rmses_u, biass_u, ngp_u, corrs_v, rmses_v, biass_v, ngp_v)
        continue
    print(
        sat,
        track_number_self,
        track_number_other,
        lons_inter1,
        lats_inter1,
        angle_acute,
        angle_obtuse,
    )
    f_guv = f"{guv_dir}/guv_at_intersection_{sat1}_{track_number_self}_{track_number_other}_{frac1}.nc"
    ds_guv = xr.open_dataset(f_guv)
    lons_track_self, lats_track_self, lon_equat_self, lat_equat_self, lon_coast_self, lat_coast_self, _ = get_track_geometry(
        sat, track_number_self
    )
    lons_track_other, lats_track_other, lon_equat_other, lat_equat_other, lon_coast_other, lat_coast_other, _ = get_track_geometry(
        sat, track_number_other
    )
    angle_self = get_track_end_to_end_angle(sat, track_number_self)
    angle_other = get_track_end_to_end_angle(sat, track_number_other)

    gu = ds_guv.gu
    gv = ds_guv.gv

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(3, 1, 3)
    ax0 = fig.add_subplot(3, 1, 2)

    gu_cut = gu.resample(time="1ME").mean()
    gu_cut_ngp = 100 * gu_cut.count(dim="time")/len(gu_cut.time)
    gv_cut = gv.resample(time="1ME").mean()
    gv_cut_ngp = 100 * gv_cut.count(dim="time")/len(gv_cut.time)

    aviso_u1 = aviso_u.sel(longitude=lon1, latitude=lat1, method="nearest")
    aviso_u2 = aviso_u1.interp(time=gu_cut.time)
    aviso_max_u = aviso_u2.max().item()
    gu_cut = gu_cut.where(gu_cut < aviso_max_u, np.nan)
    gu_cut = gu_cut.where(gu_cut > -aviso_max_u, np.nan)

    aviso_v1 = aviso_v.sel(longitude=lon1, latitude=lat1, method="nearest")
    aviso_v2 = aviso_v1.interp(time=gv_cut.time)
    aviso_max_v = aviso_v2.max().item()
    gv_cut = gv_cut.where(gv_cut < aviso_max_v, np.nan)
    gv_cut = gv_cut.where(gv_cut > -aviso_max_v, np.nan)

    aviso_u2.plot(ax=ax0, color="b", linewidth=2, label="aviso u")
    gu_cut.plot(ax=ax0, label=f"track zonal", color="r", linewidth=2)
    # ax0.set_xlim(datetime(2010, 1, 1), datetime(2020, 1, 1))
    ax0.legend()
    ax0.set_ylabel("aviso u m/s")

    aviso_v2.plot(ax=ax1, color="b", linewidth=2, label="aviso v")
    gv_cut.plot(ax=ax1, label=f"track meridional", color="r", linewidth=2)

    # percentage of good points of gu_cut
    corr_u = xs.pearson_r(gu_cut, aviso_u2, dim="time", skipna=True)
    rmse_u = xs.rmse(gu_cut, aviso_u2, dim="time", skipna=True)
    bias_var_u = aviso_u2 - gu_cut
    bias_u = bias_var_u.mean(dim="time")

    corr_v = xs.pearson_r(gv_cut, aviso_v2, dim="time", skipna=True)
    rmse_v = xs.rmse(gv_cut, aviso_v2, dim="time", skipna=True)
    bias_var_v = aviso_v2 - gv_cut
    bias_v = bias_var_v.mean(dim="time")

    corrs_u.append(corr_u.item())
    rmses_u.append(rmse_u.item())
    biass_u.append(bias_u.item())
    ngp_u.append(gu_cut_ngp.item())

    corrs_v.append(corr_v.item())
    rmses_v.append(rmse_v.item())
    biass_v.append(bias_v.item())
    ngp_v.append(gv_cut_ngp.item())

    print(f"Correlation between gu and u_aviso_at: {corr_u.item():.2f} {gu_cut_ngp.item():.2f}")
    print(f"Correlation between gv and v_aviso_at: {corr_v.item():.2f} {gv_cut_ngp.item():.2f}")
    info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat1}"
    # ax1.set_xlim(datetime(2010, 1, 1), datetime(2020, 1, 1))
    plt.text(
        0.15,
        0.90,
        info,
        transform=ax0.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.set_ylabel("aviso v m/s")
    ax2 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
    rose_cut.plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
    decorate_axis(ax2, "", *TRACKS_REG, step=5)
    ax2.grid()
    ax2.plot(
        lon1,
        lat1,
        c="r",
        marker="o",
        markersize=6,
        markerfacecolor="red",
        markeredgecolor="black",
        markeredgewidth=2,
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
        f"track_guvVSaviso_at_intersection_points/guv_timeseries_{sat1}_{track_number_self}_{track_number_other}.png",
    )
    plt.show()
    plt.close("all")
    ds_guv.close()
    # ds_aviso_u.close()

# add corrs, rmses, biass to dataframe df_out
df_out["corrs"] = corrs_u
df_out["rmses"] = rmses_u
df_out["biass"] = biass_u
df_out["ngp"] = ngp_u

df_out.to_csv(f"tracksVSaviso_corrs_rmses_biass_{sat1}_u.csv", index=False)

df_out["corrs"] = corrs_v
df_out["rmses"] = rmses_v
df_out["biass"] = biass_v
df_out["ngp"] = ngp_v

df_out.to_csv(f"tracksVSaviso_corrs_rmses_biass_{sat1}_v.csv", index=False)
dse.close()
