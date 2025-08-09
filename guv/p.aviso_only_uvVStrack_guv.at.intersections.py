import math

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import xskillscore as xs
import sys


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
    u = -1 * u
    return u, v


cmap1 = cmo.cm.diff
d = 1.5
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
aviso_msla_uv_dir = "/home/srinivasu/sealevel/data/msla_uv/"
ds_aviso_u = xr.open_dataset(f"{aviso_msla_uv_dir}/allyear_dt_global_allsat_msla_u_nio.nc")
ds_aviso_v = xr.open_dataset(f"{aviso_msla_uv_dir}/allyear_dt_global_allsat_msla_v_nio.nc")
print(ds_aviso_u)
aviso_u = ds_aviso_u.ugosa
aviso_v = ds_aviso_v.vgosa
# sys.exit()

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
count = 0
for i in range(ln):
    ds_aviso_u = xr.open_dataset(f"{aviso_msla_uv_dir}/allyear_dt_global_allsat_msla_u_nio.nc")
    ds_aviso_v = xr.open_dataset(f"{aviso_msla_uv_dir}/allyear_dt_global_allsat_msla_v_nio.nc")
    # print(ds_aviso_u)
    aviso_u = ds_aviso_u.ugosa
    aviso_v = ds_aviso_v.vgosa
    sat1 = df["sat"].values[i]
    track_number_self = df["track_self"].apply(lambda x: str(x)).values[i]
    track_number_other = df["track_other"].apply(lambda x: str(x)).values[i]
    lons_inter1 = df["lons_inter"].values[i]
    lats_inter1 = df["lats_inter"].values[i]
    lon1 = lons_inter1
    lat1 = lats_inter1
    x_from_coast_self1 = df["x_from_coast_self"].values[i]
    x_from_coast_other1 = df["x_from_coast_other"].values[i]
    angle_acute = df["angle_acute"].values[i]
    angle_obtuse = df["angle_obtuse"].values[i]
    if abs(lat1) < 2:
        corrs_u.append(np.nan)
        rmses_u.append(np.nan)
        biass_u.append(np.nan)
        ngp_u.append(np.nan)
        corrs_v.append(np.nan)
        rmses_v.append(np.nan)
        biass_v.append(np.nan)
        ngp_v.append(np.nan)
        continue
    print(
        sat,
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
    print(f"azimuths {a1:.2f} {a2:.2f}")
    gu, gv = geostrophic_components_from_a(gc_self_at, gc_other_at, a1, a2)
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(3, 1, 3)
    ax0 = fig.add_subplot(3, 1, 2)

    gu_cut = -1 * gu.resample(time="1ME").mean()
    gu_cut_ngp = 100 * gu_cut.count(dim="time")/len(gu_cut.time)
    gv_cut = gv.resample(time="1ME").mean()
    gv_cut_ngp = 100 * gv_cut.count(dim="time")/len(gv_cut.time)
    tfreq = sats_tfreq[sat]

    aviso_u = aviso_u.sel(longitude=lon1, latitude=lat1, method="nearest")
    aviso_u = aviso_u.interp(time=gu_cut.time)
    aviso_max_u = aviso_u.max().item()
    gu_cut = gu_cut.where(gu_cut < aviso_max_u, np.nan)
    gu_cut = gu_cut.where(gu_cut > -aviso_max_u, np.nan)

    aviso_v = aviso_v.sel(longitude=lon1, latitude=lat1, method="nearest")
    aviso_v = aviso_v.interp(time=gv_cut.time)
    aviso_max_v = aviso_v.max().item()
    gv_cut = gv_cut.where(gv_cut < aviso_max_v, np.nan)
    gv_cut = gv_cut.where(gv_cut > -aviso_max_v, np.nan)

    aviso_u.plot(ax=ax0, color="b", linewidth=2, label="aviso u")
    gu_cut.plot(ax=ax0, label=f"track zonal", color="r", linewidth=2)
    # ax0.set_xlim(datetime(2010, 1, 1), datetime(2020, 1, 1))
    ax0.legend()
    ax0.set_ylabel("aviso u m/s")

    aviso_v.plot(ax=ax1, color="b", linewidth=2, label="aviso v")
    gv_cut.plot(ax=ax1, label=f"track meridional", color="r", linewidth=2)

    # percentage of good points of gu_cut
    corr_u = xs.pearson_r(gu_cut, aviso_u, dim="time", skipna=True)
    rmse_u = xs.rmse(gu_cut, aviso_u, dim="time", skipna=True)
    bias_var_u = aviso_u - gu_cut
    bias_u = bias_var_u.mean(dim="time")

    corr_v = xs.pearson_r(gv_cut, aviso_v, dim="time", skipna=True)
    rmse_v = xs.rmse(gv_cut, aviso_v, dim="time", skipna=True)
    bias_var_v = aviso_v - gv_cut
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
    dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
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
    # plt.show()
    plt.close("all")
    ds_aviso_u.close()

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
