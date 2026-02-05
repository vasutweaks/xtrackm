import math
import os
import warnings

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import xskillscore as xs


cmap1 = cmo.cm.diff
d = 1.5
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
oscar_extracted_dir = f"/home/srinivasu/xtrackm/guv/oscar_at_intersections/"

sat = "TP+J1+J2+J3+S6A"
df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
df_out = df.copy()

corrs_u = []
biass_u = []
rmses_u = []

corrs_v = []
biass_v = []
rmses_v = []

frac1 = "0.05"
ln = len(df)
for i in range(ln):
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
        corrs_v.append(np.nan)
        rmses_v.append(np.nan)
        biass_v.append(np.nan)
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
    f_oscar = f"{oscar_extracted_dir}/oscar_at_intersection_{sat1}_{track_number_self}_{track_number_other}.nc"
    ds_oscar = xr.open_dataset(f_oscar)
    oscar_u = ds_oscar.u
    oscar_v = ds_oscar.v
    f_self = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
    f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
    f_self_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_self}_loess_{frac1}.nc"
    f_other_smooth = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{track_number_other}_loess_{frac1}.nc"
    ds_self = xr.open_dataset(f_self, engine="netcdf4", decode_times=False)
    ds_self_smooth = xr.open_dataset(f_self_smooth)
    sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
    sla_self_smooth = ds_self_smooth.sla_smooth
    ds_other = xr.open_dataset(f_other, engine="netcdf4", decode_times=False)
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

    gv = -1 * gv

    gu_filt = gu.where((gu <= 3.5) & (gu >= -3.5))
    gu_filtered = mad_filter_1d_xr(gu_filt, 3.5)
    gv_filt = gv.where((gv <= 3.5) & (gv >= -3.5))
    gv_filtered = mad_filter_1d_xr(gv_filt, 3.5)

    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(3, 1, 3)
    ax1 = fig.add_subplot(3, 1, 2)

    gu_cut = gu_filtered
    gv_cut = gv_filtered
    tfreq = sats_tfreq[sat]

    oscar_u = oscar_u.resample(time=f"{tfreq}D").mean()
    oscar_u = oscar_u.interp(time=gu_cut.time)
    oscar_max = oscar_u.max().item()
    gu_cut = gu_cut.where(gu_cut < oscar_max, np.nan)
    gu_cut = gu_cut.where(gu_cut > -oscar_max, np.nan)

    oscar_v = oscar_v.resample(time=f"{tfreq}D").mean()
    oscar_v = oscar_v.interp(time=gv_cut.time)
    oscar_max = oscar_v.max().item()
    gv_cut = gv_cut.where(gv_cut < oscar_max, np.nan)
    gv_cut = gv_cut.where(gv_cut > -oscar_max, np.nan)

    oscar_u.plot(ax=ax0, color="b", linewidth=2, label="oscar u")
    gu_cut.plot(ax=ax0, label=f"track zonal", color="r", linewidth=2)
    ax0.set_xlim(datetime(2010, 1, 1), datetime(2020, 1, 1))
    ax0.legend()
    ax0.set_ylabel("oscar u m/s")

    oscar_v.plot(ax=ax1, color="b", linewidth=2, label="oscar v")
    gv_cut.plot(ax=ax1, label=f"track meridional", color="r", linewidth=2)

    corr_u = xs.pearson_r(gu_cut, oscar_u, dim="time", skipna=True)
    rmse_u = xs.rmse(gu_cut, oscar_u, dim="time", skipna=True)
    bias_var_u = oscar_u - gu_cut
    bias_u = bias_var_u.mean(dim="time")

    corr_v = xs.pearson_r(gv_cut, oscar_v, dim="time", skipna=True)
    rmse_v = xs.rmse(gv_cut, oscar_v, dim="time", skipna=True)
    bias_var_v = oscar_v - gv_cut
    bias_v = bias_var_v.mean(dim="time")

    corrs_u.append(corr_u.item())
    rmses_u.append(rmse_u.item())
    biass_u.append(bias_u.item())

    corrs_v.append(corr_v.item())
    rmses_v.append(rmse_v.item())
    biass_v.append(bias_v.item())

    print(f"Correlation between gu and u_oscar_at: {corr_u.item()}")
    print(f"Correlation between gv and v_oscar_at: {corr_v.item()}")
    info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat1}"
    ax1.set_xlim(datetime(2010, 1, 1), datetime(2020, 1, 1))
    plt.text(
        0.15,
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
        f"track_guvVSoscar_at_intersection_points/guv_timeseries_{sat1}_{track_number_self}_{track_number_other}.png",
    )
    plt.show()
    plt.close("all")

# add corrs, rmses, biass to dataframe df_out
df_out["corrs"] = corrs_u
df_out["rmses"] = rmses_u
df_out["biass"] = biass_u

df_out.to_csv(f"oscar_onlyVStrack_corrs_rmses_biass_{sat1}_u.csv", index=False)

df_out["corrs"] = corrs_v
df_out["rmses"] = rmses_v
df_out["biass"] = biass_v

df_out.to_csv(f"oscar_onlyVStrack_corrs_rmses_biass_{sat1}_v.csv", index=False)
dse.close()

