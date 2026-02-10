import math

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate great-circle distance between (lon1, lat1)
    and arrays (lon2, lat2) in kilometers."""
    R = 6371  # Radius of Earth in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))


def find_n_closest_points(df, lon0, lat0, n=4):
    """Return the n rows in df closest to (lon0, lat0) based on
    lons_inter/lats_inter."""
    distances = haversine_distance(lon0, lat0, df["lons_inter"].values,
                                   df["lats_inter"].values)
    df = df.copy()
    df["distance_km"] = distances
    return df.nsmallest(n, "distance_km")


cmap1 = cmo.cm.diff
d = 1.5
dpth = 15.0
oscar_dir = "/home/srinivasu/allData/oscar_0.25/"
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
omni_data_dir = "/home/srinivasu/allData/MB_cleaned/"
frac1 = "0.05"

# ds_oscar_u = xr.open_dataset(f"{oscar_dir}/oscar_0.25_u_full_xr_concat.nc")
# ds_oscar_v = xr.open_dataset(f"{oscar_dir}/oscar_0.25_v_full_xr_concat.nc")
# print(ds_oscar_u)
# print(ds_oscar_v)
# u_oscar = ds_oscar_u.u
# v_oscar = ds_oscar_v.v

df_all_list = []
for sat in sats_new:
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    df_all_list.append(df)
df_all = pd.concat(df_all_list, ignore_index=True)

for omni_id in omni_d.keys():
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
    df_4 = find_n_closest_points(df_all, lon_omni, lat_omni, n=4)
    # print(omni_id, lon_omni, lat_omni, "\n", df_4.head(1))
    for nth in range(4):
        sat = df_4["sat"].values[nth]
        track_number_self = df_4["track_self"].apply(lambda x: str(x)).values[nth]
        track_number_other = df_4["track_other"].apply(lambda x: str(x)).values[nth]
        lons_inter1 = df_4["lons_inter"].values[nth]
        lats_inter1 = df_4["lats_inter"].values[nth]
        lon1 = lons_inter1
        lat1 = lats_inter1
        distance_km = df_4["distance_km"].values[nth]
        if abs(lat1) < 2:
            continue
        print(
            sat,
            omni_id,
            lons_inter1,
            lats_inter1,
        )
        f_self = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
        f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
        ds_self = xr.open_dataset(f_self, decode_times=False)
        sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
        ds_other = xr.open_dataset(f_other, engine="netcdf4", decode_times=False)
        sla_other = track_dist_time_asn(ds_other, var_str="sla", units_in="m")
        lons_track_self = ds_self.lon.values
        lats_track_self = ds_self.lat.values
        lon_coast_self = lons_track_self[-1]  # this on coast
        lat_coast_self = lats_track_self[-1]  # this on coast
        lon_equat_self = lons_track_self[0]  # this on equator
        lat_equat_self = lats_track_self[0]  # this on equator
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
        # u_oscar_at = u_oscar.sel(lon=lon1, lat=lat1, method="nearest")
        # v_oscar_at = v_oscar.sel(lon=lon1, lat=lat1, method="nearest")
        f_guv = f"guv_at_intersections/guv_at_intersection_{sat}_{track_number_self}_{track_number_other}_{frac1}.nc"
        # ds_guv = xr.open_dataset(f_guv)
        ds_guv = xr.open_dataset(f_guv, decode_times=False)
        ds_guv = change_time(ds_guv, "time")
        gu = ds_guv.gu
        gv = ds_guv.gv
        # other_times.append((track_number_other, abs(dist_time)))
        fig = plt.figure(figsize=(12, 8), layout="constrained")
        ax1 = fig.add_subplot(2, 1, 1)
        # gu_cut = gu.interp(time=omni_u.time)
        # gu_cut = -1 * gu.interp(time=omni_u.time)
        # gv_cut = gv.interp(time=omni_v.time)
        # gu_cut = -1 * gu
        gu_cut = gu
        gv_cut = gv
        # gu_cut.plot(ax=ax1, label=f"track zonal", color="r", linewidth=2)
        # omni_u.plot(ax=ax1, color="b", linewidth=2, label="omni")
        # u_oscar_at.plot(ax=ax1, color="b", linewidth=2, label="oscar u")
        # corr = xs.pearson_r(gu_cut, omni_u, dim="time", skipna=True)
        gu_cut.plot(ax=ax1, label=f"track zonal", color="r", linewidth=2)
        # gv_cut.plot(ax=ax1, label=f"track meridional", color="r", linewidth=2)
        tfreq = sats_tfreq[sat]
        # omni_u = omni_u.resample(time=f"{tfreq}D").mean()
        # omni_u = omni_u.interp(time=gu_cut.time)
        omni_v = omni_v.resample(time=f"{tfreq}D").mean()
        omni_v = omni_v.interp(time=gv_cut.time)
        omni_u.plot(ax=ax1, color="b", linewidth=2, label="omni zonal")
        # v_oscar_at.plot(ax=ax1, color="b", linewidth=2, label="oscar v")
        # corr = xs.pearson_r(gv_cut, omni_v, dim="time", skipna=True)
        corr = xs.pearson_r(gu_cut, omni_u, dim="time", skipna=True)
        corr1 = corr.item()
        # print(f"Correlation between gu and u_oscar_at: {corr1}")
        print(f"Correlation between gu and u_oscar_at: {corr1}")
        info = f"at ({lon1:.2f}, {lat1:.2f}) {sat} {track_number_self} {track_number_other} {distance_km:.2f} km"
        plt.text(
            0.15,
            0.95,
            info,
            transform=ax1.transAxes,
            fontsize=14,
        )
        plt.legend()
        ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
        dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
            *TRACKS_REG[2:])).plot(ax=ax2,
                                   add_colorbar=False,
                                   add_labels=False,
                                   cmap=cmap1)
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
        # plot plat form code of omni at lon_omni, lat_omni
        plt.text(lon_omni, lat_omni, f"{omni_id}", fontsize=10, color="k")
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
                color="w",
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
                color="w",
            )
        plt.show()
        plt.close("all")

dse.close()
