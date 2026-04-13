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


def filter_rama_uv_72h(rama_u, rama_v):
    rama_u_filt = butter_lowpass_1d(rama_u, cutoff_days=3, order=4)
    rama_v_filt = butter_lowpass_1d(rama_v, cutoff_days=3, order=4)
    return rama_u_filt, rama_v_filt


cmap1 = cmo.cm.diff
d = 1.5
height, width = 9, 9
frac1 = "0.05"
method = "loess"


dse = xr.open_dataset("~/allData/topo/etopo5.cdf")
rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)
rama_data_dir = "/home/srinivasu/allData/rama/cur/"
guv_dir = f"/home/srinivasu/xtrackm/guv/guv_at_intersections_{method}"
out_match_csv = "rama_guv_c_nearest_global_matches.ek.csv"
EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd/"
EKMAN_DEPTH = "0"
matched_rows_all = []

df_all_list = []
for sat in sats_new:
    df_all_list.append(pd.read_csv(f"tracks_intersections_{sat}_1.csv"))
df_all = pd.concat(df_all_list, ignore_index=True)

for rama_id, (lon_rama, lat_rama) in rama_d.items():
    if lat_rama < 5:
        continue
    rama_file = f"{rama_data_dir}/cur{rama_id}_dy.cdf"
    if not os.path.exists(rama_file):
        continue
    print(rama_id, lon_rama, lat_rama)
    with xr.open_dataset(f"{rama_data_dir}/cur{rama_id}_dy.cdf") as ds_rama:
        # print(ds_rama)
        rama_u = 0.01 * ds_rama.U_320.isel(depth=0, lon=0, lat=0, drop=True)
        rama_v = 0.01 * ds_rama.V_321.isel(depth=0, lon=0, lat=0, drop=True)
    rama_u, rama_v = filter_rama_uv_72h(rama_u, rama_v)

    # print(rama_v.count().item())
    df_4 = find_n_closest_points(df_all, lon_rama, lat_rama, n=4)
    for nth in range(4):
        row = df_4.iloc[nth]

        sat = row["sat"]
        track_number_self = str(row["track_self"])
        track_number_other = str(row["track_other"])
        lon1 = row["lons_inter"]
        lat1 = row["lats_inter"]
        distance_km = row["distance_km"]
        angle_acute = row["angle_acute"]
        angle_obtuse = row["angle_obtuse"]

        unique_str = build_unique_str(sat, row)
        f_ek = f"{EKMAN_DIR}/{unique_str}_ekman_{EKMAN_DEPTH}m_at_intersections_cd.nc"
        ds_ek = xr.open_dataset(f_ek)
        u_ek = ds_ek['u_at']
        v_ek = ds_ek['v_at']
        u_ek, v_ek = filter_rama_uv_72h(u_ek, v_ek)

        if lat1 < 5:
            continue

        print(
            sat,
            rama_id,
            track_number_self,
            track_number_other,
            lon1,
            lat1,
            angle_acute,
            angle_obtuse,
        )
        print((
        f"\nSELECT_SAT = '{sat}'"
        f"\nSELECT_RAMA_ID = '{rama_id}'"
        f"\nSELECT_TRACK_SELF = '{track_number_self}'"
        f"\nSELECT_TRACK_OTHER = '{track_number_other}'"
        ))

        f_guv = f"{guv_dir}/guv_at_intersection_{sat}_{track_number_self}_{track_number_other}_{method}_{frac1}.nc"
        if not os.path.exists(f_guv):
            print(f"missing file: {f_guv}")
            continue

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
        # rama_u_match = rama_u.reindex(
        #     time=gu_cut.time,
        #     method="nearest",
        #     tolerance=np.timedelta64(24, "h"),
        # )
        # ek_u_match = u_ek.reindex(
        #     time=gu_cut.time,
        #     method="nearest",
        #     tolerance=np.timedelta64(24, "h"),
        # )
        rama_u_match = rama_u.interp(time=gu_cut.time)
        ek_u_match = u_ek.interp(time=gu_cut.time)
        valid = np.isfinite(gu_cut) & np.isfinite(rama_u_match)
        gu_pair = gu_cut.where(valid)
        rama_u_pair = rama_u_match.where(valid)
        ek_u_pair = ek_u_match.where(valid)
        gu_ek_pair = gu_pair + ek_u_pair

        df_pairs = xr.Dataset({"gu": gu_pair, "rama_u": rama_u_pair, "u_ek": ek_u_pair}).to_dataframe().reset_index()
        df_pairs = df_pairs.dropna(subset=["gu", "rama_u", "u_ek"])
        print(df_pairs.columns)
        matched_pairs = len(df_pairs)
        if matched_pairs > 0:
            """All these will broadcasted along length of gu, rama_u"""
            df_pairs["sat"] = sat
            df_pairs["rama_id"] = rama_id
            df_pairs["distance_km"] = distance_km
            df_pairs["track_self"] = track_number_self
            df_pairs["track_other"] = track_number_other
            df_pairs["matched_pairs"] = matched_pairs
            df_pairs["timestamp"] = pd.to_datetime(df_pairs["time"]).astype(str)
            print(df_pairs.columns)
            matched_rows_all.append(
                df_pairs[
                    [
                        "sat",
                        "rama_id",
                        "distance_km",
                        "track_self",
                        "track_other",
                        "matched_pairs",
                        "timestamp",
                        "gu",
                        "rama_u",
                        "u_ek",
                    ]
                ]
            )

        # check valid points
        ngp = np.sum(valid)
        if ngp < 2:
            continue
        fig = plt.figure(figsize=(12, 8))
        ax2 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        ax1 = fig.add_subplot(2, 1, 2)
        rama_u_pair.plot(ax=ax1, color="r", linewidth=2, label="rama")
        gu_pair.plot(ax=ax1, color="k", linewidth=2, label="track")
        gu_ek_pair.plot(ax=ax1, color="b", linewidth=2, label="track+ekman")

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
        plt.text(lon_rama, lat_rama, f"{rama_id}", fontsize=10, color="k")
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

        # plt.show()
        plt.close("all")

if len(matched_rows_all) > 0:
    df_matched_all = pd.concat(matched_rows_all, ignore_index=True)
    df_matched_all.to_csv(out_match_csv, index=False)
    print(f"saved global match csv: {out_match_csv}, rows={len(df_matched_all)}")
#
dse.close()
