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


cmap1 = cmo.cm.diff
d = 1.5
height, width = 9, 9
frac1 = "0.05"


dse = xr.open_dataset("~/allData/topo/etopo5.cdf")
rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)
cb_data_dir = "/home/srinivasu/slnew/xtrackd/coastal/"
guv_dir = "/home/srinivasu/xtrackm/guv/guv_at_intersections"
out_match_csv = "cb_guv_c_nearest_global_matches.csv"
matched_rows_all = []


df_all_list = []
for sat in sats_new:
    df_all_list.append(pd.read_csv(f"tracks_intersections_{sat}_1.csv"))
df_all = pd.concat(df_all_list, ignore_index=True)

for cb_id in cb_d.keys():
    lon_cb, lat_cb = cb_d[cb_id]
    if abs(lat_cb) < 2:
        continue

    cb_file = f"{cb_data_dir}/{cb_id.upper()}_uvs.nc"
    if not os.path.exists(cb_file):
        continue
    print(cb_id, lon_cb, lat_cb)

    with xr.open_dataset(cb_file) as ds_cb:
        # print(ds_cb)
        # Coastal buoy currents are stored in cm/s; convert to m/s to match gu/gv.
        cb_u = 0.01 * ds_cb.U
        cb_v = 0.01 * ds_cb.V
        cb_s = 0.01 * ds_cb.speed
        cb_dir = ds_cb.direction

    # print(cb_v.count().item())
    df_4 = find_n_closest_points(df_all, lon_cb, lat_cb, n=4)
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

        if abs(lat1) < 2:
            continue

        print(
            sat,
            cb_id,
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

        # In precomputed guv files, exact zeros often represent filtered/missing
        # points from upstream processing; exclude them before collocation.
        gu_cut = gu.where(np.abs(gu) > 1e-12)
        gv_cut = gv.where(np.abs(gv) > 1e-12)

        # Collocate cb at track-gu timestamps; avoid sat-dependent smoothing.
        cb_u_match = cb_u.reindex(
            time=gu_cut.time,
            method="nearest",
            tolerance=np.timedelta64(24, "h"),
        )
        valid = np.isfinite(gu_cut) & np.isfinite(cb_u_match)
        gu_pair = gu_cut.where(valid)
        cb_u_pair = cb_u_match.where(valid)

        df_pairs = xr.Dataset({"gu": gu_pair, "cb_u": cb_u_pair}).to_dataframe().reset_index()
        df_pairs = df_pairs.dropna(subset=["gu", "cb_u"])
        print(df_pairs.columns)
        matched_pairs = len(df_pairs)
        if matched_pairs > 0:
            """All these will broadcasted along length of gu, cb_u"""
            df_pairs["sat"] = sat
            df_pairs["cb_id"] = cb_id
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
                        "cb_id",
                        "distance_km",
                        "track_self",
                        "track_other",
                        "matched_pairs",
                        "timestamp",
                        "gu",
                        "cb_u",
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
        cb_u_pair.plot(ax=ax1, color="r", linewidth=2, label="cb")
        gu_pair.plot(ax=ax1, color="k", linewidth=2, label="track")

        corr = xs.pearson_r(gu_pair, cb_u_pair, dim="time", skipna=True)
        corr1 = corr.item()

        info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat} {distance_km:.2f} km"
        print(f"Correlation between gu and cb u: {corr1}")
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
            lon_cb,
            lat_cb,
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
        plt.text(lon_cb, lat_cb, f"{cb_id}", fontsize=10, color="k")
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

if len(matched_rows_all) > 0:
    df_matched_all = pd.concat(matched_rows_all, ignore_index=True)
    df_matched_all.to_csv(out_match_csv, index=False)
    print(f"saved global match csv: {out_match_csv}, rows={len(df_matched_all)}")
#
dse.close()
