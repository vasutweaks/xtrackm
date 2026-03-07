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


def find_n_closest_points(df, lon0, lat0, n=4):
    """Return n rows closest to (lon0, lat0) using intersection lon/lat."""
    lons_inter = df["lons_inter"].tolist()
    lats_inter = df["lats_inter"].tolist()
    distances = [
        distance.distance((y, x), (lat0, lon0)).km
        for x, y in zip(lons_inter, lats_inter)
    ]
    out = df.copy()
    out["distance_km"] = distances
    return out.nsmallest(n, "distance_km")


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
tol_hours = 24
max_distance_km = 250
min_overlap_fraction = 0.30
min_matched_pairs = 50
w_d = 0.35
w_t = 0.45
w_n = 0.20


dse = xr.open_dataset("~/allData/topo/etopo5.cdf")
rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)
rama_data_dir = "/home/srinivasu/allData/rama/cur/"
guv_dir = "/home/srinivasu/xtrackm/guv/guv_at_intersections"


def collocate_rama_to_gu(gu, rama_u, tol_hours=24):
    rama_u_match = rama_u.reindex(
        time=gu.time,
        method="nearest",
        tolerance=np.timedelta64(tol_hours, "h"),
    )
    valid = np.isfinite(gu) & np.isfinite(rama_u_match)
    gu_pair = gu.where(valid)
    rama_u_pair = rama_u_match.where(valid)
    matched_pairs = int(valid.sum().item())
    possible_pairs = int(np.isfinite(gu).sum().item())
    overlap_fraction = (
        matched_pairs / possible_pairs if possible_pairs > 0 else 0.0
    )
    return gu_pair, rama_u_pair, matched_pairs, overlap_fraction


def norm_01(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if np.isclose(amax, amin):
        return np.zeros_like(arr, dtype=float)
    return (arr - amin) / (amax - amin)


df_all_list = []
for sat in sats_new:
    df_all_list.append(pd.read_csv(f"tracks_intersections_{sat}_1.csv"))
df_all = pd.concat(df_all_list, ignore_index=True)


for rama_id, (lon_rama, lat_rama) in rama_d.items():
    if abs(lat_rama) < 2:
        continue

    rama_file = f"{rama_data_dir}/cur{rama_id}_dy.cdf"
    if not os.path.exists(rama_file):
        continue
    print(rama_id, lon_rama, lat_rama)

    with xr.open_dataset(f"{rama_data_dir}/cur{rama_id}_dy.cdf") as ds_rama:
        # print(ds_rama)
        rama_u = 0.01 * ds_rama.U_320.isel(depth=0, drop=True)
        rama_v = 0.01 * ds_rama.V_321.isel(depth=0, drop=True)

    df_4 = find_n_closest_points(df_all, lon_rama, lat_rama, n=4)
    candidates = []
    for nth in range(min(4, len(df_4))):
        row = df_4.iloc[nth]
        sat = row["sat"]
        track_number_self = str(row["track_self"])
        track_number_other = str(row["track_other"])
        lon1 = row["lons_inter"]
        lat1 = row["lats_inter"]
        distance_km = row["distance_km"]
        if abs(lat1) < 2:
            continue

        f_guv = f"{guv_dir}/guv_at_intersection_{sat}_{track_number_self}_{track_number_other}_{frac1}.nc"
        if not os.path.exists(f_guv):
            continue

        with xr.open_dataset(f_guv) as ds_guv:
            if "gu" not in ds_guv:
                continue
            gu = ds_guv.gu.load()
            gv = ds_guv.gv.load() if "gv" in ds_guv else None

        gu_pair, rama_u_pair, matched_pairs, overlap_fraction = (
            collocate_rama_to_gu(gu, rama_u, tol_hours=tol_hours)
        )
        if matched_pairs < 2:
            continue

        candidates.append(
            {
                "row": row,
                "sat": sat,
                "track_number_self": track_number_self,
                "track_number_other": track_number_other,
                "lon1": lon1,
                "lat1": lat1,
                "distance_km": float(distance_km),
                "matched_pairs": matched_pairs,
                "overlap_fraction": float(overlap_fraction),
                "gu_pair": gu_pair,
                "rama_u_pair": rama_u_pair,
                "gv": gv,
            }
        )

    if len(candidates) == 0:
        print(f"{rama_id}: no valid candidate")
        continue

    score_df = pd.DataFrame(
        {
            "distance_km": [c["distance_km"] for c in candidates],
            "overlap_fraction": [c["overlap_fraction"] for c in candidates],
            "matched_pairs": [c["matched_pairs"] for c in candidates],
        }
    )
    hard_mask = (
        (score_df["distance_km"] <= max_distance_km)
        & (score_df["overlap_fraction"] >= min_overlap_fraction)
        & (score_df["matched_pairs"] >= min_matched_pairs)
    )
    if hard_mask.any():
        idx_pool = np.where(hard_mask.values)[0]
    else:
        idx_pool = np.arange(len(candidates))

    d_norm = norm_01(score_df.loc[idx_pool, "distance_km"].values)
    o_norm = norm_01(score_df.loc[idx_pool, "overlap_fraction"].values)
    n_norm = norm_01(score_df.loc[idx_pool, "matched_pairs"].values)
    scores = w_d * d_norm + w_t * (1.0 - o_norm) + w_n * (1.0 - n_norm)
    best_idx = idx_pool[int(np.argmin(scores))]
    best = candidates[best_idx]

    sat = best["sat"]
    track_number_self = best["track_number_self"]
    track_number_other = best["track_number_other"]
    lon1 = best["lon1"]
    lat1 = best["lat1"]
    distance_km = best["distance_km"]
    gu_pair = best["gu_pair"]
    rama_u_pair = best["rama_u_pair"]
    score_best = float(np.min(scores))

    print(
        "selected:",
        rama_id,
        sat,
        track_number_self,
        track_number_other,
        f"d={distance_km:.1f}km",
        f"overlap={best['overlap_fraction']:.2f}",
        f"n={best['matched_pairs']}",
        f"score={score_best:.3f}",
    )

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
    plt.text(lon_rama, lat_rama, f"{rama_id}", fontsize=10, color="k")
    ax2.scatter(lons_track_self, lats_track_self, marker=".", color="c", s=4)
    ax2.scatter(lons_track_other, lats_track_other, marker=".", color="c", s=4)

    lonm, latm = get_point_at_distance(
        lon_equat_self,
        lat_equat_self,
        lon_coast_self,
        lat_coast_self,
        d,
    )
    info = (
        f"{sat} {track_number_self} {track_number_other} {distance_km:.2f} km"
    )
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
