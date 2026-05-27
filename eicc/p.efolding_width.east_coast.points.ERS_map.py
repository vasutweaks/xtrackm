import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from tools_shapely import *
from tools_xtrackm import *
import pickle


def build_unique_str(sat, row):
    track_self = track_to_str(row["track_self"])
    track_other = track_to_str(row["track_other"])
    return f"{sat}_{track_self}_{track_other}"


def track_to_str(track):
    track_f = float(track)
    if track_f.is_integer():
        return str(int(track_f))
    return str(track)


def track_file(sat, track_number):
    return Path(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number.zfill(3)}.nc")


def setup_map_axis(ax, title):
    proj = ccrs.PlateCarree()
    land = cfeature.NaturalEarthFeature(
        "physical", "land", "10m", facecolor="0.93", edgecolor="none"
    )
    coastline = cfeature.NaturalEarthFeature(
        "physical", "coastline", "50m", facecolor="none", edgecolor="0.2"
    )

    ax.set_extent(MAP_EXTENT, crs=proj)
    ax.add_feature(land, zorder=0)
    ax.add_feature(coastline, linewidth=0.8, zorder=3)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="0.5",
        alpha=0.7,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title)


def plot_track(ax, ds_track, track_number, lon_equat, color):
    proj = ccrs.PlateCarree()
    lons_track = ds_track.lon.values
    lats_track = ds_track.lat.values
    in_region = (
        (lons_track >= MAP_EXTENT[0])
        & (lons_track <= MAP_EXTENT[1])
        & (lats_track >= MAP_EXTENT[2])
        & (lats_track <= MAP_EXTENT[3])
    )

    ax.plot(
        lons_track,
        lats_track,
        color="0.45",
        linewidth=0.8,
        alpha=0.7,
        transform=proj,
        zorder=4,
    )
    ax.plot(
        lons_track[in_region],
        lats_track[in_region],
        color=color,
        linewidth=2.0,
        transform=proj,
        zorder=5,
    )
    ax.scatter(
        [lon_equat],
        [lats_track[np.nanargmin(np.abs(lons_track - float(lon_equat)))]],
        marker="o",
        s=30,
        color=color,
        edgecolor="k",
        linewidth=0.5,
        transform=proj,
        zorder=6,
    )
    if np.any(in_region):
        i_mid = np.flatnonzero(in_region)[len(np.flatnonzero(in_region)) // 2]
        ax.text(
            lons_track[i_mid] + 0.08,
            lats_track[i_mid] + 0.08,
            str(track_number),
            fontsize=11,
            fontweight="bold",
            color=color,
            transform=proj,
            zorder=7,
        )


method = "loess"
frac1 = "0.05"

sat_val = "ERS1+ERS2+ENV+SRL"
INCLUDED_MONTHS = [2, 3, 4, 5]  # February-May. Use None to include all months.
# INCLUDED_MONTHS = [10,11,12]  # February-May. Use None to include all months.
# INCLUDED_MONTHS = None  # February-May. Use None to include all months.
EFOLDING_X_SLICE = slice(0, 300e3)


def select_included_months(gc):
    if INCLUDED_MONTHS is None:
        return gc
    return gc.sel(time=gc.time.dt.month.isin(INCLUDED_MONTHS))


def included_months_label():
    if INCLUDED_MONTHS is None:
        return "all_months"
    return "months_" + "_".join(f"{month:02d}" for month in INCLUDED_MONTHS)


def efolding_width_timeseries(gc):
    gc_x = select_included_months(gc).sel(x=EFOLDING_X_SLICE).sortby("x")
    gc_x = gc_x.transpose("time", "x")

    def width_1d(gc_values, x_values):
        speed = np.abs(np.asarray(gc_values, dtype=float))
        x = np.asarray(x_values, dtype=float)
        finite = np.isfinite(speed) & np.isfinite(x)
        if finite.sum() < 2:
            return np.nan

        speed = speed[finite]
        x = x[finite]
        order = np.argsort(x)
        speed = speed[order]
        x = x[order]

        coastal_speed = speed[0]
        if not np.isfinite(coastal_speed) or coastal_speed <= 0:
            return np.nan

        threshold = coastal_speed / np.e
        crossing_indices = np.flatnonzero(speed <= threshold)
        crossing_indices = crossing_indices[crossing_indices > 0]
        if crossing_indices.size == 0:
            return np.nan

        i = crossing_indices[0]
        x0, x1 = x[i - 1], x[i]
        y0, y1 = speed[i - 1], speed[i]
        if y1 == y0:
            return x1 / 1000.0

        frac = (threshold - y0) / (y1 - y0)
        return (x0 + frac * (x1 - x0)) / 1000.0

    efolding_width = xr.apply_ufunc(
        width_1d,
        gc_x,
        gc_x.x,
        input_core_dims=[["x"], ["x"]],
        vectorize=True,
        output_dtypes=[float],
    )
    efolding_width.name = "efolding_width"
    efolding_width.attrs["long_name"] = "GC speed e-folding width"
    efolding_width.attrs["units"] = "km"
    efolding_width.attrs["description"] = (
        "First distance in 0-300 km where |gc| falls below coastal |gc|/e"
    )
    return efolding_width


# open east_coast_tracks_ERS1+ERS2+ENV+SRL.pkl
with open(f"east_coast_tracks_{sat_val}.pkl", "rb") as f:
    east_coast_tracks = pickle.load(f)

N_PTS = len(east_coast_tracks)
POINT_COLORS = [plt.cm.tab10(i) for i in range(N_PTS)]
POINT_LABELS = list("ABCDEFG")

# Map extent: 10N–22N, 78E–88E
MAP_EXTENT = (78.5, 88.0, 10.0, 22.0)

for ipt, (track_number, lon_equat) in enumerate(east_coast_tracks.items()):
    color = POINT_COLORS[ipt % len(POINT_COLORS)]
    f_gc = Path(
        f"../guv/gc_along_loess/gc_along_{sat_val}_{track_number}_{method}_{frac1}.nc"
    )
    f_track = track_file(sat_val, str(track_number))

    with xr.open_dataset(f_gc) as ds_gc, xr.open_dataset(
        f_track, decode_times=False
    ) as ds_track:
        gc = ds_gc.gc.load()
        efolding_width = efolding_width_timeseries(gc)

        fig = plt.figure(figsize=(9, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.22)
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_width = fig.add_subplot(gs[1, 0])

        setup_map_axis(ax_map, f"{sat_val} track {track_number}")
        plot_track(ax_map, ds_track, track_number, lon_equat, color)
        efolding_width.plot(
            ax=ax_width, color=color, linewidth=1.4, marker="o", markersize=2
        )
        ax_width.set_ylim(0, 300)
        ax_width.grid(True, linestyle="--", alpha=0.5)
        ax_width.set_xlabel("Time")
        ax_width.set_ylabel("E-folding width (km)")
        ax_width.set_title(
            f"GC speed e-folding width, 0-300 km, {included_months_label()}"
        )

        fig.savefig(
            f"efolding_width_gc_track_{sat_val}_{track_number}_{included_months_label()}.png",
            bbox_inches="tight",
            dpi=300,
        )
        if plt.get_backend().lower() != "agg":
            plt.show()
        plt.close(fig)
