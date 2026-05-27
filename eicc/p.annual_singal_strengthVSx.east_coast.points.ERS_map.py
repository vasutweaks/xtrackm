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


def plot_annual_amplitude_on_track(ax, ds_track, full_x, annual_amp, cmap="viridis"):
    proj = ccrs.PlateCarree()
    lons_track = ds_track.lon.values
    lats_track = ds_track.lat.values
    full_x = np.asarray(full_x.values)
    selected_x = np.asarray(annual_amp.x.values)
    track_indices = np.flatnonzero(np.isin(full_x, selected_x))
    if track_indices.size == 0:
        return None

    track_indices = track_indices[: annual_amp.size]
    lons_track = lons_track[track_indices]
    lats_track = lats_track[track_indices]
    amp = annual_amp.values
    in_region = (
        (lons_track >= MAP_EXTENT[0])
        & (lons_track <= MAP_EXTENT[1])
        & (lats_track >= MAP_EXTENT[2])
        & (lats_track <= MAP_EXTENT[3])
        & np.isfinite(amp)
    )
    if not np.any(in_region):
        return None

    return ax.scatter(
        lons_track[in_region],
        lats_track[in_region],
        c=amp[in_region],
        cmap=cmap,
        s=22,
        edgecolor="none",
        transform=proj,
        zorder=8,
    )


def annual_fft_amplitude(gc):
    gc = gc.sel(x=X_SLICE).sortby("time").transpose("x", "time")
    times = pd.to_datetime(gc.time.values)
    dt_days = np.median(np.diff(times).astype("timedelta64[s]").astype(float)) / 86400.0

    def annual_amp_1d(values):
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 4:
            return np.nan

        sample_index = np.arange(values.size)
        if finite.sum() != values.size:
            values = np.interp(sample_index, sample_index[finite], values[finite])

        values = values - values.mean()
        coeffs = np.fft.rfft(values)
        freqs = np.fft.rfftfreq(values.size, d=dt_days / ANNUAL_PERIOD_DAYS)
        annual_idx = np.argmin(np.abs(freqs - 1.0))
        return 2.0 * np.abs(coeffs[annual_idx]) / values.size

    annual_amp = xr.apply_ufunc(
        annual_amp_1d,
        gc,
        input_core_dims=[["time"]],
        vectorize=True,
        output_dtypes=[float],
    )
    annual_amp.name = "annual_gc_amplitude"
    annual_amp.attrs["long_name"] = "Annual GC amplitude from FFT"
    annual_amp.attrs["units"] = gc.attrs.get("units", "m/s")
    return annual_amp


method = "loess"
frac1 = "0.05"

sat_val = "ERS1+ERS2+ENV+SRL"
INCLUDED_MONTHS = None  # February-May. Use None to include all months.
X_SLICE = slice(0, 1000e3)
ANNUAL_PERIOD_DAYS = 365.2425


def select_included_months(gc):
    if INCLUDED_MONTHS is None:
        return gc
    return gc.sel(time=gc.time.dt.month.isin(INCLUDED_MONTHS))


def included_months_label():
    if INCLUDED_MONTHS is None:
        return "all_months"
    return "months_" + "_".join(f"{month:02d}" for month in INCLUDED_MONTHS)


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

        fig = plt.figure(figsize=(9, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.22)
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_gc = fig.add_subplot(gs[1, 0])

        setup_map_axis(ax_map, f"{sat_val} track {track_number}")
        plot_track(ax_map, ds_track, track_number, lon_equat, color)
        annual_amp = annual_fft_amplitude(gc)
        annual_scatter = plot_annual_amplitude_on_track(
            ax_map, ds_track, gc.x, annual_amp
        )
        if annual_scatter is not None:
            cbar = fig.colorbar(
                annual_scatter,
                ax=ax_map,
                orientation="horizontal",
                pad=0.05,
                shrink=0.75,
            )
            cbar.set_label("Annual GC amplitude (m/s)")

        annual_amp.plot(ax=ax_gc, color=color, linewidth=1.8)
        ax_gc.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax_gc.grid(True, linestyle="--", alpha=0.5)
        ax_gc.set_xlabel("Distance along track (m)")
        ax_gc.set_ylabel("Annual GC amplitude (m/s)")
        ax_gc.set_title("Annual signal amplitude from FFT along x")

        fig.savefig(
            f"annual_gc_fft_amplitude_track_{sat_val}_{track_number}.png",
            bbox_inches="tight",
            dpi=300,
        )
        if plt.get_backend().lower() != "agg":
            plt.show()
        plt.close(fig)
