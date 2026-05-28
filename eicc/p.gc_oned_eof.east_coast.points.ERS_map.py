import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
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
INCLUDED_MONTHS = [10,11,12]  # February-May. Use None to include all months.
INCLUDED_MONTHS = None  # February-May. Use None to include all months.
X_SLICE = slice(0, 2000e3)
X_GRID_STEP = 6500.0
N_MODES = 3


def select_included_months(gc):
    if INCLUDED_MONTHS is None:
        return gc
    return gc.sel(time=gc.time.dt.month.isin(INCLUDED_MONTHS))


def included_months_label():
    if INCLUDED_MONTHS is None:
        return "all_months"
    return "months_" + "_".join(f"{month:02d}" for month in INCLUDED_MONTHS)


def uniform_x_grid(gc, step=X_GRID_STEP):
    x = gc.x.sel(x=X_SLICE)
    x_min = float(x.min())
    x_max = float(x.max())
    n_steps = int(np.floor((x_max - x_min) / step))
    return xr.DataArray(
        x_min + step * np.arange(n_steps + 1),
        dims="x",
        name="x",
        attrs={"long_name": "Distance along track", "units": "m"},
    )


def regrid_gc(gc, step=X_GRID_STEP):
    x_uniform = uniform_x_grid(gc, step=step)
    gc_uniform = (
        select_included_months(gc)
        .sel(x=X_SLICE)
        .sortby("x")
        .interp(x=x_uniform)
        .transpose("x", "time")
    )
    gc_uniform.name = "gc_uniform_x"
    gc_uniform.attrs.update(gc.attrs)
    gc_uniform.attrs["x_grid_step_m"] = step
    return gc_uniform


def pca_along_track_patterns(gc, n_modes=N_MODES):
    gc_uniform = regrid_gc(gc)
    gc_anom = gc_uniform - gc_uniform.mean("time", skipna=True)
    valid_x = gc_uniform.notnull().sum("time") >= 2
    gc_valid_x = gc_anom.where(valid_x, drop=True)
    valid_time = gc_valid_x.notnull().any("x")
    matrix = gc_valid_x.where(valid_time, drop=True).fillna(0.0).transpose("time", "x")

    if matrix.sizes.get("time", 0) < 2 or matrix.sizes.get("x", 0) < 2:
        raise ValueError("PCA needs at least two complete x points and two time samples.")

    u, singular_values, vt = np.linalg.svd(matrix.values, full_matrices=False)
    n_modes = min(n_modes, singular_values.size)
    mode = xr.DataArray(np.arange(1, n_modes + 1), dims="mode", name="mode")

    eofs = xr.DataArray(
        vt[:n_modes],
        dims=("mode", "x"),
        coords={"mode": mode, "x": matrix.x},
        name="eof",
        attrs={"long_name": "EOF spatial pattern along track"},
    )
    pcs = xr.DataArray(
        u[:, :n_modes] * singular_values[:n_modes],
        dims=("time", "mode"),
        coords={"time": matrix.time, "mode": mode},
        name="pc",
        attrs={
            "long_name": "Principal component time series",
            "units": gc.attrs.get("units", ""),
        },
    )

    max_abs_idx = np.abs(eofs).argmax("x")
    signs = xr.where(eofs.isel(x=max_abs_idx) < 0, -1.0, 1.0)
    eofs = eofs * signs
    pcs = pcs * signs

    variance = xr.DataArray(
        singular_values[:n_modes] ** 2 / (matrix.sizes["time"] - 1),
        dims="mode",
        coords={"mode": mode},
        name="explained_variance",
    )
    variance_ratio = (
        variance / (singular_values**2).sum() * (matrix.sizes["time"] - 1)
    ) * 100.0
    variance_ratio.name = "explained_variance_percent"

    return xr.Dataset(
        data_vars={
            "gc_regridded": gc_uniform,
            "gc_anomaly_filled": matrix.transpose("x", "time"),
            "eof": eofs,
            "pc": pcs,
            "explained_variance": variance,
            "explained_variance_percent": variance_ratio,
        },
        attrs={
            "x_grid_step_m": X_GRID_STEP,
            "included_months": included_months_label(),
        },
    )


def plot_pca_patterns(axes, pca, color):
    for mode, ax in zip(pca.mode.values, axes):
        eof = pca.eof.sel(mode=mode)
        variance_percent = float(pca.explained_variance_percent.sel(mode=mode))
        eof.plot(ax=ax, color=color, linewidth=1.8)
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("Distance along track (m)")
        ax.set_ylabel("EOF loading")
        ax.set_title(f"EOF {int(mode)} along x ({variance_percent:.1f}% variance)")


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
        pca = pca_along_track_patterns(gc)

        fig = plt.figure(figsize=(9, 12))
        gs = fig.add_gridspec(
            N_MODES + 1,
            1,
            height_ratios=[1.45] + [1.0] * N_MODES,
            hspace=0.38,
        )
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_eofs = [fig.add_subplot(gs[i + 1, 0]) for i in range(N_MODES)]

        setup_map_axis(ax_map, f"{sat_val} track {track_number}")
        plot_track(ax_map, ds_track, track_number, lon_equat, color)
        plot_pca_patterns(ax_eofs, pca, color)

        pca.to_netcdf(
            f"gc_oned_pca_track_{sat_val}_{track_number}_{included_months_label()}_xstep_{int(X_GRID_STEP)}m.nc"
        )
        fig.savefig(
            f"gc_oned_pca_track_{sat_val}_{track_number}_{included_months_label()}_xstep_{int(X_GRID_STEP)}m.png",
            bbox_inches="tight",
            dpi=300,
        )
        if plt.get_backend().lower() != "agg":
            plt.show()
        plt.close(fig)
