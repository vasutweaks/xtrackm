from pathlib import Path
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from geopy import distance

# copilot --resume=62ff5540-5ab6-44c0-b895-489d72a8c8cb

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from tools_xtrackm import decorate_axis, is_within_region

SAT = "ERS1+ERS2+ENV+SRL"
WBOB2 = (78.0, 97.0, 0.0, 25.0)
GC_GLOB = "../guv/guv_at_intersections_loess/gc_at_ERS1+ERS2+ENV+SRL_*_loess_0.05.nc"
SLA_TEMPLATE = "../data/ERS1+ERS2+ENV+SRL/ctoh.sla.ref.ERS1+ERS2+ENV+SRL.nindian.{track:03d}.nc"
MONTHS_TO_PLOT = ((3, "Mar"), (11, "Nov"))
ARROW_POINT_STEP = 4
ARROW_METERS_PER_MS = 150_000.0
TRACK_COLOR = "k"
ARROW_COLOR = "crimson"


def track_number_from_gc_path(path):
    return int(Path(path).stem.split("_")[-3])


def build_track_geometry(ds_sla):
    lon_rev = ds_sla["lon"].load().isel(points_numbers=slice(None, None, -1))
    lat_rev = ds_sla["lat"].load().isel(points_numbers=slice(None, None, -1))

    segment_lengths = [0.0]
    lon_vals = lon_rev.values
    lat_vals = lat_rev.values
    for idx in range(1, lon_rev.sizes["points_numbers"]):
        seg_len = distance.distance(
            (lat_vals[idx - 1], lon_vals[idx - 1]),
            (lat_vals[idx], lon_vals[idx]),
        ).m
        segment_lengths.append(seg_len)

    x_vals = np.cumsum(segment_lengths)
    geom = xr.Dataset(
        coords={"x": ("x", x_vals)},
        data_vars={
            "lon": ("x", lon_vals),
            "lat": ("x", lat_vals),
        },
    )
    return geom


def build_track_normals(track_geom):
    meters_per_degree_lat = xr.full_like(track_geom["lat"], 111_320.0)
    meters_per_degree_lon = 111_320.0 * np.cos(np.deg2rad(track_geom["lat"]))

    tangent_east = track_geom["lon"].differentiate("x") * meters_per_degree_lon
    tangent_north = track_geom["lat"].differentiate("x") * meters_per_degree_lat
    tangent_mag = np.hypot(tangent_east, tangent_north)

    normals = xr.Dataset(
        data_vars={
            "east": -tangent_north / tangent_mag,
            "north": tangent_east / tangent_mag,
        },
        coords={"x": track_geom["x"]},
    )
    return normals


def track_matches_filter(ds_sla):
    lon_track = ds_sla["lon"].values
    lat_track = ds_sla["lat"].values
    if lon_track.size == 0:
        return False
    lon_coast = float(lon_track[-1])
    lat_coast = float(lat_track[-1])
    if not is_within_region(lon_coast, lat_coast, *WBOB2):
        return False
    delta_lon = float(lon_track[-1] - lon_track[0])
    if np.isclose(delta_lon, 0.0):
        return False
    slope = float((lat_track[-1] - lat_track[0]) / delta_lon)
    return slope < 0.0


def build_plot_dataset(ds_sla, ds_gc, month_number):
    track_geom = build_track_geometry(ds_sla)
    track_normals = build_track_normals(track_geom)
    gc_clim = ds_gc["gc"].groupby("time.month").mean("time")
    gc_month = gc_clim.sel(month=month_number).interp(x=track_geom["x"])

    plot_ds = xr.Dataset(
        data_vars={
            "lon": track_geom["lon"],
            "lat": track_geom["lat"],
            "normal_east": track_normals["east"],
            "normal_north": track_normals["north"],
            "gc": gc_month,
        },
        coords={"x": track_geom["x"]},
    )
    plot_ds = plot_ds.where(np.isfinite(plot_ds["gc"]), drop=True)
    plot_ds = plot_ds.isel(x=slice(0, None, ARROW_POINT_STEP))
    return plot_ds


def add_track_and_arrows(ax, ds_sla, ds_gc, month_number):
    track_geom = build_track_geometry(ds_sla)
    plot_ds = build_plot_dataset(ds_sla, ds_gc, month_number)

    ax.plot(
        track_geom["lon"].values,
        track_geom["lat"].values,
        color=TRACK_COLOR,
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    meters_per_degree_lon = 111_320.0 * np.cos(np.deg2rad(plot_ds["lat"]))
    arrow_length_m = plot_ds["gc"] * ARROW_METERS_PER_MS
    u = plot_ds["normal_east"] * arrow_length_m / meters_per_degree_lon
    v = plot_ds["normal_north"] * arrow_length_m / 111_320.0

    ax.quiver(
        plot_ds["lon"].values,
        plot_ds["lat"].values,
        u.values,
        v.values,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0018,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
        pivot="middle",
        color=ARROW_COLOR,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )


def decorate_map_axis(ax, title):
    decorate_axis(ax, title, *WBOB2, step=5)
    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.7,
        linestyle="--",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    return gridlines


def main():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )

    gc_paths = sorted(Path(".").glob(GC_GLOB))
    plotted_tracks = 0

    for ax, (month_number, month_label) in zip(axes, MONTHS_TO_PLOT):
        decorate_map_axis(ax, f"{SAT} {month_label}")

        for gc_path in gc_paths:
            track_number = track_number_from_gc_path(gc_path)
            sla_path = Path(SLA_TEMPLATE.format(track=track_number))
            if not sla_path.exists():
                continue

            with xr.open_dataset(sla_path, decode_times=False) as ds_sla:
                if not track_matches_filter(ds_sla):
                    continue
                with xr.open_dataset(gc_path) as ds_gc:
                    add_track_and_arrows(ax, ds_sla, ds_gc, month_number)
                    if month_number == MONTHS_TO_PLOT[0][0]:
                        plotted_tracks += 1

    axes[1].gridlines(draw_labels=False)
    output_path = Path(__file__).with_suffix(".png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"saved {output_path}")
    print(f"tracks plotted: {plotted_tracks}")
    plt.close(fig)


if __name__ == "__main__":
    main()
