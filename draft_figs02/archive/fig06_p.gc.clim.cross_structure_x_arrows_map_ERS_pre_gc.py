import glob
import os
import sys
from pathlib import Path

import xarray as xr
from matplotlib.collections import LineCollection
from tools_xtrackm import *

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")

frac1 = "0.05"
sat = "ERS1+ERS2+ENV+SRL"
gc_dir = "/home/srinivasu/xtrackm/guv/guv_at_intersections"

WBOB2 = (78.0, 97.0, 0.0, 25.0)
proj = ccrs.PlateCarree()
scaling_factor = 1e5
meters_per_degree_lat = 111_320


def track_to_str(track):
    track_f = float(track)
    if track_f.is_integer():
        return str(int(track_f))
    return str(track)


def load_gc_clim_month(sat_name, track_number, month_num):
    f_gc = f"{gc_dir}/gc_at_{sat_name}_{track_number}_{frac1}.nc"
    if not os.path.exists(f_gc):
        return None
    with xr.open_dataset(f_gc) as ds_gc:
        gc = ds_gc[next(iter(ds_gc.data_vars))].load()
    gc = gc.where((gc > -2.5) & (gc < 2.5))
    gc_clim = gc.groupby("time.month").mean("time")
    return gc_clim.sel(month=month_num)


def add_month_panel(ax, month_num):
    info = f"{sat} {months[month_num + 1]}"
    ax.set_extent([*WBOB2], crs=proj)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--")
    gl.top_labels = gl.right_labels = False
    ax.text(
        0.3,
        0.95,
        info,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
    )

    for f_sla in sorted(
        glob.glob(
            f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.descending.nc"
        )
    ):
        with xr.open_dataset(f_sla, decode_times=False) as ds_sla:
            # track_number = track_to_str(ds_sla.Pass.item())
            track_number = ds_sla.Pass
            gc_clim_m = load_gc_clim_month(sat, track_number, month_num)
            if gc_clim_m is None:
                continue
            gc_val = float(gc_clim_m.values)
            if not np.isfinite(gc_val):
                continue

            lons_track_rev = ds_sla.lon.values[::-1]
            lats_track_rev = ds_sla.lat.values[::-1]

        dlon = np.gradient(lons_track_rev)
        dlat = np.gradient(lats_track_rev)
        tangent_mag = np.hypot(dlon, dlat)
        tangent_mag = np.where(tangent_mag == 0, np.nan, tangent_mag)
        tangent_u = dlon / tangent_mag
        tangent_v = dlat / tangent_mag
        normal_u = -tangent_v
        normal_v = tangent_u

        gc_deg = gc_val / meters_per_degree_lat
        delta_lon = normal_u * gc_deg * scaling_factor
        delta_lat = normal_v * gc_deg * scaling_factor

        lon_end = lons_track_rev + delta_lon
        lat_end = lats_track_rev + delta_lat
        segments = [
            [[lons_track_rev[i], lats_track_rev[i]], [lon_end[i], lat_end[i]]]
            for i in range(len(lons_track_rev))
            if np.isfinite(lon_end[i]) and np.isfinite(lat_end[i])
        ]
        if not segments:
            continue

        ax.plot(lons_track_rev, lats_track_rev, "k-", transform=proj)
        ax.add_collection(LineCollection(segments, colors="red", linewidths=1, transform=proj))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10),
                       subplot_kw={'projection': proj})
fig.subplots_adjust(wspace=0.1)
add_month_panel(ax1, month_num=2)
add_month_panel(ax2, month_num=10)
plt.savefig(f"cross_structure_x_arrows_map_{sat}_1by2.png",
            bbox_inches='tight', dpi=300)
ln_sf(f"cross_structure_x_arrows_map_{sat}_1by2.png", png_file)
plt.show()
plt.close("all")
