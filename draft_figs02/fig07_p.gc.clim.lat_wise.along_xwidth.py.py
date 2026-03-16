import xarray as xr
from tools_xtrackm import *
import glob
import os

import matplotlib.cm as cm
from pathlib import Path
import sys

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")

# sats = ["S3A"]
REG = (80.0, 89.0, 11.0, 21.0)
REG = (75.0, 92.0, 10.0, 23.0)
sats = ["S3A"]
sats = ["ERS1+ERS2+ENV+SRL"]
xwidth = 290e3
# Load the YlOrRd colormap
ncolors = 21
cmap1 = cm.get_cmap('YlOrRd', ncolors)  # Split into 10 colors
colors1 = cmap1(np.linspace(0, 1, ncolors))
cmap2 = cm.get_cmap('BuGn', ncolors)  # Split into 10 colors
colors2 = cmap2(np.linspace(0, 1, ncolors))

norm2 = plt.Normalize(10.18, 22.00)
sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
sm2.set_array([])

norm1 = plt.Normalize(10.18, 22.00)
sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
sm1.set_array([])

# fig, ax = plt.subplots(1, 1, figsize=(9, 4))
frac1 = "0.2"
frac1 = "0.05"
for sat in sats[:]:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # adjust subplots increase x spacing
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.8, wspace=0.3, hspace=0.3)
    i = 0
    lat_values = []
    for f_sla in sorted(
        glob.glob(
            f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.descending.nc"
        )
    ):
        basename = os.path.basename(f_sla)
        ds_sla = xr.open_dataset(f_sla, decode_times=False)
        trackn = ds_sla.Pass
        f = f"../computed/sla_loess_{frac1}a/track_sla_along_{sat}_{trackn}_loess_{frac1}.nc"
        lons_track = ds_sla.lon.values
        lats_track = ds_sla.lat.values
        lons_track_rev = lons_track[::-1]
        lats_track_rev = lats_track[::-1]
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        print(f, lon_coast, lat_coast)
        if is_within_region(lon_coast, lat_coast, *REG):
            lat_values.append(lat_coast)
            ds_smooth = xr.open_dataset(f)
            # print(ds_smooth)
            sla_smooth = ds_smooth.sla_smooth
            print(f"stats of sla_smooth {sat} {trackn} -------------------")
            print(f"stats of gu {sat} {trackn} -------------------")
            gc = compute_geostrophy_gc(ds_sla, sla_smooth)
            gc_clim = gc.groupby("time.month").mean("time")
            gc_clim_xwidth = gc_clim.sel(x=slice(0, xwidth))
            gc_clim_xwidth_m = gc_clim_xwidth.isel(month=2, drop=True) # march
            gc_clim_xwidth_n = gc_clim_xwidth.isel(month=10, drop=True) # november
            gc_clim_xwidth_m.plot(ax=ax, label=f"{lat_coast}",
                                  color=colors1[i])
            x0, y0 = -15, gc_clim_xwidth_m.sel(x=0).values
            plt.text(x0, y0, f"{lat_coast:.2f}", fontsize=10, color=colors1[i])
            gc_clim_xwidth_n.plot(ax=ax, label=f"{lat_coast}",
                                  color=colors2[i])
            x0, y0 = -15, gc_clim_xwidth_n.sel(x=0).values
            plt.text(x0, y0, f"{lat_coast:.2f}", fontsize=10, color=colors2[i])
            # set xticks between 0 to xwidth
            ax.set_xlim(-15, xwidth)
            ax.set_ylim(-0.7, 0.7)
            xticks1 = ax.get_xticks()
            xticks_km = np.round((xticks1 / 1000), 0)
            ax.set_xticks(xticks1, xticks_km)
            ax.set_xlabel("distance along track (km)")
            ax.set_ylabel("geostrophic velocity (m/s)")
            # ax.set_ylim(-0.3, 0.3)
            ax.axhline(0, color="k", linestyle="--")
            i += 1
            # print_stats(gc)
    # plt.legend()
    print(f"number of tracks {i}")
    minlat = min(lat_values)
    maxlat = max(lat_values)
    print(f"min lat {min(lat_values)} max lat {max(lat_values)}")
    pos = ax.get_position()
    # Adjust and add the first colorbar closer to the plot
    cbar_ax1 = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])  # Adjust 0.01 to control horizontal gap
    cbar_ax1.text(0.5, 0.5, "MARCH", fontsize=10, horizontalalignment="center", verticalalignment="center", transform=cbar_ax1.transAxes, rotation = "vertical")
    cbar = plt.colorbar(sm1, cax=cbar_ax1)
    # cbar.set_label('latitude')
    cbar.set_ticks(list(range(int(minlat), int(maxlat)+1, 2)))
    # Adjust and add the second colorbar even closer
    cbar_ax2 = fig.add_axes([pos.x1 + 0.06, pos.y0, 0.02, pos.height])  # Adjust 0.03 as needed for spacing
    cbar_ax2.text(0.5, 0.5, "NOVEMBER", fontsize=10, horizontalalignment="center", verticalalignment="center", transform=cbar_ax2.transAxes, rotation = "vertical")
    cbar = plt.colorbar(sm2, cax=cbar_ax2)
    cbar.set_label('latitude')
    cbar.set_ticks(list(range(int(minlat), int(maxlat)+1, 2)))

    plt.savefig(
        f"p.gc.clim.lat_wise.along_xwidth_{sat}a.png",
        bbox_inches="tight",
        dpi=300,
    )
    ln_sf(
        f"p.gc.clim.lat_wise.along_xwidth_{sat}a.png",
        png_file,
    )
    # plt.show()
    plt.close("all")
