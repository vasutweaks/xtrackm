from glob import glob

import cartopy.crs as ccrs
import cmaps
import cmocean as cmo
import matplotlib.pyplot as plt
import xarray as xr
from cartopy.io.img_tiles import GoogleTiles
from tools_xtrackm import *
import matplotlib as mpl


cmap = "Spectral"
cmap = "PuBuGn"
cmap = "YlGnBu"
cmap = "YlGn"
cmap_r = plt.cm.get_cmap(cmap)
cmap2 = cmo.cm.thermal
cmap3 = cmaps.BlAqGrYeOrReVi200
cmap4 = cmo.cm.phase
cmap1 = cmo.cm.topo
cmap1 = cmo.cm.topo
cmap1 = cmo.cm.diff
cmap1 = mpl.colormaps["Greys"]
cmap1 = mpl.colormaps["binary"]

# cmap1="viridis"
zone = "nindian"

IO = (30.0, 120.0, -30.0, 30.0)
NIO = (40.0, 95.0, -5.0, 30.0)
NIO = (40.0, 110.0, -5.0, 30.0)
NIO = (65.0, 95.0, -2.0, 25.0)

height, width = 9, 14
dx = 1.5

xticks = [30, 45, 60, 75, 90, 105]
yticks = [-30, -15, -5, 0, 5, 15, 30]
xticks = [45, 60, 75, 90, 105]
yticks = [0, 5, 15, 30]
tiles = GoogleTiles(style="satellite")

dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo5.cdf")  # open etopo dataset
for sat in sats_new:
    tracks_n = get_total_tracks(sat)
    tsta, tend = get_time_limits(sat)
    fig, ax1 = plt.subplots(
        1,
        1,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(width, height),
        layout="constrained",
    )
    dse.ROSE.sel(ETOPO05_X=slice(*NIO[:2])).sel(ETOPO05_Y=slice(
        *NIO[2:])).plot(ax=ax1,
                        add_colorbar=False,
                        add_labels=False,
                        cmap=cmap1)
    decorate_axis(ax1, "", *NIO)
    ax1.grid()
    info = f"MISSION: {sat} from {tsta} to {tend} with {tracks_n} tracks"
    plt.text(
        0.15,
        0.95,
        info,
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    for f in sorted(
            glob.glob(
                f"/home/srinivasu/xtrackm/data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc"
            )):
        ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        print(ds)
        sla = ds.sla
        sla1 = sla.isel(cycles_numbers=0)
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        lonm = lons_track.mean()
        latm = lats_track.mean()
        m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
        dy = m * dx
        lonm3 = lonm - dx
        latm3 = latm - dy
        if m < 0:
            angle = np.rad2deg(np.arctan(m))
            ax1.scatter(
                lons_track,
                lats_track,
                color="b",
                marker=".",
                s=4,
            )
            if is_within_region(lonm, latm, *NIO):
                plt.text(
                    lonm3,
                    latm3,
                    s=track_number,
                    fontsize=10,
                    rotation=angle,
                    color="r",
                )
        else:
            angle = np.rad2deg(np.arctan(m))
            ax1.scatter(
                lons_track,
                lats_track,
                color="r",
                s=4,
                marker=".",
            )
            if is_within_region(lonm, latm, *NIO):
                plt.text(
                    lonm3,
                    latm3,
                    s=track_number,
                    fontsize=10,
                    rotation=angle,
                    color="b",
                )
    plt.savefig(
        f"pngs_tracks/tracks_{sat}.{zone}.with_track_numbers.ascending.png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close("all")

dse.close()
