from glob import glob

import cartopy.crs as ccrs
import cmaps
import cmocean as cmo
import matplotlib.pyplot as plt
import xarray as xr
from cartopy.io.img_tiles import GoogleTiles
from tools_xtrackm import *

cmap = "Spectral"
cmap = "PuBuGn"
cmap = "YlGnBu"
cmap = "YlGn"
cmap_r = plt.cm.get_cmap(cmap)
cmap1 = cmo.cm.topo
cmap2 = cmo.cm.thermal
cmap1 = cmo.cm.topo
cmap1 = cmo.cm.diff
cmap3 = cmaps.BlAqGrYeOrReVi200
cmap4 = cmo.cm.phase

# cmap1="viridis"
zone = "nindian"

IO = (30.0, 120.0, -30.0, 30.0)
NIO = (40.0, 95.0, -5.0, 30.0)
NIO = (40.0, 110.0, -5.0, 30.0)
NIO = (60.0, 100.0, -5.0, 30.0)

height, width = 9, 14

xticks = [30, 45, 60, 75, 90, 105]
yticks = [-30, -15, -5, 0, 5, 15, 30]
xticks = [45, 60, 75, 90, 105]
yticks = [0, 5, 15, 30]
tiles = GoogleTiles(style="satellite")

sats_new = ["S3A"]
for sat in sats_new:
    tracks_n = get_total_tracks(sat)
    tsta, tend = get_time_limits(sat)
    dse = xr.open_dataset(
        "/home/srinivasu/allData/topo/etopo5.cdf"
    )  # open etopo dataset
    fig, ax1 = plt.subplots(
        1,
        1,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(width, height),
        layout="constrained",
    )
    dse.ROSE.sel(ETOPO05_X=slice(*NIO[:2])).sel(ETOPO05_Y=slice(*NIO[2:])).plot(
        ax=ax1, add_colorbar=False, add_labels=False, cmap=cmap1
    )
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
        glob.glob(f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")
    ):
        ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        # print(ds)
        sla1 = ds.sla.values
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        lon_equat = lons_track[0]  # this on equator
        lat_equat = lats_track[0]  # this on equator
        a1 = math.atan2(lat_coast - lat_equat, lon_coast - lon_equat)
        a1_d = math.degrees(a1)
        m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
        lonm = lons_track.mean()
        latm = lats_track.mean()
        dx = 1.5
        dy = m * dx
        lonm3 = lonm - dx
        latm3 = latm - dy
        print(a1_d)
        if a1_d > 90:
            ax1.scatter(
                lons_track, lats_track, s=10
            )
            if is_within_region(lonm, latm, *NIO):
                plt.text(
                    lonm,
                    latm,
                    s=track_number,
                    fontsize=10,
                    rotation=a1_d,
                    color="k",
                )
    plt.show()
    plt.close("all")
