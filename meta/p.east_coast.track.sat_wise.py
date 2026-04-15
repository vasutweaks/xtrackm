import pickle
from glob import glob

import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import xarray as xr
from cartopy.io.img_tiles import GoogleTiles
from tools_xtrackm import *

cmap1 = cmo.cm.diff

TRACKS_REG = (43.0, 99, 0, 25.4)
WBOB = (79.5, 87.5, 8.0, 21.0)
EAST_COAST_LON_EQ = (83.3, 92.9)
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

dse = xr.open_dataset(
    "/home/srinivasu/allData/topo/etopo5.cdf"
)  # open etopo dataset
rose = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)
sats_new = ["ERS1+ERS2+ENV+SRL"]
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
    rose.plot(ax=ax1, add_colorbar=False, add_labels=False, cmap=cmap1)
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
    files = sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    )
    east_coast_tracks = {}
    for f in files:
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
        # if lon_coast < WBOB[0] or lon_coast > WBOB[1]:
        # if lon_equat < WBOB[0] or lon_equat > WBOB[1]:
        if (
            lon_equat < EAST_COAST_LON_EQ[0]
            or lon_equat > EAST_COAST_LON_EQ[1]
        ):
            continue
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
        if a1_d < 90:
            continue
        east_coast_tracks[track_number] = lon_equat
        ax1.scatter(lons_track, lats_track, s=10)
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
print(east_coast_tracks)
with open(f"east_coast_tracks_{sat}.pkl", "wb") as f:
    pickle.dump(east_coast_tracks, f)
