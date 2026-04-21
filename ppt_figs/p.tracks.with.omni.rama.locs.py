from glob import glob

import cartopy.crs as ccrs
import cmaps
import cmocean as cmo
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from cartopy.io.img_tiles import GoogleTiles
from tools_xtrackm import *
from namesm import lons_omni, lats_omni
# codex resume 019dafca-5204-7c01-9d31-daad3e3b4c05

cmap1 = cmo.cm.diff

# cmap1="viridis"
zone = "nindian"

height, width = 9, 14

lons_rama = [value[0] for value in rama_d.values()]
lats_rama = [value[1] for value in rama_d.values()]
rama_buoys = list(rama_d.keys())

dse = xr.open_dataset(
    "/home/srinivasu/allData/topo/etopo5.cdf"
)  # open etopo dataset
rose = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)
# sats_new = ["TPN+J1N+J2N+J3N"]
# sats_new = ["S3A"]
# sats_new = ["TP+J1+J2+J3+S6A"]
sats_new = ["GFO"]
band_0_5n = sg.box(TRACKS_REG[0], 0.0, TRACKS_REG[1], 5.0)
lat_ref1 = 12
lat_ref2 = 7

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
    decorate_axis(ax1, "", *TRACKS_REG, step=5)
    ax1.add_geometries(
        [band_0_5n],
        crs=ccrs.PlateCarree(),
        facecolor="gray",
        edgecolor="none",
        alpha=0.5,
        zorder=2,
    )
    # ax1.grid()
    info = f"MISSION: {sat} from {tsta} to {tend} with {tracks_n} tracks"
    plt.text(
        0.50,
        0.97,
        info,
        transform=ax1.transAxes,
        fontsize=10,
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="top",
        color="b",
    )
    # sys.exit()
    # OMNI buoy locations
    ax1.scatter(
        lons_omni,
        lats_omni,
        marker="o",
        c="cyan",
        s=82,
        edgecolor="black",
        label="OMNI buoys",
        zorder=12,
    )
    # OMNI buoy names
    ln = len(lons_omni)
    for i in range(ln):
        if is_within_region(lons_omni[i], lats_omni[i], *TRACKS_REG):
            plt.text(
                lons_omni[i],
                lats_omni[i],
                omni_buoys[i],
                fontsize=10,
                ha="right",
                fontweight="bold",
                zorder=6,
                color="cyan",
            )
    ax1.scatter(
        lons_rama,
        lats_rama,
        marker="*",
        c="y",
        s=182,
        edgecolor="black",
        label="RAMA buoys",
        zorder=12,
    )
    for i in range(len(lons_rama)):
        if is_within_region(lons_rama[i], lats_rama[i], *TRACKS_REG):
            plt.text(
                lons_rama[i],
                lats_rama[i],
                rama_buoys[i],
                fontsize=10,
                ha="right",
                fontweight="bold",
                color="y",
                zorder=7,
            )
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    ax1.scatter(
        df["lons_inter"],
        df["lats_inter"],
        label=f"Satellite {sat}",
        s=60,
        marker="+",
        color="r",
        edgecolor="none",
        linewidths=1.0,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    plt.legend()
    files = sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    )
    if len(files) == 0:
        print("No files found. Check DATA_DIR / pattern.")
    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        print(ds)
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) < 6:
            continue
        ii = len(lons_track) // 2
        angle = local_angle(lons_track, lats_track, ii, ioff=3, degrees=True)
        m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
        line_track = sg.LineString(zip(lons_track, lats_track))
        ax1.add_geometries(
            [line_track],
            crs=ccrs.PlateCarree(),
            edgecolor="orange",
            facecolor="none",
            linewidth=1.0,
            alpha=1.0,
            zorder=3,
        )
        if m > 0:
            ii2 = int(0.25 * len(lons_track))
            try:
                ii2 = 120
                lonm = lons_track[ii2]
                latm = lats_track[ii2]
            except IndexError:
                ii2 = int(0.75 * len(lons_track))
                lonm = lons_track[ii2]
                latm = lats_track[ii2]
            try:
                ref_idx = np.argmin(np.abs(lats_track - lat_ref2))
                lonm = lons_track[ref_idx]
                latm = lat_ref2
            except IndexError:
                ii2 = int(0.50 * len(lons_track))
                lonm = lons_track[ii2]
                latm = lats_track[ii2]
            dx = 0.0
            dy = m * dx
            lonm3 = lonm - dx
            latm3 = latm - dy
            if is_within_region(lonm, latm, *TRACKS_REG):
                plt.text(
                    lonm3,
                    latm3,
                    s=track_number,
                    fontsize=10,
                    rotation=angle,
                    color="white",
                )
        else:
            ii2 = int(0.75 * len(lons_track))
            lonm = lons_track[ii2]
            latm = lats_track[ii2]
            try:
                ii2 = len(lons_track) - 50
                lonm = lons_track[ii2]
                latm = lats_track[ii2]
            except IndexError:
                ii2 = int(0.75 * len(lons_track))
                lonm = lons_track[ii2]
                latm = lats_track[ii2]
            try:
                ref_idx = np.argmin(np.abs(lats_track - lat_ref1))
                lonm = lons_track[ref_idx]
                latm = lat_ref1
            except IndexError:
                ii2 = int(0.5 * len(lons_track))
                lonm = lons_track[ii2]
                latm = lats_track[ii2]
            dx = 0.0
            dy = m * dx
            lonm3 = lonm - dx
            latm3 = latm - dy
            if is_within_region(lonm, latm, *TRACKS_REG):
                plt.text(
                    lonm3,
                    latm3,
                    s=track_number,
                    fontsize=10,
                    rotation=angle,
                    color="white",
                )
    plt.savefig(
        f"tracks_{sat}.{zone}.omni.rama.with_track_numbers.png",
        bbox_inches="tight",
        dpi=300
    )
    plt.show()
    plt.close("all")
    # sys.exit()
