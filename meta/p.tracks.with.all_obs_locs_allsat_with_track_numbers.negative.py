from glob import glob

import cartopy.crs as ccrs
import cmaps
import cmocean as cmo
import matplotlib.pyplot as plt
import pandas as pd
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

df = read_tide_meta()
df_nio = region_selected(df, *NIO)
lons_psmsl = df_nio["longitude"]
lats_psmsl = df_nio["latitude"]
pd.set_option("display.max_rows", None)

df_coast = pd.read_excel(
    "/home/srinivasu/forecast_e/coastal/CoastalADCPMeta_correction.xlsx",
    parse_dates=[6, 7],
)
print(df_coast.columns)
print(df_coast.info())
df1 = df_coast[["ID", "Station Name"]]
df2 = df1.drop_duplicates()
indices = df2.index.tolist()
buoys = df2["ID"].tolist()
lons_coast = df_coast["Longitude"].iloc[indices]  # .tolist()
lats_coast = df_coast["Latitude"].iloc[indices]  # .tolist()
print(df2)
dict1 = dict(zip(df2["ID"], df2["Station Name"]))

xticks = [30, 45, 60, 75, 90, 105]
yticks = [-30, -15, -5, 0, 5, 15, 30]
xticks = [45, 60, 75, 90, 105]
yticks = [0, 5, 15, 30]
tiles = GoogleTiles(style="satellite")

sats = ["TP+J1+J2+J3"]
for sat in sats_new:
    tracks_n = get_total_tracks(sat)
    tsta, tend = get_time_limits(sat)
    dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo5.cdf")  # open etopo dataset
    fig, ax1 = plt.subplots(
        1,
        1,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(width, height),
        layout="constrained",
    )
    dse.ROSE.sel(ETOPO05_X=slice(*NIO[:2])).sel(
        ETOPO05_Y=slice(*NIO[2:])
    ).plot(ax=ax1, add_colorbar=False, add_labels=False, cmap=cmap1)
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
    # plt.show()
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
    )
    # OMNI buoy names
    ln = len(lons_omni)
    for i in range(ln):
        plt.text(
            lons_omni[i],
            lats_omni[i],
            omni_buoys[i],
            fontsize=10,
            ha="right",
            fontweight="bold",
        )
    # PSMSL tide gauge locations
    ax1.scatter(
        lons_psmsl,
        lats_psmsl,
        marker="o",
        c="magenta",
        s=42,
        edgecolor="black",
        label="PSMSL buoys",
    )
    # COASTAL buoy locations
    ax1.scatter(
        lons_coast,
        lats_coast,
        marker="*",
        c="red",
        s=122,
        edgecolor="black",
        label="COASTAL buoys",
    )
    plt.legend()
    for f in sorted(glob.glob(f"/home/srinivasu/slnew/xtrackd/data/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.pass_number
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
        dx = 1.5
        dy = m * dx
        lonm3 = lonm - dx
        latm3 = latm - dy
        # lonm3 =  lonm - 1.0
        # latm3 =  latm - 1.0
        # lonm3 =  lon[0]
        # latm3 =  lat[0]
        angle = np.rad2deg(np.arctan(m))
        if m < 0:
            ax1.scatter(lons_track, lats_track, c=sla1, cmap=cmap_r, linewidths=0.0, s=10)
            if is_within_region(lonm, latm, *NIO):
                plt.text(
                    lonm,
                    latm,
                    s=track_number,
                    fontsize=10,
                    rotation=angle,
                    color="k",
                )
    plt.savefig(
        f"pngs_tracks_all_obs/tracks_{sat}.{zone}.with_track_numbers.negative.png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close("all")
    # sys.exit()
