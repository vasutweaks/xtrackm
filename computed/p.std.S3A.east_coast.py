import xarray as xr
from matplotlib.ticker import FuncFormatter
from tools_xtrackm import *
from namesm import sats_new
import cmaps
import cmocean as cmo
import math

sat = 'S3A'
cmap = 'twilight_shifted'
EAST_COAST_REG = (78.0, 99, 0, 25.4)
dse = xr.open_dataset(
    f"/home/srinivasu/allData/topo/etopo5.cdf"
)
files = sorted(
    glob.glob(
        f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc"
    )
)
tracks_n = len(files)
east_coast_tracks = []
for f in files:
    ds = xr.open_dataset(f, decode_times=False)
    lon_eq = ds.lon.values[0]
    if lon_eq < EAST_COAST_REG[0] or lon_eq > EAST_COAST_REG[1]:
        continue
    track_number = ds.Pass
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lon_equat = lons_track[0]
    lon_coast = lons_track[-1]
    lat_equat = lats_track[0]
    lat_coast = lats_track[-1]
    m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
    lonm = lons_track.mean()
    latm = lats_track.mean()
    dx = 1.5
    dy = m * dx
    lonm3 = lonm - dx
    latm3 = latm - dy
    a1 = math.atan2(lat_coast - lat_equat,
                    lon_coast - lon_equat)
    a1_d = math.degrees(a1)
    if a1_d > 90:
        # append 4 zero padded track number
        east_coast_tracks.append(track_number.zfill(4))
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        dse.ROSE.sel(ETOPO05_X=slice(*NIO[:2])).sel(ETOPO05_Y=slice(*NIO[2:])).plot(
            ax=ax1, add_colorbar=False, add_labels=False, cmap=cmap
        )
        decorate_axis(ax1, "", *NIO, step=8)
        info = f"MISSION: {sat}"
        plt.text(
            0.40,
            0.90,
            info,
            transform=ax1.transAxes,
            fontsize=10,
            fontweight="bold",
            color="b",
        )
        # need to put scatter point color to black
        ax1.scatter(
            lons_track, lats_track, s=1, c="r"
            )
        if is_within_region(lonm, latm, *NIO):
            plt.text(
                lonm,
                latm,
                s=track_number,
                fontsize=8,
                rotation=a1_d,
                color="k",
            )
        print(f"{sat} {track_number} -------------------")
        sla_track = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
        dac_track = track_dist_time_asn_midx(ds, var_str="dac", units_in="m")
        std = sla_track.std(dim="time")
        ax2 = fig.add_subplot(2, 1, 2)
        std.plot(ax=ax2)
        # Keep axis values in meters, format tick labels as kilometers.
        ax2.set_xlabel("Distance (km)")
        # draw vertical line at x=0 and name it "coast"
        ax2.axvline(x=0, color="k", linewidth=0.5, linestyle="--")
        # plt.text vertical line at x=0 and name it "coast"
        plt.text(
            0.02,
            0.5,
            "coast",
            transform=ax2.transAxes,
            fontsize=10,
            fontweight="bold",
            color="r",
            rotation=90,
        )
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
        plt.show()
        plt.close("all")
dse.close()
east_coast_tracks = sorted(east_coast_tracks)
print(east_coast_tracks)
