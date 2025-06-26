import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from tools_xtrackm import *

cmap1 = cmo.cm.topo
cmap1 = cmo.cm.diff

NIO = (60.0, 100.0, -5.0, 27.0)

width, height = 14, 9


def is_odd(track_no):
    if int(track_no) % 2 == 1:
        return True
    else:
        return False


def is_even(track_no):
    if int(track_no) % 2 == 0:
        return True
    else:
        return False


def orbit_type(track_no):
    if int(track_no) % 2 == 1:
        return "ascending"
    else:
        return "descending"


def perpendicular_line(lonm, latm, m):
    m2 = -1 / m
    x_values = np.linspace(lonm - 5, lonm + 5, 100)
    y_values = m2 * (x_values - lonm) + latm
    return x_values, y_values


def horizontal_line(lonm, latm):
    x_values = np.linspace(lonm - 5, lonm + 5, 100)
    y_values = np.full_like(x_values, latm)
    return x_values, y_values


def vertical_line(lonm, latm):
    y_values = np.linspace(latm - 5, latm + 5, 100)
    x_values = np.full_like(y_values, lonm)
    return x_values, y_values


def point_at_distance(lonm, latm, m, d):
    denom = np.sqrt(1 + m**2)
    lonm1 = lonm - d / denom
    latm1 = latm + m * (d / denom)
    return lonm1, latm1


custom_lines = [
    Line2D([0], [0], color="blue", lw=2, linestyle="dashed",
           label="Ascending"),
    Line2D([0], [0],
           color="green",
           lw=2,
           linestyle="dashed",
           label="descending"),
]

dse = xr.open_dataset(
    "/home/srinivasu/allData/topo/etopo5.cdf")  # open etopo dataset

for sat in sats:
    oi = 0
    ei = 0
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
    for f in sorted(glob.glob(f"../data/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.pass_number
        orbit_str = orbit_type(track_number)
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) < 12:
            continue
        lon_equat = lons_track[0]
        lat_equat = lats_track[0]
        lon_coast = lons_track[-1]
        lat_coast = lats_track[-1]
        m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
        # angle_radian = np.arctan(m)
        # angle_radian = np.arctan2(lats_track[-1] - lats_track[0],
        #                           lons_track[-1] - lons_track[0])
        # angle_radian = np.arctan2(lat_equat - lat_coast, lon_equat - lon_coast)
        angle_radian = np.arctan2(lat_coast - lat_equat, lon_coast - lon_equat)
        angle = np.rad2deg(angle_radian)
        angle_str = str(round(angle, 2))
        lonm = lons_track.mean()
        latm = lats_track.mean()
        xp, yp = perpendicular_line(lonm, latm, m)
        xh, yh = horizontal_line(lonm, latm)
        xy, yy = vertical_line(lonm, latm)
        d = 0.5
        lonm1, latm1 = point_at_distance(lonm, latm, m, d)
        if oi > 0 and ei > 0:
            break
        if is_odd(track_number) and oi == 0:
            print(sat, angle, orbit_str, "-----------------------------")
            oi += 1
            ax1.scatter(lons_track, lats_track, linewidths=0.0, s=2, color="b")
            ax1.scatter(xp, yp, linewidths=0.0, s=2, color="b")
            ax1.scatter(xh,
                        yh,
                        linewidths=0.0,
                        s=1,
                        color="w",
                        linestyle="dashed")
            ax1.scatter(xy,
                        yy,
                        linewidths=0.0,
                        s=1,
                        color="w",
                        linestyle="dashed")
            if is_within_region(lonm, latm, *NIO):
                plt.text(
                    lonm1,
                    latm1,
                    s=track_number,
                    fontsize=14,
                    rotation=angle,
                    fontweight="bold",
                    color="#800000",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                plt.text(
                    lon_equat,
                    lat_equat,
                    s="(" + angle_str + ")",
                    fontsize=14,
                    rotation=angle,
                    fontweight="bold",
                    color="#800000",
                    horizontalalignment="left",
                    verticalalignment="top",
                )
        if is_even(track_number) and ei == 0:
            ei += 1
            print(sat, angle, orbit_str, "-----------------------------")
            ax1.scatter(lons_track, lats_track, linewidths=0.0, s=2, color="g")
            ax1.scatter(xp, yp, linewidths=0.0, s=2, color="g")
            ax1.scatter(xh,
                        yh,
                        linewidths=0.0,
                        s=1,
                        color="w",
                        linestyle="dashed")
            ax1.scatter(xy,
                        yy,
                        linewidths=0.0,
                        s=1,
                        color="w",
                        linestyle="dashed")
            try:
                lonm = lons_track[-15]
                latm = lats_track[-15]
            except:
                lonm = 0
                latm = 0
            if is_within_region(lonm, latm, *NIO):
                plt.text(
                    lonm1,
                    latm1,
                    s=track_number,
                    fontsize=14,
                    rotation=angle,
                    fontweight="bold",
                    color="g",
                    horizontalalignment="left",
                    verticalalignment="top",
                )
                plt.text(
                    lon_equat,
                    lat_equat,
                    s="(" + angle_str + ")",
                    fontsize=14,
                    rotation=angle,
                    fontweight="bold",
                    color="g",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
    info = f"{sat}"
    ax1.text(0.4,
             0.9,
             info,
             transform=ax1.transAxes,
             fontsize=30,
             fontweight="bold")
    plt.legend(handles=custom_lines, loc="upper right")
    plt.savefig(
        f"p.track.angles.{sat}.atan2.png",
        bbox_inches="tight",
    )
    plt.show()
