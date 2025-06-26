import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tools_xtrackd import *
from matplotlib.lines import Line2D

cmap1 = cmo.cm.topo
cmap1 = cmo.cm.diff

IO = (30.0, 120.0, -30.0, 30.0)
NIO = (60.0, 100.0, -5.0, 27.0)

width, height = 14, 9


def is_odd(track_no):
    if int(track_no) % 2 == 1:
        return True
    else:
        return False


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
    # https://www.geeksforgeeks.org/find-points-at-a-given-distance-on-a-line-of-given-slope/
    # m is the slope of the line
    # lonm and latm are the coordinates of the point on the line
    # d is the distance from the point to a point above the line
    denom = np.sqrt(1 + m**2)
    lonm1 = lonm - d / denom
    latm1 = latm + m * (d / denom)
    return lonm1, latm1


dse = xr.open_dataset(
    "/home/srinivasu/allData/topo/etopo5.cdf")  # open etopo dataset

sats = ["GFO"]
for sat in sats:
    ODD = False
    EVEN = False
    fig, ax1 = plt.subplots(
        1,
        1,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(width, height),
        layout="constrained",
    )
    dse.ROSE.sel(ETOPO05_X=slice(*NIO[:2])).sel(ETOPO05_Y=slice(*NIO[2:])).plot(
        ax=ax1, add_colorbar=False, add_labels=False, cmap=cmap1)
    decorate_axis(ax1, "", *NIO)
    for f in sorted(glob.glob(f"../data/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        # print(ds)
        track_number = ds.pass_number
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        lon_eq = lons_track[0]
        lat_eq = lats_track[0]
        m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
        angle_radian = np.arctan(m)
        angle_radian = np.arctan2(lats_track[-1] - lats_track[0],
                                  lons_track[-1] - lons_track[0])
        angle = np.rad2deg(angle_radian)
        angle_str = str(round(angle, 2))
        print(angle, angle_radian, "-----------------------------")
        lonm = lons_track.mean()
        latm = lats_track.mean()
        xp, yp = perpendicular_line(lonm, latm, m)
        xh, yh = horizontal_line(lonm, latm)
        xy, yy = vertical_line(lonm, latm)
        # Horizontal and vertical arrow parameters
        x_start, x_end = lonm, lonm+5
        y_start, y_end = latm, latm+5
        # get the coordinates of a point at a distance d from the point (lonm, latm)
        # above the line with slope m
        d = 0.5
        lonm1, latm1 = point_at_distance(lonm, latm, m, d)
        if is_odd(track_number):
            ODD = True
            ax1.scatter(lons_track, lats_track, linewidths=0.0, s=2, color="b")
            ax1.scatter(xp, yp, linewidths=0.0, s=2, color="b")
            ax1.annotate(
                '', xy=(x_end, 0), xytext=(x_start, 0), 
                arrowprops=dict(arrowstyle='->', color='blue', lw=1),
                transform=ccrs.PlateCarree()  # Ensure projection compatibility
            )
            ax1.annotate(
                '', xy=(0, y_end), xytext=(0, y_start), 
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                transform=ccrs.PlateCarree()  # Ensure projection compatibility
            )
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
                    lon_eq,
                    lat_eq,
                    s="(" + angle_str + ")",
                    fontsize=14,
                    rotation=angle,
                    fontweight="bold",
                    color="#800000",
                    horizontalalignment="left",
                    verticalalignment="top",
                )
        else:
            EVEN = True
            ax1.scatter(lons_track, lats_track, linewidths=0.0, s=2, color="g")
            ax1.scatter(xp, yp, linewidths=0.0, s=2, color="g")
            ax1.annotate(
                '', xy=(x_end, 0), xytext=(x_start, 0), 
                arrowprops=dict(arrowstyle='->', color='blue', lw=1),
                transform=ccrs.PlateCarree()  # Ensure projection compatibility
            )
            ax1.annotate(
                '', xy=(0, y_end), xytext=(0, y_start), 
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                transform=ccrs.PlateCarree()  # Ensure projection compatibility
            )
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
                    lon_eq,
                    lat_eq,
                    s="(" + angle_str + ")",
                    fontsize=14,
                    rotation=angle,
                    fontweight="bold",
                    color="g",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
        if ODD and EVEN:
            break
    info = f"{sat}"
    ax1.text(0.4,
             0.9,
             info,
             transform=ax1.transAxes,
             fontsize=30,
             fontweight="bold")

# Create a custom legend with specific markers and colors
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2, linestyle='dashed', label='Ascending'),
        Line2D([0], [0], color='green', lw=2, linestyle='dashed', label='descending'),
    ]

    plt.legend(handles=custom_lines, loc='upper right')
    plt.savefig(
        f"p.track.angles.{sat}.png",
        bbox_inches="tight",
    )
    plt.show()
