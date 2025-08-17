import math
import pickle
from datetime import datetime

import cmocean as cmo
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely as shp
import xarray as xr
from tools_xtrackm import *


# function to get percentage valid point in an xarray dataarray
def perc_valid_points(da):
    return 100 * (da.count().item() / len(da.time))


xr.set_options(display_expand_data=False)
gs = gridspec.GridSpec(1,
                       2,
                       width_ratios=[3, 1],
                       height_ratios=[2],
                       wspace=0.1,
                       hspace=0.05)
PSMSL_ID = 414  # id for visakhaptnam
width, height = 13, 6
data_dir = f"/home/srinivasu/xtrackm/data/"

cmap1 = cmo.cm.topo
cmap1 = cmo.cm.diff
cmap1 = cmo.cm.balance
cmap2 = cmo.cm.thermal

# tide gauge data
df = read_tide_meta()
df_nio = region_selected(df, *NIO)
lon_psmsl, lat_psmsl, name_psmsl = get_lonlat_psmsl(PSMSL_ID, df_nio)
print(lon_psmsl, lat_psmsl)
name_psmsl = name_psmsl.strip()

delx2 = 5
lonlat_box2 = (
    lon_psmsl - delx2,
    lon_psmsl + delx2,
    lat_psmsl - delx2,
    lat_psmsl + delx2,
)
poly_box = shp.geometry.box(lonlat_box2[0], lonlat_box2[2], lonlat_box2[1],
                            lonlat_box2[3])
# get the perimeter of the box
ds_id = read_id(PSMSL_ID)
sla_tide = ds_id["height"]
sla_tide = 0.001 * sla_tide
tide_tsta_o = ds_id.time[0].values
tide_tend_o = ds_id.time[-1].values
tide_tsta_o = n64todatetime(tide_tsta_o)
tide_tend_o = n64todatetime(tide_tend_o)

tlim1 = datetime.strptime("2013-01-01", "%Y-%m-%d")
tlim2 = datetime.strptime("2021-01-01", "%Y-%m-%d")


def point_on_line_at_distance(m, c, x0, y0, d):
    """
    Find a point on the line y = mx + c at a distance d from (x0, y0).
    Parameters:
        m (float): Slope of the line.
        c (float): y-intercept of the line.
        x0 (float): x-coordinate of the starting point.
        y0 (float): y-coordinate of the starting point.
        d (float): Distance from the starting point.
    Returns:
        tuple: Coordinates (x, y) of the new point.
    """
    # Normalize the direction vector of the line
    dx = 1 / math.sqrt(1 + m**2)  # x-component of the unit direction vector
    dy = m / math.sqrt(1 + m**2)  # y-component of the unit direction vector
    # Compute the new point
    if m < 0:
        d = -d
    x_new = x0 + d * dx
    y_new = y0 + d * dy
    return x_new, y_new


ds_dt1 = xr.open_dataset(
    "/home/srinivasu/allData/aviso/allyear_dt_global_allsat_msla_h.nc",
    decode_times=False,
)
# ds_dt1 = xr.open_dataset(
#     "/media/srinivasu/allData/sealevel/data/allyear_dt_global_allsat_msla_h.nc",
#     decode_times=False,
# )
print(ds_dt1)
ds_dt = change_time(ds_dt1, "time")
sla_aviso = ds_dt.sla  # .sel(time=slice(tsta_o, tend_o))
sla_aviso_at = sla_aviso.interp(longitude=lon_psmsl,
                                method="nearest").interp(latitude=lat_psmsl,
                                                         method="nearest")

sla_tide_interp = sla_tide.interp(time=sla_aviso.time)
sla_tide_interp = sla_tide_interp - sla_tide_interp.mean(dim="time")
# sla_tide_adj = sla_tide_interp  - dac_p_smooth

chosen_index = 0
dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo5.cdf")

with open("closest_tracks_all_psmsl_new.pkl", "rb") as file:
    sorted_list = pickle.load(file)


def filter_list(sorted_list, sat=None, id=None):
    result = sorted_list
    if sat is not None:
        result = [tup for tup in result if tup[0] == sat]
    if id is not None:
        result = [tup for tup in result if tup[5] == id]
    return result


filtered_list = sorted_list[:]
filtered_list = filter_list(sorted_list, id=PSMSL_ID)
print(filtered_list)


# sys.exit(0)
for sat, trackn, x1, y1, mindist, id in filtered_list:
    if mindist > 50:
        continue
    track_tsta_o, track_tend_o = get_time_limits_o(sat)
    overlap_tsta, overlap_tend = overlap_dates(track_tsta_o, track_tend_o,
                                               tide_tsta_o, tide_tend_o)
    tsta_o, tend_o = get_time_limits_o(sat)
    print(sat, trackn, x1, y1, mindist)
    if trackn is None:
        continue
    # zero pad trackn with three digits
    trackn1 = trackn.zfill(3)
    f = f"{data_dir}/{sat}/ctoh.sla.ref.{sat}.nindian.{trackn1}.nc"
    ds = xr.open_dataset(f, decode_times=False)
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lon_coast = lons_track[-1]  # this on coast
    lat_coast = lats_track[-1]  # this on coast
    m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
    c = (lats_track[0]) - m * (lons_track[0])
    x_from_coast = distance.distance((lat_coast, lon_coast), (y1, x1)).km
    # shapely path of the track
    track_path = shapely.geometry.LineString(list(zip(lons_track, lats_track)))
    if len(lons_track) == 0:
        continue
    m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
    angle = np.rad2deg(np.arctan(m))
    c = (lats_track[0]) - m * (lons_track[0])
    ds_smooth = xr.open_dataset(
        f"/home/srinivasu/xtrackm/computed/sla_loess_0.2a/track_sla_along_{sat}_{trackn}_loess_0.2.nc"
    )
    sla_track_smooth = ds_smooth.sla_smooth
    dac_track_smooth = ds_smooth.dac_smooth
    sla_track_p_smooth = sla_track_smooth.sel(x=x_from_coast,
                                              method="nearest",
                                              drop=True)
    dac_track_p_smooth = dac_track_smooth.sel(x=x_from_coast,
                                              method="nearest",
                                              drop=True)
    sla_plus = sla_track_p_smooth + dac_track_p_smooth
    fig = plt.figure(figsize=(width, height))
    ax1 = fig.add_subplot(gs[0, 0])
    # sla_aviso_at.plot(ax=ax1,
    #                   label="extracted aviso grid data",
    #                   linestyle="--")
    sla_tide_interp.plot(ax=ax1, label="tide gauge data", linewidth=4)
    # sla_track_p_smooth.plot(ax=ax1, label=f"track {trackn} {sat}")
    sla_plus.plot(ax=ax1, label=f"track {trackn} {sat} + dac")
    # ax1.set_xlim(tlim1, tlim2)
    ax1.set_xlim(overlap_tsta, overlap_tend)
    ax1.set_ylim(-0.6, 0.6)
    info1 = f"{sat} {trackn} {mindist:.2f}km"
    ax1.text(
        0.5,
        0.9,
        info1,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax1.transAxes,
        fontweight="bold",
        fontsize=14,
    )
    ax1.legend(loc="lower right")

    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    dse.ROSE.sel(ETOPO05_X=slice(*lonlat_box2[:2])).sel(ETOPO05_Y=slice(
        *lonlat_box2[2:])).plot(ax=ax2,
                                add_colorbar=False,
                                add_labels=False,
                                cmap=cmap1)
    decorate_axis(ax2, "", *lonlat_box2, step=2)
    # get intersection of track and box
    intersection = track_path.intersection(poly_box)
    # get the intersection points
    intersection_points = intersection.coords
    # sort intesections points by latitude
    intersection_points = sorted(intersection_points, key=lambda x: x[1])
    # take the first intersection points
    if len(intersection_points) == 0:
        continue
    lonl = intersection_points[0][0]
    latl = intersection_points[0][1]
    print(lonl, latl)
    info = f"{sat} {trackn}"
    lonm, latm = point_on_line_at_distance(m, c, lonl, latl, 2)
    print(lonm, latm)
    ax2.scatter(lons_track, lats_track, linewidths=0.0, s=8, color="r")
    ax2.text(
        lonm,
        latm,
        s=f"{trackn}",
        fontsize=10,
        rotation=angle,
        color="white",
        horizontalalignment="center",
        verticalalignment="center",
        fontweight="bold",
    )
    ax2.add_geometries(
        [poly_box],
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="lightblue",
        alpha=0.5,
    )
    # PSMSL tide gauge location
    ax2.scatter(
        lon_psmsl,
        lat_psmsl,
        marker="*",
        c="yellow",
        s=150,
        edgecolor="black",
    )
    ax2.scatter(
        x1,
        y1,
        marker="x",
        c="cyan",
        s=10,
    )
    # PSMSL tide gauge name
    ax2.text(
        lon_psmsl,
        lat_psmsl,
        s=name_psmsl,
        fontsize=10,
        color="k",
        horizontalalignment="left",
    )
    ax2.set_xlim([*lonlat_box2[:2]])
    ax2.set_ylim([*lonlat_box2[2:]])
    # sla_tide_adj.plot(ax=ax1, label="tide gauge data adjusted")
    ax2.legend(loc="lower right")
    info = f"{sat}"
    ax2.text(0.1, 0.95, info, transform=ax2.transAxes, fontweight="bold")
    plt.savefig(
        f"tseries_loess_adj_allsat_map_box_separate_{sat}_{trackn}.png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close("all")
