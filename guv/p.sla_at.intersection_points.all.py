import cmocean as cmo
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import sys

cmap1 = cmo.cm.diff
sat = "TP+J1+J2+J3+S6A"
d = 1.5
height, width = 9, 9


df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")

# sys.exit()
track_self = df["track_self"].apply(lambda x: str(x)).values
track_other = df["track_other"].apply(lambda x: str(x)).values
lons_inter = df["lons_inter"].values
lats_inter = df["lats_inter"].values
x_from_coast_self = df["x_from_coast_self"].values
x_from_coast_other = df["x_from_coast_other"].values
angle_acute = df["angle_acute"].values
angle_obtuse = df["angle_obtuse"].values
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset

for i in range(len(df)):
    track_number_self = track_self[i]
    track_number_other = track_other[i]
    f_self = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
    f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
    lon1 = lons_inter[i]
    lat1 = lats_inter[i]
    x_from_coast_self1 = x_from_coast_self[i]
    x_from_coast_other1 = x_from_coast_other[i]
    angle_acute1 = angle_acute[i]
    angle_obtuse1 = angle_obtuse[i]
    ds_self = xr.open_dataset(f_self, decode_times=False)
    sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
    x_self = sla_self.x.values
    ds_other = xr.open_dataset(f_other, engine="h5netcdf", decode_times=False)
    sla_other = track_dist_time_asn(ds_other, var_str="sla", units_in="m")
    x_other = sla_other.x.values
    lons_track_self = ds_self.lon.values
    lats_track_self = ds_self.lat.values
    lon_coast_self = lons_track_self[-1]  # this on coast
    lat_coast_self = lats_track_self[-1]  # this on coast
    lon_equat_self = lons_track_self[0]  # this on equator
    lat_equat_self = lats_track_self[0]  # this on equator
    slope_self = (lats_track_self[-1] - lats_track_self[0]) / (
        lons_track_self[-1] - lons_track_self[0])
    angle_self = np.rad2deg(np.arctan(slope_self))
    track_path_self = sg.LineString(zip(lons_track_self, lats_track_self))
    lons_track_other = ds_other.lon.values
    lats_track_other = ds_other.lat.values
    lon_equat_other = lons_track_other[0]
    lat_equat_other = lats_track_other[0]
    lon_coast_other = lons_track_other[-1]
    lat_coast_other = lats_track_other[-1]
    slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other - lon_equat_other)
    angle_other = np.rad2deg(np.arctan(slope_other))
    track_path_other = sg.LineString(zip(lons_track_other, lats_track_other))
    # if track_path_self.intersects(track_path_other):
    lon_inter_self, lat_inter_self = lonlat_at_x(ds_self, x_self, x_from_coast_self1)
    lon_inter_other, lat_inter_other = lonlat_at_x(ds_other, x_other, x_from_coast_other1)
    # print(f"{lon_inter_self:.2f} {lon_inter_other:.2f} {lon1:.2f}")
    if is_land(lon1, lat1):
        print(f"{lon1:.2f} {lat1:.2f} is on land")
        continue
    print(f"{lat_inter_self:.2f} {lat_inter_other:.2f} {lat1:.2f}")
    sla_self_at = sla_self.sel(x=x_from_coast_self1,
                               method="nearest",
                               drop=True)
    sla_other_at = sla_other.sel(x=x_from_coast_other1,
                                 method="nearest",
                                 drop=True)
    sla_other_at = sla_other_at.interp(time=sla_self_at.time)
    # check if lon1, lat1 is on land
    fig1, ax1 = plt.subplots(1,
                            1,
                            figsize=(width, height),
                            layout="constrained")
    sla_self_at.plot(ax=ax1, label=f"pass {track_number_self}")
    sla_other_at.plot(ax=ax1, label=f"pass {track_number_other}")
    # text coordinate of intersection point
    plt.text(
        0.15,
        0.95,
        f"intersection point ({lon1:.2f}, {lat1:.2f})",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    plt.legend()
    fig2, ax2 = plt.subplots(
        1,
        1,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(width, height),
        layout="constrained",
    )
    dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
        *TRACKS_REG[2:])).plot(ax=ax2,
                               add_colorbar=False,
                               add_labels=False,
                               cmap=cmap1)
    decorate_axis(ax2, "", *TRACKS_REG)
    ax2.grid()
    ax2.scatter(
        lons_track_self,
        lats_track_self,
        marker=".",
        color="c",
        s=4,
    )
    ax2.scatter(
        lons_track_other,
        lats_track_other,
        marker=".",
        color="c",
        s=4,
    )
    lonm, latm = get_point_at_distance(lon_equat_self, lat_equat_self,
                                       lon_coast_self, lat_coast_self, d)
    if is_within_region(lonm, latm, *TRACKS_REG):
        plt.text(
            lonm,
            latm,
            s=track_number_self,
            fontsize=14,
            rotation=angle_self,
            color="w",
        )
    lonm, latm = get_point_at_distance(lon_equat_other, lat_equat_other, lon_coast_other, lat_coast_other, d)
    if is_within_region(lonm, latm, *TRACKS_REG):
        plt.text(
            lonm,
            latm,
            s=track_number_other,
            fontsize=14,
            rotation=angle_other,
            color="w",
        )
    fig1.savefig(
        f"sla_at_intersection_points/sla_timeseries_{sat}_{track_number_self}_{track_number_other}.png",
    )
    fig2.savefig(
        f"sla_at_intersection_points/sla_map_{sat}_{track_number_self}_{track_number_other}.png",
    )
    plt.show()
    plt.close("all")

dse.close()
