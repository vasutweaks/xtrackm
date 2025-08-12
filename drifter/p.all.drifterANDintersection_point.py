import os

import numpy as np
import xarray as xr
from tools_xtrackm import *
# import Polygon
from shapely.geometry.polygon import Polygon


def closest_index(lons_drift, lats_drift, lon1, lat1):
    dist = np.sqrt((lons_drift - lon1) ** 2 + (lats_drift - lat1) ** 2)
    return np.argmin(dist)


def is_within_region_any(lons, lats, lon_min, lon_max, lat_min, lat_max, include_boundary=True):
    """
    Return True if any (lon, lat) point falls inside the rectangular region.
    include_boundary=True counts points on the edges/corners as inside.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.shape != lats.shape:
        raise ValueError("lons and lats must have the same shape")
    if include_boundary:
        lons_in = (lons >= lon_min) & (lons <= lon_max)
        lats_in = (lats >= lat_min) & (lats <= lat_max)
    else:
        lons_in = (lons > lon_min) & (lons < lon_max)
        lats_in = (lats > lat_min) & (lats < lat_max)
    inside_mask = lons_in & lats_in
    hit = np.any(inside_mask)
    if hit:
        print("At least one point is inside the region.")
    else:
        print("No points are inside the region.")
    return bool(hit)


df_all_list = []
for sat in sats_new:
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    df_all_list.append(df)
df_all = pd.concat(df_all_list, ignore_index=True)

d = 1.5
data_loc = f"/home/srinivasu/allData/drifter1/"
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
rose = dse.ROSE
cmap1 = "Greys_r"
dist_threshold = 1 # in degrees
computed_loc = f"/home/srinivasu/slnew/xtrackm/computed/"
chunk = "15001_current"

found = False
for i, r in df_all.iterrows():
    sat1 = r["sat"]
    if sat1 != "S3A":
        continue
    track_tsta_o, track_tend_o = get_time_limits_o(sat1)
    track_number_self = str(r["track_self"])
    track_number_other = str(r["track_other"])
    lons_inter1 = r["lons_inter"]
    lats_inter1 = r["lats_inter"]
    x_from_coast_self1 = r["x_from_coast_self"]
    x_from_coast_other1 = r["x_from_coast_other"]
    angle_acute = r["angle_acute"]
    angle_obtuse = r["angle_obtuse"]
    lon1 = lons_inter1
    lat1 = lats_inter1
    if abs(lat1) < 2:
        continue
    print(
        sat1,
        track_number_self,
        track_number_other,
        lons_inter1,
        lats_inter1,
        x_from_coast_self1,
        x_from_coast_other1,
        angle_acute,
        angle_obtuse,
    )
    lon_inter_box_min = lon1 - dist_threshold
    lon_inter_box_max = lon1 + dist_threshold
    lat_inter_box_min = lat1 - dist_threshold
    lat_inter_box_max = lat1 + dist_threshold
    box = Polygon([(lon_inter_box_min, lat_inter_box_min), (lon_inter_box_max, lat_inter_box_min), (lon_inter_box_max, lat_inter_box_max), (lon_inter_box_min, lat_inter_box_max)])
    f_self = (
        f"../data/{sat1}/ctoh.sla.ref.{sat1}.nindian.{track_number_self.zfill(3)}.nc"
    )
    f_other = f"../data/{sat1}/ctoh.sla.ref.{sat1}.nindian.{track_number_other.zfill(3)}.nc"
    f_self_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat1}_{track_number_self}_loess_0.2.nc"
    f_other_smooth = f"../computed/sla_loess_0.2a/track_sla_along_{sat1}_{track_number_other}_loess_0.2.nc"
    ds_self = xr.open_dataset(f_self, engine="h5netcdf", decode_times=False)
    ds_self_smooth = xr.open_dataset(f_self_smooth)
    sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
    sla_self_smooth = ds_self_smooth.sla_smooth
    ds_other = xr.open_dataset(f_other, engine="h5netcdf", decode_times=False)
    ds_other_smooth = xr.open_dataset(f_other_smooth)
    sla_other = track_dist_time_asn(ds_other, var_str="sla", units_in="m")
    sla_other_smooth = ds_other_smooth.sla_smooth
    lons_track_self = ds_self.lon.values
    lats_track_self = ds_self.lat.values
    lon_coast_self = lons_track_self[-1]  # this on coast
    lat_coast_self = lats_track_self[-1]  # this on coast
    lon_equat_self = lons_track_self[0]  # this on equator
    lat_equat_self = lats_track_self[0]  # this on equator
    slope_self = (lats_track_self[-1] -
                  lats_track_self[0]) / (lons_track_self[-1] - lons_track_self[0])
    angle_self = np.rad2deg(np.arctan(slope_self))
    lons_track_other = ds_other.lon.values
    lats_track_other = ds_other.lat.values
    lon_equat_other = lons_track_other[0]
    lat_equat_other = lats_track_other[0]
    lon_coast_other = lons_track_other[-1]
    lat_coast_other = lats_track_other[-1]
    slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                     lon_equat_other)
    angle_other = np.rad2deg(np.arctan(slope_other))
    print(f"{angle_self} {angle_other} -------------------")
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    ax2 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    rose.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
        *TRACKS_REG[2:])).plot(ax=ax2,
                               add_colorbar=False,
                               add_labels=False,
                               cmap=cmap1)
    decorate_axis(ax2, "", *TRACKS_REG, step=5)
    ax2.grid()
    ax2.plot(
        lon1,
        lat1,
        c="r",
        marker="o",
        markersize=4,
        markerfacecolor="red",
    )
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
    ax2.add_geometries([box],
                       ccrs.PlateCarree(),
                       edgecolor="g",
                       facecolor="none",
                       linewidth=2)
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
    lonm, latm = get_point_at_distance(lon_equat_other, lat_equat_other,
                                   lon_coast_other, lat_coast_other, d)
    if is_within_region(lonm, latm, *TRACKS_REG):
        plt.text(
            lonm,
            latm,
            s=track_number_other,
            fontsize=14,
            rotation=angle_other,
            color="w",
        )
    for fd in sorted(
        glob.glob(f"{data_loc}/netcdf_{chunk}/track_reg/drifter_6h_*.nc")
    ):
        print(fd)
        basename1 = os.path.basename(fd)
        ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
        byte_id = ds_d.ID.values[0]
        str_id = byte_id.decode("utf-8")
        drifter_id = str_id
        drift_tsta_o1, drift_tend_o1 = (
            ds_d.start_date.values[0],
            ds_d.end_date.values[0],
        )
        drift_tsta_o, drift_tend_o = n64todatetime(drift_tsta_o1), n64todatetime(
            drift_tend_o1)
        overlap_tsta_o, overlap_tend_o = overlap_dates(track_tsta_o, track_tend_o,
                                                       drift_tsta_o, drift_tend_o)
        print(f"overlap period {overlap_tsta_o} {overlap_tend_o}")
        if overlap_tsta_o is None or overlap_tend_o is None:
            continue
        times_drift = ds_d.time.isel(traj=0).values
        ve_da = drifter_time_asn(ds_d, var_str="ve")
        vn_da = drifter_time_asn(ds_d, var_str="vn")
        lons_da = drifter_time_asn(ds_d, var_str="longitude")
        lats_da = drifter_time_asn(ds_d, var_str="latitude")
        lons_da2 = lons_da.rolling(time=3).mean()
        lats_da2 = lats_da.rolling(time=3).mean()
        lons_da3 = lons_da2.ffill(dim="time").bfill(dim="time")
        lats_da3 = lats_da2.ffill(dim="time").bfill(dim="time")

        lons_drift = lons_da3.values
        lats_drift = lats_da3.values
        if is_within_region_any(lons_drift,
                            lats_drift,
                            lon_inter_box_min,
                            lon_inter_box_max,
                            lat_inter_box_min,
                            lat_inter_box_max):

            nidx = closest_index(lons_drift, lats_drift, lon1, lat1)
            close_drift_lon = lons_drift[nidx]
            close_drift_lat = lats_drift[nidx]
            ax2.scatter(
                lons_drift,
                lats_drift,
                marker=".",
                color="y",
                s=4,
            )
            ax2.plot(
                close_drift_lon,
                close_drift_lat,
                marker="x",
                markersize=8,
                color="k",
            )
            plt.show()
            plt.close()
            found = True
            break
