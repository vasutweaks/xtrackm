import os

import xarray as xr
from rich.console import Console
from shapely.geometry import LineString, Point
from tools_xtrackd import *

console = Console()

topo_dir = "/home/srinivasu/allData/topo/"
dsc = xr.open_dataset(f"{topo_dir}"
                      f"GMT_intermediate_coast_distance_01d_track_reg.nc")
print(dsc)
dist_to_coast = dsc.coast_dist


def get_closest_point_index(drifter_path: LineString,
                            intersection_point: Point) -> int:
    distances = [
        Point(p).distance(intersection_point) for p in drifter_path.coords
    ]
    nearest_index = np.argmin(distances)
    return nearest_index


def rotate_to_gc(u, v, angle_r):
    u1 = u * np.sin(angle_r) - v * np.cos(angle_r)
    return u1


chunk = "15001_current"
dist_threshold = 200
computed_loc = f"/home/srinivasu/slnew/xtrackd/computed/"
data_loc = f"/home/srinivasu/allData/drifter1/"

ds_ekman = xr.open_dataset(
    f"{computed_loc}/full_series_ekman_uv_ccmp3.0_cdo.nc"
)
ekman_u = ds_ekman.ekman_u
ekman_v = ds_ekman.ekman_v

count = 0
for sat in sats[:]:
    track_tsta_o, track_tend_o = get_time_limits_o(sat)
    ekman_u_cut = ekman_u.sel(time=slice(track_tsta_o, track_tend_o))
    ekman_v_cut = ekman_v.sel(time=slice(track_tsta_o, track_tend_o))
    df_list = []
    df_e_list = []
    for fd in sorted(
            glob.glob(f"{data_loc}/netcdf_*/"
                      f"track_reg/drifter_6h_*.nc")):
        basename1 = os.path.basename(fd)
        ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
        byte_id = ds_d.ID.values[0]
        str_id = byte_id.decode("utf-8")
        drifter_id = str_id
        drift_tsta_o1, drift_tend_o1 = (
            ds_d.start_date.values[0],
            ds_d.end_date.values[0],
        )
        # drift_tsta_o, drift_tend_o = pd.to_datetime(drift_tsta_o1), pd.to_datetime(
        #     drift_tend_o1)
        drift_tsta_o, drift_tend_o = n64todatetime(drift_tsta_o1), n64todatetime(
            drift_tend_o1)
        overlap_tsta_o, overlap_tend_o = overlap_dates(track_tsta_o, track_tend_o,
                                                   drift_tsta_o, drift_tend_o)
        if overlap_tsta_o is None or overlap_tend_o is None:
            console.print(f"{overlap_tsta_o}, {overlap_tend_o}", style="red")
            continue
        # lons_drift = ds_d.longitude.isel(traj=0).values
        # lats_drift = ds_d.latitude.isel(traj=0).values
        times_drift = ds_d.time.isel(traj=0).values
        # times_drift_o = [pd.to_datetime(t) for t in times_drift]
        # times_drift_o1 = [n64todatetime(t) for t in times_drift]
        ve_da = drifter_time_asn(ds_d, var_str="ve")
        vn_da = drifter_time_asn(ds_d, var_str="vn")
        lons_da = drifter_time_asn(ds_d, var_str="longitude")
        lats_da = drifter_time_asn(ds_d, var_str="latitude")
        lons_da1 = lons_da.resample(time="1D").mean()
        lats_da1 = lats_da.resample(time="1D").mean()
        lons_da2 = lons_da1.rolling(time=3).mean()
        lats_da2 = lats_da1.rolling(time=3).mean()
        lons_da3 = lons_da2.ffill(dim="time").bfill(dim="time")
        lats_da3 = lats_da2.ffill(dim="time").bfill(dim="time")

        ve_da1 = ve_da.resample(time="1D").mean()
        vn_da1 = vn_da.resample(time="1D").mean()
        ve_da2 = ve_da1.rolling(time=3).mean()
        vn_da2 = vn_da1.rolling(time=3).mean()
        times_drift_o2 = ve_da2.time
        lons_drift = lons_da3.values
        lats_drift = lats_da3.values
        # times_drift_o = times_drift_o1[::4]
        times_drift_o = [n64todatetime(t) for t in times_drift_o2]

        # ve_drift = ds_d.ve.isel(traj=0).values
        # vn_drift = ds_d.vn.isel(traj=0).values
        # ve_drift = ve_da2.values
        # vn_drift = vn_da2.values

        # k = 0
        # for lon, lat, time in zip(lons_drift, lats_drift, times_drift):
        #     print(k, lon, lat, time)
        #     k = k + 1

        lon_start = ds_d.start_lon.isel(traj=0).values
        lon_end = ds_d.end_lon.isel(traj=0).values
        lat_start = ds_d.start_lat.isel(traj=0).values
        lat_end = ds_d.end_lat.isel(traj=0).values
        drifter_path = sg.LineString([
            (lon, lat) for lon, lat in zip(lons_drift, lats_drift)
        ])
        for ft in sorted(glob.glob(f"../data/ctoh.sla.ref.{sat}.nindian.*.nc")):
            basename_t = os.path.basename(ft)
            trackn = basename_t.split(".")[5]
            print(sat, drifter_id, trackn)
            ds_t = xr.open_dataset(ft, decode_times=False, engine="h5netcdf")
            track_number = ds_t.pass_number
            fs = (f"{computed_loc}/sla_loess_0.2a/"
                  f"track_sla_along_{sat}_{track_number}_loess_0.2.nc")
            ds_smooth = xr.open_dataset(fs, engine="h5netcdf")
            sla_smooth = ds_smooth.sla_smooth
            gc = compute_geostrophy_gc(ds_t, sla_smooth)
            lons_track = ds_t.lon.values
            lats_track = ds_t.lat.values
            lon_coast = lons_track[-1]
            lat_coast = lats_track[-1]
            lon_equat = lons_track[0]
            lat_equat = lats_track[0]
            angle_r2 = np.arctan2(lat_coast - lat_equat, lon_coast - lon_equat)
            ve_da_r = rotate_to_gc(ve_da2, vn_da2, angle_r2)
            ve_drift = ve_da_r.values
            # gc = gc.sel(time=slice(overlap_tsta, overlap_tend))
            track_path = sg.LineString([
                (lon, lat) for lon, lat in zip(lons_track, lats_track)
            ])
            if track_path.intersects(drifter_path):
                intersection1 = track_path.intersection(drifter_path)
                if intersection1.geom_type == "Point":
                    pointa = intersection1
                    ii = get_closest_point_index(drifter_path, pointa)
                    time11 = times_drift_o[ii]
                    # time11 = times_drift_o.isel(time=ii).values
                    console.print(
                        f"intersection point: {pointa}, closestindex = {ii}",
                        style="blue",
                    )
                    console.print(
                        f"lon_at_index = {lons_drift[ii]}, lat_at_index = "
                        f"{lats_drift[ii]}, time_at_index = {time11}",
                        style="blue",
                    )
                    is_between = overlap_tsta_o <= time11 <= overlap_tend_o
                    if is_between:
                        ve11 = ve_drift[ii]
                        jj = get_closest_point_index(track_path, pointa)
                        x1, y1 = lons_track[jj], lats_track[jj]
                        dist_to_coast_xy = dist_to_coast.sel(
                            lon=x1, lat=y1, method="nearest").item()
                        print(f"dist_to_coast = {dist_to_coast_xy} -------------------------")
                        if dist_to_coast_xy < dist_threshold:
                            x_from_coast = distance.distance(
                                (y1, x1), (lat_coast, lon_coast)).m
                            gc1 = gc.interp(x=x_from_coast)
                            gc11 = gc1.interp(time=time11).item()
                            ekman_at_u = ekman_u_cut.interp(lon=x1).interp(lat=y1)
                            ekman_at_v = ekman_v_cut.interp(lon=x1).interp(lat=y1)
                            ekman_at_u_r = rotate_to_gc(ekman_at_u, ekman_at_v, angle_r2)
                            ek11 = ekman_at_u_r.interp(time=time11).item()
                            gc_ek = gc11 + ek11
                            console.print(f"ve at {time11} = {ve11}",
                                          style="yellow")
                            console.print(f"gc at {time11} = {gc11}",
                                          style="yellow")
                            df = pd.DataFrame({
                                "ve": [ve11],
                                "gc": [gc11],
                                "time": [time11],
                                "lon": x1,
                                "lat": y1,
                            })
                            df_e = pd.DataFrame({
                                "ve": [ve11],
                                "gc": [gc_ek],
                                "time": [time11],
                                "lon": x1,
                                "lat": y1,
                            })
                            # df = pd.concat([df, new_row], ignore_index=True)
                            df_list.append(df)
                            df_e_list.append(df_e)
                            ii = ii + 1
                elif intersection1.geom_type == "MultiPoint":
                    for pointa in (intersection1.geoms
                                    ):  # Use .geoms to iterate over MultiPoint
                        ii = get_closest_point_index(drifter_path, pointa)
                        time11 = times_drift_o[ii]
                        console.print(
                            f"intersection point: {pointa}, closestindex = {ii}",
                            style="blue",
                        )
                        console.print(
                            f"lon_at_index = {lons_drift[ii]}, lat_at_index = "
                            f"{lats_drift[ii]}, time_at_index = {time11}",
                            style="blue",
                        )
                        is_between = overlap_tsta_o <= time11 <= overlap_tend_o
                        if is_between:
                            ve11 = ve_drift[ii]
                            jj = get_closest_point_index(track_path, pointa)
                            x1, y1 = lons_track[jj], lats_track[jj]
                            dist_to_coast_xy = dist_to_coast.sel(
                                lon=x1, lat=y1, method="nearest").item()
                            print(f"dist_to_coast = {dist_to_coast_xy} -------------------------")
                            if dist_to_coast_xy < dist_threshold:
                                x_from_coast = distance.distance(
                                    (y1, x1), (lat_coast, lon_coast)).m
                                gc1 = gc.interp(x=x_from_coast)
                                gc11 = gc1.interp(time=time11).item()
                                ekman_at_u = ekman_u_cut.interp(lon=x1).interp(lat=y1)
                                ekman_at_v = ekman_v_cut.interp(lon=x1).interp(lat=y1)
                                ekman_at_u_r = rotate_to_gc(ekman_at_u, ekman_at_v, angle_r2)
                                ek11 = ekman_at_u_r.interp(time=time11).item()
                                gc_ek = gc11 + ek11
                                console.print(f"ve at {time11} = {ve11}",
                                              style="yellow")
                                console.print(f"gc at {time11} = {gc11}",
                                              style="yellow")
                                df = pd.DataFrame({
                                    "ve": [ve11],
                                    "gc": [gc11],
                                    "time": [time11],
                                    "lon": x1,
                                    "lat": y1,
                                })
                                df_e = pd.DataFrame({
                                    "ve": [ve11],
                                    "gc": [gc_ek],
                                    "time": [time11],
                                    "lon": x1,
                                    "lat": y1,
                                })
                                    # df = pd.concat([df, new_row], ignore_index=True)
                                df_list.append(df)
                                df_e_list.append(df_e)
                                count = count + 1

    df_out = pd.concat(df_list, ignore_index=True)
    df_e_out = pd.concat(df_e_list, ignore_index=True)
    df_out.to_csv(f"ve_gc_all_{sat}.within.isobath.{dist_threshold}km.csv")
    df_e_out.to_csv(f"ve_gc_ek_all_{sat}.within.isobath.{dist_threshold}km.csv")
