import os
import sys

import numpy as np
import xarray as xr
import xskillscore as xs
from tools_xtrackd import *

np.seterr(invalid="ignore")

radar_str = "TN"
width, height = 13, 6
REG = radar_extents[radar_str + "_BLOCK"]

gdf = gpd.read_file(f"polygon_shapefile_around_{radar_str}.shp")
polygon = gdf.geometry.iloc[0]

radar_tsta_o, radar_tend_o = radar_times[radar_str]

dsr = xr.open_dataset(
    f"/home/srinivasu/allData/radar/{radar_str}Codar_daily.nc")
# print(dsr)
dsr = dsr.rename({
    "XAXS": "longitude",
    "YAXS": "latitude",
    "ZAXS": "lev",
    "TAXIS1D": "time"
})
# print(dsr)
u_radar = 0.01 * dsr.U_RADAR.isel(lev=0, drop=True)
v_radar = 0.01 * dsr.V_RADAR.isel(lev=0, drop=True)
s_radar = 0.01 * dsr.S_RADAR.isel(lev=0, drop=True)
sizes = u_radar.sizes
ln = sizes["time"]

ds_ekman = xr.open_dataset(f"../computed/full_series_ekman_uv_ccmp3.0_cdo.nc")
# print(ds_ekman)
ekman_u = ds_ekman.ekman_u
ekman_v = ds_ekman.ekman_v

ii = 0
corr_means = []
corr_ek_means = []
rmse_means = []
rmse_ek_means = []

for sat in sats[:]:
    corr_list = []
    rmse_list = []
    p_list = []
    corr_ek_list = []
    rmse_ek_list = []
    pe_list = []
    track_tsta_o, track_tend_o = get_time_limits_o(sat)
    overlap_tsta, overlap_tend = overlap_dates(track_tsta_o, track_tend_o,
                                               radar_tsta_o, radar_tend_o)
    eu_cut = ekman_u.sel(time=slice(overlap_tsta, overlap_tend))
    ru_cut = u_radar.sel(time=slice(overlap_tsta, overlap_tend))
    ev_cut = ekman_v.sel(time=slice(overlap_tsta, overlap_tend))
    rv_cut = v_radar.sel(time=slice(overlap_tsta, overlap_tend))
    # print(sat, overlap_tsta, overlap_tend, "-------------------------")
    for fs in sorted(
            glob.glob(
                f"../computed/sla_loess_0.2a/track_sla_along_{sat}_*_loess_0.2.nc"
            )):
        bn = os.path.basename(fs)
        bn_cut = bn.split("_")
        track_number = bn_cut[4]
        # print(track_number)
        dss = xr.open_dataset(fs)
        sla_smooth = dss.sla_smooth

        f_sla = f"../data/ctoh.sla.ref.{sat}.nindian.{track_number}.nc"
        ds = xr.open_dataset(f_sla, decode_times=False)
        # sla = track_dist_time_asn(ds, var_str="sla")
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        lon_equat = lons_track[0]  # this is a point on equator
        lat_equat = lats_track[0]
        lon_coast = lons_track[-1]
        lat_coast = lats_track[-1]
        m = (lat_coast - lat_equat) / (lon_coast - lon_equat)
        # angle_r = np.arctan(m)
        angle_r = np.arctan2(lat_coast - lat_equat, lon_coast - lon_equat)
        angle = np.rad2deg(angle_r)
        print(f"angle is {angle_r}")
        gc = compute_geostrophy_gc(ds, sla_smooth)

        track_path = sg.LineString([
            (lon, lat) for lon, lat in zip(lons_track, lats_track)
        ])

        if polygon.intersects(track_path):
            intersection_points = track_path.intersection(polygon)
            points = [sg.Point(coord) for coord in intersection_points.coords]

            gc_cut = gc.sel(time=slice(overlap_tsta, overlap_tend))
            # eu_interp = eu_cut.interp(time=gu_cut.time)
            # ev_interp = ev_cut.interp(time=gu_cut.time)
            # ru_interp = ru_cut.interp(time=gu_cut.time)

            for point in points:
                # print(point.x, point.y)
                x_from_coast = distance.distance((point.y, point.x),
                                                 (lat_coast, lon_coast)).m
                # gu_at = gu_cut.sel(x=x_from_coast, method="nearest")
                gc_at = gc_cut.interp(x=x_from_coast)
                ekman_u_at = eu_cut.sel(lon=point.x,
                                           lat=point.y,
                                           method="nearest")
                ekman_v_at = ev_cut.sel(lon=point.x,
                                           lat=point.y,
                                           method="nearest")
                ru_at1 = ru_cut.sel(longitude=point.x,
                                      latitude=point.y,
                                      method="nearest")
                rv_at1 = rv_cut.sel(longitude=point.x,
                                      latitude=point.y,
                                      method="nearest")
                rc = ru_at1 * np.sin(angle_r) - rv_at1 * np.cos(
                    angle_r)
                ec = ekman_u_at * np.sin(angle_r) - ekman_v_at * np.cos(
                    angle_r)
                # rc_at_filter = butter_lowpass_1d(rc, cutoff_days=3, order=4)
                # ru_at = ru_at_filter.interp(time=gu_at.time)
                rc_at = rc.interp(time=gc_at.time)
                ec_at = ec.interp(time=gc_at.time)
                gc_ek = gc_at + ec_at
                # gu_cut.plot(label="gu")
                # gu_ek.plot(label="gu + ekman")
                # ru_at_interp.plot(label="radar")
                # plt.legend()
                # plt.show()
                p = xs.pearson_r_p_value(gc_at, rc_at, dim="time",
                                         skipna=True).item()
                corr = xs.pearson_r(gc_at, rc_at, dim="time",
                                    skipna=True).item()
                rmse = xs.rmse(gc_at, rc_at, dim="time", skipna=True).item()
                # print(rmse, "------------------")
                # corr_valid = corr.where(p < 0.05)
                pe = xs.pearson_r_p_value(gc_ek, rc_at, dim="time",
                                          skipna=True).item()
                corr_ek = xs.pearson_r(gc_ek, rc_at, dim="time",
                                       skipna=True).item()
                rmse_ek = xs.rmse(gc_ek, rc_at, dim="time", skipna=True).item()
                # print(rmse_ek, "------------------")
                # corr_ek_valid = corr_ek.where(pe < 0.05)
                p_list.append(p)
                pe_list.append(pe)
                corr_list.append(corr)
                corr_ek_list.append(corr_ek)
                rmse_list.append(rmse)
                rmse_ek_list.append(rmse_ek)
    corr_valid = [x for x, p in zip(corr_list, p_list) if p < 0.05]
    corr_ek_valid = [x for x, pe in zip(corr_ek_list, pe_list) if pe < 0.05]
    # corr_valid = corr_list
    # print(f"len of corr_valid = {len(corr_valid)}")
    # corr_ek_valid = corr_ek_list
    # if len(corr_valid) > 0 and len(corr_ek_valid) > 0:
    # print(f"mean valid correlation of {sat}: {np.nanmean(corr_valid)}")
    # print(f"mean valid correlation+ek of {sat}: {np.nanmean(corr_ek_valid)}")
    corr_means.append(np.nanmean(corr_valid))
    corr_ek_means.append(np.nanmean(corr_ek_valid))
    print(f"mean rmse of {sat}: {np.nanmean(rmse_list)}")
    print(f"mean rmse+ek of {sat}: {np.nanmean(rmse_ek_list)}")
    rmse_means.append(np.nanmean(rmse_list))
    rmse_ek_means.append(np.nanmean(rmse_ek_list))

# print("overall mean across all sats excluding GFO:")
# print(np.nanmean(corr_means[1:]), np.nanmean(corr_ek_means[1:]))
print("overall mean across all sats excluding GFO:")
print(np.nanmean(rmse_means[1:]), np.nanmean(rmse_ek_means[1:]))

sys.exit()
