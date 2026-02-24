import math
import os

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
# codex resume 019c8e05-4ac3-7611-b3f5-b58c982002e0
ds_coast = xr.open_dataset(
    f"~/allData/topo/GMT_intermediate_coast_distance_01d_track_reg.nc"
)
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")
rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(*TRACKS_REG[2:]))
cmap1 = cmo.cm.diff
COAST_DISTANCE_KM = 100
d = 1.5
out_dir = "track_guvVSoscar_at_intersection_points/coastal_u_timeseries"
os.makedirs(out_dir, exist_ok=True)


def is_close_to_coast(ds_coast, lon, lat, dist_km):
    dist1 = ds_coast.coast_distance.sel(lon=lon, lat=lat, method="nearest").item()
    return abs(dist1) <= dist_km


def format_track_id(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


frac1 = "0.05"
track_dir = "/home/srinivasu/xtrackm/guv/guv_at_intersections/"
oscar_dir = "/home/srinivasu/allData/oscar_0.25/oscar_at_intersection_points"
std_rows = []
print(sats_new)
for sat in sats_new[:]:
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    for i, r in df.iterrows():
        sat1 = r["sat"]
        track_number_self = format_track_id(r["track_self"])
        track_number_other = format_track_id(r["track_other"])
        if track_number_self is None or track_number_other is None:
            continue
        lons_inter1 = r["lons_inter"]
        lats_inter1 = r["lats_inter"]
        x_from_coast_self1 = r["x_from_coast_self"]
        x_from_coast_other1 = r["x_from_coast_other"]
        angle_acute = r["angle_acute"]
        angle_obtuse = r["angle_obtuse"]
        lon1 = lons_inter1
        lat1 = lats_inter1
        if not is_close_to_coast(ds_coast, lon1, lat1, COAST_DISTANCE_KM):
            continue
        if abs(lat1) < 2:
            continue
        track_uv = f"{track_dir}/guv_at_intersection_{sat1}_{track_number_self}_{track_number_other}_{frac1}.nc"
        oscar_uv = f"{oscar_dir}/oscar_uv_at_intersection_{sat1}_{track_number_self}_{track_number_other}.nc"
        f_self = f"../data/{sat1}/ctoh.sla.ref.{sat1}.nindian.{track_number_self.zfill(3)}.nc"
        f_other = f"../data/{sat1}/ctoh.sla.ref.{sat1}.nindian.{track_number_other.zfill(3)}.nc"
        ds_track = xr.open_dataset(track_uv)
        ds_oscar = xr.open_dataset(oscar_uv)
        ds_self = xr.open_dataset(f_self, engine="netcdf4", decode_times=False)
        ds_other = xr.open_dataset(f_other, engine="netcdf4", decode_times=False)

        gu = ds_track.gu
        oscar_u = ds_oscar.u
        tfreq = sats_tfreq.get(sat1, None)
        # if tfreq is not None:
        #     oscar_u = oscar_u.resample(time=f"{tfreq}D").mean()
        # oscar_u = oscar_u.interp(time=gu.time)
        oscar_u = oscar_u.sel(time=gu.time, method="nearest")
        std_rows.append(
            {
                "sat": sat1,
                "track_self": track_number_self,
                "track_other": track_number_other,
                "std_oscar": oscar_u.std(dim="time", skipna=True).item(),
                "std_track": gu.std(dim="time", skipna=True).item(),
            }
        )

        lons_track_self = ds_self.lon.values
        lats_track_self = ds_self.lat.values
        lons_track_other = ds_other.lon.values
        lats_track_other = ds_other.lat.values

        lon_equat_self = lons_track_self[0]
        lat_equat_self = lats_track_self[0]
        lon_coast_self = lons_track_self[-1]
        lat_coast_self = lats_track_self[-1]
        slope_self = (lat_coast_self - lat_equat_self) / (lon_coast_self - lon_equat_self)
        angle_self = np.rad2deg(np.arctan(slope_self))

        lon_equat_other = lons_track_other[0]
        lat_equat_other = lats_track_other[0]
        lon_coast_other = lons_track_other[-1]
        lat_coast_other = lats_track_other[-1]
        slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other - lon_equat_other)
        angle_other = np.rad2deg(np.arctan(slope_other))

        fig = plt.figure(figsize=(12, 10))
        ax_map = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        ax_ts = fig.add_subplot(2, 1, 2)
        rose_cut.plot(
            ax=ax_map, add_colorbar=False, add_labels=False, cmap=cmap1
        )
        decorate_axis(ax_map, "", *TRACKS_REG, step=5)
        ax_map.grid()
        ax_map.plot(
            lon1,
            lat1,
            c="r",
            marker="o",
            markersize=6,
            markerfacecolor="red",
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        ax_map.scatter(lons_track_self, lats_track_self, marker=".", color="c", s=4)
        ax_map.scatter(lons_track_other, lats_track_other, marker=".", color="c", s=4)

        lonm, latm = get_point_at_distance(
            lon_equat_self, lat_equat_self, lon_coast_self, lat_coast_self, d
        )
        if is_within_region(lonm, latm, *TRACKS_REG):
            ax_map.text(lonm, latm, s=track_number_self, fontsize=12, rotation=angle_self, color="k")

        lonm, latm = get_point_at_distance(
            lon_equat_other, lat_equat_other, lon_coast_other, lat_coast_other, d
        )
        if is_within_region(lonm, latm, *TRACKS_REG):
            ax_map.text(lonm, latm, s=track_number_other, fontsize=12, rotation=angle_other, color="k")

        oscar_u.plot(ax=ax_ts, color="b", linewidth=2, label="oscar u")
        gu.plot(ax=ax_ts, color="r", linewidth=2, label="track u")
        ax_ts.legend()
        ax_ts.set_ylabel("u (m/s)")
        ax_ts.set_xlabel("Time")
        ax_ts.set_title(
            f"{sat1}: ({lon1:.2f}, {lat1:.2f}), coast <= {COAST_DISTANCE_KM} km"
        )

        fig.tight_layout()
        plt.savefig(
            f"{out_dir}/u_timeseries_{sat1}_{track_number_self}_{track_number_other}.png",
            dpi=180,
        )
        # plt.show()
        plt.close(fig)

        ds_track.close()
        ds_oscar.close()
        ds_self.close()
        ds_other.close()

ds_coast.close()
dse.close()
df_std = pd.DataFrame(
    std_rows, columns=["sat", "track_self", "track_other", "std_oscar", "std_track"]
)
df_std.to_csv(f"{out_dir}/std_oscar_track_u_coastal.csv", index=False)
