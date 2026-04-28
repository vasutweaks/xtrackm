import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from tools_xtrackm import *

ioff = 3
frac1 = "0.05"
method = "loess"
out_dir = "../computed/sla_corr_at_intersection_points"
os.makedirs(out_dir, exist_ok=True)
print(sats_new)
# sats_new = ["S3B", "HY2A", "HY2B", "GFO"]
for sat in sats_new[:]:
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    corr_records = []
    for i, r in df.iterrows():
        sat1 = r["sat"]
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
        print(
            sat,
            track_number_self,
            track_number_other,
            lons_inter1,
            lats_inter1,
            x_from_coast_self1,
            x_from_coast_other1,
            angle_acute,
            angle_obtuse,
        )
        f_self = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
        f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
        f_self_smooth = f"../computed/sla_{method}_{frac1}a/track_sla_along_{sat}_{track_number_self}_{method}_{frac1}.nc"
        f_other_smooth = f"../computed/sla_{method}_{frac1}a/track_sla_along_{sat}_{track_number_other}_{method}_{frac1}.nc"
        ds_self = xr.open_dataset(f_self, engine="netcdf4", decode_times=False)
        ds_self_smooth = xr.open_dataset(f_self_smooth)
        sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
        sla_self_smooth = ds_self_smooth.sla_smooth
        ds_other = xr.open_dataset(
            f_other, engine="netcdf4", decode_times=False
        )
        ds_other_smooth = xr.open_dataset(f_other_smooth)
        sla_other = track_dist_time_asn(ds_other, var_str="sla", units_in="m")
        sla_other_smooth = ds_other_smooth.sla_smooth
        sla_self_at = sla_self.sel(
            x=x_from_coast_self1, method="nearest", drop=True
        )
        sla_other_at = sla_other.sel(
            x=x_from_coast_other1, method="nearest", drop=True
        )
        lons_track_self = ds_self.lon.values
        lats_track_self = ds_self.lat.values
        lon_coast_self = lons_track_self[-1]  # this on coast
        lat_coast_self = lats_track_self[-1]  # this on coast
        lon_equat_self = lons_track_self[0]  # this on equator
        lat_equat_self = lats_track_self[0]  # this on equator
        lons_track_other = ds_other.lon.values
        lats_track_other = ds_other.lat.values
        lon_equat_other = lons_track_other[0]
        lat_equat_other = lats_track_other[0]
        lon_coast_other = lons_track_other[-1]
        lat_coast_other = lats_track_other[-1]
        ii_self = index_at_lat(ds_self, lats_inter1)
        a1 = local_angle(lons_track_self, lats_track_self, ii_self, ioff=ioff)
        a1_d = math.degrees(a1)
        ii_other = index_at_lat(ds_other, lats_inter1)
        a2 = local_angle(
            lons_track_other, lats_track_other, ii_other, ioff=ioff
        )
        a2_d = math.degrees(a2)
        # print(f"heading angle {sat}: a1 = {a1_d:.2f}, a2 = {a2_d:.2f}")
        print(f"intersection angle {sat}: a1 = {angle_acute:.2f}, a2 = {angle_obtuse:.2f}")
        sla_other_at = sla_other_at.interp(time=sla_self_at.time)
        corr = xr.corr(sla_self_at, sla_other_at, dim="time").item()
        corr_records.append(
            {
                "sat": sat,
                "intersection_index": i,
                "track_self": track_number_self,
                "track_other": track_number_other,
                "lon_inter": lons_inter1,
                "lat_inter": lats_inter1,
                "x_from_coast_self": x_from_coast_self1,
                "x_from_coast_other": x_from_coast_other1,
                "angle_acute": angle_acute,
                "angle_obtuse": angle_obtuse,
                "corr_sla_self_other": corr,
            }
        )

    df_corr = pd.DataFrame(corr_records)
    csv_out = os.path.join(out_dir, f"sla_self_other_corr_at_intersections_{sat}.csv")
    png_out = os.path.join(out_dir, f"sla_self_other_corr_at_intersections_{sat}.png")
    df_corr.to_csv(csv_out, index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        df_corr["intersection_index"],
        df_corr["corr_sla_self_other"],
        marker=".",
        linewidth=1.5,
        label=sat,
    )
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xlabel("Intersection index")
    ax.set_ylabel("corr(sla_self_at, sla_other_at)")
    ax.set_title(f"{sat} SLA correlation at intersection points")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_out, dpi=200)
    plt.show()
    plt.close(fig)
    print(f"saved {csv_out}")
    print(f"saved {png_out}")
