from glob import glob

import cmocean as cmo
import matplotlib.pyplot as plt
import xarray as xr
from tools_xtrackm import *

height, width = 7, 12

sats = ["ERS1+ERS2+ENV+SRL"]
sats = ["GFO"]
sat = sats[0]

lons_inter = []
lats_inter = []
track_self = []
track_other = []

for f_self in sorted(
        glob.glob(
            f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
    ds_self = xr.open_dataset(f_self, decode_times=False)
    track_number_self = ds_self.Pass
    print(ds_self)
    # sla_self = ds_self.sla
    sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
    lns = ds_self["sla"].sizes["points_numbers"]
    lns2 = lns // 2
    lons_track_self = ds_self.lon.values
    lats_track_self = ds_self.lat.values
    lon_coast_self = lons_track_self[-1]  # this on coast
    lat_coast_self = lats_track_self[-1]  # this on coast
    lon_eqaut_self = lons_track_self[0]  # this on equator
    lat_eqaut_self = lats_track_self[0]  # this on equator
    if len(lons_track_self) == 0:
        continue
    slope_self = (lats_track_self[-1] - lats_track_self[0]) / (
        lons_track_self[-1] - lons_track_self[0])
    angle_self = np.rad2deg(np.arctan(slope_self))
    if slope_self < 0:
        continue
    track_path_self = sg.LineString(zip(lons_track_self, lats_track_self))
    for f_other in sorted(
            glob.glob(
                f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        # print(f)
        ds_other = xr.open_dataset(f_other,
                                   engine="h5netcdf",
                                   decode_times=False)
        if len(ds_other.points_numbers) == 0:
            continue
        track_number_other = ds_other.Pass
        # sla_other = ds_other.sla
        sla_other = track_dist_time_asn(ds_other, var_str="sla", units_in="m")
        lno = ds_other["sla"].sizes["points_numbers"]
        lno2 = lno // 2
        lons_track_other = ds_other.lon.values
        lats_track_other = ds_other.lat.values
        lon_equat_other = lons_track_other[0]
        lat_equat_other = lats_track_other[0]
        lon_coast_other = lons_track_other[-1]
        lat_coast_other = lats_track_other[-1]
        slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other -
                                                             lon_equat_other)
        if slope_other > 0:
            continue
        # define a linestring for the track
        track_path_other = sg.LineString(
            zip(lons_track_other, lats_track_other))
        other_times = []
        if track_path_self.intersects(track_path_other):
            point = track_path_self.intersection(track_path_other)
            print(sat, track_number_self, track_number_other, point)
            x_from_coast_self = distance.distance(
                (lat_coast_self, lon_coast_self), (point.y, point.x)).km
            x_from_coast_other = distance.distance(
                (lat_coast_other, lon_coast_other), (point.y, point.x)).km
            sla_self_at = sla_self.sel(x=x_from_coast_self, method="nearest", drop=True)
            sla_other_at = sla_other.sel(x=x_from_coast_other,
                                         method="nearest", drop=True)
            # other_times.append((track_number_other, abs(dist_time)))
            track_other.append(track_number_other)
            track_self.append(track_number_self)
            lons_inter.append(point.x)
            lats_inter.append(point.y)
            fig, ax1 = plt.subplots(1, 1, figsize=(width, height), layout="constrained")
            sla_self_at.plot(ax=ax1, label=f"pass {track_number_self}")
            sla_other_at.plot(ax=ax1, label=f"pass {track_number_other}")
            # text coordinate of intersection point
            plt.text(
                0.15,
                0.95,
                f"intersection point ({point.x:.2f}, {point.y:.2f})",
                transform=ax1.transAxes,
                fontsize=14,
                fontweight="bold",
            )
            plt.legend()
            plt.savefig(
                f"pngs_sla_at_intersection_points/tracks_intersections_{sat}_{track_number_self}_{track_number_other}.png"
            )
            plt.close("all")
            # plt.show()

df = pd.DataFrame({
    "track_self": track_self,
    "track_other": track_other,
    "lons_inter": lons_inter,
    "lats_inter": lats_inter,
})
df.to_csv(f"tracks_intersections_{sat}1.csv")

plt.show()
plt.close("all")
