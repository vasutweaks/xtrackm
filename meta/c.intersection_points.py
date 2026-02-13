from glob import glob

import xarray as xr
from tools_xtrackm import *


def intersection_angles_rad(m1, m2):
    """Return both angles of intersection (acute and obtuse) in radians between lines of slopes m1 and m2."""
    if 1 + m1 * m2 == 0:
        return (math.pi / 2, math.pi / 2)  # Perpendicular lines
    tan_theta = abs((m1 - m2) / (1 + m1 * m2))
    theta = math.atan(tan_theta)  # Always gives acute angle
    # Second angle is the obtuse supplementary angle
    other_theta = math.pi - theta
    return (theta, other_theta)


lons_inter = []
lats_inter = []
track_self = []
track_other = []
angle_acute = []
angle_obtuse = []
x_from_coast_self = []
x_from_coast_other = []

sat = "S3A"
sat = "ERS1+ERS2+ENV+SRL"
sat = "TP+J1+J2+J3+S6A"

for f_self in sorted(
        glob.glob(
            f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
    ds_self = xr.open_dataset(f_self, decode_times=False)
    track_number_self = ds_self.Pass
    lons_track_self = ds_self.lon.values
    lats_track_self = ds_self.lat.values
    lon_coast_self = lons_track_self[-1]  # this on coast
    lat_coast_self = lats_track_self[-1]  # this on coast
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
        track_number_other = ds_other.Pass
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
        if track_path_self.intersects(track_path_other):
            point = track_path_self.intersection(track_path_other)
            acute, obtuse = intersection_angles_rad(slope_self, slope_other)
            x_from_coast_self1 = distance.distance(
                (lat_coast_self, lon_coast_self), (point.y, point.x)).m
            x_from_coast_other1 = distance.distance(
                (lat_coast_other, lon_coast_other), (point.y, point.x)).m
            print(sat, track_number_self, track_number_other, point.x, point.y, x_from_coast_self1, x_from_coast_other1)
            if is_land(point.x, point.y):
                continue
            track_other.append(track_number_other)
            track_self.append(track_number_self)
            lons_inter.append(point.x)
            lats_inter.append(point.y)
            x_from_coast_self.append(x_from_coast_self1)
            x_from_coast_other.append(x_from_coast_other1)
            angle_acute.append(acute)
            angle_obtuse.append(obtuse)

df = pd.DataFrame({
    "track_self": track_self,
    "track_other": track_other,
    "lons_inter": lons_inter,
    "lats_inter": lats_inter,
    "x_from_coast_self": x_from_coast_self,
    "x_from_coast_other": x_from_coast_other,
    "angle_acute": angle_acute,
    "angle_obtuse": angle_obtuse
})
df.to_csv(f"tracks_intersections_{sat}_1.csv")
