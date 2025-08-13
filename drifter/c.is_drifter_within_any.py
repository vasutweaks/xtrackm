import os

import xarray as xr
from rich.console import Console
from tools_xtrackm import *

console = Console()

dist_threshold = 2
computed_loc = f"/home/srinivasu/slnew/xtrackm/computed/"
data_loc = f"/home/srinivasu/allData/drifter1/"


def is_within_region_all(lons_drifter, lats_drifter, lon_min, lon_max, lat_min,
                         lat_max):
    lons_within_range = np.logical_and(lons_drifter >= lon_min, lons_drifter
                                       <= lon_max)
    # Check if all latitudes are within the range
    lats_within_range = np.logical_and(lats_drifter >= lat_min, lats_drifter
                                       <= lat_max)
    # Combine both conditions
    all_within_region = np.logical_and(lons_within_range, lats_within_range)
    # Check if all points are within the region
    if np.all(all_within_region):
        print("All points are within the region.")
        return True
    else:
        # print("Not all points are within the region.")
        return False


def is_within_region_any(lons,
                         lats,
                         lon_min,
                         lon_max,
                         lat_min,
                         lat_max,
                         include_boundary=True):
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


sat, track_number_self, track_number_other, lons_inter, lats_inter = (
    "S3A",
    "8",
    "79",
    94.14047103085358,
    8.36326782978126,
)
# a square polygon with lons_inter, lats_inter as the center
box = create_box_patch(lons_inter, lats_inter, dist_threshold)
lon_max, lat_max = lons_inter + dist_threshold, lats_inter + dist_threshold
lat_min, lon_min = lats_inter - dist_threshold, lons_inter - dist_threshold
track_tsta_o, track_tend_o = get_time_limits_o(sat)

chunks = ["1_5000", "5001_10000", "10001_15000", "15001_current"]
found = False
for chunk in chunks:
    for fd in sorted(
            glob.glob(f"{data_loc}/netcdf_{chunk}/"
                      f"track_reg/drifter_6h_*.nc")):
        # print(fd)
        basename1 = os.path.basename(fd)
        ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
        byte_id = ds_d.ID.values[0]
        str_id = byte_id.decode("utf-8")
        drifter_id = str_id
        drift_tsta_o1, drift_tend_o1 = (
            ds_d.start_date.values[0],
            ds_d.end_date.values[0],
        )
        drift_tsta_o, drift_tend_o = n64todatetime(
            drift_tsta_o1), n64todatetime(drift_tend_o1)
        overlap_tsta_o, overlap_tend_o = overlap_dates(track_tsta_o,
                                                       track_tend_o,
                                                       drift_tsta_o,
                                                       drift_tend_o)
        times_drift = ds_d.time.isel(traj=0).values
        lons_da = drifter_time_asn(ds_d, var_str="longitude")
        lats_da = drifter_time_asn(ds_d, var_str="latitude")
        lons_da2 = lons_da.rolling(time=3).mean()
        lats_da2 = lats_da.rolling(time=3).mean()
        lons_da3 = lons_da2.ffill(dim="time").bfill(dim="time")
        lats_da3 = lats_da2.ffill(dim="time").bfill(dim="time")

        lons_drift = lons_da3.values
        lats_drift = lats_da3.values

        lon_start = ds_d.start_lon.isel(traj=0).values
        lon_end = ds_d.end_lon.isel(traj=0).values
        lat_start = ds_d.start_lat.isel(traj=0).values
        lat_end = ds_d.end_lat.isel(traj=0).values
        if overlap_tsta_o is None or overlap_tend_o is None:
            # console.print(f"{overlap_tsta_o}, {overlap_tend_o}", style="red")
            continue
        else:
            print(f"{overlap_tsta_o}, {overlap_tend_o}")
        if is_within_region_any(lons_drift, lats_drift, lon_min, lon_max,
                                lat_min, lat_max):
            print(f"{fd}")
            print(f"Drifter {drifter_id} is within the region.")
            found = True
            break
    if found:
        break
