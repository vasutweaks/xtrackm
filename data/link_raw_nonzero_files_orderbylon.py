import glob
import os

import xarray as xr
from namesm import sats_new
from tools_xtrackm import *

for sat in sats_new[:]:
    os.makedirs(f"{sat}_lon_ordered", exist_ok=True)
    for f in sorted(
            glob.glob(
                f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
            )):
        f_base = os.path.basename(f)
        ds = xr.open_dataset(f, decode_times=False)
        trackn = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            print(f, "zero length")
            continue
        dvals = ds.time.isel(points_numbers=1).values
        lon_coast = lons_track[-1]
        lat_coast = lats_track[-1]
        lon_equat = lons_track[0]
        lat_equat = lats_track[0]
        if not is_monotonic(dvals):
            print(f, "not monotonic")
            continue
        # lon_track_equat_str = str(int(lon_equat*1000))
        lon_equat_str = str(int(lon_equat * 10000))
        m = (lat_coast - lat_equat) / (lon_coast - lon_equat)
        if m > 0:
            f_new = f"{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.{lon_equat_str}.ascending.nc"
            print(f, f_new)
        else:
            f_new = f"{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.{lon_equat_str}.descending.nc"
            print(f, f_new)
        if not os.path.exists(f_new):
            # pass
            os.symlink(f, f_new)
        else:
            # pass
            os.remove(f_new)
            os.symlink(f, f_new)
            # print(f,f_new)
        # break
    # break
