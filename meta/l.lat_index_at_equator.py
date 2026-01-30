import glob
import xarray as xr
from namesm import *

for sat in sats_new:
    for f in sorted(glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")):
        # print(f)
        ds = xr.open_dataset(f, decode_times=False)
        lons = ds.lon
        lats = ds.lat
        if len(lats.values) == 0:
            continue
        lat_equat = lats.values[0]
        print(lat_equat)
