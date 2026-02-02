import glob
from collections import Counter

import xarray as xr
from geopy import distance
from namesm import *
import time
import os

# cmap1="viridis"

# Get the process ID
process_id = os.getpid()
# Print the process ID
print(f"The process ID of the current Python script is: {process_id}")

sats_ints = {}

start_time0 = time.time()
for sat in sats_new:
    start_time = time.time()
    delds = []
    for f in sorted(glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")):
        # print(f)
        try:
            ds = xr.open_dataset(f)
        except:
            continue
        if len(ds.points_numbers) == 0:
            continue
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        time1 = ds.time.isel(cycles_numbers=0).values
        if len(lons_track) > 20:
            lat_diff = lats_track[20] - lats_track[10]
            # print(sat, track_number, lat_diff)
            time_s = time1[20]
            time_e = time1[10]
            scnds = (time_e - time_s).seconds
            print(sat, track_number, scnds)

