import glob
from collections import Counter

import xarray as xr
from geopy import distance
from namesm import *
import time
import os

def split_files(list_files, n):
    chunks = []
    for i in range(0, len(list_files), n):
        chunk = list_files[i*n:(i+1)*n]
        chunks.append(chunk)
    return chunks

# Get the process ID
process_id = os.getpid()
# Print the process ID
print(f"The process ID of the current Python script is: {process_id}")

sats_ints = {}

sat = "ERS1+ERS2+ENV+SRL"
start_time0 = time.time()
delds = []
for f in sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
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
    time_s = ds.time.isel(cycles_numbers=0).isel(points_numbers=0).item()
    time_e = ds.time.isel(cycles_numbers=0).isel(points_numbers=-1).item()
    scnds = (time_e - time_s).seconds
    lon_s = lons_track[0]
    lon_e = lons_track[-1]
    lat_s = lats_track[0]
    lat_e = lats_track[-1]
    if len(lons_track) > 100:
        for i in range(len(lons_track)-1):
            lat1 = lats_track[i]
            lon1 = lons_track[i]
            lat2 = lats_track[i+1]
            lon2 = lons_track[i+1]
            deld = distance.distance((lat1, lon1), (lat2, lon2)).km
            delds.append(deld)
    ds.close()
delds1 = delds[:] # [x for x in delds if x!=0]
mode_value = sum(delds1)/len(delds1)
print(f"{sat} {mode_value}")
sats_ints[sat] = mode_value
print(sats_ints)
print(f"total time {(time.time() - start_time0)} seconds ---")
