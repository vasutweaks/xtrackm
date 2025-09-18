import glob
from collections import Counter

import xarray as xr
from geopy import distance
from namesm import *
import time
import os
from multiprocessing import Pool, cpu_count
from itertools import chain

# Get the process ID
process_id = os.getpid()
# Print the process ID
print(f"The process ID of the current Python script is: {process_id}")

def split_list(lst, n):
    size = len(lst) // n
    return [lst[i*size:(i+1)*size] for i in range(n-1)] + [lst[(n-1)*size:]]


def process_chunk(chunk):
    delds = []
    for f in chunk:
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
    return delds

sats_ints = {}
start_time0 = time.time()
files = sorted(glob.glob(f"../data/*_lon_ordered/ctoh.sla.ref.*.nindian.*.nc"))

number_of_workers = 4
chunks = split_list(files, number_of_workers)
with Pool(number_of_workers) as pool:
    list_of_lists = pool.map(process_chunk, chunks)
    delds_all = list(chain.from_iterable(list_of_lists))

delds1 = delds_all[:] # [x for x in delds if x!=0]
mode_value = sum(delds1)/len(delds1)
print(mode_value)
print(f"total time {(time.time() - start_time0)} seconds ---")
