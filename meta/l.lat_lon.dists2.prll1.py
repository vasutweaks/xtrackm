import glob
from collections import Counter

import xarray as xr
from geopy import distance
from namesm import *
import time
import numpy as np

# cmap1="viridis"

def haversine_km_between_consecutive(lats, lons):
    lat = np.radians(lats)
    lon = np.radians(lons)
    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    a = np.sin(dlat/2.0)**2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon/2.0)**2
    R = 6371.0088
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

sats_ints = {}

sat = "GFO"
delds = []
start_time = time.time()
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
    deld = haversine_km_between_consecutive(lats_track, lons_track)
    delds.extend(deld)
    ds.close()
delds1 = delds[:] # [x for x in delds if x!=0]
mode_value = sum(delds1)/len(delds1)
print(f"{sat} {mode_value}")
sats_ints[sat] = mode_value

print(sats_ints)
print(f"time taken {time.time() - start_time}")
