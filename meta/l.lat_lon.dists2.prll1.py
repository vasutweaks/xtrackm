# l.lat_lon.dists2.parallel.notry.py
# Parallelized: threading for I/O, multiprocessing across satellites.
# process_file contains NO try/except/finally blocks (uses context manager instead).

import glob
import time
from collections import Counter

import xarray as xr
from geopy import distance
from namesm import *   # provides sats_new etc.

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

sats_ints = {}

def process_file(path):
    delds = []
    # No try/except here — let errors propagate. Using context manager to ensure close.
    with xr.open_dataset(path) as ds:
        if len(ds.points_numbers) == 0:
            return delds
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
    return delds

def process_sat(sat):
    start_time = time.time()
    delds = []
    pattern = f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc"
    files = sorted(glob.glob(pattern))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_file, fpath): fpath for fpath in files}
        for fut in as_completed(futures):
            try:
                out = fut.result()
                if out:
                    delds.extend(out)
            except Exception:
                # keep outer-level minimal error skipping (per your dirty request)
                pass
    if len(delds) == 0:
        mode_value = 0.0
    else:
        delds1 = delds[:]
        mode_value = sum(delds1) / len(delds1)
    elapsed = time.time() - start_time
    print(f"{sat} {mode_value}")
    print(f"{sat} took {elapsed} seconds ---")
    return (sat, mode_value)

if __name__ == "__main__":
    ncores = max(1, cpu_count() - 1)
    sats_list = list(sats_new)
    with Pool(processes=ncores) as pool:
        results = pool.map(process_sat, sats_list)
    sats_ints = dict(results)
    print(sats_ints)
