# l.lat_lon.dists2.prll_files.py
# Parallel (faster): multiprocessing over files + vectorized haversine per file.
# Dirty style on purpose. Worker has no try/except around dataset reading.

import glob
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import xarray as xr
from namesm import *   # provides sats_new etc.

# Get the process ID
process_id = os.getpid()
# Print the process ID
print(f"The process ID of the current Python script is: {process_id}")

# adjust these paths/patterns to match your layout
DATA_DIR = "../data"
PATTERN = "ctoh.sla.ref.{sat}.nindian.*.nc"

def haversine_sum_and_count(lats, lons):
    """
    Vectorized haversine distances between successive points.
    Returns (sum_of_distances_km, number_of_pairs)
    lats, lons are 1D numpy arrays (degrees).
    """
    # require at least 2 points
    if lats.size < 2:
        return 0.0, 0
    # convert to radians
    lat1 = np.deg2rad(lats[:-1])
    lat2 = np.deg2rad(lats[1:])
    lon1 = np.deg2rad(lons[:-1])
    lon2 = np.deg2rad(lons[1:])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # numerical safety
    a = np.minimum(1.0, np.maximum(0.0, a))
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    R = 6371.0088  # mean Earth radius in km
    dists = R * c
    return float(dists.sum()), int(dists.size)

def worker_process_file(args):
    """
    Worker that reads one file and returns:
      (sat_name, sum_delds_km, n_pairs)
    No try/except (user requested).
    """
    fpath, sat = args
    with xr.open_dataset(fpath) as ds:
        # follow original logic: skip tiny tracks / use only ones with >100 points
        if len(ds.points_numbers) == 0:
            return (sat, 0.0, 0)
        lons = ds.lon.values
        lats = ds.lat.values
        if lons.size <= 100:
            return (sat, 0.0, 0)
        s, n = haversine_sum_and_count(lats, lons)
        return (sat, s, n)

def gather_file_list_for_all_sats(sats):
    tasks = []
    for sat in sats:
        pattern = os.path.join(DATA_DIR, f"{sat}_lon_ordered", PATTERN.format(sat=sat))
        files = sorted(glob.glob(pattern))
        for f in files:
            tasks.append((f, sat))
    return tasks

if __name__ == "__main__":
    start_all = time.time()
    sats = list(sats_new)
    tasks = gather_file_list_for_all_sats(sats)
    ncores = max(1, cpu_count() - 1)   # leave one core free
    with Pool(processes=ncores) as pool:
        # map over files; this returns a list of (sat, sum, count)
        results = pool.map(worker_process_file, tasks)
    # aggregate per-satellite sums and counts
    sums = {}
    counts = {}
    for sat, s, n in results:
        if n == 0:
            continue
        sums[sat] = sums.get(sat, 0.0) + s
        counts[sat] = counts.get(sat, 0) + n

    sats_ints = {}
    for sat in sats:
        if counts.get(sat, 0) == 0:
            sats_ints[sat] = 0.0
        else:
            sats_ints[sat] = sums[sat] / counts[sat]
    # print in same style as your scripts
    for sat in sats:
        print(f"{sat} {sats_ints[sat]}")
        # We don't print per-sat timings here; measure whole run
    print(sats_ints)
    print("TOTAL TIME:", time.time() - start_all)
