# fast_netCDF4_worker.py
import glob, os, time
from multiprocessing import Pool, cpu_count
import numpy as np
from netCDF4 import Dataset
from namesm import *   # sats_new

DATA_DIR = "../data"
PATTERN = "ctoh.sla.ref.{sat}.nindian.*.nc"

R = 6371.0088
def haversine_sum_and_count(lats, lons):
    if lats.size < 2:
        return 0.0, 0
    lat1 = np.deg2rad(lats[:-1]); lat2 = np.deg2rad(lats[1:])
    lon1 = np.deg2rad(lons[:-1]); lon2 = np.deg2rad(lons[1:])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    a = np.minimum(1.0, np.maximum(0.0, a))
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    d = R * c
    return float(d.sum()), int(d.size)

def worker(args):
    fpath, sat = args
    # NO try/except here by user request
    ds = Dataset(fpath, mode="r")
    # adapt variable names to your files if different
    lons = ds.variables['lon'][:].astype('float64')
    lats = ds.variables['lat'][:].astype('float64')
    ds.close()
    if lons.size <= 100:
        return (sat, 0.0, 0)
    return (sat, ) + haversine_sum_and_count(lats, lons)

def gather():
    tasks=[]
    for sat in list(sats_new):
        pattern = os.path.join(DATA_DIR, f"{sat}_lon_ordered", PATTERN.format(sat=sat))
        for f in sorted(glob.glob(pattern)):
            tasks.append((f, sat))
    return tasks

if __name__ == "__main__":
    t0=time.time()
    tasks=gather()
    n= max(1, cpu_count()-1)
    with Pool(n) as p:
        res = p.map(worker, tasks)
    sums, counts = {}, {}
    for sat, s, c in res:
        if c==0: continue
        sums[sat]=sums.get(sat,0.0)+s
        counts[sat]=counts.get(sat,0)+c
    sats_ints = {sat: (sums.get(sat,0.0)/counts.get(sat,1)) if counts.get(sat,0)>0 else 0.0 for sat in list(sats_new)}
    print(sats_ints)
    print("TOTAL", time.time()-t0)
