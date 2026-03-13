# bench_read_xr_vs_netcdf4.py
# Dirty benchmark: read lon/lat from all files using xarray and netCDF4 and compare times.
# Only reading — no further processing.

import glob, os, time
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from namesm import *   # sats_new

DATA_DIR = "../data"
PATTERN = "ctoh.sla.ref.{sat}.nindian.*.nc"

# Toggle this to test xarray decode_times True/False
DECODE_TIMES = True
DECODE_TIMES = False

def gather_files(sats):
    files = []
    for sat in sats:
        p = os.path.join(DATA_DIR, f"{sat}_lon_ordered", PATTERN.format(sat=sat))
        files.extend(sorted(glob.glob(p)))
    return files

def read_with_xarray(files, decode_times=True):
    t0 = time.perf_counter()
    sizes = []
    for f in files:
        # open, read lon/lat into memory, then close
        with xr.open_dataset(f, decode_times=decode_times) as ds:
            # force load to memory
            lons = ds.lon.values
            lats = ds.lat.values
            sizes.append((lats.size, lons.size))
    t1 = time.perf_counter()
    total = t1 - t0
    return total, sizes

def read_with_netcdf4(files):
    t0 = time.perf_counter()
    sizes = []
    for f in files:
        ds = Dataset(f, mode="r")
        lons = ds.variables['lon'][:].astype('float64')
        lats = ds.variables['lat'][:].astype('float64')
        ds.close()
        sizes.append((lats.size, lons.size))
    t1 = time.perf_counter()
    total = t1 - t0
    return total, sizes

def summarize(name, total_time, sizes):
    nfiles = len(sizes)
    counts = [s[0] for s in sizes]  # lat counts (should equal lon)
    npoints = sum(counts)
    avg_file = total_time / nfiles if nfiles else float('nan')
    avg_point = total_time / npoints if npoints else float('nan')
    print(f"--- {name} ---")
    print(f"files: {nfiles}, total points read: {npoints}")
    print(f"total time: {total_time:.4f} s")
    print(f"avg time / file: {avg_file:.6f} s")
    print(f"avg time / point: {avg_point:.9f} s")
    print()

if __name__ == "__main__":
    sats = list(sats_new)
    files = gather_files(sats)
    if not files:
        print("No files found. Check DATA_DIR / pattern.")
        raise SystemExit(1)

    print(f"Found {len(files)} files. Running benchmarks...")
    print("Xarray decode_times =", DECODE_TIMES)
    xr_time, xr_sizes = read_with_xarray(files, decode_times=DECODE_TIMES)
    summarize(f"xarray (decode_times={DECODE_TIMES})", xr_time, xr_sizes)

    nc_time, nc_sizes = read_with_netcdf4(files)
    summarize("netCDF4.Dataset", nc_time, nc_sizes)

    diff = xr_time - nc_time
    rel = diff / nc_time if nc_time else float('nan')
    print("Summary comparison:")
    print(f"xarray time: {xr_time:.4f} s")
    print(f"netCDF4 time: {nc_time:.4f} s")
    print(f"absolute difference (xarray - netCDF4): {diff:.4f} s")
    print(f"relative difference: {rel*100:.2f}% (positive => xarray slower)")

