#!/usr/bin/env python3
"""
Parallelized distance calculator
Based on your l.lat_lon.dists2.prll1.py (kept behavior, replaced geopy loop with vectorized haversine
and added multiprocessing across files).
"""

import glob
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import xarray as xr
import numpy as np

# -----------------------
# Parameters (tweakable)
# -----------------------
FILE_GLOB = f"../data/*_lon_ordered/ctoh.sla.ref.*.nindian.*.nc"
N_WORKERS = 8  # set to 8, or use cpu_count()
CHUNKSIZE = 1  # tune for performance; larger chunksize reduces scheduling overhead
# -----------------------

def haversine_km_between_consecutive(lats, lons):
    """Vectorized haversine distances (km) between consecutive lat/lon points."""
    # If arrays are short, keep safe behaviour
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if lats.size < 2 or lons.size < 2:
        return np.array([], dtype=float)
    lat = np.radians(lats)
    lon = np.radians(lons)
    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    a = np.sin(dlat * 0.5)**2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon * 0.5)**2
    R = 6371.0088  # Earth radius in km
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def process_one_file(fname, require_min_points=2):
    """
    Open dataset, extract lon/lat arrays, compute consecutive haversine distances,
    return list (or flattened array) of distances for that file or None on failure.
    """
    try:
        ds = xr.open_dataset(fname, mask_and_scale=False)  # avoid automatic masking if not needed
    except Exception as e:
        # Optionally print or log
        # print(f"Failed to open {fname}: {e}")
        return None

    try:
        # Try multiple likely coord names if necessary
        # Original script used ds.lon and ds.lat
        if hasattr(ds, "lon") and hasattr(ds, "lat"):
            lons = ds.lon.values
            lats = ds.lat.values
        else:
            # try other names / dims (adjust if your files use other naming)
            # fallback: attempt to find first 1D coordinate-like arrays
            coords = [v for v in ds.variables if getattr(ds[v], 'dims', None) and ds[v].ndim == 1]
            # minimal fallback - this is heuristic; adapt if your files differ
            if len(coords) >= 2:
                lons = ds[coords[0]].values
                lats = ds[coords[1]].values
            else:
                ds.close()
                return None

        # ensure 1D arrays
        lons = np.asarray(lons).ravel()
        lats = np.asarray(lats).ravel()

        if lons.size < require_min_points or lats.size < require_min_points:
            ds.close()
            return None

        distances = haversine_km_between_consecutive(lats, lons)
        # convert to python list for pickling/transport via multiprocessing
        out = distances.tolist()
        ds.close()
        return out

    except Exception as e:
        # print(f"Error processing {fname}: {e}")
        try:
            ds.close()
        except Exception:
            pass
        return None

def flatten_and_filter(list_of_lists):
    """Flatten lists of floats and filter out None/empty results."""
    out = []
    for item in list_of_lists:
        if not item:
            continue
        out.extend(item)
    return out

def main():
    t0 = time.time()
    filelist = sorted(glob.glob(FILE_GLOB))
    if not filelist:
        print("No files found for pattern:", FILE_GLOB)
        return

    n_workers = N_WORKERS or min(8, cpu_count())
    print(f"Found {len(filelist)} files. Using {n_workers} workers.")

    # Use multiprocessing.Pool to process files in parallel
    worker = partial(process_one_file)
    all_distances = []

    # Pool.map returns results in input order; for large number of files, you can use imap_unordered
    with Pool(processes=n_workers) as pool:
        # Using map with chunksize to amortize scheduling cost
        results = pool.map(worker, filelist, chunksize=CHUNKSIZE)

    # Flatten
    all_distances = flatten_and_filter(results)

    if len(all_distances) == 0:
        print("No distances computed (all files failed or had too few points).")
    else:
        # compute global stats
        arr = np.asarray(all_distances, dtype=float)
        mean_value = float(np.nanmean(arr))
        median_value = float(np.nanmedian(arr))
        n_pts = arr.size
        print(f"Processed {len(filelist)} files, total {n_pts} distances.")
        print(f"Mean distance (km): {mean_value:.6f}")
        print(f"Median distance (km): {median_value:.6f}")

    print(f"Elapsed time: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()

