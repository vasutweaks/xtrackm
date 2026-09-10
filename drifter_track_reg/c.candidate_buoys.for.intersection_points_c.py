"""Candidate drifters near every track intersection, for ALL satellites.

Rewrite of c.candidate_buoys.for.intersection_points.py.  Same selection rule --
a drifter is a candidate for an intersection if any of its fixes falls inside a
+-CELL deg box around it, and its lifetime overlaps the mission -- but:

  1. ALL EIGHT MISSIONS in one pass.  The original hardcoded a single `sat`, and
     only S3A (438 pickles), TP+J1+J2+J3+S6A (87) and TPN+J1N+J2N+J3N (88) were
     ever generated.  ERS1+ERS2+ENV+SRL, GFO, HY2A, HY2B and S3B had none, which
     is why the drifter comparison existed for only three missions.
  2. THE DRIFTER POSITIONS ARE READ ONCE.  The original re-opened every
     bbox-surviving drifter file inside the per-intersection loop, so each file
     was read once per nearby intersection.  Here every file is read once and
     the fixes are indexed into a CELL-degree spatial hash, so each intersection
     only tests the handful of drifters that visited its own cells.
  3. The `print(dmeta)` inside the inner loop is gone -- it printed the whole
     metadata dict for every drifter of every intersection.

Output: candidate_pickles_c/{sat}_{track_self}_{track_other}.pkl
Existing candidate_pickles/ is left untouched.
"""

import os
import pickle
import sys
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "/home/srinivasu/xtrackm/lib")
from namesm import sats_new
from tools_xtrackm import get_time_limits_np64, overlap_dates_np64

DRIFTER_DIR = "/home/srinivasu/allData/drifter_track_reg/netcdf_all/"
OUT_DIR = "candidate_pickles_c"
CELL = 0.30          # the half-width of the box, and the spatial-hash cell size
EQUATOR_BAND = 5.0   # matches the Ekman mask; intersections inside are skipped


def load_all_drifters():
    """Read every drifter once: id, lifetime, positions, and the cells visited."""
    files = sorted(glob(os.path.join(DRIFTER_DIR, "drifter_6h_*.nc")))
    if not files:
        raise FileNotFoundError(f"no drifter_6h_*.nc under {DRIFTER_DIR}")
    drifters, cell_index = [], defaultdict(set)
    for k, fd in enumerate(files):
        with xr.open_dataset(fd, drop_variables=["WMO"]) as ds:
            did = ds.ID.values[0]
            did = did.decode("utf-8").strip() if isinstance(did, (bytes, np.bytes_)) else str(did).strip()
            t0 = np.datetime64(ds.start_date.values[0], "s")
            t1 = np.datetime64(ds.end_date.values[0], "s")
            lon = np.asarray(ds.longitude.values, dtype=float)
            lat = np.asarray(ds.latitude.values, dtype=float)
        good = np.isfinite(lon) & np.isfinite(lat)
        lon, lat = lon[good], lat[good]
        if lon.size == 0:
            continue
        drifters.append({"id": did, "t0": t0, "t1": t1, "lon": lon, "lat": lat})
        for c in set(zip(np.floor(lon / CELL).astype(int),
                         np.floor(lat / CELL).astype(int))):
            cell_index[c].add(k)
    return drifters, cell_index


def candidates_for(lon1, lat1, drifters, cell_index, allowed):
    """Drifter ids with a fix inside the +-CELL box around (lon1, lat1)."""
    ci = range(int(np.floor((lon1 - CELL) / CELL)), int(np.floor((lon1 + CELL) / CELL)) + 1)
    cj = range(int(np.floor((lat1 - CELL) / CELL)), int(np.floor((lat1 + CELL) / CELL)) + 1)
    near = set()
    for i in ci:
        for j in cj:
            near |= cell_index.get((i, j), set())
    out = []
    for k in sorted(near & allowed):
        d = drifters[k]
        if np.any((np.abs(d["lon"] - lon1) <= CELL) & (np.abs(d["lat"] - lat1) <= CELL)):
            out.append(d["id"])
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("reading drifters ...")
    drifters, cell_index = load_all_drifters()
    print(f"  {len(drifters)} drifters, {len(cell_index)} occupied {CELL} deg cells")

    for sat in sats_new:
        csv = f"tracks_intersections_{sat}_1.csv"
        if not os.path.exists(csv):
            print(f"{sat:<22} no {csv}, skipped")
            continue
        inter = pd.read_csv(csv)
        t0, t1 = get_time_limits_np64(sat)
        allowed = {k for k, d in enumerate(drifters)
                   if not np.isnat(overlap_dates_np64(t0, t1, d["t0"], d["t1"])[0])}

        written = skipped_eq = empty = 0
        for _, r in inter.iterrows():
            lon1, lat1 = float(r["lons_inter"]), float(r["lats_inter"])
            if abs(lat1) < EQUATOR_BAND:
                skipped_eq += 1
                continue
            ts = str(int(r["track_self"])) if float(r["track_self"]).is_integer() else str(r["track_self"])
            to = str(int(r["track_other"])) if float(r["track_other"]).is_integer() else str(r["track_other"])
            ids = candidates_for(lon1, lat1, drifters, cell_index, allowed)
            if not ids:
                empty += 1
                continue
            with open(os.path.join(OUT_DIR, f"{sat}_{ts}_{to}.pkl"), "wb") as fh:
                pickle.dump(ids, fh)
            written += 1
        print(f"{sat:<22} {len(inter):>4} intersections | {len(allowed):>4} drifters overlap "
              f"| wrote {written:>4} | no candidates {empty:>4} | inside {EQUATOR_BAND:g}N/S {skipped_eq:>4}")


if __name__ == "__main__":
    main()
