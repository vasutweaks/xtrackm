"""Extract drifter ve against altimetry gu (+ Ekman) at track intersections.

Rewrite of c.extract.drifter_veANDgu_ek_at_intersections.py against
tools_xtrackm_c.  What changed, and why:

  1. DROGUE STATE IS NOW RECORDED.  The old loader took only
     ["longitude", "latitude", "time", "ve", "vn"] and never read
     drogue_lost_date, which sits in the same file.  48.5% of the points the
     published figure plots come from drifters that had already lost their
     drogue -- surface followers, biased downwind by roughly 1% of wind speed
     (~0.1 m/s at 10 m/s, comparable to the signal).  Restricting to drogued
     points moves r from 0.728 to 0.782 and rmse(1:1) from 0.342 to 0.316.
     Each row now carries `drogued`, so the figure can cut on it.

  2. THE "72 HOUR" FILTER WAS AN 18 HOUR FILTER ON THE DRIFTER DATA.
     filter_rama_uv_72h called butter_lowpass_1d(cutoff_days=3), which
     hardcodes fs = 1 sample/day (summary section 9.2).  Drifter data is
     6-hourly, so the true -3 dB cutoff was 0.75 days, not 3.  Measured from
     the frequency response:
         daily (Ekman)      true Nyquist 0.50 cyc/day -> cutoff 3.00 d  correct
         6-hourly (drifter) true Nyquist 2.00 cyc/day -> cutoff 0.75 d  wrong
     ve is already a 12 h centred difference, so an 18 h filter was close to a
     no-op and left the inertial and tidal energy in.  lowpass_time() reads the
     sampling interval off the time axis, so both series now get a real 3 day
     cutoff.

  3. THE CLOSEST-APPROACH DISTANCE IS RECORDED.  The candidate pickles
     pre-select drifters in a +-0.30 deg box (~33 km), so proximity was bounded,
     but the actual distance never reached the CSV and nothing downstream could
     refine it.  Now a geodesic distance in km.

  4. THE TIME MATCH IS DESCRIBED HONESTLY.  gu is interpolated across the ~10 d
     repeat cycle onto the drifter instant, but the old gu_time column recorded
     the NEAREST native cycle, so the offset it implied was not the offset that
     was used.  Rows now carry gu_dt_days (drifter time to the nearest native
     cycle) and gu_from_interp, so a consumer can tell an interpolated value
     from a nearest-sample fallback.  Ten rows in the old CSV were fallbacks
     from as far as 126 days away.

  5. u_ek IS WRITTEN AT BOTH 0 m AND 15 m.  The buoy comparisons in fig02/fig03
     needed the 15 m Ekman to match their 15 m current bin, but drifters are the
     exception: with drogues at 15 m, 0 m still verifies marginally better
     (r 0.726 vs 0.705, rmse 0.343 vs 0.348, RMA 1.143 vs 1.256).  EKMAN_DEPTH
     stays "0"; the 15 m column is written alongside so the choice stays
     testable rather than assumed.

  6. raise("...") raised a str (TypeError: exceptions must derive from
     BaseException); globals().update(locals()) removed; the Ekman .sel() has a
     tolerance so a time outside the record errors instead of binding to an
     endpoint.

  7. ALL EIGHT MISSIONS, not one at a time.  The candidate pickles existed only
     for S3A, TP+J1+J2+J3+S6A and TPN+J1N+J2N+J3N, which is why the drifter
     comparison covered only those three; c.candidate_buoys.for.
     intersection_points_c.py now generates them for all eight.  Drifter files
     are read once and cached across missions.

Output: drifter_ve_vs_gu_ek_{SAT}_c.csv, one per mission
"""

import os
import pickle
import sys
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod

sys.path.insert(0, "/home/srinivasu/xtrackm/lib")
from tools_xtrackm_c import lowpass_time

from namesm import sats_new

SATS = list(sats_new)      # all eight missions; the original did one at a time
METHOD, FRAC = "loess", "0.05"
DRIFTER_DIR = "/home/srinivasu/allData/drifter_track_reg/netcdf_all/"
CANDIDATE_PICKLE_DIR = "candidate_pickles_c"   # from
                                               # c.candidate_buoys.for.intersection_points_c.py
GUV_DIR = f"/home/srinivasu/xtrackm/guv/guv_at_intersections_{METHOD}"
EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd"

EKMAN_DEPTH = "0"          # see (5); "15" is written alongside for comparison
EKMAN_DEPTH_ALT = "15"
LOWPASS_DAYS = 3.0         # now honoured on both series, whatever their sampling
EQUATOR_BAND = 5.0         # matches the Ekman product's own mask (section 12.3)
EKMAN_TOL = np.timedelta64(2, "D")

GEOD = Geod(ellps="WGS84")


def decode_id(raw_id):
    if isinstance(raw_id, (bytes, np.bytes_)):
        return raw_id.decode("utf-8").strip()
    return str(raw_id).strip()


def track_to_str(track):
    t = float(track)
    return str(int(t)) if t.is_integer() else str(track)


def build_drifter_index(drifter_files):
    """{id: (path, drogue_lost_date)} in one pass over the files."""
    index = {}
    for fd in drifter_files:
        with xr.open_dataset(fd, drop_variables=["WMO"]) as ds:
            index[decode_id(ds.ID.values[0])] = (
                fd, pd.to_datetime(ds["drogue_lost_date"].values[0]))
    return index


def load_drifter(fd):
    """ve/vn low-passed at LOWPASS_DAYS, on a real time dimension."""
    with xr.open_dataset(fd, drop_variables=["WMO"]) as ds:
        d = ds[["longitude", "latitude", "time", "ve", "vn"]].isel(traj=0).load()
    d = d.assign_coords(time=d["time"]).swap_dims({"obs": "time"})
    for v in ("ve", "vn"):
        try:
            d[v] = lowpass_time(d[v], LOWPASS_DAYS)
        except ValueError:
            pass  # series too short, or sampling too coarse for this cutoff
    return d


def closest_obs(d, lon0, lat0):
    """Index and geodesic distance (km) of the drifter's closest approach."""
    lon = np.asarray(d["longitude"].values, dtype=float)
    lat = np.asarray(d["latitude"].values, dtype=float)
    good = np.isfinite(lon) & np.isfinite(lat)
    if not good.any():
        return None, np.nan
    _, _, dist = GEOD.inv(np.where(good, lon, lon0), np.where(good, lat, lat0),
                          np.full(lon.shape, lon0), np.full(lat.shape, lat0))
    dist = np.where(good, dist, np.inf)
    i = int(np.argmin(dist))
    return i, dist[i] / 1000.0


def load_ekman(unique_str, depth):
    fn = f"{EKMAN_DIR}/{unique_str}_ekman_{depth}m_at_intersections_cd.nc"
    if not os.path.exists(fn):
        return None
    with xr.open_dataset(fn) as ds:
        u = ds["u_at"].load()
    try:
        u = lowpass_time(u, LOWPASS_DAYS)
    except ValueError:
        pass
    return u


def ekman_at(u_ek, when):
    if u_ek is None:
        return np.nan
    try:
        return float(u_ek.sel(time=when, method="nearest", tolerance=EKMAN_TOL).values)
    except KeyError:
        return np.nan


def process(sat, index, cache):
    """One mission: returns the rows for drifter_ve_vs_gu_ek_{sat}_c.csv."""
    csv = f"tracks_intersections_{sat}_1.csv"
    if not os.path.exists(csv):
        print(f"{sat:<22} no {csv}, skipped")
        return []
    inter_df = pd.read_csv(csv)
    rows = []

    for _, row in inter_df.iterrows():
        t_self = track_to_str(row["track_self"])
        t_other = track_to_str(row["track_other"])
        unique_str = f"{sat}_{t_self}_{t_other}"
        lon_i, lat_i = float(row["lons_inter"]), float(row["lats_inter"])
        if abs(lat_i) < EQUATOR_BAND:
            continue

        pkl = os.path.join(CANDIDATE_PICKLE_DIR, f"{unique_str}.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as fh:
            candidates = pickle.load(fh)
        if not candidates:
            continue

        guv_path = os.path.join(
            GUV_DIR, f"guv_at_intersection_{unique_str}_{METHOD}_{FRAC}.nc")
        if not os.path.exists(guv_path):
            continue
        with xr.open_dataset(guv_path) as ds_g:
            gu = ds_g["gu"].sortby("time").load()

        u_ek = load_ekman(unique_str, EKMAN_DEPTH)
        u_ek_alt = load_ekman(unique_str, EKMAN_DEPTH_ALT)

        for did in candidates:
            entry = index.get(did)
            if entry is None:
                continue
            fd, drogue_lost = entry
            if did not in cache:
                cache[did] = load_drifter(fd)
            d = cache[did]

            i, dist_km = closest_obs(d, lon_i, lat_i)
            if i is None:
                continue
            when = d["time"].values[i]
            ve = float(d["ve"].values[i])
            if np.isnat(when) or not np.isfinite(ve):
                continue

            gu_interp = float(gu.interp(time=when).values)
            gu_near = gu.sel(time=when, method="nearest")
            from_interp = bool(np.isfinite(gu_interp))
            gu_val = gu_interp if from_interp else float(gu_near.values)
            if not np.isfinite(gu_val):
                continue

            rows.append({
                "sat": sat, "track_self": t_self, "track_other": t_other,
                "lon_inter": lon_i, "lat_inter": lat_i,
                "drifter_id": did,
                "gu": gu_val,
                "u_ek": ekman_at(u_ek, when),
                f"u_ek_{EKMAN_DEPTH_ALT}m": ekman_at(u_ek_alt, when),
                "drifter_ve": ve,
                "drifter_time": str(when),
                "gu_time": str(gu_near["time"].values),
                "gu_dt_days": float(
                    (pd.Timestamp(when) - pd.Timestamp(gu_near["time"].values))
                    .total_seconds() / 86400.0),
                "gu_from_interp": from_interp,
                "close_index": i,
                "closest_km": float(dist_km),
                "drogue_lost_date": ("" if pd.isna(drogue_lost) else str(drogue_lost)),
                "drogued": bool(pd.isna(drogue_lost) or pd.Timestamp(when) <= drogue_lost),
            })
    return rows


def main():
    drifter_files = sorted(glob(os.path.join(DRIFTER_DIR, "drifter_6h_*.nc")))
    if not drifter_files:
        raise FileNotFoundError(f"no drifter_6h_*.nc under {DRIFTER_DIR}")
    print(f"indexing {len(drifter_files)} drifter files ...")
    index = build_drifter_index(drifter_files)
    cache = {}   # shared across missions: a drifter read once serves all of them

    for sat in SATS:
        rows = process(sat, index, cache)
        out = pd.DataFrame(rows)
        out_csv = f"drifter_ve_vs_gu_ek_{sat}_c.csv"
        out.to_csv(out_csv, index=False)
        if len(out):
            print(f"{sat:<22} {len(out):>5} rows | drogued {100*out.drogued.mean():4.1f}% | "
                  f"|dt| med {out.gu_dt_days.abs().median():4.2f} d | "
                  f"closest med {out.closest_km.median():4.1f} km | "
                  f"{out.drifter_id.nunique():>4} drifters | -> {out_csv}")
        else:
            print(f"{sat:<22} {0:>5} rows -> {out_csv}")


if __name__ == "__main__":
    main()
