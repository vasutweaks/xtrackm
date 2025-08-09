import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta, timezone
from geographiclib.geodesic import Geodesic

# ------------------------
# User-configurable inputs
# ------------------------

# Intersection CSVs: all satellites use the same header
# Example files: tracks_intersections_S3A_1.csv, tracks_intersections_GFO_1.csv, etc.
INTERSECTION_CSV_GLOB = "tracks_intersections_*.csv"

# Folder containing guv time series files:
# Files are named guv_at_intersection_{sat}_{track_self}_{track_other}.nc
GUV_FOLDER = "/home/srinivasu/xtrackm/guv/guv_at_intersections"

# Drifter files (traj dimension always present)
# Example glob given in earlier context:
DRIFTER_GLOB = "/home/srinivasu/allData/drifter1/netcdf_15001_current/track_reg/drifter_6h_*.nc"

# Output CSV of collocations
OUTPUT_CSV = "collocations_guv_vs_drifter.csv"

# ------------------------
# Search and QC parameters (defaults)
# ------------------------
# Max great-circle distance to accept a drifter sample, in km
MAX_DISTANCE_KM = 20.0
# Max absolute time difference to accept a drifter sample, in hours (set None to ignore time filter)
MAX_TIME_HOURS = 12.0
# If multiple candidates exist, keep nearest in space (ties broken by smaller |dt|)
# If no candidate within radius/time window, no match recorded for that satellite time

# For track-direction estimation at an intersection, define a small offset to form a local bearing
# We approximate the ground-track direction locally by taking two points displaced along meridian/parallel
# Note: If per-track along-track neighbor coordinates are available, prefer those instead.
BEARING_OFFSET_KM = 5.0  # used to compute bearing at the intersection lon/lat

# ------------------------
# Helpers
# ------------------------

geod = Geodesic.WGS84

def decode_sat_time_days_since_2000_01_11(days_array):
    """Convert float days since 2000-01-11 (proleptic_gregorian) to timezone-naive UTC datetimes."""
    epoch = datetime(2000, 1, 11, tzinfo=timezone.utc)
    # Ensure float conversion and handle NaNs
    out = []
    for t in np.asarray(days_array):
        if np.isnan(t):
            out.append(None)
        else:
            out.append((epoch + timedelta(days=float(t))).replace(tzinfo=None))
    return np.array(out, dtype=object)

def decode_unix_seconds_to_datetime(sec_array):
    """Convert seconds since 1970-01-01 UTC to naive datetimes."""
    base = datetime(1970, 1, 1, tzinfo=timezone.utc)
    out = []
    for s in np.asarray(sec_array):
        if np.isnan(s):
            out.append(None)
        else:
            out.append((base + timedelta(seconds=float(s))).replace(tzinfo=None))
    return np.array(out, dtype=object)

def bearing_deg(lat1, lon1, lat2, lon2):
    """Forward azimuth (bearing) from point1 to point2 in degrees [0,360)."""
    inv = geod.Inverse(lat1, lon1, lat2, lon2)
    azi = inv['azi1']
    if azi < 0:
        azi += 360.0
    return azi

def destination_point(lat, lon, bearing_deg_val, distance_km):
    """Compute destination point given start, bearing, and distance on WGS84."""
    res = geod.Direct(lat, lon, bearing_deg_val, distance_km * 1000.0)
    return res['lat2'], res['lon2']

def local_track_bearing(lat, lon):
    """
    Estimate a local ground-track bearing at (lat,lon).
    We approximate as the N-S direction (descending/ascending ambiguity) by probing a short meridional segment
    and choosing the azimuth from south-to-north segment as the 'along-track' reference.
    If along-track data per track are available, replace with that for precision.
    """
    # Use small meridional offset to get a stable bearing
    # South point:
    lat_s, lon_s = destination_point(lat, lon, 180.0, BEARING_OFFSET_KM)
    # North point:
    lat_n, lon_n = destination_point(lat, lon, 0.0, BEARING_OFFSET_KM)
    # Bearing from south to north point approximates a meridional "track-like" direction
    return bearing_deg(lat_s, lon_s, lat_n, lon_n)

def en_to_along_cross(ve, vn, bearing_deg_val):
    """
    Project east/north components (m/s) into (along, cross) relative to a track bearing.
    Convention:
      - along is positive in the forward track direction (bearing).
      - cross is positive to the right of the track direction.
    Unit vectors:
      along: ax = cos(b), ay = sin(b)
      right: rx = sin(b), ry = -cos(b)
    Cross (right-of-track) = ve*rx + vn*ry
    """
    b = np.deg2rad(bearing_deg_val)
    ax, ay = np.cos(b), np.sin(b)
    rx, ry = np.sin(b), -np.cos(b)
    valong = ve * ax + vn * ay
    vcross = ve * rx + vn * ry
    return valong, vcross

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance using geographiclib (authoritative); kept as backup if needed."""
    return geod.Inverse(lat1, lon1, lat2, lon2)['s12'] / 1000.0

def build_drifter_pool(drifter_glob):
    """
    Load all drifter files, extract traj=0 by default (extend if multiple trajectories are needed),
    collect arrays: longitude, latitude, time (datetime), ve, vn, and keep file and index references.
    """
    pool = []
    fnames = sorted(glob.glob(drifter_glob))
    for f in fnames:
        try:
            ds = xr.open_dataset(f)
            # Always a traj dim; if multiple trajs exist, process all
            ntraj = ds.dims.get('traj', 1)
            for it in range(ntraj):
                sub = ds.isel(traj=it)
                # Variables expected: longitude, latitude, time (sec since 1970), ve, vn
                if not all(v in sub.variables for v in ['longitude','latitude','time','ve','vn']):
                    continue
                lon = sub['longitude'].values
                lat = sub['latitude'].values
                tsec = sub['time'].values
                ve = sub['ve'].values
                vn = sub['vn'].values

                # Replace fill values -999999 with NaN
                for arr in (lon, lat, tsec, ve, vn):
                    bad = (arr == -999999) if np.issubdtype(arr.dtype, np.number) else np.zeros_like(arr, dtype=bool)
                    arr[bad] = np.nan

                # Decode time
                tdt = decode_unix_seconds_to_datetime(tsec)

                mask = (~np.isnan(lon) & ~np.isnan(lat) & (tdt != None) &
                        ~np.isnan(ve) & ~np.isnan(vn))
                if np.any(mask):
                    idxs = np.nonzero(mask)[0]
                    pool.append({
                        'file': Path(f).name,
                        'traj_index': it,
                        'lon': lon[mask],
                        'lat': lat[mask],
                        'time': np.array([tdt[k] for k in idxs], dtype=object),
                        've': ve[mask],
                        'vn': vn[mask],
                        'obs_index': idxs
                    })
            ds.close()
        except Exception as e:
            print(f"Warning: could not read drifter file {f}: {e}")
            continue
    return pool

def find_best_drifter_match(lon0, lat0, t0, pool, max_dist_km, max_dt_hours):
    """
    Among all drifter samples, find the nearest in space (within max_dist_km) and optionally within max_dt_hours.
    Returns dict with match details or None.
    """
    best = None
    best_dist = 1e18
    best_dt = 1e18

    for chunk in pool:
        # Time filter if enabled
        if t0 is not None and max_dt_hours is not None:
            dt_hours = np.array([abs((tt - t0).total_seconds())/3600.0 for tt in chunk['time']])
            valid_time = dt_hours <= max_dt_hours
            if not np.any(valid_time):
                continue
            cand_idx = np.nonzero(valid_time)[0]
        else:
            cand_idx = np.arange(len(chunk['lon']))

        # Compute distances for candidates
        for j in cand_idx:
            dkm = haversine_km(lat0, lon0, chunk['lat'][j], chunk['lon'][j])
            if dkm <= max_dist_km:
                # Favor smaller distance; break ties with smaller |dt|
                if t0 is not None:
                    dt_h = abs((chunk['time'][j] - t0).total_seconds())/3600.0
                else:
                    dt_h = np.nan
                better = (dkm < best_dist) or (np.isclose(dkm, best_dist) and (dt_h < best_dt))
                if better:
                    best = {
                        'drifter_file': chunk['file'],
                        'traj_index': chunk['traj_index'],
                        'obs_index': int(chunk['obs_index'][j]),
                        'drifter_lon': float(chunk['lon'][j]),
                        'drifter_lat': float(chunk['lat'][j]),
                        'drifter_time': chunk['time'][j],
                        'drifter_ve': float(chunk['ve'][j]),
                        'drifter_vn': float(chunk['vn'][j]),
                        'distance_km': float(dkm),
                        'time_diff_hours': float(dt_h) if t0 is not None else np.nan
                    }
                    best_dist = dkm
                    best_dt = dt_h
    return best

def open_guv_timeseries(sat, track_self, track_other):
    """
    Open guv_at_intersection_{sat}_{track_self}_{track_other}.nc
    Expect variables:
      time (days since 2000-01-11, proleptic_gregorian)
      gu(time), gv(time)
    """
    fpath = os.path.join(GUV_FOLDER, f"guv_at_intersection_{sat}_{int(track_self)}_{int(track_other)}.nc")
    if not os.path.exists(fpath):
        return None, f"File not found: {fpath}"
    try:
        ds = xr.open_dataset(fpath)
        time_vals = ds['time'].values
        tdt = decode_sat_time_days_since_2000_01_11(time_vals)
        gu = ds['gu'].values if 'gu' in ds.variables else np.full_like(time_vals, np.nan, dtype=float)
        gv = ds['gv'].values if 'gv' in ds.variables else np.full_like(time_vals, np.nan, dtype=float)
        ds.close()
        return {'time': tdt, 'gu': gu, 'gv': gv, 'path': fpath}, None
    except Exception as e:
        return None, f"Error reading {fpath}: {e}"

# ------------------------
# Main processing
# ------------------------

def main():
    # Build drifter pool once
    print("Loading drifter samples...")
    drifter_pool = build_drifter_pool(DRIFTER_GLOB)
    total_samples = sum(len(ch['lon']) for ch in drifter_pool)
    print(f"Drifter pool: {len(drifter_pool)} traj-chunks, {total_samples} samples")

    # Collect results
    rows = []

    csv_files = sorted(glob.glob(INTERSECTION_CSV_GLOB))
    print(f"Found {len(csv_files)} intersection CSV files")
    for csv in csv_files:
        print(f"Processing intersections from {csv} ...")
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"Could not read {csv}: {e}")
            continue

        # Expect columns: sat, track_self, track_other, lons_inter, lats_inter, ...
        needed = {'sat','track_self','track_other','lons_inter','lats_inter'}
        if not needed.issubset(df.columns):
            print(f"Missing required columns in {csv}, skipping")
            continue

        for _, row in df.iterrows():
            sat = str(row['sat'])
            track_self = int(row['track_self'])
            track_other = int(row['track_other'])
            lon_inter = float(row['lons_inter'])
            lat_inter = float(row['lats_inter'])

            # Load the satellite time series at this intersection
            guv, err = open_guv_timeseries(sat, track_self, track_other)
            if err is not None:
                print(err)
                continue

            # Estimate a local track bearing at the intersection for projection
            track_bearing = local_track_bearing(lat_inter, lon_inter)

            # Iterate over satellite times
            times = guv['time']
            gu_vals = guv['gu']
            gv_vals = guv['gv']

            for i in range(len(times)):
                t_sat = times[i]
                if t_sat is None:
                    continue
                gu_i = gu_vals[i]
                gv_i = gv_vals[i]
                if np.isnan(gu_i) and np.isnan(gv_i):
                    continue

                # Find nearest drifter sample
                match = find_best_drifter_match(
                    lon_inter, lat_inter, t_sat, drifter_pool,
                    max_dist_km=MAX_DISTANCE_KM,
                    max_dt_hours=MAX_TIME_HOURS
                )
                if match is None:
                    # Optionally record no-match for diagnostics; here we skip
                    continue

                # Project drifter to along/cross of the track frame
                v_along, v_cross = en_to_along_cross(match['drifter_ve'], match['drifter_vn'], track_bearing)

                rows.append({
                    'sat': sat,
                    'track_self': track_self,
                    'track_other': track_other,
                    'lon_inter': lon_inter,
                    'lat_inter': lat_inter,
                    'track_bearing_deg': track_bearing,
                    'sat_time': t_sat.isoformat(),
                    'sat_time_index': i,
                    'gu_along': float(gu_i),  # along-track satellite component
                    'gv_cross': float(gv_i),  # cross-track satellite component
                    'drifter_file': match['drifter_file'],
                    'drifter_traj_index': match['traj_index'],
                    'drifter_obs_index': match['obs_index'],
                    'drifter_lon': match['drifter_lon'],
                    'drifter_lat': match['drifter_lat'],
                    'drifter_time': match['drifter_time'].isoformat(),
                    'drifter_ve_east': match['drifter_ve'],
                    'drifter_vn_north': match['drifter_vn'],
                    'drifter_valong': v_along,
                    'drifter_vcross': v_cross,
                    'separation_km': match['distance_km'],
                    'abs_dt_hours': match['time_diff_hours']
                })

    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(out_df)} collocations to {OUTPUT_CSV}")
        # Print simple diagnostics
        print("Quick diagnostics on matched pairs:")
        print(f"  Mean separation: {out_df['separation_km'].mean():.2f} km")
        print(f"  Median |dt|: {out_df['abs_dt_hours'].median():.2f} h")
        # Optional: correlations (drop NaNs)
        with pd.option_context('mode.use_inf_as_na', True):
            valid = out_df[['gv_cross','drifter_vcross']].dropna()
            if len(valid) >= 10:
                r = valid['gv_cross'].corr(valid['drifter_vcross'])
                print(f"  Corr(gv vs drifter_vcross): {r:.3f}")
            valid2 = out_df[['gu_along','drifter_valong']].dropna()
            if len(valid2) >= 10:
                r2 = valid2['gu_along'].corr(valid2['drifter_valong'])
                print(f"  Corr(gu vs drifter_valong): {r2:.3f}")
    else:
        print("No collocations found with current thresholds.")

if __name__ == "__main__":
    main()

