# Copilot Instructions — xtrackm

## Project Overview

This is an oceanographic research codebase for computing and validating **satellite altimetry-derived geostrophic currents** in the North Indian Ocean (NIO) using X-TRACK data (v2.2). It processes sea level anomaly (SLA) from 8 satellite missions, derives cross-track geostrophic velocities, decomposes them into zonal/meridional components at track intersection points, and validates against in-situ observations (drifters, HF radar, RAMA/OMNI buoys, OSCAR model).

## Running Scripts

Scripts are standalone Python files, not a package. Run from the directory containing the script:

```bash
cd computed && python c.loess_0.05_smooth.all.py
cd guv && python c.guv_at_intersection_points_loess_0.05.py
```

The `lib/` directory must be on `PYTHONPATH` (or scripts are run from directories where Python can resolve `from tools_xtrackm import *`). There is no `setup.py`, `requirements.txt`, or test suite.

## Architecture

### Directory Pipeline

Work progresses through directories in this order (defined in `workflow.txt`):

1. **`data/`** — Download and organize raw X-TRACK NetCDF files (8 satellite missions)
2. **`meta/`** — Extract metadata, find track intersection points, compute temporal coverage
3. **`schema/`** — Schematic diagrams explaining cross-track geometry and vector decomposition
4. **`computed/`** — Smooth SLA along-track, compute metrics comparing smoothing methods
5. **`guv/`** — Compute geostrophic velocity components (gu, gv) at intersection points, validate against OSCAR/buoys
6. **`eicc/`** — East Indian Coastal Current analysis (transects, trends, Hovmöller diagrams)
7. **`ekman/`** — Ekman current computation from wind data (CCMP), comparison with geostrophic
8. **`drifter/`**, **`drifter_track_reg/`** — Drifter buoy validation
9. **`radar/`** — HF radar (CODAR/WERA) validation
10. **`selected/`** — Curated final results for publication

### Core Library (`lib/`)

- **`tools_xtrackm.py`** (~2300 lines) — Main utility library: smoothing (LOESS, Gaussian, box, Butterworth, optimal difference operator), geostrophic current computation, MAD filtering, track geometry, buoy matching, plotting helpers
- **`namesm.py`** (~500 lines) — All project constants: region bounds, satellite mission names/frequencies, buoy station IDs (160+), data paths, physical constants
- **`tools_shapely.py`** — Coastline geometry operations (smoothing, tangent calculation)
- **`dump.py`** — Debugging/serialization utilities

### Geostrophic Current Pipeline

The core scientific computation:

1. Load raw SLA along a satellite ground track
2. Reorient to distance-from-coast coordinate using `track_dist_time_asn_midx()`
3. Smooth SLA along-track (LOESS frac=0.05 is the primary method)
4. Compute cross-track geostrophic current: `gc = (g/f) × dSLA/ds` using `compute_geostrophy_atan2()`
5. At track intersection points, solve two cross-track currents into (u, v) using `geostrophic_components_from_a(gc1, gc2, a1, a2)`
6. Filter outliers with `mad_filter_1d_xr()` (threshold 3.5)
7. Validate against observations (OSCAR, drifters, buoys, radar)

## Conventions

### File Naming Prefixes

| Prefix | Purpose | Example |
|--------|---------|---------|
| `c.` | **Compute** — data processing, produces intermediate output | `c.loess_0.05_smooth.all.py` |
| `p.` | **Plot** — visualization scripts that generate PNG figures | `p.std.S3A.common_tracks.with_mean.py` |
| `f.` | **Figure** — publication-quality figure scripts | `f.xtr.S3A.common_tracks.map.py` |
| `l.` | **List/Learn** — exploratory/diagnostic scripts | `l.lat_lon.dists2.py` |
| `d.` | **Debug/Display** — debugging visualization | `d.plot_candidate_drifters.for.intersections.py` |

### Variable Name Abbreviations

- **sla** — sea level anomaly (the primary satellite measurement)
- **gc** — cross-track geostrophic current (scalar along one track)
- **gu, gv** — zonal and meridional geostrophic velocity components
- **ve, vn** — eastward/northward velocity from drifters or models
- **x** or **x_from_coast** — distance from coast along track (meters or km)
- **sat** — satellite mission identifier string (e.g., `"S3A"`, `"TP+J1+J2+J3+S6A"`)
- **ioff** — index offset for local angle computation (typically 3)
- **frac** — LOESS smoothing fraction parameter

### Import Pattern

All scripts use wildcard imports from the library:

```python
from tools_xtrackm import *   # pulls in namesm.py and tools_shapely.py transitively
```

### Script Structure Pattern

Compute scripts typically follow:

```python
from tools_xtrackm import *

# Configuration at top
method = "loess"
frac1 = "0.05"
out_dir = f"/home/srinivasu/xtrackm/guv/guv_at_intersections_{method}/"
os.makedirs(out_dir, exist_ok=True)

# Loop over satellites
for sat in sats_new:
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    for i, r in df.iterrows():
        # Load track data, process, save
```

### Data Formats

- **NetCDF (.nc)** — primary format for satellite tracks and computed fields (xarray)
- **CSV** — intersection metadata, validation metrics
- **Pickle (.pkl)** — cached Python objects (DataFrames, track lists)
- **PNG** — all figures (dpi=300)

### Key Constants (from `namesm.py`)

- **Region**: NIO = (40°–110°E, 5°S–30°N)
- **Equatorial mask**: ±2° latitude excluded (Coriolis → 0)
- **8 satellites**: `sats_new` list in namesm.py
- **Satellite data path pattern**: `data/{SAT}/ctoh.sla.ref.{SAT}.nindian.{PASS}.nc`
- **Track chunks**: `["1_5000", "5001_10000", "10001_15000", "15001_current"]`

### Git Policy

Large binary files are gitignored (*.nc, *.pkl, *.png). Exception: `draft_figs02/f*.png` is tracked for publication figures.
