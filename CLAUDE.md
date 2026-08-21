# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Oceanographic research codebase for computing and validating **satellite altimetry-derived
geostrophic currents** in the North Indian Ocean (NIO) from X-TRACK data (v2.2). It processes
sea level anomaly (SLA) from 8 satellite missions, derives cross-track geostrophic velocities,
decomposes them into zonal/meridional components at track intersection points, and validates
against in-situ observations (drifters, HF radar, RAMA/OMNI/coastal buoys) and gridded products
(OSCAR, AVISO MSLA-UV).

This is a collection of ~380 standalone analysis scripts sharing one library — not a package.
There is no `setup.py`, `requirements.txt`, or test suite.

## Running Scripts

```bash
# The venv is auto-activated by ~/.zshrc; PYTHONPATH already includes lib/
python <script>.py
```

Scripts **must be run from the directory that contains them** — they use relative paths
(`../data/{SAT}/...`, `../computed/sla_loess_0.05a/...`) and read per-directory copies of
`tracks_intersections_{sat}_1.csv`:

```bash
cd computed && python c.loess_0.05_smooth.all.py
cd guv     && python c.guv_at_intersection_points_loess_0.05.py
```

Environment (already configured on this machine, needed if reproducing elsewhere):
- venv `~/venvs/pybase` (Python 3.13)
- `export PYTHONPATH=$PYTHONPATH:/home/srinivasu/xtrackm/lib`
- Raw data under `data/` and external datasets under `/home/srinivasu/allData/`
  (aviso, oscar, ccmp31, drifter1, topo, OMNI/RAMA, radar) — none of it is in git.

`code2pdf.py` bundles all `.py`/`.txt` into a single syntax-highlighted PDF
(`python code2pdf.py --dir lib/ guv/ --exclude archive/`).

## Directory Pipeline

Work progresses through directories in this order (see `workflow.txt`):

1. **`data/`** — download and lon-ordered linking of raw X-TRACK NetCDF (8 missions)
2. **`meta/`** — metadata, track intersection points, temporal/spatial coverage, outlier detection
3. **`schema/`** — schematic diagrams of cross-track geometry and vector decomposition
4. **`computed/`** — along-track SLA smoothing (outputs `computed/sla_{method}_{frac}a/`), metrics comparing smoothing methods
5. **`guv/`** — geostrophic velocity components at intersection points; validation vs OSCAR/AVISO/buoys
6. **`eicc/`** — East India Coastal Current analysis (transects, trends, Hovmöllers, periodograms)
7. **`ekman/`** — Ekman currents from CCMP winds, combined with geostrophic
8. **`drifter/`, `drifter_track_reg/`** — drifter validation
9. **`radar/`** — HF radar (CODAR/WERA) validation for AP/OR/TN regions
10. **`selected/`, `draft_figs02/`, `ppt_figs/`** — curated results and publication figures

Every directory has an `archive/` subdir of superseded scripts — ignore it unless asked.

## Core Library (`lib/`)

- **`tools_xtrackm.py`** (~2300 lines) — the working library: smoothing (LOESS, Gaussian, box,
  Butterworth, optimal difference operator), geostrophy, MAD filtering, track geometry,
  buoy/track matching, plotting helpers
- **`namesm.py`** — all constants: region bounds, mission names/repeat periods, 160+ buoy station
  IDs and coordinates, data paths
- **`tools_shapely.py`** — coastline geometry (smoothing, tangents)
- **`tools_xtrackm_c.py`** — an **in-progress simplified rewrite** of the along-track reshaping
  helpers (see `tools_xtrackm_c_summary.txt` and `lib/prompts.txt`). Not yet used by any script;
  don't route new work through it unless asked.
- **`dump.py`** — debugging/serialization helpers

## Geostrophic Current Pipeline

1. Load raw SLA along a satellite ground track (`decode_times=False`; X-TRACK time is non-standard)
2. Reorient to distance-from-coast coordinate with `track_dist_time_asn()` /
   `track_dist_time_asn_midx()` — `x` increases **offshore**, arrays are reversed so index `-1`
   is nearest the coast
3. Smooth SLA along-track — LOESS `frac=0.05` is the primary method (`track_smooth_loess_x`)
4. Cross-track geostrophic current `gc = (g/f)·d(sla)/dx` via `compute_geostrophy_gc()`;
   it masks |lat| < 2° (Coriolis → 0, per Sudre 2013)
5. At an intersection, get the local track heading with `local_angle(lons, lats, ii, ioff=3)`
   (radians) and solve the two cross-track currents into (u, v) with
   `geostrophic_components(gc1, gc2, a1_r, a2_r)` — asserts angles are in radians
6. Filter outliers with `mad_filter_1d_xr(da, 3.5, dim="time")`
7. Validate against OSCAR / AVISO / drifters / buoys / radar, optionally after adding Ekman
   (`compute_ekman_currents_wenbo_wang`)

Several older scripts define a local `geostrophic_components_from_a(gc1, gc2, a1, a2)` copy
instead of importing `geostrophic_components` — they are equivalent; prefer the library one.

## Conventions

### Script name prefixes

| Prefix | Meaning |
|--------|---------|
| `c.` | **compute** — processing that writes NetCDF/CSV/PKL output |
| `p.` | **plot** — figures (the bulk of the repo) |
| `f.` | **figure** — publication-quality figure |
| `l.` | **list/learn** — exploratory diagnostics, prints to stdout |
| `d.` | **debug** visualization |

In `draft_figs02/`, publication scripts are prefixed `figNN_` plus the original script name, with
the matching `.png` and a `.txt` caption alongside.

### Import pattern

```python
from tools_xtrackm import *   # pulls in namesm.py and tools_shapely.py transitively
```

### Variable abbreviations

`sla` sea level anomaly · `gc` cross-track geostrophic current · `gu`/`gv` zonal/meridional
geostrophic components · `ve`/`vn` eastward/northward from drifters or models · `x` /
`x_from_coast` distance from coast · `sat` mission id string · `ioff` index offset for local
angle (usually 3) · `frac` LOESS fraction · `ek` Ekman.

### Key constants (`namesm.py`)

- `NIO = (40, 110, -5, 30)`, `TRACKS_REG = (43, 99, 0, 25.4)`
- `sats_new = ("GFO", "TP+J1+J2+J3+S6A", "S3A", "S3B", "TPN+J1N+J2N+J3N", "ERS1+ERS2+ENV+SRL", "HY2A", "HY2B")`
  — scripts often shadow this with a short list for testing; check before assuming a full run
- `sats_tfreq` — repeat period in days per mission
- Track file pattern: `data/{SAT}/ctoh.sla.ref.{SAT}.nindian.{PASS:03d}.nc`
- `chunks = ["1_5000", "5001_10000", "10001_15000", "15001_current"]` (drifter data chunks)

### Data formats

NetCDF via xarray (tracks, computed fields) · CSV (intersection metadata, validation metrics) ·
Pickle (cached DataFrames/track lists) · PNG at dpi=300 · shapefiles for radar coverage polygons.

## Git

`.gitignore` excludes all data and figures (`*.nc`, `*.pkl`, `*.png`, …). The one exception is
`draft_figs02/f*.png`, tracked for publication figures. Commit messages in this repo are terse
("update"); match that unless asked otherwise.

## Open issues and notes

`TODO.txt`, `Issues.txt` (and per-directory `Notes.txt`, `Issues.txt`, `prompts.txt`) carry the
active research questions — e.g. whether `gc_other_at.interp(time=gc_self_at.time)` is the right
way to align the two tracks' time axes at an intersection. Read them before changing the
intersection or validation logic.
