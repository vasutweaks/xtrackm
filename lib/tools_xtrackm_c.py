"""Simplified X-TRACK along-track helpers (rewrite of tools_xtrackm.py).

X-TRACK L2P files store every variable on (points_numbers, cycles_numbers),
with the *first* point on the equator and the *last* point on the coast.
That layout is awkward to slice and to plot.  The functions here reshape a
track onto physically meaningful axes:

    x     along-track distance from the coast, increasing offshore
    time  one time stamp per cycle
    lat   latitude of each along-track point   (indexed, so ``.sel`` works)
    lon   longitude of each along-track point  (indexed, so ``.sel`` works)

``x`` is the geodesic distance between the coastal end point of the track and
each track point on the WGS84 ellipsoid, computed with Karney (2013) via
``pyproj.Geod`` -- the same quantity the draft calls ``s``.
"""

import os

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod

__all__ = [
    "REF_EPOCH",
    "along_track_dist",
    "track_time",
    "track_dist_time_asn_midx",
    "sel_track",
    "local_angle",
    "track_angle_at",
]

REF_EPOCH = "1950-01-01"  # X-TRACK time units: "days since 1950-1-1"

_GEOD = Geod(ellps="WGS84")
_DIST_SCALE = {"m": 1.0, "km": 1.0e-3}

_POINT_DIM = "points_numbers"
_CYCLE_DIM = "cycles_numbers"


def _scale(units_in):
    try:
        return _DIST_SCALE[units_in]
    except KeyError:
        raise ValueError(
            f"units_in must be one of {sorted(_DIST_SCALE)}, got {units_in!r}"
        ) from None


def along_track_dist(ds, units_in="m", dist_from="track"):
    """Distance to the coast for every track point, in file (equator->coast) order.

    dist_from="track"  geodesic distance to the coastal end point of the track
    dist_from="gshhg"  the ``dist_to_coast_gshhg`` variable shipped with the file
                       (distance to the *nearest* coastline, so not monotonic)
    """
    if ds.sizes[_POINT_DIM] == 0:
        raise ValueError("this track has no along-track points")

    if dist_from == "gshhg":
        dists = np.asarray(ds["dist_to_coast_gshhg"].values, dtype=float)
    elif dist_from == "track":
        lons = np.asarray(ds["lon"].values, dtype=float)
        lats = np.asarray(ds["lat"].values, dtype=float)
        _, _, dists = _GEOD.inv(
            np.full_like(lons, lons[-1]),
            np.full_like(lats, lats[-1]),
            lons,
            lats,
        )
    else:
        raise ValueError(
            f"dist_from must be 'track' or 'gshhg', got {dist_from!r}"
        )
    return dists * _scale(units_in)


def track_time(ds):
    """One ``datetime64`` stamp per cycle, as a pandas DatetimeIndex.

    A pass takes a few tens of seconds, so the median of the valid point times
    represents the whole cycle.  Cycles with no valid time at all are filled by
    linear interpolation/extrapolation against the cycle number, which is far
    safer than the ffill/bfill the old code used (repeated stamps would make the
    time index non-unique).
    """
    time = ds["time"].transpose(_POINT_DIM, _CYCLE_DIM)
    values = time.values

    if np.issubdtype(values.dtype, np.number):
        # decode_times=False: raw "days since REF_EPOCH"
        days = np.asarray(values, dtype=float)
    elif np.issubdtype(values.dtype, np.datetime64):
        days = (values - np.datetime64(REF_EPOCH)) / np.timedelta64(1, "D")
    else:
        # cftime objects (the files declare a julian calendar); go back to the
        # raw numbers so that decoded and undecoded input give the same answer
        calendar = getattr(values.flat[0], "calendar", "standard")
        days = cftime.date2num(
            values.ravel(), f"days since {REF_EPOCH}", calendar=calendar
        ).reshape(values.shape)

    days = np.where(np.isfinite(days), days, np.nan)
    valid = np.isfinite(days).any(axis=0)  # cycles that have at least one stamp
    per_cycle = np.full(days.shape[1], np.nan)
    per_cycle[valid] = np.nanmedian(days[:, valid], axis=0)

    bad = ~np.isfinite(per_cycle)
    if bad.any():
        if bad.all():
            raise ValueError("no valid time stamps in this track")
        cycles = np.asarray(ds[_CYCLE_DIM].values, dtype=float)
        per_cycle = np.interp(cycles, cycles[~bad], per_cycle[~bad])

    return pd.DatetimeIndex(
        pd.Timestamp(REF_EPOCH) + pd.to_timedelta(per_cycle, unit="D"), name="time"
    )


def track_dist_time_asn_midx(ds, var_str="sla", units_in="m", dist_from="track"):
    """Reshape one X-TRACK variable onto (x, time) with lat/lon also indexed.

    Returns a DataArray with dim ``x`` (distance from coast, increasing
    offshore) and dim ``time``.  ``lat``, ``lon`` and the cycle number ride
    along as coordinates and carry their own indexes, so the same array can be
    sliced along distance, along time or along latitude::

        da = track_dist_time_asn_midx(ds, "sla", units_in="km")
        da.sel(x=slice(0, 200))                    # first 200 km from the coast
        da.sel(time=slice("2005", "2010"))         # a date range
        da.sel(lat=15.0, method="nearest")         # the point nearest 15 N
        da.sel(lat=slice(20, 10))                  # a latitude band

    ``ds`` may be opened with or without ``decode_times``.
    """
    dists = along_track_dist(ds, units_in=units_in, dist_from=dist_from)
    times = track_time(ds)

    var = ds[var_str]
    if _POINT_DIM not in var.dims:
        raise ValueError(f"{var_str!r} has no {_POINT_DIM} dimension")

    has_cycles = _CYCLE_DIM in var.dims

    # Flip so index 0 sits on the coast; transpose so dims are (point, cycle).
    src_dims = [_POINT_DIM] + ([_CYCLE_DIM] if has_cycles else [])
    out_dims = ["x"] + (["time"] if has_cycles else [])
    values = (
        var.load()
        .isel({_POINT_DIM: slice(None, None, -1)})
        .transpose(*src_dims)
        .values
    )

    coords = {
        "x": dists[::-1],
        "lon": ("x", np.asarray(ds["lon"].values, dtype=float)[::-1]),
        "lat": ("x", np.asarray(ds["lat"].values, dtype=float)[::-1]),
        "points_numbers": ("x", np.asarray(ds[_POINT_DIM].values)[::-1]),
    }
    if has_cycles:
        coords["time"] = times
        coords["cycles_numbers"] = ("time", np.asarray(ds[_CYCLE_DIM].values))

    out = xr.DataArray(
        values, dims=out_dims, coords=coords, name=var.name, attrs=dict(var.attrs)
    )

    out.coords["x"].attrs.update(
        units=units_in,
        long_name="distance from coast",
        dist_from=dist_from,
    )
    out.coords["lat"].attrs.update(units="degrees_north", long_name="Latitude")
    out.coords["lon"].attrs.update(units="degrees_east", long_name="Longitude")

    # Extra indexes let .sel() work on lat/lon as well as on x and time.
    for name in ("lat", "lon"):
        out = out.set_xindex(name)
    return out


def sel_track(da, x=None, lat=None, lon=None, time=None, method="nearest"):
    """Select along distance, latitude, longitude and/or time in one call.

    Each argument takes either a scalar (nearest point) or a ``(lo, hi)`` pair
    (inclusive range, given in natural order regardless of how the coordinate
    runs along the track)::

        sel_track(da, x=(0, 200_000))          # 0-200 km from the coast
        sel_track(da, lat=12.5)                # nearest along-track point
        sel_track(da, lat=(10, 20), time=("2005", "2010"))
    """
    for name, arg in (("x", x), ("lat", lat), ("lon", lon), ("time", time)):
        if arg is None:
            continue
        if isinstance(arg, (tuple, list)) and len(arg) == 2:
            lo, hi = arg
            if not da.indexes[name].is_monotonic_increasing:
                lo, hi = hi, lo  # coordinate runs the other way along the track
            da = da.sel({name: slice(lo, hi)})
        else:
            da = da.sel({name: arg}, method=method)
    return da


def local_angle(lons, lats, ii, ioff=3, degrees=False):
    """Track heading at point ``ii``, measured counter-clockwise from east.

    The heading points in the direction of **increasing index**, so the order of
    ``lons``/``lats`` is part of the answer: reversing them rotates the result by
    180 deg.  X-TRACK files run equator -> coast, so passing ``ds.lon.values``
    straight in gives the heading of the equator -> coast direction, which is the
    convention ``compute_geostrophy_gc`` and ``geostrophic_components`` expect.

    The angle is the true geodesic azimuth (Karney, 2013), taken across a
    centred ``2 * ioff`` stencil and averaged between its two ends so it refers
    to ``ii`` rather than to the start of the stencil.  The old flat-earth
    ``atan2(dlat, dlon)`` ignored the convergence of the meridians, which biases
    the heading by up to ~2 deg at 25N in this region.

    Returns NaN rather than a wrong number if the stencil is degenerate: fewer
    than two points, ``ii`` out of range, coincident end points, or a NaN at
    ``ii`` or at either end of the stencil.

    Prefer :func:`track_angle_at`, which takes a latitude/longitude/distance
    instead of an index and so cannot be given an index from the wrong ordering.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    ln = lons.size
    if ln < 2 or not (0 <= ii < ln):
        return np.nan

    ioff = max(int(ioff), 1)
    if ii - ioff < 0:  # near the start: one-sided, forward
        i0, i1 = ii, min(ii + ioff, ln - 1)
    elif ii + ioff >= ln:  # near the end: one-sided, backward
        i0, i1 = max(ii - ioff, 0), ii
    else:  # centred
        i0, i1 = ii - ioff, ii + ioff

    pts = (lons[i0], lats[i0], lons[i1], lats[i1])
    if i0 == i1 or not np.all(np.isfinite(pts + (lons[ii], lats[ii]))):
        return np.nan

    # az12 is the azimuth at i0, az21 the azimuth at i1 looking back; averaging
    # az12 with (az21 + 180) gives the azimuth in the middle of the stencil.
    az12, az21, dist = _GEOD.inv(*pts)
    if dist == 0.0:
        return np.nan
    az = np.degrees(
        np.arctan2(
            np.sin(np.radians(az12)) + np.sin(np.radians(az21 + 180.0)),
            np.cos(np.radians(az12)) + np.cos(np.radians(az21 + 180.0)),
        )
    )

    angle = (90.0 - az + 180.0) % 360.0 - 180.0  # CW from north -> CCW from east
    return angle if degrees else np.radians(angle)


def track_angle_at(ds, lat=None, lon=None, x=None, units_in="m", ioff=3,
                   degrees=False, order="file"):
    """Track heading at the point nearest a given latitude, longitude or distance.

    Give exactly one of ``lat``, ``lon`` or ``x`` (``x`` is distance from the
    coast, in ``units_in``).  The index is resolved and the heading computed in
    the same array order, which is what ``index_at_lat`` + ``local_angle`` in the
    old module got wrong: ``index_at_lat`` returned an index into the reversed
    (coast-first) array while the call sites passed the forward (equator-first)
    coordinates, so the heading came from the mirror-image point on the track.

    order="file"   heading of the equator -> coast direction (the default, and
                   the convention ``geostrophic_components`` expects)
    order="coast"  heading of the coast -> offshore direction, i.e. 180 deg away
                   and aligned with the ``x`` axis of
                   :func:`track_dist_time_asn_midx`

    Returns ``(angle, ii)`` -- the heading and the file-order index used.
    """
    given = [k for k, v in (("lat", lat), ("lon", lon), ("x", x)) if v is not None]
    if len(given) != 1:
        raise ValueError(f"give exactly one of lat, lon, x -- got {given}")

    if lat is not None:
        target = np.asarray(ds["lat"].values, dtype=float) - lat
    elif lon is not None:
        target = np.asarray(ds["lon"].values, dtype=float) - lon
    else:
        target = along_track_dist(ds, units_in=units_in) - x

    if not np.isfinite(target).any():
        raise ValueError("no finite coordinates to search")
    ii = int(np.nanargmin(np.abs(target)))

    angle = local_angle(
        ds["lon"].values, ds["lat"].values, ii, ioff=ioff, degrees=degrees
    )
    if order == "coast":
        half = 180.0 if degrees else np.pi
        angle = (angle + half + half) % (2 * half) - half
    elif order != "file":
        raise ValueError(f"order must be 'file' or 'coast', got {order!r}")
    return angle, ii



# ----------------------------------------------------------------------------
# Usage examples.  Run this file directly to print the examples and write the
# demo figures:   python tools_xtrackm_c.py [track_file.nc] [outdir]
# ----------------------------------------------------------------------------

DEMO_FILE = "/home/srinivasu/xtrackm/data/GFO/ctoh.sla.ref.GFO.nindian.008.nc"


def demo_selections(ds):
    """Print the different ways a reshaped track can be sliced."""
    show = lambda label, obj: print(f"  {label:<46} {dict(obj.sizes)}")

    # --- 1. reshape one variable ---------------------------------------------
    da = track_dist_time_asn_midx(ds, "sla", units_in="km")
    print("\n1. reshaped SLA")
    print(f"  dims    {da.dims} {da.shape}")
    print(f"  indexes {list(da.indexes)}")
    print(f"  x       {da.x.values[0]:.1f} -> {da.x.values[-1]:.1f} {da.x.units}")
    print(f"  lat     {da.lat.values[0]:.2f} -> {da.lat.values[-1]:.2f}")
    print(f"  time    {da.time.values[0]} -> {da.time.values[-1]}")

    # --- 2. along distance from the coast ------------------------------------
    print("\n2. along distance from coast (x)")
    show("da.sel(x=slice(0, 200))", da.sel(x=slice(0, 200)))
    show("da.sel(x=50, method='nearest')", da.sel(x=50, method="nearest"))
    show("da.sel(x=slice(0, 100)).mean('time')", da.sel(x=slice(0, 100)).mean("time"))
    print(f"  {'std of SLA vs distance (first 3)':<46} "
          f"{np.round(da.std('time').values[:3], 4)}")

    # --- 3. along time --------------------------------------------------------
    print("\n3. along time")
    show("da.sel(time=slice('2005', '2010'))", da.sel(time=slice("2005", "2010")))
    show("da.sel(time='2003-06', method='nearest')",
         da.sel(time="2003-06-15", method="nearest"))
    show("da.groupby('time.month').mean()", da.groupby("time.month").mean())
    show("da.resample(time='1YE').mean()", da.resample(time="1YE").mean())

    # --- 4. along latitude (and longitude) ------------------------------------
    print("\n4. along latitude / longitude")
    show("da.sel(lat=8.0, method='nearest')", da.sel(lat=8.0, method="nearest"))
    show("da.sel(lat=slice(12, 5))   # lat runs downward", da.sel(lat=slice(12, 5)))
    show("da.sel(lon=95.0, method='nearest')", da.sel(lon=95.0, method="nearest"))
    print(f"  {'point nearest 8N sits at':<46} "
          f"x={float(da.sel(lat=8.0, method='nearest').x):.1f} km")

    # --- 5. sel_track: the same thing without worrying about direction --------
    print("\n5. sel_track (scalars or (lo, hi) pairs, natural order)")
    show("sel_track(da, x=(0, 200))", sel_track(da, x=(0, 200)))
    show("sel_track(da, lat=(5, 12))", sel_track(da, lat=(5, 12)))
    show("sel_track(da, x=(0, 300), time=('2005', '2010'))",
         sel_track(da, x=(0, 300), time=("2005", "2010")))
    show("sel_track(da, lat=12.0)", sel_track(da, lat=12.0))

    # --- 6. any other variable ------------------------------------------------
    print("\n6. other variables (same call, just change var_str)")
    for name in ("sla", "swh", "dac", "ocean_tide", "sigma0_ku"):
        v = track_dist_time_asn_midx(ds, name, units_in="km")
        print(f"  {name:<12} {str(v.dims):<18} {v.attrs.get('units',''):<4} "
              f"{v.attrs.get('long_name','')}")
    for name in ("mssh", "mdt", "dist_to_coast_gshhg"):
        v = track_dist_time_asn_midx(ds, name, units_in="km")
        print(f"  {name:<12} {str(v.dims):<18} {v.attrs.get('units',''):<4} "
              f"{v.attrs.get('long_name','')}  (1-D, no time)")

    # --- 7. units and distance source ----------------------------------------
    print("\n7. units_in and dist_from")
    for units_in in ("m", "km"):
        v = track_dist_time_asn_midx(ds, "sla", units_in=units_in)
        print(f"  units_in={units_in!r:<5} x[-1] = {v.x.values[-1]:14.2f} {v.x.units}")
    g = track_dist_time_asn_midx(ds, "sla", units_in="km", dist_from="gshhg")
    print(f"  dist_from='gshhg'  x[:3] = {np.round(g.x.values[:3], 2)}"
          "  (nearest coastline, not monotonic)")

    # --- 8. the pieces on their own ------------------------------------------
    print("\n8. helpers used directly")
    d = along_track_dist(ds, units_in="km")
    print(f"  along_track_dist(ds, 'km')  equator {d[0]:.1f} -> coast {d[-1]:.1f}")
    print(f"  track_time(ds)              {len(track_time(ds))} cycles, "
          f"median gap {np.median(np.diff(track_time(ds))) / np.timedelta64(1, 'D'):.1f} d")

    # --- 9. track heading ----------------------------------------------------
    print("\n9. track heading (CCW from east)")
    for lat0 in (5.0, 10.0, 12.0):
        ang, ii = track_angle_at(ds, lat=lat0, degrees=True)
        print(f"  track_angle_at(ds, lat={lat0:<5}) {ang:8.3f} deg  "
              f"at point {ii:3d} ({float(ds.lat[ii]):.2f}N)")
    ang_c, _ = track_angle_at(ds, lat=10.0, degrees=True, order="coast")
    print(f"  {'same point, order=coast':<28} {ang_c:8.3f} deg  (180 deg away)")
    ang_x, ii = track_angle_at(ds, x=200.0, units_in="km", degrees=True)
    print(f"  {'selected by distance instead':<28} {ang_x:8.3f} deg  "
          f"at x = 200 km ({float(ds.lat[ii]):.2f}N)")
    print(f"  {'local_angle, explicit index':<28} "
          f"{local_angle(ds.lon.values, ds.lat.values, ii, degrees=True):8.3f} deg")
    return da


def demo_plots(ds, outdir="."):
    """Write the demo figures and return their paths."""
    import matplotlib.pyplot as plt

    da = track_dist_time_asn_midx(ds, "sla", units_in="km")
    near = sel_track(da, x=(0, 400))  # the coastal band
    paths = []

    # --- figure 1: the three axes the reshaping buys you ---------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    near.plot(ax=ax[0, 0], x="time", y="x", vmin=-0.3, vmax=0.3, cmap="RdBu_r")
    ax[0, 0].set_title("(a) along distance and time")

    near.mean("time").plot(ax=ax[0, 1], marker="o", ms=3)
    near.std("time").plot(ax=ax[0, 1], marker="o", ms=3)
    ax[0, 1].legend(["mean", "std dev"])
    ax[0, 1].set_ylabel(f"SLA [{da.attrs.get('units', '')}]")
    ax[0, 1].set_title("(b) along distance from coast")
    ax[0, 1].axhline(0, lw=0.6, color="k")

    for lat0 in (12.0, 10.0, 8.0):
        s = sel_track(da, lat=lat0)
        s.plot(ax=ax[1, 0], lw=0.9, label=f"{float(s.lat):.1f}N (x={float(s.x):.0f} km)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("(c) along time, at fixed latitudes")

    near.mean("time").plot(ax=ax[1, 1], y="lat", marker="o", ms=3)
    ax[1, 1].set_title("(d) along latitude")
    ax[1, 1].axvline(0, lw=0.6, color="k")

    fig.suptitle(f"track_dist_time_asn_midx: SLA on (x, time, lat) "
                 f"-- {os.path.basename(getattr(ds, 'encoding', {}).get('source', ''))}")
    paths.append(os.path.join(outdir, "tools_xtrackm_c_demo1_axes.png"))
    fig.savefig(paths[-1], dpi=110)
    plt.show()
    plt.close(fig)

    # --- figure 2: the same call for other variables -------------------------
    variables = [("sla", -0.3, 0.3, "RdBu_r"), ("swh", 0, 3, "viridis"),
                 ("dac", -0.3, 0.3, "RdBu_r"), ("sigma0_ku", 8, 20, "magma")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axi, (name, vmin, vmax, cmap) in zip(axes.ravel(), variables):
        v = sel_track(track_dist_time_asn_midx(ds, name, units_in="km"), x=(0, 400))
        v.plot(ax=axi, x="time", y="x", vmin=vmin, vmax=vmax, cmap=cmap)
        axi.set_title(f"{name} -- {v.attrs.get('long_name', '')}", fontsize=10)
    fig.suptitle("any variable, same call: track_dist_time_asn_midx(ds, var_str=...)")
    paths.append(os.path.join(outdir, "tools_xtrackm_c_demo2_variables.png"))
    fig.savefig(paths[-1], dpi=110)
    plt.show()
    plt.close(fig)

    # --- figure 3: what the time and distance axes make easy -----------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    near.groupby("time.month").mean().plot(
        ax=ax[0, 0], x="month", y="x", vmin=-0.15, vmax=0.15, cmap="RdBu_r")
    ax[0, 0].set_title("(a) seasonal cycle vs distance from coast")

    for band, label in (((0, 50), "0-50 km"), ((50, 150), "50-150 km"),
                        ((150, 400), "150-400 km")):
        sel_track(da, x=band).mean("x").rolling(
            time=6, min_periods=3, center=True).mean().plot(ax=ax[0, 1], lw=1, label=label)
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("(b) band means vs time (6-cycle running mean)")
    ax[0, 1].axhline(0, lw=0.6, color="k")

    da.count("time").plot(ax=ax[1, 0], marker=".", ls="none")
    ax[1, 0].set_ylabel("number of valid cycles")
    ax[1, 0].set_title("(c) valid cycles vs distance from coast")

    da.std("time").plot(ax=ax[1, 1], marker=".", ls="none")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_ylabel(f"SLA std dev [{da.attrs.get('units', '')}]")
    ax[1, 1].set_title("(d) SLA std dev vs distance (log x)")

    fig.suptitle("distance-from-coast diagnostics")
    paths.append(os.path.join(outdir, "tools_xtrackm_c_demo3_diagnostics.png"))
    fig.savefig(paths[-1], dpi=110)
    plt.show()
    plt.close(fig)
    return paths


if __name__ == "__main__":
    import sys

    fname = sys.argv[1] if len(sys.argv) > 1 else DEMO_FILE
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."

    print(f"file: {fname}")
    with xr.open_dataset(fname, decode_times=False) as ds:
        demo_selections(ds)
        print("\n10. figures")
        for path in demo_plots(ds, outdir):
            print(f"  wrote {path}")
