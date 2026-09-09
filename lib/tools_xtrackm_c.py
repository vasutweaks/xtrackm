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

import math
import os

import cftime
import numpy as np
import statsmodels.api as sm
import pandas as pd
import xarray as xr
from pyproj import Geod
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from scipy.stats import median_abs_deviation

__all__ = [
    "REF_EPOCH",
    "along_track_dist",
    "track_time",
    "track_dist_time_asn_midx",
    "sel_track",
    "local_angle",
    "track_angle_at",
    "OMEGA",
    "coriolis",
    "gravity",
    "cross_track_geostrophy",
    "resolve_uv",
    "mad_filter",
    "smooth_box",
    "smooth_gaussian",
    "smooth_loess",
    "lowpass_time",
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
# Outlier rejection and smoothing.
#
# Two rules the old implementations broke, and that everything here follows:
#   1. never invent data -- a point that was NaN on input stays NaN on output,
#      unless the caller explicitly asks for gaps up to a stated width to be
#      filled;
#   2. never silently destroy it either -- a single NaN inside a kernel must
#      not blank out the whole window.
# ----------------------------------------------------------------------------


def _mad_mask_1d(y, thresh):
    """Boolean outlier mask for one series, NaN treated as not-an-outlier."""
    good = np.isfinite(y)
    if good.sum() < 3:
        return np.zeros(y.shape, dtype=bool)
    v = y[good]
    med = np.median(v)
    sigma = median_abs_deviation(v, scale="normal")  # already MAD / 0.6745
    if sigma == 0:
        return np.zeros(y.shape, dtype=bool)
    z = np.abs(y - med) / sigma
    return good & (z > thresh)


def mad_filter(da, thresh=3.5, dim=None):
    """Blank median-absolute-deviation outliers along ``dim``.

    ``thresh`` is in robust standard deviations, so 3.5 means 3.5 sigma.
    ``tools_xtrackm.mad_filter_1d_xr`` computed
    ``0.6745 * (x - median) / median_abs_deviation(..., scale="normal")``,
    but ``scale="normal"`` has already divided by 0.6745, so the z-scores came
    out 0.6745x too small and a nominal 3.5 actually cut at 5.19 sigma -- on
    Gaussian noise that flags 0.000% of points where 3.5 sigma should flag
    0.057%.  It also rebuilt the output with only ``dim`` as a coordinate,
    dropping lat/lon/attrs, and only ever worked on 1-D input.

    Works on any number of dimensions and keeps every coordinate and attribute.
    Pass a tuple for ``dim`` to filter along each in turn, which is what
    ``mad_filter_2d_xr`` did.
    """
    if dim is None:
        for cand in ("time", "x"):
            if cand in da.dims:
                dim = cand
                break
        else:
            if da.ndim != 1:
                raise ValueError(
                    f"could not infer dim from {da.dims}; pass dim='time' or 'x'"
                )
            dim = da.dims[0]
    dims = (dim,) if isinstance(dim, str) else tuple(dim)

    values = np.asarray(da.values, dtype=float)
    for d in dims:
        if d not in da.dims:
            raise ValueError(f"dimension {d!r} not in {da.dims}")
        axis = da.get_axis_num(d)
        mask = np.apply_along_axis(_mad_mask_1d, axis, values, thresh)
        values = np.where(mask, np.nan, values)
    # rebuilt with .copy so that extra indexes (lat/lon) and attrs survive
    # untouched -- going through .where() aligns coordinates and breaks on a
    # DataArray whose lat/lon index no longer matches its x axis
    return da.copy(data=values)


def smooth_box(da, window=6, min_count=None, dim="x", keep_coverage=True):
    """Centred running mean over ``window`` samples along ``dim``.

    ``min_count`` defaults to half the window, so an output point needs real
    support rather than the single valid sample ``track_smooth_box_x`` accepted
    (its ``min_periods=1``).  Note also that ``track_smooth_box_x`` defaulted to
    ``smooth=1``, which is a no-op -- calling it without an explicit window
    returned the data unsmoothed.

    keep_coverage : with the default True the output is NaN wherever the input
        was, so smoothing never extends the data.  Set False to let the running
        mean bridge short gaps.
    """
    if min_count is None:
        min_count = max(1, window // 2)
    out = da.rolling({dim: window}, center=True, min_periods=min_count).mean()
    if keep_coverage:
        out = out.where(np.isfinite(da))
    out.attrs = dict(da.attrs)
    out.name = da.name
    return out


def smooth_gaussian(da, sigma=3, dim="x", min_weight=0.5, keep_coverage=True):
    """Gaussian smoothing that tolerates gaps (normalised convolution).

    ``scipy.ndimage.gaussian_filter1d`` turns the whole kernel footprint to NaN
    around every missing sample: on GFO pass 008 ``track_smooth_gaussian_x``
    returned only 41% of the valid points it was given.  Here the data and a
    validity mask are filtered separately and divided, so a gap costs only the
    points it actually covers.

    min_weight : an output point is kept only if at least this fraction of the
        kernel weight landed on valid samples, so heavily extrapolated points
        are dropped rather than reported.
    sigma is in samples, as before -- along-track spacing is 6-7 km, but see
        ``smooth_loess`` if you need a window defined in distance.
    keep_coverage : with the default True the output is NaN wherever the input
        was, so smoothing never extends the data.
    """
    axis = da.get_axis_num(dim)
    values = np.asarray(da.values, dtype=float)
    good = np.isfinite(values)

    num = gaussian_filter1d(np.where(good, values, 0.0), sigma=sigma, axis=axis,
                            mode="nearest")
    den = gaussian_filter1d(good.astype(float), sigma=sigma, axis=axis,
                            mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den >= min_weight, num / den, np.nan)
    if keep_coverage:
        out = np.where(good, out, np.nan)

    return xr.DataArray(out, dims=da.dims, coords=da.coords,
                        name=da.name, attrs=dict(da.attrs))


def smooth_loess(da, frac=0.05, dim="x", min_points=10, fill_gaps=None):
    """LOESS smoothing along ``dim`` that neither crashes nor invents data.

    This is the project's primary smoother (frac=0.05, per the draft).  Both old
    versions called ``interpolate_na(dim="x", method="linear",
    fill_value="extrapolate")`` first and then kept the result everywhere, so
    the output had *more* finite values than the input: on GFO pass 008,
    ``track_smooth_loess_statsmodels_x`` turned 28695 valid points into 32550,
    fabricating 3855 of them -- including at x=0, the coastal point, where 37
    valid cycles became 155.  ``track_smooth_loess_x`` additionally raised
    ``LinAlgError: SVD did not converge`` on that same file, because 31 of its
    186 cycles are entirely empty and ``interpolate_na`` cannot fill an all-NaN
    column, so ``loess_1d`` was handed all-NaN.

    Here each series is fitted on its finite samples only, series with fewer
    than ``min_points`` are returned as NaN, and the fit is written back only
    where the input was finite.

    fill_gaps : if given (in the units of the ``dim`` coordinate), also emit
        values at missing samples whose nearest valid neighbour is within that
        distance.  Leave it None to preserve the input's coverage exactly.
    """
    coord = np.asarray(da[dim].values, dtype=float)
    axis = da.get_axis_num(dim)
    values = np.moveaxis(np.asarray(da.values, dtype=float), axis, -1)
    flat = values.reshape(-1, values.shape[-1])
    out = np.full_like(flat, np.nan)

    for i, y in enumerate(flat):
        good = np.isfinite(y)
        if good.sum() < min_points:
            continue
        fit = sm.nonparametric.lowess(
            endog=y[good], exog=coord[good], frac=frac, return_sorted=False
        )
        keep = good
        if fill_gaps is not None:
            near = np.min(np.abs(coord[:, None] - coord[good][None, :]), axis=1)
            keep = near <= fill_gaps
            fit = np.interp(coord[keep], coord[good], fit)
            out[i, keep] = fit
            continue
        out[i, keep] = fit

    out = np.moveaxis(out.reshape(values.shape), -1, axis)
    return xr.DataArray(out, dims=da.dims, coords=da.coords,
                        name=da.name, attrs=dict(da.attrs))


def lowpass_time(da, cutoff_days, order=4, dim="time"):
    """Zero-phase Butterworth low pass along ``dim``, sampling read from the axis.

    ``butter_lowpass_1d`` / ``butter_lowpass_2d`` hardcode ``fs = 1`` with the
    comment "1 sample per day", but an along-track series is one sample per
    satellite repeat cycle.  A nominal 3-day cutoff was therefore really 3
    cycles: 30 days for TP/Jason, 51 for GFO, 105 for ERS/Envisat/Saral -- the
    physical cutoff silently changed with the mission, so cross-mission
    comparisons were filtered at different scales.

    Here the sampling interval is the median spacing of the ``dim`` coordinate,
    and a cutoff shorter than the Nyquist period is refused instead of being
    quietly meaningless.  NaN gaps are bridged for the filter and restored
    afterwards, as before; series too short for ``filtfilt`` are returned
    untouched (``butter_lowpass_2d`` raised on those).
    """
    t = np.asarray(da[dim].values)
    if np.issubdtype(t.dtype, np.datetime64):
        step = np.median(np.diff(t)) / np.timedelta64(1, "D")
    else:
        step = float(np.median(np.diff(np.asarray(t, dtype=float))))
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"could not determine sampling interval from {dim!r}")

    nyquist_days = 2.0 * step
    if cutoff_days <= nyquist_days:
        raise ValueError(
            f"cutoff_days={cutoff_days} is at or below the Nyquist period "
            f"({nyquist_days:.1f} d for {step:.1f} d sampling); nothing to filter"
        )

    wn = (1.0 / cutoff_days) / (0.5 / step)  # normalised to Nyquist
    b, a = butter(order, wn, btype="low", analog=False)
    padlen = 3 * max(len(a), len(b))

    def _filt(y):
        good = np.isfinite(y)
        if good.sum() < 2 or y.size <= padlen:
            return y.astype(float)
        idx = np.arange(y.size)
        out = filtfilt(b, a, np.interp(idx, idx[good], y[good]))
        return np.where(good, out, np.nan)

    out = xr.apply_ufunc(
        _filt, da.astype(float),
        input_core_dims=[[dim]], output_core_dims=[[dim]],
        vectorize=True, dask="parallelized", output_dtypes=[float],
    ).transpose(*da.dims)
    out.attrs = dict(da.attrs)
    out.attrs["lowpass_cutoff_days"] = cutoff_days
    out.attrs["sampling_days"] = step
    out.name = da.name
    return out



# ----------------------------------------------------------------------------
# Geostrophy.
#
# Sign convention, stated once and used everywhere below:
#
#   alpha  is the track heading in the direction of INCREASING x, i.e. pointing
#          offshore -- this is track_angle_at(..., order="coast").
#   gc     is the current component 90 deg counter-clockwise from that heading,
#              gc = -u sin(alpha) + v cos(alpha)
#          which for a geostrophic flow is
#              gc = +(g/f) dh/dx        (x offshore, in metres)
#   u, v   follow by inverting the two projection equations at a crossing:
#              u = (gc1 cos a2 - gc2 cos a1) / sin(a2 - a1)
#              v = (gc1 sin a2 - gc2 sin a1) / sin(a2 - a1)
#
# This reproduces tools_xtrackm.compute_geostrophy_gc +
# geostrophic_components numerically, but without the two compensating sign
# errors those carry (see tools_xtrackm_c_summary.txt section 8).
# ----------------------------------------------------------------------------

OMEGA = 7.292115e-5  # Earth rotation rate, rad/s (Groten, 2004)


def coriolis(lat):
    """Coriolis parameter f in 1/s for latitude in degrees (plain ndarray)."""
    return 2.0 * OMEGA * np.sin(np.radians(np.asarray(lat, dtype=float)))


def gravity(lat):
    """Gravity in m/s2 from the international gravity formula, latitude in deg."""
    sin2 = np.sin(np.radians(np.asarray(lat, dtype=float))) ** 2
    return 9.780327 * (1.0 + (5.2792e-3 + 2.32e-5 * sin2) * sin2)


def cross_track_geostrophy(da, equator_band=5.0, max_gap=None):
    """Cross-track geostrophic current from a reshaped, smoothed SLA track.

    ``da`` is what :func:`track_dist_time_asn_midx` returns (dim ``x``, optional
    dim ``time``, ``lat`` riding along on ``x``) after along-track smoothing --
    LOESS frac=0.05 is this project's primary method.  Because ``lat`` travels
    with the data as a coordinate, f and g are aligned to the SLA by coordinate
    rather than by array position, which is what
    ``tools_xtrackm.compute_geostrophy_gc`` did (it multiplied by a separately
    reversed ``ds.lat.values[::-1]`` and so silently depended on ``da`` still
    being in the original order and orientation).

    gc = +(g/f) dh/dx, positive 90 deg counter-clockwise from the offshore
    heading.  The x axis is converted to metres from ``da.x.attrs['units']``, so
    a track built with ``units_in='km'`` no longer yields currents 1000x too
    large.

    equator_band : latitudes with ``|lat| < equator_band`` are masked, where f
        vanishes.  Defaults to the 5S-5N band used in the draft; the old code
        used the one-sided test ``lat > 2``, which happens to be harmless only
        because no NINDIAN track reaches the southern hemisphere.
    max_gap : if given (metres), gc is masked wherever the gap to the nearest
        neighbouring point exceeds it.  Adjacent 1 Hz points are 6-7 km apart,
        but 0.2% of intervals are gaps of up to ~1400 km, across which
        ``differentiate`` happily reports a long-baseline average slope as
        though it were a local gradient.
    """
    if "x" not in da.dims:
        raise ValueError("expected a track reshaped onto an 'x' dimension")
    if "lat" not in da.coords:
        raise ValueError("expected a 'lat' coordinate on 'x'")

    scale = _scale(da["x"].attrs.get("units", "m"))  # -> metres
    da_m = da.assign_coords(x=da["x"].values / scale) if scale != 1.0 else da

    lat = np.asarray(da_m["lat"].values, dtype=float)
    f = coriolis(lat)
    f = np.where(np.abs(lat) >= equator_band, f, np.nan)

    ratio = xr.DataArray(gravity(lat) / f, dims=["x"], coords={"x": da_m["x"]})
    gc = ratio * da_m.differentiate("x")

    if max_gap is not None:
        xv = np.asarray(da_m["x"].values, dtype=float)
        step = np.diff(xv)
        near = np.minimum(
            np.r_[np.inf, step], np.r_[step, np.inf]
        )  # gap to the closer neighbour
        gc = gc.where(xr.DataArray(near <= max_gap, dims=["x"], coords={"x": da_m["x"]}))

    gc = gc.assign_coords(x=da["x"])  # restore the caller's units on the axis
    gc.attrs = {
        "units": "m s-1",
        "long_name": "cross-track geostrophic current",
        "description": (
            "component 90 deg counter-clockwise from the offshore track heading; "
            "gc = +(g/f) dh/dx"
        ),
        "equator_band": equator_band,
    }
    gc.name = "gc"
    return gc


def resolve_uv(gc1, gc2, a1, a2, min_sin=0.1):
    """Zonal and meridional current from two crossing tracks' cross-track currents.

    ``a1``/``a2`` are the offshore headings in radians, as returned by
    ``track_angle_at(ds, ..., order="coast")``.  ``gc1``/``gc2`` may be scalars,
    arrays or DataArrays.

        u = (gc1 cos a2 - gc2 cos a1) / sin(a2 - a1)
        v = (gc1 sin a2 - gc2 sin a1) / sin(a2 - a1)

    These are the exact inverse of the projection equations in the module
    header.  Note the draft (v02.2a) writes the v numerator as
    ``gc2 sin a1 - gc1 sin a2``, which is negated and recovers the true v in 0%
    of random test cases; ``tools_xtrackm.geostrophic_components`` instead
    writes the determinant as ``sin(a1 - a2)``, negating both u and v, which
    cancels against the sign of its own gc and so comes out right.

    min_sin : the tracks must cross at ``|sin(a2 - a1)| >= min_sin`` (about 5.7
        deg for the default) or ValueError is raised -- near-parallel tracks
        cannot resolve two components.  The old code used a 1e-6 threshold,
        which admits crossings so shallow that the inversion is pure noise.
    """
    a1 = float(a1)
    a2 = float(a2)
    if not (np.isfinite(a1) and np.isfinite(a2)):
        raise ValueError("heading angles must be finite")
    for name, a in (("a1", a1), ("a2", a2)):
        if abs(a) > 2 * np.pi:
            raise ValueError(f"{name}={a:.3f} looks like degrees, not radians")

    den = np.sin(a2 - a1)
    if abs(den) < min_sin:
        raise ValueError(
            f"tracks cross at {np.degrees(np.arcsin(abs(den))):.2f} deg "
            f"(|sin| {abs(den):.3g} < min_sin {min_sin}); "
            "cannot resolve both components"
        )

    u = (gc1 * np.cos(a2) - gc2 * np.cos(a1)) / den
    v = (gc1 * np.sin(a2) - gc2 * np.sin(a1)) / den
    for arr, long_name in ((u, "zonal geostrophic current"),
                           (v, "meridional geostrophic current")):
        if isinstance(arr, xr.DataArray):
            arr.attrs = {"units": "m s-1", "long_name": long_name}
    if isinstance(u, xr.DataArray):
        u.name, v.name = "gu", "gv"
    return u, v



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

    # --- 10. cross-track geostrophy ------------------------------------------
    print("\n10. cross-track geostrophy")
    dam = track_dist_time_asn_midx(ds, "sla", units_in="m")
    smooth = dam.rolling(x=6, center=True, min_periods=3).mean()  # stand-in smoother
    for label, arr, kw in (("raw sla", dam, {}),
                           ("smoothed", smooth, {}),
                           ("smoothed, max_gap=20 km", smooth, {"max_gap": 20_000})):
        gc = cross_track_geostrophy(arr, **kw)
        good = np.abs(gc.values[np.isfinite(gc.values)])
        print(f"  {label:<26} |gc| median {np.median(good):6.3f}  "
              f"p99 {np.percentile(good, 99):7.3f} m/s   n={good.size}")
    print("  (raw differentiation gives unphysical currents -- hence the smoothing step)")
    a_off, ii = track_angle_at(ds, lat=10.0, order="coast")
    print(f"  offshore heading at 10N: {np.degrees(a_off):.2f} deg"
          "  ->  gc = -u sin(a) + v cos(a)")
    print("  resolve_uv(gc1, gc2, a1, a2) then needs a second, crossing track")

    # --- 11. outlier rejection and smoothing ---------------------------------
    print("\n11. outlier rejection and smoothing")
    n_in = int(np.isfinite(dam.values).sum())
    print(f"  input: {n_in} finite of {dam.size}")
    for label, out in (("smooth_box(window=6)", smooth_box(dam, window=6)),
                       ("smooth_gaussian(sigma=3)", smooth_gaussian(dam, sigma=3)),
                       ("smooth_loess(frac=0.05)", smooth_loess(dam, frac=0.05)),
                       ("smooth_loess(fill_gaps=20 km)",
                        smooth_loess(dam, frac=0.05, fill_gaps=20_000))):
        k = int(np.isfinite(out.values).sum())
        print(f"  {label:<32} finite {k:6d}  ({k - n_in:+d} vs input)")
    for th in (3.0, 3.5):
        k = int(np.isfinite(mad_filter(dam, thresh=th, dim="time").values).sum())
        print(f"  mad_filter(thresh={th}, dim='time')      flags {n_in - k:5d} outliers")
    tt = track_time(ds)
    print(f"  lowpass_time: sampling read from the time axis = "
          f"{np.median(np.diff(tt)) / np.timedelta64(1, 'D'):.1f} d")
    try:
        lowpass_time(dam.assign_coords(time=tt), cutoff_days=3)
    except ValueError as exc:
        print(f"  lowpass_time(cutoff_days=3) refused: {exc}")
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

    # --- figure 4: cross-track geostrophy ------------------------------------
    dam = track_dist_time_asn_midx(ds, "sla", units_in="m")
    smooth = dam.rolling(x=6, center=True, min_periods=3).mean()
    gc_raw = cross_track_geostrophy(dam)
    gc = cross_track_geostrophy(smooth, max_gap=20_000)
    to_km = lambda g: g.assign_coords(x=g.x / 1000.0)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    sel_track(to_km(gc), x=(0, 400)).plot(
        ax=ax[0, 0], x="time", y="x", vmin=-1.5, vmax=1.5, cmap="RdBu_r")
    ax[0, 0].set_ylabel("distance from coast [km]")
    ax[0, 0].set_title("(a) cross-track geostrophic current [m s-1]")

    for g, lbl in ((gc_raw, "raw sla"), (gc, "smoothed")):
        v = np.abs(g.values[np.isfinite(g.values)])
        ax[0, 1].hist(v, bins=np.logspace(-3, 2, 60), histtype="step", label=lbl)
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlabel("|gc| [m s-1]")
    ax[0, 1].set_ylabel("count")
    ax[0, 1].legend()
    ax[0, 1].set_title("(b) why the smoothing step exists")

    sel_track(to_km(gc), x=(0, 400)).mean("time").plot(ax=ax[1, 0], marker="o", ms=3)
    ax[1, 0].axhline(0, lw=0.6, color="k")
    ax[1, 0].set_xlabel("distance from coast [km]")
    ax[1, 0].set_title("(c) time-mean gc vs distance from coast")

    wide = cross_track_geostrophy(smooth, equator_band=0.0, max_gap=20_000)
    ax[1, 1].plot(wide.lat.values, np.abs(wide).median("time").values, ".",
                  ms=4, label="|gc| (equator_band=0)")
    ax[1, 1].axvspan(0, 2, color="0.75", alpha=0.6, label="masked by lat > 2 (old)")
    ax[1, 1].axvspan(2, 5, color="0.88", alpha=0.9,
                     label="also masked by equator_band=5")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("latitude [degrees_north]")
    ax[1, 1].set_ylabel("median |gc| [m s-1]")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("(d) g/f blow-up near the equator, and what each mask cuts")

    fig.suptitle("cross_track_geostrophy: gc = +(g/f) dh/dx, x offshore")
    paths.append(os.path.join(outdir, "tools_xtrackm_c_demo4_geostrophy.png"))
    fig.savefig(paths[-1], dpi=110)
    plt.show()
    plt.close(fig)

    # --- figure 5: smoothing -------------------------------------------------
    import tools_xtrackm as _old  # for the side-by-side comparison only

    loess = smooth_loess(dam, frac=0.05)
    loess_old = _old.track_smooth_loess_statsmodels_x(dam, frac=0.05)
    xk = dam.x.values / 1000.0

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    cyc = int(np.nanargmax(np.isfinite(dam.values).sum(axis=0)))
    ax[0, 0].plot(xk, dam.isel(time=cyc), ".", ms=3, color="0.6", label="sla")
    ax[0, 0].plot(xk, smooth_box(dam, window=6).isel(time=cyc), lw=1, label="box 6")
    ax[0, 0].plot(xk, smooth_gaussian(dam, sigma=3).isel(time=cyc), lw=1,
                  label="gaussian 3")
    ax[0, 0].plot(xk, loess.isel(time=cyc), lw=1.6, label="loess 0.05")
    ax[0, 0].set_xlim(0, 500)
    ax[0, 0].set_xlabel("distance from coast [km]")
    ax[0, 0].set_ylabel("sla [m]")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title(f"(a) the smoothers, cycle {cyc}")

    names = ["input", "box 6", "gaussian 3", "loess 0.05", "old gaussian", "old loess"]
    counts = [np.isfinite(v).sum() for v in
              (dam.values, smooth_box(dam, window=6).values,
               smooth_gaussian(dam, sigma=3).values, loess.values,
               _old.track_smooth_gaussian_x(dam, sigma=3).values, loess_old.values)]
    bars = ax[0, 1].bar(names, counts,
                        color=["0.4"] + ["C0"] * 3 + ["C3"] * 2)
    ax[0, 1].axhline(counts[0], color="k", lw=0.8, ls="--")
    ax[0, 1].tick_params(axis="x", rotation=30, labelsize=8)
    ax[0, 1].set_ylabel("finite points")
    ax[0, 1].set_title("(b) coverage: red = old, destroys or invents data")

    diff = np.abs(loess.values - loess_old.values)
    both = np.isfinite(diff)
    ax[1, 0].plot(np.repeat(xk[:, None], dam.sizes["time"], 1)[both],
                  diff[both] * 100, ".", ms=1.5, alpha=0.3)
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("distance from coast [km]")
    ax[1, 0].set_ylabel("|new - old| [cm]")
    ax[1, 0].set_title("(c) old loess is contaminated at REAL observations")

    z = np.linspace(0, 6, 300)
    ax[1, 1].plot(z, 2 * (1 - 0.5 * (1 + np.vectorize(math.erf)(z / np.sqrt(2)))) * 100,
                  label="correct (mad_filter)")
    ax[1, 1].plot(z, 2 * (1 - 0.5 * (1 + np.vectorize(math.erf)(z / 0.6745 / np.sqrt(2))))
                  * 100, label="as coded (0.6745x too small)")
    ax[1, 1].axvline(3.5, color="k", lw=0.8, ls="--")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("nominal threshold [sigma]")
    ax[1, 1].set_ylabel("% of Gaussian noise flagged")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("(d) the MAD double-scaling, at thresh=3.5")

    fig.suptitle("smoothing and outlier rejection: never invent, never destroy")
    paths.append(os.path.join(outdir, "tools_xtrackm_c_demo5_smoothing.png"))
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
        print("\n12. figures")
        for path in demo_plots(ds, outdir):
            print(f"  wrote {path}")
