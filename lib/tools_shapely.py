import geopandas as gpd
import cartopy.io.shapereader as shpreader
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, box
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

_COAST_CACHE = {}


def apply_to_geom(geom, func):
    if isinstance(geom, LineString):
        return func(geom)
    elif isinstance(geom, MultiLineString):
        return MultiLineString([func(g) for g in geom.geoms])
    else:
        return geom


def gaussian_smooth(line, sigma=10):
    coords = np.array(line.coords)
    if len(coords) < 7:
        return line
    x = gaussian_filter1d(coords[:, 0], sigma=sigma, mode="nearest")
    y = gaussian_filter1d(coords[:, 1], sigma=sigma, mode="nearest")
    return LineString(np.column_stack([x, y]))


def moving_average_smooth(line, window=15):
    coords = np.array(line.coords)
    if len(coords) < window:
        return line
    kernel = np.ones(window) / window
    x = np.convolve(coords[:, 0], kernel, mode="same")
    y = np.convolve(coords[:, 1], kernel, mode="same")
    return LineString(np.column_stack([x, y]))


def savgol_smooth(line, window=21, poly=2):
    coords = np.array(line.coords)
    if len(coords) < window:
        return line
    x = savgol_filter(coords[:, 0], window, poly)
    y = savgol_filter(coords[:, 1], window, poly)
    return LineString(np.column_stack([x, y]))


def polyfit_smooth(line, window_km=200, degree=2):
    coords = np.array(line.coords)
    if len(coords) < 10:
        return line
    x = coords[:, 0]
    y = coords[:, 1]
    # cumulative arclength (meters)
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    x_new = np.zeros_like(x)
    y_new = np.zeros_like(y)
    half_window = window_km * 1000 / 2
    for i in range(len(s)):
        mask = np.abs(s - s[i]) <= half_window
        if np.sum(mask) < degree + 2:
            x_new[i] = x[i]
            y_new[i] = y[i]
            continue
        # Fit polynomial y(s), x(s)
        px = np.polyfit(s[mask], x[mask], degree)
        py = np.polyfit(s[mask], y[mask], degree)
        x_new[i] = np.polyval(px, s[i])
        y_new[i] = np.polyval(py, s[i])
    return LineString(np.column_stack([x_new, y_new]))


def _get_coastline(resolution="50m"):
    if resolution not in _COAST_CACHE:
        shpfilename = shpreader.natural_earth(
            resolution=resolution,
            category="physical",
            name="coastline",
        )
        _COAST_CACHE[resolution] = gpd.read_file(shpfilename)
    return _COAST_CACHE[resolution]


def _tangent_on_line_near_point(line, point_xy, neighborhood=2):
    coords = np.asarray(line.coords, dtype=float)
    if coords.shape[0] < 2:
        raise ValueError("LineString must have at least 2 coordinates.")

    p = np.array(point_xy, dtype=float)
    best = {"dist2": np.inf, "seg_idx": 0, "proj": coords[0].copy(), "t": 0.0}

    for i in range(len(coords) - 1):
        a = coords[i]
        b = coords[i + 1]
        ab = b - a
        denom = np.dot(ab, ab)
        if denom == 0:
            t = 0.0
            q = a
        else:
            t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
            q = a + t * ab

        d2 = np.dot(p - q, p - q)
        if d2 < best["dist2"]:
            best = {"dist2": d2, "seg_idx": i, "proj": q, "t": float(t)}

    i_center = best["seg_idx"] + (1 if best["t"] >= 0.5 else 0)
    i0 = max(0, i_center - neighborhood)
    i1 = min(len(coords) - 1, i_center + neighborhood)
    tangent = coords[i1] - coords[i0]

    if np.allclose(tangent, 0.0):
        tangent = coords[best["seg_idx"] + 1] - coords[best["seg_idx"]]

    return {
        "tangent_vector": (float(tangent[0]), float(tangent[1])),
        "nearest_point": (float(best["proj"][0]), float(best["proj"][1])),
        "distance": float(np.sqrt(best["dist2"])),
    }


def coastline_gaussian_tangent(
    query_point,
    sigma=2,
    region_bbox_tuple=(70, 5, 95, 30),
):
    """
    One-call coastline tangent helper for lon/lat inputs.

    Parameters
    ----------
    query_point : tuple(float, float)
        Longitude/latitude point as (lon, lat).
    sigma : float, optional
        Gaussian smoothing sigma. Default is 2.
    region_bbox_tuple : tuple(float, float, float, float), optional
        Clipping box as (lon_min, lat_min, lon_max, lat_max).
        Default is (70, 5, 95, 30).

    Returns
    -------
    dict
        {
            "heading_angle_deg": float,   # 0-360 deg, measured CCW from +lon (east)
            "tangent_vector": (tx, ty),
            "nearest_point": (lon, lat),
            "distance": float
        }
    """
    if len(region_bbox_tuple) != 4:
        raise ValueError(
            "region_bbox_tuple must be (lon_min, lat_min, lon_max, lat_max)."
        )

    region_bbox = box(*region_bbox_tuple)
    coast = _get_coastline("50m")
    coast_clip = gpd.clip(coast, region_bbox)

    if coast_clip.empty:
        raise ValueError("No coastline found in the provided bounding box.")

    coast_gauss = coast_clip.copy()
    coast_gauss["geometry"] = coast_clip.geometry.apply(
        lambda g: apply_to_geom(g, lambda line: gaussian_smooth(line, sigma=sigma))
    )

    pt = Point(query_point)
    best_geom = min(coast_gauss.geometry, key=lambda g: g.distance(pt))

    if isinstance(best_geom, MultiLineString):
        best_geom = min(best_geom.geoms, key=lambda g: g.distance(pt))
    if not isinstance(best_geom, LineString):
        raise TypeError("Nearest coastline geometry is not line-like.")

    tangent_info = _tangent_on_line_near_point(best_geom, query_point, neighborhood=2)

    tx, ty = tangent_info["tangent_vector"]
    if tx < 0:
        tx, ty = -tx, -ty
    heading_angle_deg = float((np.degrees(np.arctan2(ty, tx)) + 360) % 360)

    return {
        "heading_angle_deg": heading_angle_deg,
        "tangent_vector": tangent_info["tangent_vector"],
        "nearest_point": tangent_info["nearest_point"],
        "distance": tangent_info["distance"],
    }
