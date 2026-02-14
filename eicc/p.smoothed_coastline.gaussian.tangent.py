import geopandas as gpd
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, box
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from shapely_smooth_funcs import *

lon_min, lon_max = 70, 95
lat_min, lat_max = 5, 30
region_bbox = box(lon_min, lat_min, lon_max, lat_max)

# Load Natural Earth 110m coastline
shpfilename = shpreader.natural_earth(
    resolution='50m',
    category='physical',
    name='coastline'
)

coast = gpd.read_file(shpfilename)
coast_clip = gpd.clip(coast, region_bbox)

# Reproject to metric CRS
coast_proj = coast_clip.to_crs("EPSG:3857")


def tangent_slope_on_line_near_point(line, point_xy, neighborhood=2):
    """
    Compute local tangent slope dy/dx on a LineString near a query point.
    Parameters
    ----------
    line : shapely.geometry.LineString
        Smoothed coastline line.
    point_xy : tuple(float, float)
        Query point (x, y) in the same CRS as `line`.
    neighborhood : int
        Number of points on each side used for a local finite-difference tangent.
    Returns
    -------
    dict with keys:
      slope : float (np.inf for vertical tangent)
      tangent_vector : tuple(float, float)
      nearest_point : tuple(float, float)
      distance : float
    """
    coords = np.asarray(line.coords, dtype=float)
    if coords.shape[0] < 2:
        raise ValueError("LineString must have at least 2 coordinates.")
    px, py = point_xy
    p = np.array([px, py], dtype=float)
    best = {
        "dist2": np.inf,
        "seg_idx": 0,
        "proj": coords[0].copy(),
        "t": 0.0,
    }
    for i in range(len(coords) - 1):
        a = coords[i]
        b = coords[i + 1]
        ab = b - a
        denom = np.dot(ab, ab)
        if denom == 0:
            t = 0.0
            q = a
        else:
            t = np.dot(p - a, ab) / denom
            t = np.clip(t, 0.0, 1.0)
            q = a + t * ab
        d2 = np.dot(p - q, p - q)
        if d2 < best["dist2"]:
            best.update({"dist2": d2, "seg_idx": i, "proj": q, "t": t})
    i_center = best["seg_idx"] + (1 if best["t"] >= 0.5 else 0)
    i0 = max(0, i_center - neighborhood)
    i1 = min(len(coords) - 1, i_center + neighborhood)
    tangent = coords[i1] - coords[i0]
    if np.allclose(tangent, 0.0):
        tangent = coords[best["seg_idx"] + 1] - coords[best["seg_idx"]]
    dx, dy = tangent[0], tangent[1]
    slope = np.inf if np.isclose(dx, 0.0) else dy / dx
    return {
        "slope": slope,
        "tangent_vector": (float(dx), float(dy)),
        "nearest_point": (float(best["proj"][0]), float(best["proj"][1])),
        "distance": float(np.sqrt(best["dist2"])),
    }


def tangent_slope_near_point(geom, point_xy, neighborhood=2):
    """
    Compute tangent slope near a point for LineString or MultiLineString geometry.
    """
    if isinstance(geom, LineString):
        return tangent_slope_on_line_near_point(geom, point_xy, neighborhood=neighborhood)
    if isinstance(geom, MultiLineString):
        pt = Point(point_xy)
        line = min(geom.geoms, key=lambda g: g.distance(pt))
        return tangent_slope_on_line_near_point(line, point_xy, neighborhood=neighborhood)
    raise TypeError("Geometry must be LineString or MultiLineString.")


def tangent_slope_near_point_in_geoseries(geoms, point_xy, neighborhood=2):
    """
    Find nearest geometry in a GeoSeries and compute tangent slope near point.
    """
    pt = Point(point_xy)
    best_geom = min(geoms, key=lambda g: g.distance(pt))
    return tangent_slope_near_point(best_geom, point_xy, neighborhood=neighborhood)


coast_gauss = coast_proj.copy()
coast_gauss["geometry"] = coast_proj.geometry.apply(
    lambda g: apply_to_geom(g, lambda line: gaussian_smooth(line, sigma=2))
)

# Reproject back to lat/lon

coast_orig = coast_clip
coast_gauss = coast_gauss.to_crs("EPSG:4326")

# Query point in lon/lat for tangent visualization
query_point = (83.62, 17.17)

tangent_info = tangent_slope_near_point_in_geoseries(
    coast_gauss.geometry, query_point, neighborhood=2
)
nearest_point = tangent_info["nearest_point"]
tx, ty = tangent_info["tangent_vector"]
tangent_norm = np.hypot(tx, ty)
if tangent_norm > 0:
    ux, uy = tx / tangent_norm, ty / tangent_norm
else:
    ux, uy = 1.0, 0.0

# Tangent segment length in degree units (lon/lat plot)
tangent_half_len = 0.35
tangent_x = [
    nearest_point[0] - ux * tangent_half_len,
    nearest_point[0] + ux * tangent_half_len,
]
tangent_y = [
    nearest_point[1] - uy * tangent_half_len,
    nearest_point[1] + uy * tangent_half_len,
]

def angle_in_degrees(tangent_info):
    tx, ty = tangent_info["tangent_vector"]
    if tx < 0:
        tx, ty = -tx, -ty
    theta_east = (np.degrees(np.arctan2(ty, tx)) + 360) % 360
    print(theta_east, "---------------------------------------------------")
    return theta_east

theta_east = angle_in_degrees(tangent_info)

# Plot

fig, ax = plt.subplots(figsize=(10, 8))

coast_orig.plot(ax=ax, linewidth=0.8, color="black", label="Original 110m")
coast_gauss.plot(ax=ax, linewidth=2, color="red", label="Gaussian")
ax.scatter(*query_point, color="dodgerblue", s=60, zorder=6, label="Input point")
ax.scatter(*nearest_point, color="orange", s=60, zorder=7, label="Nearest point")
ax.plot(tangent_x, tangent_y, color="green", linewidth=5, zorder=8, label="Tangent segment")

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

ax.set_title("Coastline Smoothing Comparison (70–95E, 5–30N)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.tight_layout()
plt.show()
