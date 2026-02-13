import geopandas as gpd
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, box
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

lon_min, lon_max = 70, 95
lat_min, lat_max = 5, 30
region_bbox = box(lon_min, lat_min, lon_max, lat_max)

# Load Natural Earth 110m coastline
shpfilename = shpreader.natural_earth(
    resolution='110m',
    category='physical',
    name='coastline'
)

coast = gpd.read_file(shpfilename)
coast_clip = gpd.clip(coast, region_bbox)

# Reproject to metric CRS
coast_proj = coast_clip.to_crs("EPSG:3857")


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


coast_gauss = coast_proj.copy()
coast_gauss["geometry"] = coast_proj.geometry.apply(
    lambda g: apply_to_geom(g, lambda line: gaussian_smooth(line, sigma=1))
)

coast_simplify = coast_proj.copy()
coast_simplify["geometry"] = coast_proj.geometry.simplify(
    tolerance=30000, preserve_topology=True
)

coast_movavg = coast_proj.copy()
coast_movavg["geometry"] = coast_proj.geometry.apply(
    lambda g: apply_to_geom(g, lambda line: moving_average_smooth(line, window=15))
)

coast_savgol = coast_proj.copy()
coast_savgol["geometry"] = coast_proj.geometry.apply(
    lambda g: apply_to_geom(g, lambda line: savgol_smooth(line, window=21, poly=2))
)

coast_poly = coast_proj.copy()
coast_poly["geometry"] = coast_proj.geometry.apply(
    lambda g: apply_to_geom(g, lambda line: polyfit_smooth(line, window_km=300, degree=2))
)

# Reproject back to lat/lon

coast_orig = coast_clip
coast_gauss = coast_gauss.to_crs("EPSG:4326")
coast_simplify = coast_simplify.to_crs("EPSG:4326")
coast_movavg = coast_movavg.to_crs("EPSG:4326")
coast_savgol = coast_savgol.to_crs("EPSG:4326")
coast_poly = coast_poly.to_crs("EPSG:4326")

# Plot

fig, ax = plt.subplots(figsize=(10, 8))

coast_orig.plot(ax=ax, linewidth=0.8, color="black", label="Original 110m")
coast_gauss.plot(ax=ax, linewidth=2, color="red", label="Gaussian")
# coast_simplify.plot(ax=ax, linewidth=2, color="blue", label="Simplify")
# coast_movavg.plot(ax=ax, linewidth=2, color="green", label="Moving Avg")
# coast_savgol.plot(ax=ax, linewidth=2, color="purple", label="Savitzky-Golay")
# coast_poly.plot(ax=ax, linewidth=2, color="purple", label="Polynomial")

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

ax.set_title("Coastline Smoothing Comparison (70–95E, 5–30N)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.tight_layout()
plt.show()
