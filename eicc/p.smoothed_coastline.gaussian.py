import geopandas as gpd
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, box
from scipy.ndimage import gaussian_filter1d

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

# Clip to Indian Ocean region

coast_clip = gpd.clip(coast, region_bbox)

# Reproject to metric CRS for smoothing

coast_proj = coast_clip.to_crs("EPSG:3857")

# Gaussian smoothing

def smooth_linestring(line, sigma=20):
    coords = np.array(line.coords)
    if len(coords) < 5:
        return line
    x = coords[:, 0]
    y = coords[:, 1]
    x_s = gaussian_filter1d(x, sigma=sigma, mode='nearest')
    y_s = gaussian_filter1d(y, sigma=sigma, mode='nearest')
    return LineString(np.column_stack([x_s, y_s]))


def smooth_geometry(geom, sigma=20):
    if isinstance(geom, LineString):
        return smooth_linestring(geom, sigma)
    elif isinstance(geom, MultiLineString):
        return MultiLineString([smooth_linestring(g, sigma) for g in geom.geoms])
    else:
        return geom


sigma_value = 1   # Increase for stronger smoothing

coast_smooth = coast_proj.copy()
coast_smooth["geometry"] = coast_proj.geometry.apply(
    lambda g: smooth_geometry(g, sigma=sigma_value)
)

# Reproject back to lat/lon

coast_orig = coast_clip
coast_smooth = coast_smooth.to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(10, 8))

coast_orig.plot(ax=ax, linewidth=0.8, color="black", label="Original 110m")
coast_smooth.plot(ax=ax, linewidth=2.0, color="red", label="Smoothed")

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

ax.set_title("Indian Ocean Coastline\nOriginal vs Gaussian Smoothed")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.tight_layout()
plt.show()
