import cmaps
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from shapely.geometry import MultiPoint

cmap_rmse = "YlGn"
cmap3 = cmaps.BlAqGrYeOrReVi200

width, height = 10, 10

dsr = xr.open_dataset("/home/srinivasu/allData/radar/TNCodar_daily.nc")
dsr = dsr.rename(
    {"XAXS": "longitude", "YAXS": "latitude", "ZAXS": "lev", "TAXIS1D": "time"}
)
print(dsr)

radar_da = 0.01 * dsr.U_RADAR.isel(lev=0, drop=True)
sizes = dsr.sizes
ln = sizes["time"]
ngp_u = 100 * (radar_da.count(dim="time") / ln)
# Find the indices where the data array is non-zero
non_zero_indices = np.nonzero(ngp_u.values)

# Extract the corresponding latitude and longitude values
lat_vals = ngp_u.latitude.values[non_zero_indices[0]]
lon_vals = ngp_u.longitude.values[non_zero_indices[1]]

# Assuming 'points' is a list of (x, y) tuples of the non-zero coordinates
points = [(lon, lat) for lon, lat in zip(lon_vals, lat_vals)]

# Create a MultiPoint object and then create the smallest polygon that can enclose the points (Convex Hull)
polygon = MultiPoint(points).convex_hull
polygon = polygon.buffer(0.05)

x, y = polygon.exterior.xy

gdf = gpd.GeoDataFrame(
    index=[0],
    geometry=[polygon],
    crs="EPSG:4326",
)
#
gdf.to_file("polygon_shapefile_around_TN.shp")
#
fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")
c = ngp_u.plot(ax=ax, cmap=cmap3, add_colorbar=False)
ax.plot(x, y, color="k", label="Polygon Outline")
cb = plt.colorbar(c)
cb.set_label("% of valid data points")
# plt.savefig(f"TNradar_valid_points.png")
plt.show()
