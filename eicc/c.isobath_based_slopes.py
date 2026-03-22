import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tools_xtrackm import *
# codex resume 019d1654-1b08-70c0-adfb-1fd3acdb5f72

def extract_isobath_all_pts(topo, level, lon_name="lon", lat_name="lat"):
    """
    Extract all (lon, lat) points along an isobath.
    Parameters
    ----------
    topo : xarray.DataArray
        2D bathymetry/topography field.
    level : float
        Isobath level (e.g., -200 for 200 m isobath).
    lon_name, lat_name : str
        Coordinate names in topo.
    Returns
    -------
    list[tuple[float, float]]
        Flat list of (lon, lat) points from all isobath segments.
    """
    lon = topo[lon_name].values
    lat = topo[lat_name].values
    z = topo.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig, ax = plt.subplots()
    cs = ax.contour(lon2d, lat2d, z, levels=[level])
    plt.close(fig)
    all_pts = []
    # Matplotlib versions differ in contour internals; allsegs is stable.
    # For a single requested level, cs.allsegs[0] holds all disconnected segments.
    segments = cs.allsegs[0] if cs.allsegs else []
    for seg in segments:
        all_pts.extend(map(tuple, seg))
    return all_pts


def smooth_and_subsample_pts(points, window=5, step=10):
    """
    Smooth (lon, lat) points with a moving average and subsample.
    """
    if not points:
        return []
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] < window:
        return [tuple(xy) for xy in arr[::step]]
    kernel = np.ones(window, dtype=float) / window
    lon_smooth = np.convolve(arr[:, 0], kernel, mode="same")
    lat_smooth = np.convolve(arr[:, 1], kernel, mode="same")
    smooth_pts = np.column_stack((lon_smooth, lat_smooth))
    return [tuple(xy) for xy in smooth_pts[::step]]


def nearest_point_and_tangent(
    points, query_point, return_degrees_from_east=False
):
    """
    Find nearest point on a polyline and local tangent slope at that point.
    Parameters
    ----------
    points : list[tuple[float, float]]
        Polyline points as (lon, lat), ideally smoothed/subsampled.
    query_point : tuple[float, float]
        Query (lon, lat).
    return_degrees_from_east : bool
        If True, include tangent angle in degrees measured CCW from east.
    Returns
    -------
    dict
        nearest_point: (lon, lat)
        index: index in points
        distance: euclidean distance in lon/lat units
        slope_dy_dx: tangent slope dy/dx (np.inf for vertical tangent)
        degrees_from_east: optional, range [-180, 180]
    """
    if not points:
        raise ValueError("points is empty")
    if len(points) == 1:
        raise ValueError(
            "at least two points are required to estimate tangent"
        )
    arr = np.asarray(points, dtype=float)
    q = np.asarray(query_point, dtype=float)
    d2 = np.sum((arr - q) ** 2, axis=1)
    idx = int(np.argmin(d2))
    nearest = arr[idx]
    if idx == 0:
        v = arr[1] - arr[0]
    elif idx == len(arr) - 1:
        v = arr[-1] - arr[-2]
    else:
        v = arr[idx + 1] - arr[idx - 1]
    dx, dy = float(v[0]), float(v[1])
    slope = np.inf if np.isclose(dx, 0.0) else (dy / dx)
    out = {
        "nearest_point": (float(nearest[0]), float(nearest[1])),
        "index": idx,
        "distance": float(np.sqrt(d2[idx])),
        "slope_dy_dx": float(slope) if np.isfinite(slope) else np.inf,
    }
    if return_degrees_from_east:
        out["degrees_from_east"] = float(np.degrees(np.arctan2(dy, dx)))
    return out


REG = (62, 92, 0, 30)
dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo_60s_io.nc")
topo = dse.rose.sel(lon=slice(*REG[:2])).sel(lat=slice(*REG[2:]))

cmap = "twilight_shifted"
isobath_levels = [-200, -1000, -2000]
fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
topo.plot.contourf(ax=ax1, add_colorbar=False, add_labels=False, cmap=cmap)

# Continental slope zone: 200 m to 2000 m depth
slope_band = topo.where((topo <= -200) & (topo >= -2000))
slope_band.plot.contourf(
    ax=ax1,
    levels=[-2000, -200],
    colors=["gold"],
    alpha=0.25,
    add_colorbar=False,
    add_labels=False,
)
# Plot each isobath separately to avoid level-color reordering issues.
isobath_style = {
    -200: ("k", 1.2),
    -1000: ("darkorange", 1.0),
    -2000: ("red", 1.0),
}
for level in isobath_levels:
    color, lw = isobath_style[level]
    topo.plot.contour(
        ax=ax1,
        levels=[level],
        add_colorbar=False,
        add_labels=False,
        colors=[color],
        linewidths=[lw],
    )
xsta, xend, ysta, yend = REG
step = 8
ax1.add_feature(cfeature.COASTLINE, linewidth=2.0)
xticks = list(range(int(xsta), int(xend), step))
yticks = list(range(int(ysta), int(yend), step))
ax1.axhline(y=0.0, linestyle="--", color="k")
ax1.set_xticks(xticks, crs=ccrs.PlateCarree())
ax1.set_yticks(yticks, crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(LongitudeFormatter())
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.set_extent([xsta, xend, ysta, yend], crs=ccrs.PlateCarree())
info = f"name of cmap: {cmap}, slope: 200-2000 m"
plt.text(
    0.15,
    0.90,
    info,
    transform=ax1.transAxes,
    fontsize=14,
    fontweight="bold",
    color="w",
)

legend_handles = [
    Patch(
        facecolor="gold",
        edgecolor="none",
        alpha=0.25,
        label="Slope zone (200-2000 m)",
    ),
    Line2D([0], [0], color="k", lw=1.2, label="200 m isobath"),
    Line2D([0], [0], color="darkorange", lw=1.0, label="1000 m isobath"),
    Line2D([0], [0], color="red", lw=1.0, label="2000 m isobath"),
]
ax1.legend(
    handles=legend_handles,
    loc="lower left",
    frameon=True,
    framealpha=0.85,
    fontsize=10,
)

topo.plot.contourf(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap)
all_pts = extract_isobath_all_pts(topo, level=-1)
all_pts = smooth_and_subsample_pts(all_pts, window=10, step=15)
if all_pts:
    lons, lats = zip(*all_pts)
    ax2.scatter(
        lons, lats, color="k", marker=".", s=1, label="200 m isobath points"
    )
else:
    print("No points found for the requested isobath level.")
xsta, xend, ysta, yend = REG
step = 8
xticks = list(range(int(xsta), int(xend), step))
yticks = list(range(int(ysta), int(yend), step))
ax2.axhline(y=0.0, linestyle="--", color="k")
ax2.set_xticks(xticks, crs=ccrs.PlateCarree())
ax2.set_yticks(yticks, crs=ccrs.PlateCarree())
ax2.xaxis.set_major_formatter(LongitudeFormatter())
ax2.yaxis.set_major_formatter(LatitudeFormatter())
ax2.set_extent([xsta, xend, ysta, yend], crs=ccrs.PlateCarree())

query_points = (
    (83.62, 17.17),
    (88.92, 19.00),
    (80.203, 12.015),
    (82.982, 16.837),
    (85.442, 18.856),
)
for q in query_points:
    tangent_info = nearest_point_and_tangent(
        all_pts, q, return_degrees_from_east=True
    )
    nearest_point = tangent_info["nearest_point"]
    degrees_from_east = tangent_info["degrees_from_east"]
    print(f"{q}, {degrees_from_east:.1f}°")

plt.savefig("topo.isobath.png", dpi=300)
plt.show()
plt.close(fig)
