import glob
import os

import xarray as xr
from matplotlib.collections import LineCollection
from tools_xtrackm import *

sat = "ERS1+ERS2+ENV+SRL"

REG = (80.0, 92.0, 11.0, 21.0)
WBOB = (78.0, 87.0, 8.0, 21.0)
WBOB2 = (78.0, 97.0, 0.0, 25.0)
proj = ccrs.PlateCarree()

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 10), subplot_kw={"projection": proj}
)
# adjust subplots to have minimum space between them
month_num = 2
month_name = months[month_num + 1]
info = f"a) {sat} {months[month_num + 1]}"
fig.subplots_adjust(wspace=0.1)
ax1.set_extent([*WBOB2], crs=proj)
ax1.coastlines()
gl = ax1.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--"
)
gl.top_labels = gl.right_labels = False  # Hide top and right labels
plt.text(
    0.3,
    0.95,
    info,
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=15,
    fontweight="bold",
    transform=ax1.transAxes,
)

for f_sla in sorted(
    glob.glob(
        f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.descending.nc"
    )
):
    basename = os.path.basename(f_sla)
    ds_sla = xr.open_dataset(f_sla, decode_times=False)
    trackn = ds_sla.Pass
    f = f"../guv/guv_at_intersections_loess/gc_at_ERS1+ERS2+ENV+SRL_{trackn}_loess_0.05.nc"
    ds_gc = xr.open_dataset(f)
    gc = ds_gc.gc
    lons_track = ds_sla.lon.values
    lats_track = ds_sla.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    lon_coast = lons_track[-1]  # this on coast
    lat_coast = lats_track[-1]  # this on coast
    lon_equat = lons_track[0]
    lat_equat = lats_track[0]
    # compute tangent vectors or slope of track at each point
    dlon = np.gradient(lons_track_rev)
    dlat = np.gradient(lats_track_rev)
    tangent_mag = np.hypot(dlon, dlat)
    tangent_u = dlon / tangent_mag
    tangent_v = dlat / tangent_mag
    # Convert latitude to radians
    phi = np.deg2rad(lats_track_rev)
    # Length of one degree of latitude and longitude in meters
    meters_per_degree_lat = 111_320  # Approximate value
    meters_per_degree_lon = 111_320 * np.cos(phi)
    # Handle division by zero at the poles
    meters_per_degree_lon = np.where(
        meters_per_degree_lon == 0, np.nan, meters_per_degree_lon
    )
    # Compute normal vectors
    normal_u = -tangent_v
    normal_v = tangent_u
    theta = np.arctan2(lat_coast - lat_equat, lon_coast - lon_equat)
    print(theta, "---------------------------------------------------")
    # if is_within_region(lon_coast, lat_coast, *REG):
    if True:
        gc_clim = gc.groupby("time.month").mean("time")
        gc_clim_m = gc_clim.sel(month=month_num)
        # Convert gc from m/s to degrees (assuming cross-track direction)
        gc_deg = gc_clim_m.values / meters_per_degree_lat
        # Calculate displacements
        delta_lon = normal_u * gc_deg
        delta_lat = normal_v * gc_deg
        # Apply scaling factor for visualization
        scaling_factor = 1e5  # Adjust as needed
        delta_lon *= scaling_factor
        delta_lat *= scaling_factor
        # Compute endpoints
        lon_start = lons_track_rev
        lat_start = lats_track_rev
        lon_end = lon_start + delta_lon
        lat_end = lat_start + delta_lat
        # Create line segments
        segments = [
            [[lon_start[i], lat_start[i]], [lon_end[i], lat_end[i]]]
            for i in range(len(lon_start))
        ]
        # Create a LineCollection
        lc = LineCollection(
            segments, colors="red", linewidths=1, transform=proj
        )
        ax1.plot(
            lons_track_rev,
            lats_track_rev,
            "k-",
            transform=proj,
            label="Ground Track",
        )
        ax1.add_collection(lc)

month_num = 10
month_name = months[month_num + 1]
info = f"{sat} {months[month_num + 1]}"
ax2.set_extent([*WBOB2], crs=proj)
ax2.coastlines()
# ax1.legend()
gl = ax2.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--"
)
gl.top_labels = gl.right_labels = False  # Hide top and right labels
plt.text(
    0.3,
    0.95,
    info,
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=15,
    fontweight="bold",
    transform=ax2.transAxes,
)

for f_sla in sorted(
    glob.glob(
        f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.descending.nc"
    )
):
    basename = os.path.basename(f_sla)
    ds_sla = xr.open_dataset(f_sla, decode_times=False)
    trackn = ds_sla.Pass
    f = f"../guv/guv_at_intersections_loess/gc_at_ERS1+ERS2+ENV+SRL_{trackn}_loess_0.05.nc"
    ds_gc = xr.open_dataset(f)
    gc = ds_gc.gc
    lons_track = ds_sla.lon.values
    lats_track = ds_sla.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    lon_coast = lons_track[-1]  # this on coast
    lat_coast = lats_track[-1]  # this on coast
    lon_equat = lons_track[0]
    lat_equat = lats_track[0]
    # compute tangent vectors or slope of track at each point
    dlon = np.gradient(lons_track_rev)
    dlat = np.gradient(lats_track_rev)
    tangent_mag = np.hypot(dlon, dlat)
    tangent_u = dlon / tangent_mag
    tangent_v = dlat / tangent_mag
    # Convert latitude to radians
    phi = np.deg2rad(lats_track_rev)
    # Length of one degree of latitude and longitude in meters
    meters_per_degree_lat = 111_320  # Approximate value
    meters_per_degree_lon = 111_320 * np.cos(phi)
    # Handle division by zero at the poles
    meters_per_degree_lon = np.where(
        meters_per_degree_lon == 0, np.nan, meters_per_degree_lon
    )
    # Compute normal vectors
    normal_u = -tangent_v
    normal_v = tangent_u
    theta = np.arctan2(lat_coast - lat_equat, lon_coast - lon_equat)
    print(theta, "---------------------------------------------------")
    # if is_within_region(lon_coast, lat_coast, *REG):
    if True:
        gc_clim = gc.groupby("time.month").mean("time")
        gc_clim_m = gc_clim.sel(month=month_num)
        # Convert gc from m/s to degrees (assuming cross-track direction)
        gc_deg = gc_clim_m.values / meters_per_degree_lat
        # Calculate displacements
        delta_lon = normal_u * gc_deg
        delta_lat = normal_v * gc_deg
        # Apply scaling factor for visualization
        scaling_factor = 1e5  # Adjust as needed
        delta_lon *= scaling_factor
        delta_lat *= scaling_factor
        # Compute endpoints
        lon_start = lons_track_rev
        lat_start = lats_track_rev
        lon_end = lon_start + delta_lon
        lat_end = lat_start + delta_lat
        # Create line segments
        segments = [
            [[lon_start[i], lat_start[i]], [lon_end[i], lat_end[i]]]
            for i in range(len(lon_start))
        ]
        # Create a LineCollection
        lc = LineCollection(
            segments, colors="red", linewidths=1, transform=proj
        )
        ax2.plot(
            lons_track_rev,
            lats_track_rev,
            "k-",
            transform=proj,
            label="Ground Track",
        )
        ax2.add_collection(lc)
plt.savefig(
    f"cross_structure_x_arrows_map_{sat}_1by2.png",
    bbox_inches="tight",
    dpi=300,
)
ln_sf(f"cross_structure_x_arrows_map_{sat}_1by2.png", "fig06.png")
plt.show()
plt.close("all")
