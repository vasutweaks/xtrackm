import os
import sys

import xarray as xr
from rich.console import Console
from shapely.geometry import LineString, Point, box
from tools_xtrackd import *

console = Console()

data_loc = "/home/srinivasu/allData/drifter1/"
REG = (42.0, 99.0, 0.0, 23.0)
topo_dir = "/home/srinivasu/allData/topo"


def clip_lines_within_box(shapefile_path, minx, maxx, miny, maxy):
    """
    Clips LineString geometries from a shapefile within a specified bounding box.
    Parameters:
        shapefile_path (str): Path to the input shapefile.
        minx (float): Minimum x-coordinate of the bounding box.
        miny (float): Minimum y-coordinate of the bounding box.
        maxx (float): Maximum x-coordinate of the bounding box.
        maxy (float): Maximum y-coordinate of the bounding box.
        output_file_path (str, optional): Path to save the clipped shapefile (default is None).
    Returns:
        GeoDataFrame: A GeoDataFrame containing the clipped LineString geometries.
    """
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    # Create the bounding box
    bounding_box = box(minx, miny, maxx, maxy)
    # Clip the LineString geometries within the bounding box
    clipped_gdf = gdf[gdf.intersects(bounding_box)]
    clipped_gdf["geometry"] = clipped_gdf["geometry"].intersection(bounding_box)
    # Save the clipped data to a new shapefile if an output path is provided
    return clipped_gdf


minx, maxx, miny, maxy = REG
bounding_box = box(minx, miny, maxx, maxy)
clipped_gdf = clip_lines_within_box(f"{topo_dir}/ne_50m_coastline.shp", *REG)

lakshadweep_bbox = (71, 8, 74, 12)  # (minx, miny, maxx, maxy)


# Remove geometries that fall within the bounding box of Lakshadweep
def is_within_bbox(geometry, bbox):
    minx, miny, maxx, maxy = bbox
    return (geometry.bounds[0] >= minx and geometry.bounds[2] <= maxx and
            geometry.bounds[1] >= miny and geometry.bounds[3] <= maxy)


# Filter out unwanted islands (keeping only geometries outside the Lakshadweep region)
coastline_filtered = clipped_gdf[~clipped_gdf["geometry"].
                                 apply(is_within_bbox, bbox=lakshadweep_bbox)]
print(type(clipped_gdf.geometry.iloc[0]))

# ind_buffer = clipped_gdf.geometry.buffer(2)
ind_buffer = coastline_filtered.geometry.buffer(2)
ind_buffer1 = ind_buffer.unary_union


def get_closest_point_index(drifter_path: LineString,
                            intersection_point: Point) -> int:
    """
    Finds the index of the point on a LineString that is closest to the given Point.
    Args:
        line (LineString): The LineString consisting of multiple points.
        point (Point): The Point for which the closest point on the LineString is found.
    Returns:
        int: The index of the closest point on the LineString.
    """
    # Find the closest point on the LineString to the given Point
    distances = [
        Point(p).distance(intersection_point) for p in drifter_path.coords
    ]
    nearest_index = np.argmin(distances)
    return nearest_index


def find_file_by_id(id, root_dir):
    """
    Search for the drifter_6h_id.nc file in all subdirectories.
    Returns the full path if found, else None.
    """
    target_filename = f"drifter_6h_{id}.nc"
    # Walk through all folders and files from the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_filename in filenames:
            # Return the full path if the file is found
            return os.path.join(dirpath, target_filename)
    # If the file is not found, return None
    return None


dse = xr.open_dataset(
    "/home/srinivasu/allData/topo/etopo_60s_io.nc")  # open etopo dataset
print(dse)

dsc = xr.open_dataset(f"{topo_dir}/"
                      f"GMT_intermediate_coast_distance_01d_track_reg.nc")
print(dsc)
dist_to_coast = dsc.coast_dist

# REG = (63.0, 99, 0, 25.4)
chunk = "15001_current"
sat = "S3A"
track_tsta_o, track_tend_o = get_time_limits_o(sat)
drifter_id = "114945"
selected_drifters = [
    "114737",
    "147132",
    "300234065481940",
    "300234065515670",
    "300234065718080",
    "300234065718110",
]

selected_drifters = [
    "114737",
]
xsta = 78
xend = REG[1]
ysta = REG[2]
yend = REG[3]
title1 = ""
step = 5
lon_range = xend - xsta
lat_range = yend - ysta
aspect_ratio = lon_range / lat_range if lat_range != 0 else 1
base_height = 8
fig_width = round(aspect_ratio * base_height) + 1
fig_height = round(base_height)
print(fig_width, fig_height)
# sys.exit(0)
fig, ax = plt.subplots(
    1,
    1,
    subplot_kw={"projection": ccrs.PlateCarree()},
    figsize=(fig_width, fig_height),
)
fig.tight_layout(pad=1.7, h_pad=1.7, w_pad=1.2)
ax.set_extent([xsta, xend, ysta, yend], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
xticks = list(range(int(xsta), int(xend), step))
yticks = list(range(int(ysta), int(yend), step))
ax.axhline(y=0.0, linestyle="--", color="k")
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
ax.set_title(title1)
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

# dist_to_coast.plot.contour(ax=ax, level=200, cmap="Greys", linewidth=0.5)
# ax.add_geometries(ind_buffer1,
#                   ccrs.PlateCarree(), facecolor="none", edgecolor="gray")
dse.rose.sel(lon=slice(xsta, xend)).sel(lat=slice(ysta, yend)).plot(
    ax=ax, add_colorbar=False, add_labels=False)
cs = (dist_to_coast.sel(lon=slice(xsta, xend)).sel(
    lat=slice(ysta, yend)).plot.contour(
        ax=ax,
        levels=[200],
        colors="gray",
        linestyle="--",
        linewidths=0.5,
        add_labels=False,
    ))

label_locations = [(86.15, 17.74)]
ax.clabel(cs, cs.levels, inline=True, fontsize=10, manual=label_locations)
# plt.show()

for id in selected_drifters:
    fd = find_file_by_id(id, data_loc)
    # fd = f"{data_loc}/netcdf_*/track_reg/drifter_6h_*.nc"
    ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
    byte_id = ds_d.ID.values[0]
    str_id = byte_id.decode("utf-8")
    drifter_id = str_id
    drift_tsta_o1, drift_tend_o1 = (
        ds_d.start_date.values[0],
        ds_d.end_date.values[0],
    )
    drift_tsta_o, drift_tend_o = pd.to_datetime(drift_tsta_o1), pd.to_datetime(
        drift_tend_o1)
    overlap_tsta, overlap_tend = overlap_dates(track_tsta_o, track_tend_o,
                                               drift_tsta_o, drift_tend_o)
    lons_drift = ds_d.longitude.isel(traj=0).values
    lats_drift = ds_d.latitude.isel(traj=0).values
    times_drift = ds_d.time.isel(traj=0).values
    times_drift_o = [pd.to_datetime(t) for t in times_drift]
    vn_drift = ds_d.vn.isel(traj=0).values

    drifter_path = LineString([
        (lon, lat) for lon, lat in zip(lons_drift, lats_drift)
    ])
    print(f"{overlap_tsta}, {overlap_tend}, {drifter_id}")
    if (drifter_path.intersects(ind_buffer1) and overlap_tsta is not None and
            overlap_tend is not None):
        intersection = drifter_path.intersection(ind_buffer1)
        sc = ax.scatter(
            lons_drift,
            lats_drift,
            c=vn_drift,
            cmap="viridis",
            s=8,
            transform=ccrs.PlateCarree(),
        )
    for ft in sorted(
            glob.glob(
                f"../data/ctoh.sla.ref.{sat}.nindian.*.nc"
            )):
        ds_t = xr.open_dataset(ft, decode_times=False, engine="h5netcdf")
        track_number = ds_t.pass_number
        lons_track = ds_t.lon.values
        lats_track = ds_t.lat.values
        m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
        track_path = LineString([(lon, lat)
                                    for lon, lat in zip(lons_track, lats_track)])
        if track_path.intersects(drifter_path) and m < 0:
            intersection1 = track_path.intersection(drifter_path)
            print(track_number, "---------------------------")
            # ax.scatter(
            #     lons_track,
            #     lats_track,
            #     c="k",
            #     s=1,
            #     transform=ccrs.PlateCarree(),
            # )
            ax.add_geometries(
                [track_path],
                ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="black")

            if intersection1.geom_type == "Point":
                pointa = intersection1
                if ind_buffer1.contains(pointa):
                    ii = get_closest_point_index(drifter_path, pointa)
                    ax.plot(
                        intersection1.x,
                        intersection1.y,
                        "x",
                        markersize=8,
                        markeredgewidth=1,
                        color="r",
                    )
                    time11 = times_drift_o[ii]
                    console.print(
                        f"intersection point: {pointa}, closestindex = {ii}",
                        style="blue",
                    )
                    console.print(
                        f"lon_at_index = {lons_drift[ii]}, lat_at_index = "
                        f"{lats_drift[ii]}, time_at_index = {time11}",
                        style="blue",
                    )
            elif intersection1.geom_type == "MultiPoint":
                for (
                        pointa
                ) in intersection1.geoms:  # Use .geoms to iterate over MultiPoint
                    if ind_buffer1.contains(pointa):
                        ii = get_closest_point_index(drifter_path, pointa)
                        time11 = times_drift_o[ii]
                        console.print(
                            f"intersection point: {pointa}, closestindex = {ii}",
                            style="blue",
                        )
                        console.print(
                            f"lon_at_index = {lons_drift[ii]}, lat_at_index = "
                            f"{lats_drift[ii]}, time_at_index = {time11}",
                            style="blue",
                        )
                        ax.plot(
                            pointa.x,
                            pointa.y,
                            "x",
                            markersize=8,
                            markeredgewidth=1,
                            color="r",
                        )
plt.savefig(f"p.drifters_near_coast.intersecting.tracks.selected.png", bbox_inches="tight", dpi=300)
plt.show()
plt.close("all")
