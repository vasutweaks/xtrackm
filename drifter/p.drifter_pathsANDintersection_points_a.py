import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap

# ------------------------
# User/configurable inputs
# ------------------------
# Region: (lon_min, lon_max, lat_min, lat_max)
REG = (43.0, 99.0, 0.0, 25.4)
cmap1 = ListedColormap(["white"])

# ETOPO/topo file
ETOPO_PATH = "/home/srinivasu/allData/topo/etopo_60s_io.nc"

# Drifter files
CHUNK = "15001_current"
DRIFTER_GLOB = f"/home/srinivasu/allData/drifter1/netcdf_{CHUNK}/track_reg/drifter_6h_*.nc"

sats_new = [
    "GFO",
    "TP+J1+J2+J3+S6A",
    "S3A",
    "S3B",
    "TPN+J1N+J2N+J3N",
    "ERS1+ERS2+ENV+SRL",
    "HY2A",
    "HY2B",
]
# Marker per satellite (you can adjust)
sat_marker_dict = {
    "GFO": "*",
    "TP+J1+J2+J3+S6A": "o",
    "S3A": "^",
    "S3B": "^",
    "TPN+J1N+J2N+J3N": "o",
    "ERS1+ERS2+ENV+SRL": "x",
    "HY2A": "s",
    "HY2B": "s",
}


# ------------------------
# Helper to pick topo var
# ------------------------
def get_topo_and_coords(dset):
    # Try common variable names for etopo/topo height
    for v in ["ROSE", "rose", "z", "elevation", "topo", "ETOPO", "Band1"]:
        if v in dset.variables:
            topo = dset[v]
            break
    else:
        topo = None

    # Try coordinate names
    lon_names = ["ETOPO05_X", "x", "lon", "longitude"]
    lat_names = ["ETOPO05_Y", "y", "lat", "latitude"]

    lon_name = next((n for n in lon_names if n in dset.coords), None)
    lat_name = next((n for n in lat_names if n in dset.coords), None)

    return topo, lon_name, lat_name


# ------------------------
# Load topo
# ------------------------
dse = xr.open_dataset(ETOPO_PATH)
topo, lon_name, lat_name = get_topo_and_coords(dse)

# ------------------------
# Create figure
# ------------------------
fig, ax = plt.subplots(1,
                       1,
                       figsize=(12, 9),
                       subplot_kw=dict(projection=ccrs.PlateCarree()))

# Add simple land/coatlines if desired
ax.add_feature(cfeature.LAND, facecolor="0.9")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.gridlines(draw_labels=True,
             linewidth=0.3,
             color="gray",
             alpha=0.5,
             linestyle="--")

# Plot topo if available
topo_sel = topo.sel({
    lon_name: slice(REG[0], REG[1]),
    lat_name: slice(REG[2], REG[3])
})
# A neutral background colormap to keep points visible
topo_sel.plot(ax=ax, add_colorbar=False, cmap="Greys_r")
# Set extent
ax.set_extent(REG, crs=ccrs.PlateCarree())

# ------------------------
# Plot drifter tracks
# ------------------------
# Expecting variables: longitude, latitude, ve in each drifter file
# and potentially traj dimension; adjust as needed
# white cmap for driter paths
# cmap = cm.get_cmap("white")
lon_var = "longitude"
lat_var = "latitude"
ve_var = "ve"
for fd in sorted(glob.glob(DRIFTER_GLOB)):
    ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
    lons = ds_d[lon_var].isel(traj=0).values
    lats = ds_d[lat_var].isel(traj=0).values
    ve = ds_d[ve_var].isel(traj=0).values
    sc = ax.scatter(
        lons,
        lats,
        c=ve,
        cmap=cmap1,
        s=4,
        transform=ccrs.PlateCarree(),
        alpha=0.9,
    )

# Optional: add a colorbar for ve if the last scatter produced it
# Comment this block if you have heterogeneous presence of ve
try:
    cbar = plt.colorbar(sc,
                        ax=ax,
                        orientation="vertical",
                        fraction=0.04,
                        pad=0.03)
    cbar.set_label("Drifter velocity (ve)")
except Exception:
    pass

# ------------------------
# Plot intersection points from all satellites
# ------------------------
colors = plt.cm.get_cmap("tab10", len(sats_new))

any_points = False
for i, sat in enumerate(sats_new):
    csv_path = f"tracks_intersections_{sat}_1.csv"
    df = pd.read_csv(csv_path)
    ax.scatter(
        df["lons_inter"],
        df["lats_inter"],
        color=colors(i),
        s=18,
        marker=sat_marker_dict.get(sat, "o"),
        transform=ccrs.PlateCarree(),
        label=sat,
    )
    any_points = True

# Legend for satellites
if any_points:
    ax.legend(title="Satellite", loc="upper right", frameon=True)

plt.title("Drifter Tracks with All Satellite Intersection Points")
plt.tight_layout()
plt.show()
