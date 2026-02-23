import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *

cmap1 = "twilight_shifted"
cmap1 = ListedColormap(["white"])

# A map that is 100% white for every single value
d = 1.5
height, width = 9, 9
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset

df_all_list = []
for sat in sats_new:
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    df_all_list.append(df)
df_all = pd.concat(df_all_list, ignore_index=True)

satellites = df_all["sat"].unique()
colors = plt.cm.get_cmap("tab10", len(satellites))
# Unique satellites for coloring
ds_coast = xr.open_dataset(
    f"~/allData/topo/GMT_intermediate_coast_distance_01d_track_reg.nc"
)
COAST_DISTANCE_KM = 100


def is_close_to_coast(ds_coast, lon, lat, dist_km):
    dist1 = ds_coast.coast_distance.sel(lon=lon, lat=lat, method="nearest").item()
    return abs(dist1) <= dist_km

rose = dse.ROSE
fig2, ax2 = plt.subplots(
    1,
    1,
    subplot_kw={"projection": ccrs.PlateCarree()},
    figsize=(width, height),
    layout="constrained",
)
rose_cut = rose.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)
rose_cut.plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG, step=15)
ax2.set_extent([*TRACKS_REG], crs=ccrs.PlateCarree())
ax2.coastlines()
# list of 8 different of markers
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

for i, sat in enumerate(satellites):
    sat_data = df_all[df_all["sat"] == sat]
    close_mask = sat_data.apply(
        lambda row: is_close_to_coast(
            ds_coast, row["lons_inter"], row["lats_inter"], COAST_DISTANCE_KM
        ),
        axis=1,
    )
    sat_data_close = sat_data[close_mask]
    ax2.scatter(
        sat_data_close["lons_inter"],
        sat_data_close["lats_inter"],
        color=colors(i),
        label=f"Satellite {sat}",
        s=18,
        marker=sat_marker_dict[sat],
        transform=ccrs.PlateCarree(),
    )

# edgecolors='black',
ax2.legend(title="Satellite", loc="upper right")
plt.title("Intersection Points of Tracks in North Indian Ocean")
plt.show()
dse.close()
ds_coast.close()
