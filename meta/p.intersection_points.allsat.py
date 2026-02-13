import math
from matplotlib.colors import ListedColormap

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm

cmap1 = cmo.cm.diff
cmap1 = cmo.cm.tarn
cmap1 = cmo.cm.topo
cmap1 = cmo.cm.deep
cmap1 = cm.get_cmap('viridis')
cmap1 = cm.get_cmap('binary_r')
cmap1 = ListedColormap(['white'])

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

rose = dse.ROSE
fig2, ax2 = plt.subplots(
    1,
    1,
    subplot_kw={"projection": ccrs.PlateCarree()},
    figsize=(width, height),
    layout="constrained",
)
rose.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    *TRACKS_REG[2:])).plot(ax=ax2,
                           add_colorbar=False,
                           add_labels=False,
                           cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG)
ax2.set_extent([*TRACKS_REG], crs=ccrs.PlateCarree())
# list of 8 different of markers
sat_marker_dict = {"GFO": "*", "TP+J1+J2+J3+S6A": "o", "S3A": "^", "S3B": "^", "TPN+J1N+J2N+J3N": "o", "ERS1+ERS2+ENV+SRL": "x", "HY2A": "s", "HY2B": "s"}

for i, sat in enumerate(satellites):
    sat_data = df_all[df_all['sat'] == sat]
    ax2.scatter(
        sat_data['lons_inter'], 
        sat_data['lats_inter'],
        color=colors(i),
        label=f'Satellite {sat}',
        s=18,
        marker=sat_marker_dict[sat],
        transform=ccrs.PlateCarree()
    )

# edgecolors='black',
ax2.legend(title='Satellite', loc='upper right')
plt.title("Intersection Points of Tracks in North Indian Ocean")
plt.show()
dse.close()
