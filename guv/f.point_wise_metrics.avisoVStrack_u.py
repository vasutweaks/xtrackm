import cmocean as cmo
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from tools_xtrackm import plot_etopo_subplot, TRACKS_REG
from matplotlib.colors import ListedColormap
import math

cmap1 = cmo.cm.diff
sat = "S3B"
sat = "TP+J1+J2+J3+S6A"
cmap1 = ListedColormap(["white"])
cmap2 = "bwr"
fig = plt.figure(figsize=(10, 10), layout="constrained")
long_part = "correlations between aviso zonal current and track zonal current"
details = "at intersection points of tracks TP+J1+J2+J3+S6A"
title = f"{long_part}\n{details}"
ax = plot_etopo_subplot(fig, 111, TRACKS_REG, cmap1, title=title, step=5,
                       data_path="~/allData/topo/etopo5.cdf")
df = pd.read_csv(f"tracksVSaviso_corrs_rmses_biass_{sat}_u.csv")

ln = len(df)
for i in range(ln):
    sat1 = df["sat"].values[i]
    lons_inter1 = df["lons_inter"].values[i]
    lats_inter1 = df["lats_inter"].values[i]
    lon1 = lons_inter1
    lat1 = lats_inter1
    corr = df["corrs"].values[i]
    rmse = df["rmses"].values[i]
    bias = df["biass"].values[i]
    # size of marker proportional to correlation
    # check if corr is a valid number
    if not math.isnan(corr):
        ax.scatter(
            lon1,
            lat1,
            s=100 * abs(corr),
            c=[corr],
            vmin=-1,
            vmax=1,
            cmap=cmap2,
        )
# sm.set_array([])
# plt.colorbar(sm, ax=ax, orientation="horizontal", label="Correlation")

# Choose a few example correlation values to represent
legend_corrs = [0.2, 0.5, 0.8]  # example correlation magnitudes
legend_sizes = [100 * abs(c) for c in legend_corrs]
# Create legend handles manually
legend_handles = [
    plt.scatter([], [], s=size, edgecolors='k', facecolors='red', label=f"{corr:.1f}")
    for corr, size in zip(legend_corrs, legend_sizes)
]
ax.legend(
    handles=legend_handles,
    title="|Correlation|",
    loc="lower right",  # or another suitable position
    frameon=True,
)

import matplotlib.cm as cm
sm = cm.ScalarMappable(cmap=cmap2, norm=plt.Normalize(vmin=-1, vmax=1))
plt.subplots_adjust(bottom=0.2)
# Add horizontal colorbar below
cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Correlation')
plt.savefig(f"tracksVSaviso_corrs_rmses_biass_{sat}_u.png")
plt.show()
