import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from tools_xtrackm import *
import matplotlib.pyplot as plt

cmap1 = "twilight_shifted"

d = 1.5
height, width = 9, 12

sat = "TP+J1+J2+J3+S6A"
df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")

dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
rose = dse.ROSE
rose_cut = rose.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:]))

fig2, ax2 = plt.subplots(
    1,
    1,
    subplot_kw={"projection": ccrs.PlateCarree()},
    figsize=(width, height),
    layout="constrained",
)

rose_cut.plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG, step=5)
ax2.set_extent([*TRACKS_REG], crs=ccrs.PlateCarree())
ax2.scatter(
    df["lons_inter"],
    df["lats_inter"],
    color="w",
    label=f"Satellite {sat}",
    s=18,
    marker="o",
    edgecolors='black',
    transform=ccrs.PlateCarree(),
)
for i, row in df.iterrows():
    # plot self and other tracks as text
    tself = row["track_self"]
    tothr = row["track_other"]
    txt = f"({tself}, {tothr})"
    ax2.text(
        row["lons_inter"],
        row["lats_inter"],
        txt,
        color="k",
        transform=ccrs.PlateCarree(),
        fontsize=8,
        rotation=-45,
    )

# edgecolors='black',
ax2.legend(loc="upper right")
# plt.title("Intersection Points of Tracks in North Indian Ocean")
plt.show()
dse.close()
