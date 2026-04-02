import glob
import math

import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.ticker import FuncFormatter
from tools_xtrackm import *
import pickle

SAT = "S3A"
cmap = "twilight_shifted"
EAST_COAST_REG = (78.0, 95, 0, 25.4)

dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo5.cdf")
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*EAST_COAST_REG[:2])).sel(
    ETOPO05_Y=slice(*EAST_COAST_REG[2:]))

rose_cut.plot(
    ax=ax1, add_colorbar=False, add_labels=False, cmap=cmap
)

decorate_axis(ax1, "", *EAST_COAST_REG, step=8)
ax1.text(
    0.25,
    0.90,
    f"MISSION: {SAT}",
    transform=ax1.transAxes,
    fontsize=10,
    fontweight="bold",
    color="b",
)

with open("common_passes_l2pVSxtrack_selected.pkl", "rb") as f:
    common = pickle.load(f)

colors = colormaps["tab20"](np.linspace(0, 1, len(common)))
i=0
for pass1 in common:
    pass2 = pass1[1:]
    f = f"../data/{SAT}/ctoh.sla.ref.{SAT}.nindian.{pass2}.nc"
    with xr.open_dataset(f, decode_times=False) as ds:
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        ln = len(lons_track)
        middle_lon = lons_track[ln // 2]
        middle_lat = lats_track[ln // 2]
        mid_lon_i = np.argmin(np.abs(lons_track - middle_lon))
        mid_lat_i = np.argmin(np.abs(lats_track - middle_lat))
        lonm = middle_lon
        latm = middle_lat
        a1 = local_angle(lons_track, lats_track, mid_lat_i, ioff=3, degrees=False)
        a1_d = math.degrees(a1)
        color = colors[i]
        print(f"{SAT} {track_number} -------------------")
        ax1.scatter(lons_track, lats_track, s=38, color="w")
        if is_within_region(lonm, latm, *NIO):
            ax1.text(
                lonm,
                latm-0.5,
                s=str(track_number),
                fontsize=10,
                fontweight="bold",
                rotation=a1_d,
                color='r',
                ha='center',
                va='center',
            )
        i+=1

plt.tight_layout()
plt.savefig(f"f.l2p.S3A.common_tracks.map.png", dpi=300)
plt.show()
plt.close("all")

dse.close()
