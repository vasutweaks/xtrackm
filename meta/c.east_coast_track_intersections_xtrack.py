import math
import pickle

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
from tools_xtrackm import *

TOPO_DIR = "/home/srinivasu/allData/topo/"
TRACKS_REG = (43.0, 99, 0, 25.4)
EAST_COAST_REG = (75.0, 99, 0, 25.4)
dse = xr.open_dataset(f"{TOPO_DIR}/etopo_60s_io.nc")
fig, ax1 = create_topo_map(dse, *EAST_COAST_REG, step=5)

SAT = "S3A"
files = sorted(glob.glob(f"/home/srinivasu/xtrackm/data/{SAT}_lon_ordered/ctoh.sla.ref.{SAT}.nindian.*.nc"))
east_coast_tracks = []
for f in files:
    ds = xr.open_dataset(f, decode_times=False)
    track_number = ds.Pass
    sla = ds.sla
    sla1 = sla.isel(cycles_numbers=0)
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lon_eq = lons_track[0]
    ln = len(lons_track)
    middle_lon = lons_track[int(0.3 * ln)]
    middle_lat = lats_track[int(0.3 * ln)]
    mid_lon_i = np.argmin(np.abs(lons_track - middle_lon))
    mid_lat_i = np.argmin(np.abs(lats_track - middle_lat))
    slope = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
    a1 = local_angle(lons_track, lats_track, mid_lat_i, ioff=3, degrees=False)
    a1_d = math.degrees(a1)
    if slope < 0:
        ax1.scatter(lons_track, lats_track, color="k", s=1, alpha=0.5)
        ax1.text(middle_lon, middle_lat, track_number, fontsize=12, ha='center', va='center',
                 transform=ccrs.PlateCarree(), color='w', rotation=a1_d)
        if lon_eq > EAST_COAST_REG[0] and lon_eq < EAST_COAST_REG[1]:
            east_coast_tracks.append(track_number)

plt.savefig(f"east_coast_intersections_a_{SAT}_xtrack.png", dpi=300)
plt.show()
plt.close("all")

print(east_coast_tracks)
with open(f"east_coast_tracks_xtrack.pkl", "wb") as f:
    pickle.dump(east_coast_tracks, f)
