import os

import numpy as np
import xarray as xr
from tools_xtrackm import *

TRACKS_REG = (73.0, 99.0, 0.0, 25.4)
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
rose = dse.ROSE
cmap1 = "Greys_r"
dist_threshold = 2
data_loc = f"/home/srinivasu/allData/drifter1/"
fd = f"{data_loc}/netcdf_15001_current/track_reg/drifter_6h_133666.nc"

basename1 = os.path.basename(fd)
ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
print(ds_d)
# print attributes
for k, v in ds_d.attrs.items():
    print(f"{k}: {v}")
byte_id = ds_d.ID.values[0]
str_id = byte_id.decode("utf-8")
drifter_id = str_id
drift_tsta_o1, drift_tend_o1 = (
    ds_d.start_date.values[0],
    ds_d.end_date.values[0],
)
print(f"{drifter_id} {drift_tsta_o1} {drift_tend_o1}")
drift_tsta_o, drift_tend_o = n64todatetime(drift_tsta_o1), n64todatetime(drift_tend_o1)

times_drift = ds_d.time.isel(traj=0).values
ve_da = drifter_time_asn(ds_d, var_str="ve")
vn_da = drifter_time_asn(ds_d, var_str="vn")
lons_da = drifter_time_asn(ds_d, var_str="longitude")
lats_da = drifter_time_asn(ds_d, var_str="latitude")
lons_da1 = lons_da.resample(time="1D").mean()
lats_da1 = lats_da.resample(time="1D").mean()
lons_da2 = lons_da1.rolling(time=3).mean()
lats_da2 = lats_da1.rolling(time=3).mean()
lons_da3 = lons_da2.ffill(dim="time").bfill(dim="time")
lats_da3 = lats_da2.ffill(dim="time").bfill(dim="time")

lons_drift = lons_da3.values
lats_drift = lats_da3.values

fig = plt.figure(figsize=(12, 10), layout="constrained")
ax2 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
rose.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    *TRACKS_REG[2:])).plot(ax=ax2,
                           add_colorbar=False,
                           add_labels=False,
                           cmap=cmap1)
decorate_axis(ax2, "", *TRACKS_REG, step=5)
ax2.grid()
ax2.scatter(
    lons_drift,
    lats_drift,
    marker=".",
    color="b",
    s=4,
)
# add polygon box as geometry
# ax2.add_geometries([box],
#                    ccrs.PlateCarree(),
#                    edgecolor="g",
#                    facecolor="none",
#                    linewidth=2)
plt.show()
