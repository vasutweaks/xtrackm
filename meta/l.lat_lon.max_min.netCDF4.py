import glob
import xarray as xr
from namesm import *
import time
from netCDF4 import Dataset
import numpy as np

lon_mins=[]
lon_maxs=[]
lat_mins=[]
lat_maxs=[]

start_time = time.time()
for sat in sats_new:
    for f in sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        # print(f)
        ds = Dataset(f, mode="r")
        # adapt variable names to your files if different
        lons = ds.variables['lon'][:].astype('float64')
        lats = ds.variables['lat'][:].astype('float64')
        ds.close()
        lon_min = np.min(lons)
        lon_max = np.max(lons)
        lat_min = np.min(lats)
        lat_max = np.max(lats)
        lon_mins.append(lon_min)
        lon_maxs.append(lon_max)
        lat_mins.append(lat_min)
        lat_maxs.append(lat_max)

print(f"lon min is: {min(lon_mins)}")
print(f"lon max is: {max(lon_maxs)}")
print(f"lat min is: {min(lat_mins)}")
print(f"lat min is: {max(lat_maxs)}")

print("--- %s seconds ---" % (time.time() - start_time))
