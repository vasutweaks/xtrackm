import glob
import xarray as xr
from namesm import *

lon_mins=[]
lon_maxs=[]
lat_mins=[]
lat_maxs=[]

for sat in sats_new:
    for f in sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        print(f)
        ds = xr.open_dataset(f,decode_times=False)
        lon_min = ds.lon.min().item()
        lon_max = ds.lon.max().item()
        lat_min = ds.lat.min().item()
        lat_max = ds.lat.max().item()
        lon_mins.append(lon_min)
        lon_maxs.append(lon_max)
        lat_mins.append(lat_min)
        lat_maxs.append(lat_max)

print(f"lon min is: {min(lon_mins)}")
print(f"lon max is: {max(lon_maxs)}")
print(f"lat min is: {min(lat_mins)}")
print(f"lat min is: {max(lat_maxs)}")
