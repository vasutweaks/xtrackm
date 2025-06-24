import glob
import xarray as xr
from namesm import *

for sat in sats_new:
    for f in sorted(
        glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")
    ):
        ds = xr.open_dataset(f, decode_times=False)
        trackn = ds.Pass
        if len(ds.lon.values) == 0:
            print(f"{sat} {trackn}")
