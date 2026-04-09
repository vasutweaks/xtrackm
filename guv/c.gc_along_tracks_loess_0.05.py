import os

import xarray as xr
from tools_xtrackm import *

frac1 = "0.05"
method = "loess"
out_dir = f"/home/srinivasu/xtrackm/guv/gc_along_{method}/"
os.makedirs(out_dir, exist_ok=True)
print(sats_new)
for sat in sats_new:
    files = sorted(glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"))
    for f in files:
        ds = xr.open_dataset(f, engine="netcdf4", decode_times=False)
        track_number = ds.Pass
        if len(ds.lat.values) < 3:
            continue
        f_smooth = f"../computed/sla_{method}_{frac1}a/track_sla_along_{sat}_{track_number}_{method}_{frac1}.nc"
        ds_smooth = xr.open_dataset(f_smooth)
        sla_smooth = ds_smooth.sla_smooth
        gc = compute_geostrophy_gc(ds, sla_smooth)
        gc = mad_filter_2d_xr(gc, 3.5, dims=("time", "x"))
        ds_out = xr.Dataset({"gc": gc})
        print(
            f"saving to {out_dir}/gc_along_{sat}_{track_number}_{method}_{frac1}.nc")
        ds_out.to_netcdf(
            f"{out_dir}/gc_along_{sat}_{track_number}_{method}_{frac1}.nc"
        )
