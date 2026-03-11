import sys

import numpy as np
import xarray as xr
from tools_xtrackd import *
import os


def extract_oscar_xy(ds, da_oscar):
    dists = compute_dist(ds, units="m")  # to convert to meters
    dists_rev = dists[::-1]
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    times = da_oscar.time.values
    vals = np.full((len(lons_track_rev), len(times)), np.nan)
    vals[:, :] = da_oscar.sel(lon=ds.lon, lat=ds.lat, method="nearest").values.T
    var2 = xr.DataArray(
        vals,
        coords=[dists_rev, times],
        dims=["x", "time"],
    )
    return var2


ds_oscar = xr.open_dataset("/home/srinivasu/allData/oscar_0.25/oscar_0.25_u_full_xr_concat.nc")
print(ds_oscar)
oscar_u = ds_oscar.u

# sats = ["S3A"]
for sat in sats[:]:
    for f in sorted(
            glob.glob(
                f"../data/ctoh.sla.ref.{sat}.nindian.*.nc"
            )):
        ds = xr.open_dataset(f, decode_times=False)
        # print(ds)
        track_number = ds.pass_number
        fe = f"oscar_e_py/oscar_along_{sat}_{track_number}_u.nc"
        # check if fe exists
        if os.path.exists(fe):
            print(fe)
            continue
        u_oscar_e = extract_oscar_xy(ds, oscar_u)
        # u_oscar_e = oscar_u.sel(lon=ds.lon, lat=ds.lat, method="nearest")
        # v_oscar_e = extract_oscar_xy(ds, oscar_v)
        # s_oscar_e = extract_oscar_xy(ds, oscar_s)
        ds_out = xr.Dataset({
            "u_oscar_e": u_oscar_e,
        })
        ds_out.to_netcdf(
            f"oscar_e_py/oscar_along_{sat}_{track_number}_u.nc")

sys.exit()
# ds_out = xr.Dataset({
#     "u_oscar_e": u_oscar_e,
#     "v_oscar_e": v_oscar_e,
#     "s_oscar_e": s_oscar_e,
# })
