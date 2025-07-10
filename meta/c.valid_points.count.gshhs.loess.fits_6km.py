import glob
import sys

import numpy as np
import xarray as xr
from loess.loess_1d import loess_1d
from scipy.signal import savgol_filter
from tools_xtrackm import sats_new, get_first_file, get_total_tracks
import matplotlib.pyplot as plt

frac1 = 0.05
# check if argument is supplied
if len(sys.argv) > 1:
    frac1 = float(sys.argv[1])
else:
    frac1 = 0.05

print(f"loess frac = {frac1}")

part1 = np.arange(0, 20, 1)

j = 0
for sat in sats_new[:]:
    i = 0
    # we are making a huge pile of gshhg distances across all tracks
    gshhg_np = np.empty(0)
    counts_np = np.empty(0)
    for f in sorted(glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        gshhg = 0.001 * ds.dist_to_coast_gshhg.values
        counts = ds.sla.count(dim="cycles_numbers").values
        count_perc = 100 * counts / len(ds.cycles_numbers)
        gshhg_np = np.append(gshhg_np, gshhg)
        counts_np = np.append(counts_np, count_perc)
        i = i + 1
    print(len(gshhg_np), i)
    max_dist = max(gshhg_np)
    # Next values with step 6, starting from 20
    # Example: generate 10 more values
    part2 = np.arange(20, max_dist, 6)
    # Combine both parts
    xnew = np.concatenate([part1, part2])
    xout, yout, wout = loess_1d(
        gshhg_np,
        counts_np,
        xnew=xnew,
        degree=1,
        frac=frac1,
        npoints=None,
        rotate=False,
        sigy=None,
    )
    da_out = xr.DataArray(yout, coords={"standard_6km": xout})
    ds_out = xr.Dataset({"valid_count": da_out})
    ds_out.to_netcdf(f"xtrack_sla_valid_count_{sat}_loess_fit_6km.nc")
    j = j + 1
