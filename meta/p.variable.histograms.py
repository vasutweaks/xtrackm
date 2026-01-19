import datetime
from collections import Counter

import xarray as xr
from tools_xtrackm import *
import numpy as np
from math import sqrt, erf

# initiate empyt sla numpy array
sla_np = np.array([])
dac_np = np.array([])
ocean_tide_np = np.array([])
ssb_np = np.array([])

# print min, max, mean and standard deviation of numpy array
# of array containing nans
def stats_nan(sla_np):
    print(f"min is: {np.nanmin(sla_np)}")
    print(f"max is: {np.nanmax(sla_np)}")
    print(f"mean is: {np.nanmean(sla_np)}")
    print(f"std is: {np.nanstd(sla_np)}")


def perc_outside(std_limit = 3.5):
    """Percentage of values outside 3.5 standard deviations in a standard normal distribution"""
    phi0 = 0.5 * (1 + erf(std_limit / sqrt(2)))
    perc0 = (2 * phi0 - 1) * 100
    return 100 - perc0


std_limit = 3.5
for sat in sats_new:
    tracks_n = get_total_tracks(sat)
    print(f"number of tracks of {sat}: {tracks_n}")
    for f in sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        sla = ds.sla
        dac = ds.dac
        ocean_tide = ds.ocean_tide
        ssb = ds.ssb
        sla_np = np.append(sla_np, sla.values)
        dac_np = np.append(dac_np, dac.values)
        ocean_tide_np = np.append(ocean_tide_np, ocean_tide.values)
        ssb_np = np.append(ssb_np, ssb.values)
    stats_nan(sla_np)
    print(40 * "*")
    stats_nan(dac_np)
    print(40 * "*")
    stats_nan(ocean_tide_np)
    print(40 * "*")
    stats_nan(ssb_np)
    mean = np.nanmean(sla_np)
    # print outliers above 3.5 standard deviations
    std = np.nanstd(sla_np)
    outliers = np.abs(sla_np - mean) > std_limit * std
    # percentage of outliers
    perc = outliers.sum() / len(sla_np) * 100
    # normal percentage of outliers
    perc_out = perc_outside(std_limit)
    print(f"normal percentage of outliers: {perc_out}")
    print(f"percentage of outliers: {perc}")
    break

    # plot histogram
    fig, ax = plt.subplots()
    ax.hist(sla_np, bins=100, label=f"{sat}")
    plt.legend()
    plt.show()
