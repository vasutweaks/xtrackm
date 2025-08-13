import os
import glob
import xarray as xr
import numpy as np
import numpy
from tools_xtrackm import *

chunk_time_dict = {'1_5000': (numpy.datetime64('1985-10-30T00:00:00.000000000'), numpy.datetime64('1999-11-01T00:00:00.000000000')), '5001_10000': (numpy.datetime64('1997-03-20T00:00:00.000000000'), numpy.datetime64('2006-09-16T00:00:00.000000000')), '10001_15000': (numpy.datetime64('2006-05-24T00:00:00.000000000'), numpy.datetime64('2011-01-04T00:00:00.000000000')), '15001_current': (numpy.datetime64('2010-10-30T00:00:00.000000000'), numpy.datetime64('2024-12-31T00:00:00.000000000'))}

chunks = ["1_5000", "5001_10000", "10001_15000", "15001_current"]

for chunk in chunks:
    drift_tsta_o1, drift_tend_o1 = chunk_time_dict[chunk]
    drift_tsta_o, drift_tend_o = n64todatetime1(drift_tsta_o1), n64todatetime1(
        drift_tend_o1)
    # print(type(track_tsta_o), type(track_tend_o))
    for sat in sats_new:
        track_tsta_o, track_tend_o = get_time_limits_o(sat)
        # print(type(drift_tsta_o), type(drift_tend_o))
        overlap_tsta_o, overlap_tend_o = overlap_dates(track_tsta_o, track_tend_o,
                                                       drift_tsta_o, drift_tend_o)
        if overlap_tsta_o is None or overlap_tend_o is None:
            continue
        print(f"{chunk} overlap for {sat} is {overlap_tsta_o} to {overlap_tend_o}")
