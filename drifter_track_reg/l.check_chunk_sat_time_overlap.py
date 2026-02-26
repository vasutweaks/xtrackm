import numpy
from tools_xtrackm import *

chunk_time_dict = {
    "1_5000": (
        np.datetime64("1985-10-30T00:00:00.000000000"),
        np.datetime64("2001-09-26T00:00:00.000000000"),
    ),
    "5001_10000": (
        np.datetime64("1997-03-20T00:00:00.000000000"),
        np.datetime64("2009-04-25T00:00:00.000000000"),
    ),
    "10001_15000": (
        np.datetime64("2006-03-05T00:00:00.000000000"),
        np.datetime64("2012-11-04T00:00:00.000000000"),
    ),
    "15001_current": (
        np.datetime64("2010-09-30T00:00:00.000000000"),
        np.datetime64("2025-05-19T00:00:00.000000000"),
    ),
}

chunks = ["1_5000", "5001_10000", "10001_15000", "15001_current"]
for chunk in chunks:
    drift_tsta_o1, drift_tend_o1 = chunk_time_dict[chunk]
    # print(type(track_tsta_o), type(track_tend_o))
    for sat in sats_new:
        track_tsta_o, track_tend_o = get_time_limits_o_np64(sat)
        # print(type(drift_tsta_o), type(drift_tend_o))
        overlap_tsta_o, overlap_tend_o = overlap_dates_np64(
            track_tsta_o, track_tend_o, drift_tsta_o1, drift_tend_o1
        )
        # if overlap_tsta_o is nat or overlap_tend_o is nat:
        if np.isnat(overlap_tsta_o) or np.isnat(overlap_tend_o):
            continue
        print(
            f"{chunk} overlap for {sat} is {overlap_tsta_o} to {overlap_tend_o}"
        )
