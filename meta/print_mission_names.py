import datetime
from collections import Counter

import xarray as xr
from tools_xtrackm import *

for sat in sats_new:
    f = get_first_file(sat)
    tracks_n = get_total_tracks(sat)
    ds = xr.open_dataset(f)
    mission = ds.Mission
    passn = ds.Pass
    cycle = ds.cycle
    print(f"mission and pass : {mission} {passn} {cycle}")
