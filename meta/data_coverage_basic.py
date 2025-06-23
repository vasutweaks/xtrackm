import datetime

import pandas as pd
import xarray as xr
from tools_xtrackm import *
from geopy import distance
from namesm import *

sats = sats_new

df = pd.DataFrame(
    columns=["sat", "trackn", "first_date", "last_date", "tfreq"]
)

for sat in sats_new:
    f = get_first_file(sat)
    trackn = get_total_tracks(sat)
    ds = xr.open_dataset(f)
    track_number = ds.Pass
    time = ds.time.isel(points_numbers=0)
    time1 = time.values
    first_time = time1[0]
    last_time = time1[-1]
    first_time_str = first_time.strftime("%Y-%m-%d")
    last_time_str = last_time.strftime("%Y-%m-%d")
    time_datetime = [
        datetime(year=t.year, month=t.month, day=t.day) for t in time1
    ]
    tfreq = xr.infer_freq(time_datetime)
    print(f"{sat} {trackn} {first_time_str} {last_time_str} {tfreq}")
    df = df._append(
        {
            "sat": sat,
            "trackn": trackn,
            "first_date": first_time_str,
            "last_date": last_time_str,
            "tfreq": tfreq,
        },
        ignore_index=True,
    )

print(df)
