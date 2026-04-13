import datetime
from collections import Counter

import xarray as xr
from tools_xtrackm import *

rows = []

for sat in sats_new:
    tracks_n = get_total_tracks(sat)
    print(f"number of tracks of {sat}: {tracks_n}")

    n_intersections = len(pd.read_csv(f"tracks_intersections_{sat}_1.csv"))

    delds = []
    intervals = []
    first_time_str = None
    last_time_str = None

    for f in sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f)
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        time = ds.time.isel(points_numbers=0)
        time1 = time.values
        first_time = time1[0]
        last_time = time1[-1]
        first_time_str_cur = first_time.strftime("%Y-%m-%d")
        last_time_str_cur = last_time.strftime("%Y-%m-%d")
        if first_time_str is None or first_time_str_cur < first_time_str:
            first_time_str = first_time_str_cur
        if last_time_str is None or last_time_str_cur > last_time_str:
            last_time_str = last_time_str_cur
        time_datetime = [
            datetime(year=t.year, month=t.month, day=t.day) for t in time1
        ]
        for i in range(1, len(time_datetime)):
            interval = time_datetime[i] - time_datetime[i - 1]
            intervals.append(interval.days)
        for i in range(len(lons_track) - 1):
            lat1 = lats_track[i]
            lon1 = lons_track[i]
            lat2 = lats_track[i + 1]
            lon2 = lons_track[i + 1]
            deld = distance.distance((lat1, lon1), (lat2, lon2)).km
            delds.append(deld)

    try:
        xfreq = sum(delds) / len(delds)
    except ZeroDivisionError:
        xfreq = 0

    data_counts = Counter(intervals)
    most_common = data_counts.most_common(1)
    tfreq = most_common[0][0]

    rows.append(
        {
            "sat": sat,
            "trackn": tracks_n,
            "n_intersections": n_intersections,
            "first_date": first_time_str,
            "last_date": last_time_str,
            "tfreq": tfreq,
            "xfreq": xfreq,
        }
    )

df = pd.DataFrame(rows, columns=["sat", "trackn", "n_intersections", "first_date", "last_date", "tfreq", "xfreq"])
print(df)
df.to_csv("data_coverage_df.csv")
