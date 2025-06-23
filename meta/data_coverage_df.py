import datetime
from collections import Counter

import xarray as xr
from tools_xtrackm import *

df = pd.DataFrame(
    columns=["sat", "trackn", "first_date", "last_date", "tfreq", "xfreq"]
)

for sat in sats_new:
    tracks_n = get_total_tracks(sat)
    intervals = []
    delds = []
    for f in sorted(glob.glob(f"../data/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f)
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        time_s = ds.time.isel(cycles_numbers=0).isel(points_numbers=0).item()
        time_e = ds.time.isel(cycles_numbers=0).isel(points_numbers=-1).item()
        scnds = (time_e - time_s).seconds
        time = ds.time.isel(points_numbers=0)
        time1 = time.values
        first_time = time1[0]
        last_time = time1[-1]
        first_time_str = first_time.strftime("%Y-%m-%d")
        last_time_str = last_time.strftime("%Y-%m-%d")
        time_datetime = [
            datetime(year=t.year, month=t.month, day=t.day) for t in time1
        ]
        lon_s = lons_track[0]
        lon_e = lons_track[-1]
        lat_s = lats_track[0]
        lat_e = lats_track[-1]
        for i in range(1, len(time_datetime)):
            interval = time_datetime[i] - time_datetime[i - 1]
            intervals.append(interval.days)
        if len(lons_track) > 10:
            for i in range(len(lons_track) - 1):
                lat1 = lats_track[i]
                lon1 = lons_track[i]
                lat2 = lats_track[i + 1]
                lon2 = lons_track[i + 1]
                deld = distance.distance((lat1, lon1), (lat2, lon2)).km
                delds.append(deld)
    delds1 = delds[:]  # [x for x in delds if x!=0]
    try:
        xfreq = sum(delds1) / len(delds1)
    except ZeroDivisionError:
        xfreq = 0
    data_counts = Counter(intervals)
    most_common = data_counts.most_common(1)
    tfreq = most_common[0][0]
    df = df._append(
        {
            "sat": sat,
            "trackn": tracks_n,
            "first_date": first_time_str,
            "last_date": last_time_str,
            "tfreq": tfreq,
            "xfreq": xfreq,
        },
        ignore_index=True,
    )

print(df)
df.to_csv("data_coverage_df.csv")
