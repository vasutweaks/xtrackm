import glob
from collections import Counter

import xarray as xr
from geopy import distance
from namesm import *

frac1 = 0.05
frac1 = 0.2


for sat in sats_new:
    lens = []
    frac_lens = []
    for f in sorted(glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")):
        # print(f)
        ds = xr.open_dataset(f, decode_times=False)
        if len(ds.points_numbers) == 0:
            continue
        track_number = ds.pass_number
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        # time_s = ds.time.isel(cycles_numbers=0).isel(points_numbers=0)
        # time_e = ds.time.isel(cycles_numbers=0).isel(points_numbers=-1)
        # scnds = (time_e - time_s).seconds
        lon_equat = lons_track[0]
        lon_coast = lons_track[-1]
        lat_equat = lats_track[0]
        lat_coast = lats_track[-1]
        total_len = distance.distance((lat_equat, lon_equat),
                                      (lat_coast, lon_coast)).km
        lens.append(total_len)
        frac_lens.append(frac1 * total_len)
        print(f"frac of {sat} {track_number} is {frac1 * sum(lens)/len(lens)}")
    print(f"mean len of {sat} is {sum(lens)/len(lens)} km -----------")
    print(f"mean frac len of {sat} is {sum(frac_lens)/len(frac_lens)} km -----------")

