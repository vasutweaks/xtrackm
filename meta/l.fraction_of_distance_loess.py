import glob
import xarray as xr
from namesm import *
from geopy import distance

fixed_dist = 75_000 # m

for sat in sats_new:
    for f in sorted(glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")):
        # print(f)
        ds = xr.open_dataset(f, decode_times=False)
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        track_number = ds.Pass
        if len(lats_track) == 0:
            continue
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        lon_equat = lons_track[0]  # this on equator
        lat_equat = lats_track[0]  # this on equator
        dist = distance.distance((lat_coast, lon_coast), (lat_equat, lon_equat)).m
        frac = fixed_dist/dist
        info = f"{dist} {frac} {sat} {track_number}"
        print(info)
