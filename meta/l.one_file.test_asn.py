import xarray as xr
from tools_xtrackm import track_dist_time_asn_midx

sat = "GFO"
passn = "008"
file1 = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{passn}.nc"
ds = xr.open_dataset(file1,decode_times=False)
print(ds)
sla_asn = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")

# `track_dist_time_asn_midx` returns dims ("x", "time") with lat as a coordinate on "x".
# Swap "x" with "lat" so we can select directly along latitude.
sla_asn_lat = sla_asn.swap_dims({"x": "lat"})#.sortby("lat")

# Example latitude-band selection.
lat_min, lat_max = 11.0, 5.0
sla_asn_lat_band = sla_asn_lat.sel(lat=slice(lat_min, lat_max))
print(sla_asn_lat_band)
