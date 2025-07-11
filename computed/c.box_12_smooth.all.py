import xarray as xr
from tools_xtrackm import *

for sat in sats_new[:]:
    for f in sorted(
        glob.glob(
            f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) < 12:
            continue
        print(f"{sat} {track_number} -------------------")
        sla_track = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
        dac_track = track_dist_time_asn_midx(ds, var_str="dac", units_in="m")
        sla_smooth = track_smooth_box_x(sla_track, smooth=12, min_periods=6)
        dac_smooth = track_smooth_box_x(dac_track, smooth=12, min_periods=6)
        # sla_smooth = track_smooth_loess_statsmodels_x_parallel(sla_track, frac=0.2)
        # dac_smooth = track_smooth_loess_statsmodels_x_parallel(dac_track, frac=0.2)
        ds_out = xr.Dataset({"sla_smooth": sla_smooth, "dac_smooth": dac_smooth})
        ds_out.to_netcdf(
        f"sla_box_12/track_sla_along_{sat}_{track_number}_box_12.nc"
        )
