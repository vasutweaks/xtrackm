import xarray as xr
from tools_xtrackm import *
from namesm import sats_new

for sat in sats_new[:]:
    for f in sorted(
        glob.glob(
            f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        print(f"{sat} {track_number} -------------------")
        sla_track = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
        dac_track = track_dist_time_asn_midx(ds, var_str="dac", units_in="m")
        # sla_smooth = track_smooth_loess_statsmodels_x_parallel(sla_track, frac=0.2)
        # dac_smooth = track_smooth_loess_statsmodels_x_parallel(dac_track, frac=0.2)
        sla_smooth = track_smooth_loess_statsmodels_x(sla_track, frac=0.05)
        dac_smooth = track_smooth_loess_statsmodels_x(dac_track, frac=0.05)
        ds_out = xr.Dataset({"sla_smooth": sla_smooth, "dac_smooth": dac_smooth})
        ds_out.to_netcdf(
        f"sla_loess_0.05a/track_sla_along_{sat}_{track_number}_loess_0.05.nc"
        )
