import xarray as xr
from tools_xtrackm import *

for sat in sats_new[:]:
    for f in sorted(
        glob.glob(
            f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.pass_number
        print(f"{sat} {track_number} -------------------")
        sla_track = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
        sla_slope = sla_track.differentiate("x")
        # What is slope for dac, following is wrong
        # dac_slope = dac_track.differentiate("x")
        sla_smooth = slope_smooth_optimal_filter_x(sla_slope, p=-7, q=8)
        # dac_smooth = slope_smooth_optimal_filter_x(dac_slope, p=-7, q=8)
        # sla_smooth.plot()
        # plt.show()
        ds_out = xr.Dataset({"sla_smooth": sla_smooth})
        ds_out.to_netcdf(
        f"sla_optim_78/track_sla_slope_along_{sat}_{track_number}_optim_78.nc"
        )
