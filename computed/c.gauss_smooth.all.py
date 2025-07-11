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
        print(f"{sat} {track_number} -------------------")
        if len(ds.lon.values) < 6:
            continue
        sla_track = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
        dac_track = track_dist_time_asn_midx(ds, var_str="dac", units_in="m")
        sla_smooth = track_smooth_gaussian_x(sla_track, sigma=3)
        dac_smooth = track_smooth_gaussian_x(dac_track, sigma=3)
        ds_out = xr.Dataset({"sla_smooth": sla_smooth, "dac_smooth": dac_smooth})
        ds_out.to_netcdf(
        f"sla_gauss/track_sla_along_{sat}_{track_number}_gauss3.nc"
        )
        sla_smooth = track_smooth_gaussian_x(sla_track, sigma=6)
        dac_smooth = track_smooth_gaussian_x(dac_track, sigma=6)
        ds_out = xr.Dataset({"sla_smooth": sla_smooth, "dac_smooth": dac_smooth})
        ds_out.to_netcdf(
        f"sla_gauss/track_sla_along_{sat}_{track_number}_gauss6.nc"
        )
