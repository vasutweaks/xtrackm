import math
import os

import pandas as pd
import xarray as xr
from tools_xtrackm import *

frac1 = "0.05"
method = "loess"
out_dir = f"/home/srinivasu/xtrackm/guv/guv_at_intersections_{method}/"
os.makedirs(out_dir, exist_ok=True)
print(sats_new)
sat = "ERS1+ERS2+ENV+SRL"
files = sorted(glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"))
for f in files:
    ds = xr.open_dataset(f, engine="netcdf4", decode_times=False)
    track_number = ds.Pass
    if len(ds.lat.values) == 0:
        continue
    f_smooth = f"../computed/sla_{method}_{frac1}/track_sla_along_{sat}_{track_number}_{method}_{frac1}.nc"
    ds_smooth = xr.open_dataset(f_smooth)
    sla = track_dist_time_asn(ds, var_str="sla", units_in="m")
    sla_smooth = ds_smooth.sla_smooth
    gc = compute_geostrophy_gc(ds, sla_smooth)
    # print nan stats before and after smoothing
    print(f"Track {track_number} - {sat}:")
    print(f"  gc (raw) - NaN count: {np.isnan(gc.values).sum()}, Total count: {gc.size}")
    # print nammin, nanmax, and mean before and after smoothing
    print(f"  gc (raw) - min: {np.nanmin(gc.values)}, max: {np.nanmax(gc.values)}, mean: {np.nanmean(gc.values)}")
    # print nanmedian and nanmad before and after smoothing
    print(f"  gc (raw) - nanmedian: {np.nanmedian(gc.values)}, nanmad: {np.nanmedian(np.abs(gc.values - np.nanmedian(gc.values)))}")
    gc_filt = mad_filter_2d_xr(gc, 3.5, dims=("time", "x"))
    print(f"  gc (filt) - NaN count: {np.isnan(gc_filt.values).sum()}, Total count: {gc_filt.size}")
    # print nammin, nanmax, and mean before and after smoothing
    print(f"  gc (filt) - min: {np.nanmin(gc_filt.values)}, max: {np.nanmax(gc_filt.values)}, mean: {np.nanmean(gc_filt.values)}")
    # print nanmedian and nanmad before and after smoothing
    print(f"  gc (filt) - nanmedian: {np.nanmedian(gc_filt.values)}, nanmad: {np.nanmedian(np.abs(gc_filt.values - np.nanmedian(gc_filt.values)))}")
    # plot gc before and after smoothing
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # gc.plot(ax=ax[0], label="gc (raw)", vmin=-3.5, vmax=3.5)
    # gc_filt.plot(ax=ax[1], label="gc (mad-filtered)", vmin=-3.5, vmax=3.5)
    gc.plot(ax=ax[0], label="gc (raw)", robust = True)
    gc_filt.plot(ax=ax[1], label="gc (mad-filtered)", robust = True)
    ax[0].set_title(f"Geostrophic Current (raw) - {sat} Track {track_number}")
    ax[1].set_title(f"Geostrophic Current (mad-filtered) - {sat} Track {track_number}")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Geostrophic Current (m/s)")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Geostrophic Current (m/s)")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    ds_out = xr.Dataset({"gc": gc_filt, "gc_raw": gc})
    print(f"saving to {out_dir}/gc_at_{sat}_{track_number}_{method}_{frac1}.nc")
    ds_out.to_netcdf(
        f"{out_dir}/gc_at_{sat}_{track_number}_{method}_{frac1}.nc"
    )
    break
