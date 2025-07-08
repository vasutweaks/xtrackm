import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tools_xtrackm import *

intervals = np.arange(0, 200)

fig, axes = plt.subplots(3, 2, figsize=(12, 14), sharex=True)

j = 0
for sat in sats_new[:]:
    tsta, tend = get_time_limits(sat=sat)
    f_ref = get_first_file(sat)
    ds_ref = xr.open_dataset(f_ref, decode_times=False)
    cycle_len_ref = len(ds_ref.cycles_numbers)
    # sla_ref = track_dist_asn_gshhs(ds_ref, "sla")
    sla_ref = track_dist_time_asn_gshhg(ds_ref, "sla", units_in="km")
    print(sla_ref)
    # first counted and percentage
    count_ref = sla_ref.count(dim="time")
    count_ref = (count_ref/cycle_len_ref) * 100
    # and then interpolated
    count_ref_interp = count_ref.interp(x=intervals)
    count_ref_interp = count_ref_interp * 0.0

    i = 0
    for f in sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        cycle_len = len(ds.cycles_numbers)
        if len(ds.lon) == 0:
            continue
        TRACK_NUMBER = ds.Pass
        try:
            # sla_track = track_dist_asn_gshhs(ds, "sla")
            sla_track = track_dist_time_asn_gshhg(ds, "sla", units_in="km")
        except ValueError:
            continue
        count_track = sla_track.count(dim="time")
        count_track = (count_track/cycle_len) * 100
        # and then interpolated
        count_track_interp = count_track.interp(x=intervals)
        count_ref_interp = xr.DataArray(
            np.nansum(
                [count_ref_interp, count_track_interp], axis=0
            ),
            dims="x",
        )
        axes.flat[j].scatter(
            count_track.x.values,
            count_track.values,
            marker=".",
            s=1.0,
            color="c",
        )
        i = i + 1
    count_ref_interp = count_ref_interp / float(i)
    print(count_ref_interp.values)
    axes.flat[j].plot(
        count_ref_interp.x.values,
        count_ref_interp.values,
        color="b",
    )
    axes.flat[j].set_xlim(-5, 75)
    axes.flat[j].set_ylim(0, 110)
    axes.flat[j].axvline(x=0, color="gray", linestyle="--")
    axes.flat[j].set_xlabel("distance from coast (km)")
    axes.flat[j].set_ylabel("% valid points")
    axes.flat[j].text(
        0.5,
        0.2,
        f"{sat}",
        transform=axes.flat[j].transAxes,
        fontweight="bold",
        color="r",
    )
    j = j + 1
plt.savefig(f"p.count.alltracks_allsat.gshhg.3by2.png", bbox_inches="tight")
plt.show()
