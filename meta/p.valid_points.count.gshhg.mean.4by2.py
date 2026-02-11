import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# import funcformatter
from matplotlib.ticker import FuncFormatter
from tools_xtrackm import sats_new, track_dist_time_asn_gshhg

fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)

xlim1, xlim2 = (-10e3, 500e3)

x_common = xr.DataArray(np.arange(0, 2_300_001, 6_000), dims=["x"],
  attrs={"units": "m"})  # 0..1300 km, 1 km step


subplot_n = 0
for sat in sats_new[:]:
    counts_array = []
    ax1 = axes.flat[subplot_n]
    files = sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc"))
    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        sla_asn = track_dist_time_asn_gshhg(ds, var_str="sla", units_in="m")
        counts = sla_asn.count(dim="time")
        count_perc = 100 * counts / len(sla_asn.time)
        count_perc1 = count_perc.interp(x=x_common)
        counts_array.append(count_perc1)
        ax1.scatter(
            count_perc.x.values,
            count_perc.values,
            marker=".",
            s=1.0,
            color="c",
        )
    counts_mean = xr.concat(counts_array, dim="track").mean(dim="track")
    counts_mean.plot(ax = ax1)
    subplot_n = subplot_n + 1
    ax1.set_xlabel("distance from coast")
    ax1.set_ylabel("% valid points")
    ax1.set_xlim(xlim1, xlim2)
    ax1.set_ylim(0, 110)
    ax1.axvline(x=0, color='gray', linestyle='--')
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
    ax1.text(
        0.5,
        0.8,
        f"{sat}",
        transform=ax1.transAxes,
        fontweight="bold",
        color="r",
    )
plt.show()
