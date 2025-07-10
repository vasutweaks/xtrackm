import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tools_xtrackm import sats_new

fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)

frac1 = 0.05
xlim1, xlim2 = (-8.0, 80.0)

j = 0
for sat in sats_new[:]:
    i = 0
    gshhg_np = np.empty(0)
    counts_np = np.empty(0)
    for f in sorted(glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        gshhg = 0.001 * ds.dist_to_coast_gshhg.values
        counts = ds.sla.count(dim="cycles_numbers").values
        count_perc = 100 * counts / len(ds.cycles_numbers)
        gshhg_np = np.append(gshhg_np, gshhg)
        counts_np = np.append(counts_np, count_perc)
        i = i + 1
    ax = axes.flat[j]
    ax.scatter(
        gshhg_np,
        counts_np,
        marker=".",
        s=1.0,
        color="c",
    )
    print(len(gshhg_np), i)
    f_lfit = f"xtrack_sla_valid_count_{sat}_loess_fit.nc"
    ds_lfit = xr.open_dataset(f_lfit, decode_times=False)
    print(ds_lfit)
    xout = ds_lfit.gshhg_resampled.values
    # xout = ds_lfit.standard_6km.values
    yout = ds_lfit.valid_count.values
    ax.plot(xout, yout, label=f"lowess {frac1}")
    ax.set_xlabel("distance from coast")
    ax.set_ylabel("% valid points")
    ax.set_xlim(xlim1, xlim2)
    ax.set_ylim(0, 110)
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.set_xticks(np.arange(0, xlim2, 10))
    ax.text(
        0.5,
        0.8,
        f"{sat}",
        transform=ax.transAxes,
        fontweight="bold",
        color="r",
    )
    j = j + 1
plt.savefig(
    f"p.count.alltracks_allsat.gshhg.loess.allsat.4by2.png",
    bbox_inches="tight",
)
plt.show()
