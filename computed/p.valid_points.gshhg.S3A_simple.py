import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import FuncFormatter
from tools_xtrackm import sats_new, track_dist_time_asn_gshhg
import pickle

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

xlim1, xlim2 = (-10e3, 500e3)

x_common = xr.DataArray(np.arange(0, 1_300_001, 6_000), dims=["x"],
  attrs={"units": "m"})  # 0..1300 km, 1 km step


sat = "S3A"
counts_array = []
with open("common_passes_l2pVSxtrack.pkl", "rb") as f:
    common = pickle.load(f)

for pass1 in common:
    pass2 = pass1[1:]
    f = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{pass2}.nc"
    ds = xr.open_dataset(f, decode_times=False)
    track_number = ds.Pass
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
    )
counts_mean = xr.concat(counts_array, dim="track").mean(dim="track")
counts_mean.plot(ax = ax1)
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
