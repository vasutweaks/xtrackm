import glob

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import xarray as xr
# import funcformatter
from tools_xtrackm import sats_new, track_dist_time_asn
from geopy import distance
cmap = plt.get_cmap("tab10")
# copilot --resume=02600381-a35e-4bce-abcc-c0d243785bc8
rows = []

for sat in sats_new[:]:
    counts_array1 = []
    counts_array2 = []
    for f in sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        track_len = distance.distance((lats_track[0], lons_track[0]), (lats_track[-1], lons_track[-1])).km
        if len(lons_track) == 0:
            print(f, "zero length")
            continue
        lon_equat = lons_track[0]
        info = f"{sat} {track_number}"
        # print(info)
        sla_asn = track_dist_time_asn(ds, var_str="sla", units_in="m")

        n_x = len(sla_asn.x)
        n_time = len(sla_asn.time)
        n_xt = sla_asn.size

        # median % valid in x-direction (across time steps)
        perc_valid_x = float(100 * (sla_asn.count(dim="x") / n_x).median())
        # median % valid in time-direction (across x-positions)
        perc_valid_time = float(100 * (sla_asn.count(dim="time") / n_time).median())
        # % of time steps with fully valid x-profile (no spatial gaps)
        perc_full_x = float(100 * (sla_asn.count(dim="x") == n_x).sum() / n_time)
        # % of x-positions with fully valid time series (no temporal gaps)
        perc_full_time = float(100 * (sla_asn.count(dim="time") == n_time).sum() / n_x)
        perc_valid_xt = float(100 * sla_asn.count() / n_xt)

        rows.append({
            "sat": sat,
            "track_number": int(track_number),
            "perc_valid_x": round(perc_valid_x, 2),
            "perc_valid_time": round(perc_valid_time, 2),
            "perc_full_x": round(perc_full_x, 2),
            "perc_full_time": round(perc_full_time, 2),
            "perc_valid_xt": round(perc_valid_xt, 2),
            "lon_equat": lon_equat,
            "track_len": track_len
        })

        # counts = sla_asn.count(dim="time")
        # count_perc = 100 * counts / n_time
        # counts_array.append(count_perc)
        # fig, ax1 = plt.subplots(figsize=(10, 8))
        # sla_asn.plot.contourf(robust=True, ax=ax1, levels=np.linspace(-1.0, 1.0, 30))
        # sla_asn.plot.contourf(ax=ax1, levels=np.linspace(-0.3, 0.3, 30), extend="both")
        # ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
        # ax1.set_ylabel("distance from coast (km)")
        # ax1.axhline(400e3, color="black", linestyle="dashed", linewidth=0.5)
        # plt.title(info)
        # print(f"Saving {info}.png")
        # plt.savefig(f"x_time_plots/{info}.png", dpi=300, bbox_inches="tight")
        # plt.show()
        counts_array1.append(perc_valid_x)
        counts_array2.append(perc_valid_time)
        ds.close()

    mean_valid_x = np.median(counts_array1)
    mean_valid_time = np.median(counts_array2)
    dominant = "x" if mean_valid_x > mean_valid_time else "time"
    print(f"{sat}: x-dir {mean_valid_x:.1f}% valid (median), time-dir {mean_valid_time:.1f}% valid (median) → {dominant} direction is more valid")

    x = range(len(counts_array1))
    y1 = counts_array1
    y2 = counts_array2
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y1, label=f"{sat} x", linewidth=2.8)
    ax.plot(x, y2, label=f"{sat} time", linewidth=2.8)
    plt.scatter(x, y2, label=f"{sat} time", marker="x", s=20)
    # plt.show()
    plt.close("all")

out_csv = "spatio_temporal_valid_perc.csv"
df_valid = pd.DataFrame(rows)
df_valid.to_csv(out_csv, index=False)
print(f"Saved {out_csv} ({len(df_valid)} rows)")
