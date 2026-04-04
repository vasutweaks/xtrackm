import glob

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import xarray as xr
# import funcformatter
from tools_xtrackm import sats_new, track_dist_time_asn
# copilot --resume=ec9cb701-6d76-436d-a10f-d288254df66d
rows = []

for sat in sats_new[:]:
    counts_array = []
    for f in sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            print(f, "zero length")
            continue
        lon_equat = lons_track[0]
        info = f"{sat} {track_number}"
        print(info)
        sla_asn = track_dist_time_asn(ds, var_str="sla", units_in="m")

        n_x = len(sla_asn.x)
        n_time = len(sla_asn.time)
        n_xt = n_x * n_time

        # min % valid along x (worst time step: fewest valid x-points)
        perc_valid_x = float(100 * sla_asn.count(dim="x").median() / n_x)
        # min % valid along time (worst x-point: fewest valid time steps)
        perc_valid_time = float(100 * sla_asn.count(dim="time").median() / n_time)
        perc_valid_xt = float(100 * sla_asn.count() / n_xt)

        rows.append({
            "sat": sat,
            "track_number": int(track_number),
            "perc_valid_x": round(perc_valid_x, 2),
            "perc_valid_time": round(perc_valid_time, 2),
            "perc_valid_xt": round(perc_valid_xt, 2),
            "lon_equat": lon_equat
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
        ds.close()
        # plt.close("all")

out_csv = "spatio_temporal_valid_perc.csv"
df_valid = pd.DataFrame(rows)
df_valid.to_csv(out_csv, index=False)
print(f"Saved {out_csv} ({len(df_valid)} rows)")
