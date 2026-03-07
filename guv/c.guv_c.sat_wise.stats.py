import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import os


# sats_new = ("TP+J1+J2+J3+S6A", "S3A")

frac1 = "0.05"
guv_dir = "/home/srinivasu/xtrackm/guv/guv_at_intersections"
for sat in sats_new:
    gu_all = []
    df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
    for _, row in df.iterrows():
        sat1 = row["sat"]
        track_number_self = str(row["track_self"])
        track_number_other = str(row["track_other"])
        lons_inter1 = row["lons_inter"]
        lat1 = row["lats_inter"]
        if abs(lat1) < 2:
            continue
        f_guv = (
            f"{guv_dir}/guv_at_intersection_{sat1}_{track_number_self}_"
            f"{track_number_other}_{frac1}.nc"
        )
        if not os.path.exists(f_guv):
            print(f"missing file: {f_guv}")
            continue
        with xr.open_dataset(f_guv) as ds_guv:
            gu = ds_guv["gu"].values
        gu = gu[np.isfinite(gu)]
        if gu.size == 0:
            continue
        print(
            sat,
            track_number_self,
            track_number_other,
            lons_inter1,
            lat1,
            gu.size,
        )
        gu_all.extend(gu)

    # plot histogram of gu_all
    plt.hist(gu_all, bins=100, label=f"{sat}")
    np.savez(f"guv_at_intersections_{sat}_{frac1}.npz", gu_all=gu_all)
    plt.show()
