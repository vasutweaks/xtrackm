import pandas as pd
import xarray as xr
from pathlib import Path
from namesm import sats_new
import matplotlib.pyplot as plt

ekman_dir = "/home/srinivasu/allData/ccmp31/"
out_dir = Path("/home/srinivasu/xtrackm/ekman/ekman_at_intersections/")
out_dir_cd = Path("/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd/")

SAT = "TP+J1+J2+J3+S6A"
SAT = "S3A"
DEPTH = "0"
inter_df = pd.read_csv(f"tracks_intersections_{SAT}_1.csv")

for i, r in inter_df.iterrows():
    sat = r["sat"]
    track_number_self = str(r["track_self"])
    track_number_other = str(r["track_other"])
    lons_inter1 = r["lons_inter"]
    lats_inter1 = r["lats_inter"]
    unique_str = f"{sat}_{track_number_self}_{track_number_other}"
    f_ek = f"{out_dir}/{unique_str}_ekman_{DEPTH}m_at_intersections.nc"
    f_ek_cd = f"{out_dir_cd}/{unique_str}_ekman_{DEPTH}m_at_intersections_cd.nc"
    ds_ek = xr.open_dataset(f_ek)
    print(ds_ek)
    ds_ek_cd = xr.open_dataset(f_ek_cd)
    u_ek = ds_ek["u_at"].resample(time="ME").mean()
    u_ek_cd = ds_ek_cd["u_at"].resample(time="ME").mean()
    u_ek.plot(color="b", label="ekman")
    u_ek_cd.plot(color="r", label="ekman cd")
    plt.legend()
    plt.show()
    
