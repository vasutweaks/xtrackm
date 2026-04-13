import os
import pickle

import xarray as xr
import xskillscore as xs
from tools_xtrackm import *

RADARS = ("TN", "AP", "OR")
IN_DIR = "radar_rotatedVStrack_gc_along_intersections"
META_DIR="/home/srinivasu/xtrackm/meta/"


def select_heading_angle(sat, track_number, meta_dir=META_DIR):
    f = os.path.join(meta_dir, f"slopes_coastal_{sat}.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing slope file: {f}")

    df = pd.read_csv(f)
    track_number = int(track_number)
    row = df[(df["sat"] == sat) & (df["track_number"] == track_number)]
    if row.empty:
        raise ValueError(f"No heading angle found for sat={sat}, track_number={track_number}")

    return float(row.iloc[0]["heading_deg"])


def select_heading(row):
    return select_heading_angle(row["sat"], row["track_number"])


dfs = []
for sat in sats_new:
    f = f"{META_DIR}/slopes_coastal_{sat}.csv"
    df = pd.read_csv(f)
    dfs.append(df)

df_slopes = pd.concat(dfs, ignore_index=True)

def print_rsq_diff(corr, corr_ek):
    r_square = corr**2
    r_square_ek = corr_ek**2
    r_mean = r_square.mean()
    r_mean_ek = r_square_ek.mean()
    diff = r_mean_ek - r_mean
    print(f"{r_mean}, {r_mean_ek}, {100*diff}")


dfs = []
dfs_ek = []
for rstr in RADARS:
    f = f"radar_rotatedVStrack_gc_along_intersections/radar_rotatedVStrack_gc_along_summary_{rstr}.csv"
    f_ek = f"radar_rotatedVStrack_gc_along_intersections/radar_rotatedVStrack_gc_along_summary_{rstr}_ek.csv"
    df = pd.read_csv(f)
    df_ek = pd.read_csv(f_ek)
    corr = df["corr_at_x"]
    corr_ek = df_ek["corr_at_x_ek"]
    print_rsq_diff(corr, corr_ek)
    dfs.append(df)
    dfs_ek.append(df_ek)

df_all = pd.concat(dfs, ignore_index=True)
df_all_ek = pd.concat(dfs_ek, ignore_index=True)
df_all["heading_deg"] = df_all.apply(select_heading, axis=1)
df_all_ek["heading_deg"] = df_all_ek.apply(select_heading, axis=1)
df_all["angle_track_minus_coast_deg"] = (
    (df_all["local_angle_deg"] - df_all["heading_deg"] + 180) % 360
) - 180
df_all_ek["angle_track_minus_coast_deg"] = (
    (df_all_ek["local_angle_deg"] - df_all_ek["heading_deg"] + 180) % 360
) - 180

print("all radars: ---------------")
df_filt = df_all[df_all["dist_to_coast"] > 100_000]
df_filt_ek = df_all_ek[df_all_ek["dist_to_coast"] > 100_000]
corr = df_filt["corr_at_x"]
corr_ek = df_filt_ek["corr_at_x_ek"]
print_rsq_diff(corr, corr_ek)
