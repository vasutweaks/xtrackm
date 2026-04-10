import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = "radar_rotatedVStrack_gc_along_intersections"
RADARS = ("TN", "AP", "OR")
OUT_PNG = f"{BASE_DIR}/scatter_dist_to_coast_vs_rmse_normal_ek.png"

pattern = f"{BASE_DIR}/radar_rotatedVStrack_gc_along_summary_*.csv"
files = glob.glob(pattern)
files_normal = [f for f in files if not f.endswith("_ek.csv")]
files_ek = [f for f in files if f.endswith("_ek.csv")]

dfs1 = [pd.read_csv(f) for f in files_normal]
df_total = pd.concat(dfs1, ignore_index=True)
dfs2 = [pd.read_csv(f) for f in files_ek]
df_ek_total = pd.concat(dfs2, ignore_index=True)

# radar_str,sat,track_number,x,dist_to_coast,local_angle_deg,corr_at_x,rmse_at_x,bias_x
STAT_COLS = ("dist_to_coast", "local_angle_deg", "corr_at_x", "rmse_at_x", "bias_x")

dist_thresh = 100_000 # in m
# df_total_filt = df_total[df_total["dist_to_coast"] < dist_thresh]
# df_ek_total_filt = df_ek_total[df_ek_total["dist_to_coast"] < dist_thresh]
df_total_filt1 = df_total[df_total["local_angle_deg"] < -100]
df_ek_total_filt1 = df_ek_total[df_ek_total["local_angle_deg"] < -100]
df_total_filt2 = df_total[df_total["local_angle_deg"] > -100]
df_ek_total_filt2 = df_ek_total[df_ek_total["local_angle_deg"] > -100]


def print_stat_one(label, df, col):
    print(f"{label} (n={len(df)}) for {col}:")
    print(
        f"  {col}: min={df[col].min():.6f}, "
        f"mean={df[col].mean():.6f}, max={df[col].max():.6f}"
    )


# print_stat_one("total normal", df_total, "corr_at_x")
print_stat_one("total normal", df_total_filt1, "corr_at_x")
# print_stat_one("total normal", df_total, "corr_at_x")
print_stat_one("total normal", df_total_filt2, "corr_at_x")
# print_stat_one("total normal", df_ek_total, "corr_at_x")
# print_stat_one("total normal", df_ek_total_filt, "corr_at_x")
# print_stat_one("total normal", df_ek_total, "bias_x")
# print_stat_one("total normal", df_ek_total_filt, "corr_at_x")

# print_stat_one("total normal", df_total, "rmse_at_x")
# print_stat_one("total normal", df_total_filt, "rmse_at_x")
# print_stat_one("total normal", df_ek_total, "rmse_at_x")
# print_stat_one("total normal", df_ek_total_filt, "rmse_at_x")
