import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = "radar_rotatedVStrack_gc_along_intersections"
RADARS = ("TN", "AP", "OR")
OUT_PNG = f"{BASE_DIR}/scatter_dist_to_coast_vs_rmse_normal_ek.png"
META_DIR = "/home/srinivasu/xtrackm/meta"


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

fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

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
# df_total = df_total[df_total["dist_to_coast"] < dist_thresh]
# df_ek_total = df_ek_total[df_ek_total["dist_to_coast"] < dist_thresh]


def print_stats(label, df):
    print(f"{label} (n={len(df)}):")
    for col in STAT_COLS:
        print(
            f"  {col}: min={df[col].min():.6f}, "
            f"mean={df[col].mean():.6f}, max={df[col].max():.6f}"
        )


def print_stat_one(label, df, col):
    print(f"{label} (n={len(df)}) for {col}:")
    print(
        f"  {col}: min={df[col].min():.6f}, "
        f"mean={df[col].mean():.6f}, max={df[col].max():.6f}"
    )


print_stat_one("total normal", df_total, "dist_to_coast")
print_stat_one("total ek", df_ek_total, "dist_to_coast")



# for r in RADARS:
#     f = f"{BASE_DIR}/radar_rotatedVStrack_gc_along_summary_{r}.csv"
#     f_ek = f"{BASE_DIR}/radar_rotatedVStrack_gc_along_summary_{r}_ek.csv"
#     df = pd.read_csv(f)
#     df_ek = pd.read_csv(f_ek)
#     df = df[df["dist_to_coast"] < dist_thresh]
#     df_ek = df_ek[df_ek["dist_to_coast"] < dist_thresh]
#     mean_corr = df["corr_at_x"].mean()
#     mean_corr_ek = df_ek["corr_at_x"].mean()
#     improvement = mean_corr_ek- mean_corr 
#     # print(f"{r}: {mean_corr}, {mean_corr_ek} {improvement}")
#     mean_rmse = df["rmse_at_x"].mean()
#     mean_rmse_ek = df_ek["rmse_at_x"].mean()
#     improvement = mean_rmse_ek- mean_rmse 
#     # print(f"{r}: {mean_rmse}, {mean_rmse_ek} {improvement}")
#     print_stats(f"{r} normal", df)
#     print_stats(f"{r} ek", df_ek)
