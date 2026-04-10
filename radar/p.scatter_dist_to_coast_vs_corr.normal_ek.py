import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = "radar_rotatedVStrack_gc_along_intersections"
RADARS = ("TN", "AP", "OR")
OUT_PNG = f"{BASE_DIR}/scatter_dist_to_coast_vs_rmse_normal_ek.png"

fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

pattern = f"{BASE_DIR}/radar_rotatedVStrack_gc_along_summary_*.csv"
files = glob.glob(pattern)
files_normal = [f for f in files if not f.endswith("_ek.csv")]
files_ek = [f for f in files if f.endswith("_ek.csv")]

dfs = [pd.read_csv(f) for f in files_normal]
df_normal = pd.concat(dfs, ignore_index=True)
dfs = [pd.read_csv(f) for f in files_ek]
df_ek = pd.concat(dfs, ignore_index=True)

axn = axs[0]
axn.scatter(
    df_normal["dist_to_coast"].values / 1000.0,
    df_normal["rmse_at_x"].values,
    s=8,
    alpha=0.5,
    color="tab:blue",
)
axn.set_ylabel("rmse_at_x")
axn.grid(alpha=0.3, linestyle="--")
axn.set_ylim(-1.0, 1.0)

axe = axs[1]
axn.scatter(
    df_ek["dist_to_coast"].values / 1000.0,
    df_ek["rmse_at_x"].values,
    s=8,
    alpha=0.5,
    color="tab:red",
)

axn.set_ylabel("rmse_at_x")
axe.grid(alpha=0.3, linestyle="--")
axe.set_ylim(-1.0, 1.0)

fig.tight_layout()
plt.show()
