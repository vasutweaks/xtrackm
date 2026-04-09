import sys

import matplotlib.pyplot as plt
import pandas as pd
from namesm import sats_new

fig, ax = plt.subplots(figsize=(12, 5))
df_all = pd.read_csv("spatio_temporal_valid_perc.csv")
cmap = plt.get_cmap("tab10")

for i, sat_name in enumerate(sats_new):
    df = df_all[df_all["sat"] == sat_name].sort_values("perc_valid_xt")
    # df = df_all[df_all["sat"] == sat_name].sort_values("lon_equat")
    if df.empty:
        continue

    x = range(len(df))
    y = df["perc_valid_xt"]
    color = cmap(i % cmap.N)

    ax.plot(x, y, label=sat_name, color=color, linewidth=2.8)

    middle_idx = len(df) // 2
    if sat_name == "S3A":
        middle_idx = len(df) // 3
    ax.text(
        middle_idx,
        y.iloc[middle_idx],
        sat_name,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="bottom",
        color=color,
    )

ax.set_xlabel("track number")
ax.set_ylabel("min % valid")

title_sat = "all"

ax.set_title(f"{title_sat} — min % valid points along x and time")
ax.legend()
plt.tight_layout()
out = f"spatio_temporal_valid_perc_{title_sat}.png"
plt.savefig(out, dpi=300)
print(f"Saved {out}")
plt.show()
