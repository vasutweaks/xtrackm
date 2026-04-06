import sys

import matplotlib.pyplot as plt
import pandas as pd
from namesm import sats_new

fig, (ax, ax_bar) = plt.subplots(
    2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}
)
df_all = pd.read_csv("spatio_temporal_valid_perc.csv")
cmap = plt.get_cmap("tab10")
bar_labels = []
bar_values = []
bar_colors = []

for i, sat_name in enumerate(sats_new):
    df = df_all[df_all["sat"] == sat_name].sort_values("perc_valid_xt")
    # df = df_all[df_all["sat"] == sat_name].sort_values("lon_equat")
    if df.empty:
        continue

    x = range(len(df))
    y = df["perc_valid_xt"]
    color = cmap(i % cmap.N)
    mean_perc = y.mean()

    ax.plot(x, y, label=sat_name, color=color, linewidth=2.8)
    bar_labels.append(sat_name)
    bar_values.append(mean_perc)
    bar_colors.append(color)

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

bar_data = sorted(zip(bar_values, bar_labels, bar_colors), key=lambda t: t[0])
sorted_values = [v for v, _, _ in bar_data]
sorted_labels = [l for _, l, _ in bar_data]
sorted_colors = [c for _, _, c in bar_data]

ax_bar.bar(sorted_labels, sorted_values, color=sorted_colors)
ax_bar.set_ylabel("mean % valid")
ax_bar.set_xlabel("satellite")
ax_bar.set_title("Mean perc_valid_xt across all tracks")
ax_bar.tick_params(axis="x", rotation=45)

plt.tight_layout()
out = f"spatio_temporal_valid_perc_{title_sat}.png"
plt.savefig(out, dpi=300)
print(f"Saved {out}")
plt.show()
