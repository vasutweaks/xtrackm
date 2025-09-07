import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

only_df = pd.read_csv(
    "oscar_onlyVStrack_corrs_rmses_biass_TP+J1+J2+J3+S6A_u.csv")
ekmn_df = pd.read_csv(
    "oscar_ekmnVStrack_corrs_rmses_biass_TP+J1+J2+J3+S6A_u.csv")

# Create unique column combining zero-padded sat and track_self, track_other
only_df["unique_id"] = (only_df["sat"].apply(lambda s: s.zfill(20)) + "_" +
                        only_df["track_self"].astype(str) + "_" +
                        only_df["track_other"].astype(str))
only_df = only_df.set_index("unique_id")

ekmn_df["unique_id"] = (ekmn_df["sat"].apply(lambda s: s.zfill(20)) + "_" +
                        ekmn_df["track_self"].astype(str) + "_" +
                        ekmn_df["track_other"].astype(str))
ekmn_df = ekmn_df.set_index("unique_id")

common_ids = only_df.index.intersection(ekmn_df.index)
only_df = only_df.loc[common_ids]
ekmn_df = ekmn_df.loc[common_ids]

# Extract metrics for plotting
metrics = ["biass", "corrs", "rmses"]
titles = ["Bias", "Correlation", "Rmse"]

# Prepare figure with 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("comparison metrics with oscar u: track gu only VS track gu + ekman", fontsize=14)

summary_lines = []

for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]
    valid_mask = only_df[metric].notna() & ekmn_df[metric].notna()
    only_df1 = only_df[valid_mask]
    ekmn_df1 = ekmn_df[valid_mask]
    # Get data
    only_df_data = only_df1[metric].values
    ekmn_df_data = ekmn_df1[metric].values
    indices = np.arange(len(only_df_data))
    ax.plot(indices, only_df_data, label="track gu only")
    ax.plot(indices, ekmn_df_data, label="track gu+ekman")
    # draw zero line
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Data Point Index")
    ax.set_ylabel(titles[i])
    ax.set_title(titles[i])
    ax.legend()
    # collect summary text for this metric
    if len(only_df_data) > 0:
        mean_o = np.nanmean(only_df_data)
        mean_e = np.nanmean(ekmn_df_data)
        diff = mean_e - mean_o
        cnt = len(only_df_data)
        summary_lines.append(f"\n{titles[i]}:")
        summary_lines.append(f" count = {cnt}")
        summary_lines.append(f" mean track u = {mean_o:.4f}")
        summary_lines.append(f" mean track u + ekman = {mean_e:.4f}")
        summary_lines.append(f" mean improvement = {diff:.4f}")
    else:
        summary_lines.append(f"\n{titles[i]}: no valid paired data")

ax_sum = axs[1, 1]
ax_sum.axis("off")
text = " ".join(summary_lines)

ax_sum.annotate(
    "\n".join(summary_lines),
    xy=(0, 1),
    xycoords="axes fraction",
    va="top",
    ha="left",
    fontsize=10,
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    annotation_clip=False,
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"track_guvVSoscar_ww_ekman_u.png")
plt.show()
