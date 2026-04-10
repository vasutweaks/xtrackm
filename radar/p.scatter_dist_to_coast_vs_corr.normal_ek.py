import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = "radar_rotatedVStrack_gc_along_intersections"
RADARS = ("TN", "AP", "OR")
OUT_PNG = f"{BASE_DIR}/scatter_dist_to_coast_vs_corr_normal_ek.png"


def gather_csvs(radar_str, ek=False):
    if ek:
        pattern = f"{BASE_DIR}/radar_rotatedVStrack_gc_along_summary_{radar_str}*_ek.csv"
    else:
        pattern = f"{BASE_DIR}/radar_rotatedVStrack_gc_along_summary_{radar_str}*.csv"
    files = sorted(glob.glob(pattern))
    if ek:
        return files
    # exclude ek files for normal branch
    return [f for f in files if not f.endswith("_ek.csv")]


def concat_for_group(files):
    if not files:
        return pd.DataFrame(
            columns=[
                "radar_str",
                "sat",
                "track_number",
                "x",
                "dist_to_coast",
                "corr_at_x",
            ]
        )
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def main():
    fig, axs = plt.subplots(len(RADARS), 2, figsize=(12, 12), sharex=True, sharey=True)

    for i, radar_str in enumerate(RADARS):
        files_normal = gather_csvs(radar_str, ek=False)
        files_ek = gather_csvs(radar_str, ek=True)

        df_normal = concat_for_group(files_normal)
        df_ek = concat_for_group(files_ek)

        axn = axs[i, 0]
        axe = axs[i, 1]

        if not df_normal.empty:
            axn.scatter(
                df_normal["dist_to_coast"].values / 1000.0,
                df_normal["corr_at_x"].values,
                s=8,
                alpha=0.5,
                color="tab:blue",
            )
        else:
            axn.text(0.5, 0.5, "No data", ha="center", va="center", transform=axn.transAxes)

        if not df_ek.empty:
            axe.scatter(
                df_ek["dist_to_coast"].values / 1000.0,
                df_ek["corr_at_x"].values,
                s=8,
                alpha=0.5,
                color="tab:green",
            )
        else:
            axe.text(0.5, 0.5, "No data", ha="center", va="center", transform=axe.transAxes)

        axn.set_title(f"{radar_str} Normal")
        axe.set_title(f"{radar_str} EK")
        axn.set_ylabel("corr_at_x")
        axn.grid(alpha=0.3, linestyle="--")
        axe.grid(alpha=0.3, linestyle="--")
        axn.set_ylim(-1.0, 1.0)
        axe.set_ylim(-1.0, 1.0)

    axs[-1, 0].set_xlabel("dist_to_coast (km)")
    axs[-1, 1].set_xlabel("dist_to_coast (km)")

    fig.tight_layout()
    plt.show()
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
