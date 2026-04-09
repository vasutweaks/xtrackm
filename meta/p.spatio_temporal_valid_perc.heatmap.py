import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from tools_xtrackm import sats_new

CSV_FILE = "spatio_temporal_valid_perc.csv"
OUT_FILE = "spatio_temporal_valid_perc_heatmap_all.png"
SORT_COL = "lon_equat"  # use "track_number" if preferred


def make_panel_data(df_sat: pd.DataFrame) -> np.ndarray:
    x_gap = 100.0 - df_sat["perc_valid_x"].to_numpy(dtype=float)
    time_gap = 100.0 - df_sat["perc_valid_time"].to_numpy(dtype=float)
    return np.vstack([x_gap, time_gap])


def main() -> None:
    df_all = pd.read_csv(CSV_FILE)

    sats_present = [sat for sat in sats_new if sat in set(df_all["sat"])]
    if not sats_present:
        raise SystemExit("No satellites found in CSV.")

    n_sats = len(sats_present)
    ncols = min(3, n_sats)
    nrows = math.ceil(n_sats / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 2.1 * nrows + 0.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    im = None
    for i, sat in enumerate(sats_present):
        ax = axes[i]
        df_sat = df_all[df_all["sat"] == sat].copy()
        if SORT_COL in df_sat.columns:
            df_sat = df_sat.sort_values(SORT_COL)
        else:
            df_sat = df_sat.sort_values("track_number")

        panel = make_panel_data(df_sat)

        im = ax.imshow(
            panel,
            aspect="auto",
            interpolation="nearest",
            cmap="magma_r",
            vmin=0,
            vmax=100,
        )

        ax.set_title(f"{sat} (n={panel.shape[1]} tracks)")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["x_gap", "time_gap"])

        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.set_xlabel("track index")

    for j in range(n_sats, len(axes)):
        axes[j].axis("off")

    cbar = fig.colorbar(im, ax=axes[:n_sats], shrink=0.95, pad=0.02)
    cbar.set_label("gap (%) = 100 - valid (%)")

    fig.suptitle("Satellite-wise spatio-temporal gaps", fontsize=14)
    fig.savefig(OUT_FILE, dpi=300)
    print(f"Saved {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
