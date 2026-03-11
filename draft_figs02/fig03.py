import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# codex resume 019cd8ab-eee9-72e0-9952-b9e4a8bf1b11

RAMA_NO_EK_CSV = "rama_guv_c_nearest_global_matches.csv"
RAMA_EK_CSV = "rama_guv_c_nearest_global_matches.ek.csv"
OMNI_NO_EK_CSV = "omni_guv_c_nearest_global_matches.csv"
OMNI_EK_CSV = "omni_guv_c_nearest_global_matches.ek.csv"

OUT_PNG = "scatter_omniVSrama_uVSgu_ww_ek.png"
# get file name

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")

MIN_DISTANCE_KM = None
MAX_DISTANCE_KM = 250.0


def load_xy(csv_path, y_col, with_ekman=False):
    df = pd.read_csv(csv_path)

    dist = df["distance_km"].to_numpy(dtype=float)
    keep = np.isfinite(dist)
    if MIN_DISTANCE_KM is not None:
        keep &= dist >= float(MIN_DISTANCE_KM)
    if MAX_DISTANCE_KM is not None:
        keep &= dist <= float(MAX_DISTANCE_KM)
    df = df.loc[keep].copy()

    x = df["gu"].to_numpy(dtype=float)
    if with_ekman:
        x = x + df["u_ek"].to_numpy(dtype=float)

    y = df[y_col].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    return x, y, len(df)


def add_panel(ax, x, y, title, ylabel, xlabel, extent=None):
    if x.size < 3:
        ax.text(
            0.5,
            0.98,
            f"{title} (insufficient points)",
            transform=ax.transAxes,
            va="top",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.5, edgecolor="white"),
            color="white",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        return None

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r = np.corrcoef(x, y)[0, 1]
    stderr = np.sqrt(np.sum((y - yhat) ** 2) / (x.size - 2))

    hb = ax.hexbin(x, y, gridsize=55, cmap="viridis", mincnt=1, extent=extent)

    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ax.plot(xs, slope * xs + intercept, color="crimson", linewidth=2.0, label="Linear fit")
    ax.plot(xs, xs, color="white", linestyle="--", linewidth=1.2, label="1:1 line")

    text = (
        f"N = {x.size}\n"
        f"r = {r:.3f}\n"
        f"Std error = {stderr:.3f} m/s\n"
        f"y = {slope:.3f}x + {intercept:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.55, edgecolor="white"),
        color="white",
    )

    ax.text(
        0.6,
        0.98,
        title,
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.5, edgecolor="white"),
        color="black",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    return hb


def main():
    min_lbl = MIN_DISTANCE_KM if MIN_DISTANCE_KM is not None else "-inf"
    max_lbl = MAX_DISTANCE_KM if MAX_DISTANCE_KM is not None else "+inf"
    x_label_no_ek = "Altimetry gu (m/s)"
    x_label_ek = "Altimetry gu + u_ek (m/s)"
    none = ""

    x_rama_no, y_rama_no, n_rama_no = load_xy(RAMA_NO_EK_CSV, "rama_u", with_ekman=False)
    x_rama_ek, y_rama_ek, n_rama_ek = load_xy(RAMA_EK_CSV, "rama_u", with_ekman=True)
    x_omni_no, y_omni_no, n_omni_no = load_xy(OMNI_NO_EK_CSV, "omni_u", with_ekman=False)
    x_omni_ek, y_omni_ek, n_omni_ek = load_xy(OMNI_EK_CSV, "omni_u", with_ekman=True)
    all_x = np.concatenate([x_rama_no, x_rama_ek, x_omni_no, x_omni_ek])
    all_y = np.concatenate([y_rama_no, y_rama_ek, y_omni_no, y_omni_ek])
    global_extent = (
        float(np.nanmin(all_x)),
        float(np.nanmax(all_x)),
        float(np.nanmin(all_y)),
        float(np.nanmax(all_y)),
    )

    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)

    hb00 = add_panel(
        axs[0, 0],
        x_rama_no,
        y_rama_no,
        f"RAMA u vs gu (no Ekman) | distance_km in [{min_lbl}, {max_lbl}]",
        "RAMA u (m/s)",
        none,
        extent=global_extent,
    )
    hb01 = add_panel(
        axs[0, 1],
        x_rama_ek,
        y_rama_ek,
        f"RAMA u vs gu+u_ek (with Ekman) | distance_km in [{min_lbl}, {max_lbl}]",
        none,
        none,
        extent=global_extent,
    )
    hb10 = add_panel(
        axs[1, 0],
        x_omni_no,
        y_omni_no,
        f"OMNI u vs gu (no Ekman) | distance_km in [{min_lbl}, {max_lbl}]",
        "OMNI u (m/s)",
        x_label_no_ek,
        extent=global_extent,
    )
    hb11 = add_panel(
        axs[1, 1],
        x_omni_ek,
        y_omni_ek,
        f"OMNI u vs gu+u_ek (with Ekman) | distance_km in [{min_lbl}, {max_lbl}]",
        none,
        x_label_ek,
        extent=global_extent,
    )

    top_hbs = [hb for hb in [hb00, hb01] if hb is not None]
    if top_hbs:
        top_vmax = max(float(np.nanmax(hb.get_array())) for hb in top_hbs)
        for hb in top_hbs:
            hb.set_clim(1.0, top_vmax)
        cbar_top = fig.colorbar(top_hbs[0], ax=axs[0, :], location="right", fraction=0.03, pad=0.02)
        cbar_top.set_label("Count")

    bottom_hbs = [hb for hb in [hb10, hb11] if hb is not None]
    if bottom_hbs:
        bottom_vmax = max(float(np.nanmax(hb.get_array())) for hb in bottom_hbs)
        for hb in bottom_hbs:
            hb.set_clim(1.0, bottom_vmax)
        cbar_bottom = fig.colorbar(bottom_hbs[0], ax=axs[1, :], location="right", fraction=0.03, pad=0.02)
        cbar_bottom.set_label("Count")

    fig.savefig(OUT_PNG, dpi=180)
    # plt.show()
    plt.close(fig)

    print(f"Rows after distance filter | RAMA no-ek: {n_rama_no}, RAMA ek: {n_rama_ek}")
    print(f"Rows after distance filter | OMNI no-ek: {n_omni_no}, OMNI ek: {n_omni_ek}")
    print(f"Saved {OUT_PNG}")
    # link OUT_PNG to png_file
    if os.path.exists(png_file):
        os.remove(png_file)
    png_file.symlink_to(OUT_PNG)


if __name__ == "__main__":
    main()
