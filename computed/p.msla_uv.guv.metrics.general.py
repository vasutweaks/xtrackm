#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import sys

# codex resume 019ce71b-c364-7f20-9c6b-6e2a7108c17c

# -------------------- USER CONFIG --------------------
# If CSV_FILES is empty, files are discovered using CSV_GLOB.
SAT = "TP+J1+J2+J3+S6A"
CSV_GLOB = f"metrics_msla_uvVSguv_{SAT}_*.csv"

# Column to compare across files.
METRIC = "corr_u"

# Column used to align rows across CSVs.
KEY_COL = "unique_str"

# "inner" keeps only matching keys in all files; "outer" keeps union.
JOIN_MODE = "inner"

# Use "index" to show 1..N on x-axis, or "key" to show KEY_COL values.
XLABEL_MODE = "index"
PLOT_MODE = "line"

FIGSIZE = (18, 6)
OUTPUT_PNG = f"compare_{METRIC}.png"
DRAW_GROUP_FRAMES = True
# -----------------------------------------------------


def label_from_filename(path: Path) -> str:
    # Expected pattern (example):
    # metrics_msla_uvVSguv_TP+J1+J2+J3+S6A_gauss_3.csv
    tokens = path.stem.split("_")
    if len(tokens) >= 2:
        method = tokens[-2]
        param = tokens[-1]
        return f"{method}:{param}"
    return path.stem


def load_metric_frame(path: Path, key_col: str, metric: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if key_col not in df.columns:
        raise KeyError(f"{path}: missing key column '{key_col}'")
    if metric not in df.columns:
        raise KeyError(f"{path}: missing metric column '{metric}'")
    return df[[key_col, metric]].rename(columns={metric: label})


def build_aligned_frame(
    csv_paths: list[Path],
    labels: list[str],
    key_col: str,
    metric: str,
    join_mode: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for path, label in zip(csv_paths, labels):
        frame = load_metric_frame(path, key_col, metric, label)
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=key_col, how=join_mode)
    if merged is None:
        raise RuntimeError("No data loaded.")
    return merged.sort_values(key_col).reset_index(drop=True)


def plot_line_common_only(
    aligned_df: pd.DataFrame,
    key_col: str,
    labels: list[str],
    metric: str,
    xlabel_mode: str,
    figsize: tuple[float, float],
    output_png: str,
) -> None:
    common_df = aligned_df.dropna(subset=labels).reset_index(drop=True)
    if common_df.empty:
        raise ValueError("No rows where all metric values exist across all files.")

    x = np.arange(len(common_df))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for label in labels:
        ax.plot(x, common_df[label].to_numpy(), marker="o", linewidth=1.5, label=label)

    ax.set_title(f"{metric} line comparison (common rows only)")
    ax.set_ylabel(metric)
    ax.set_xlabel("Intersection index" if xlabel_mode == "index" else key_col)
    ax.grid(alpha=0.3)
    ax.legend()

    ax.set_xticks(x)
    if xlabel_mode == "index":
        ax.set_xticklabels((x + 1).astype(str))
    else:
        ax.set_xticklabels(common_df[key_col].astype(str), rotation=90)

    out_path = Path(output_png)
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)
    print(f"Saved line plot: {out_path.resolve()}")
    print(f"Rows plotted (all files valid): {len(common_df)}")


def main() -> None:
    csv_paths = sorted(Path(".").glob(CSV_GLOB))
    labels = [label_from_filename(p) for p in csv_paths]
    if len(labels) != len(csv_paths):
        raise ValueError("LABELS length must match number of CSV files.")

    aligned_df = build_aligned_frame(
        csv_paths=csv_paths,
        labels=labels,
        key_col=KEY_COL,
        metric=METRIC,
        join_mode=JOIN_MODE,
    )
    print(f"Using files: {[str(p) for p in csv_paths]}")
    line_output = str(Path(OUTPUT_PNG).with_name(f"{Path(OUTPUT_PNG).stem}_line.png"))
    plot_line_common_only(
        aligned_df=aligned_df,
        key_col=KEY_COL,
        labels=labels,
        metric=METRIC,
        xlabel_mode=XLABEL_MODE,
        figsize=FIGSIZE,
        output_png=line_output,
    )
    globals().update(locals())
    sys.exit()


if __name__ == "__main__":
    main()
