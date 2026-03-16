#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# codex resume 019ce71b-c364-7f20-9c6b-6e2a7108c17c

# -------------------- USER CONFIG --------------------
SAT = "TP+J1+J2+J3+S6A"

# Explicit (method, para) pairs to compare.
# Example: [("gauss", "3"), ("loess", "0.05"), ("odo", "56")]
METHOD_PARA_PAIRS: list[tuple[str, str]] = [
    ("loess", "0.05"),
    ("xsmooth", "0.05"),
]

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


def csv_path_from_method_para(sat: str, method: str, para: str) -> Path:
    return Path(f"metrics_msla_uvVSguv_{sat}_{method}_{para}.csv")


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


def companion_msla_std_metric(metric: str) -> str | None:
    if metric == "rmse_u":
        return "std_msla_u"
    if metric == "rmse_v":
        return "std_msla_v"
    return None


def plot_line_common_only(
    aligned_df: pd.DataFrame,
    key_col: str,
    labels: list[str],
    metric: str,
    xlabel_mode: str,
    figsize: tuple[float, float],
    output_png: str,
    companion_df: pd.DataFrame | None = None,
    companion_metric: str | None = None,
) -> None:
    common_df = aligned_df.dropna(subset=labels).reset_index(drop=True)
    if companion_df is not None and companion_metric is not None:
        companion_common = companion_df[[key_col] + labels]
        common_df = common_df.merge(companion_common, on=key_col, suffixes=("", "_comp"))
        all_required_cols = labels + [f"{label}_comp" for label in labels]
        common_df = common_df.dropna(subset=all_required_cols).reset_index(drop=True)

    if common_df.empty:
        raise ValueError("No rows where all metric values exist across all files.")

    x = np.arange(len(common_df))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if companion_df is not None and companion_metric is not None:
        show_companion_legend = True
        for label in labels:
            ax.plot(
                x,
                common_df[f"{label}_comp"].to_numpy(),
                color="gray",
                alpha=0.35,
                linewidth=4.0,
                zorder=1,
                label=companion_metric if show_companion_legend else None,
            )
            show_companion_legend = False

    for label in labels:
        ax.plot(x, common_df[label].to_numpy(), linewidth=1.5, label=label, zorder=2)

    ax.set_title(f"{metric} line comparison (common rows only)")
    ylabel = metric if companion_metric is None else f"{metric} (+ {companion_metric})"
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()

    ax.set_xlabel("Intersection index" if xlabel_mode == "index" else key_col)
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
    if not METHOD_PARA_PAIRS:
        raise ValueError("METHOD_PARA_PAIRS is empty. Add at least one (method, para) pair.")

    csv_paths = []
    labels = []
    for method, para in METHOD_PARA_PAIRS:
        path = csv_path_from_method_para(SAT, str(method), str(para))
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for ({method}, {para}): {path}")
        csv_paths.append(path)
        labels.append(f"{method}:{para}")

    if len(labels) != len(csv_paths):
        raise ValueError("LABELS length must match number of CSV files.")

    aligned_df = build_aligned_frame(
        csv_paths=csv_paths,
        labels=labels,
        key_col=KEY_COL,
        metric=METRIC,
        join_mode=JOIN_MODE,
    )

    companion_metric = companion_msla_std_metric(METRIC)
    companion_df = None
    if companion_metric is not None:
        companion_df = build_aligned_frame(
            csv_paths=csv_paths,
            labels=labels,
            key_col=KEY_COL,
            metric=companion_metric,
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
        companion_df=companion_df,
        companion_metric=companion_metric,
    )
    globals().update(locals())
    sys.exit()


if __name__ == "__main__":
    main()
