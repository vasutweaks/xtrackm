from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# codex resume 019ce71b-c364-7f20-9c6b-6e2a7108c17c

SAT = "TP+J1+J2+J3+S6A"
METHOD = "loess"
PARA = "0.05"
GUV_ROOT = Path("/home/srinivasu/xtrackm/guv")
MSLA_DIR = GUV_ROOT / "msla_uv_at_intersections"
GUV_DIR = GUV_ROOT / f"guv_at_intersections_{METHOD}"
MONTHLY_RULE = "1ME"


def compute_metrics(track_da: xr.DataArray, ref_da: xr.DataArray) -> dict[str, float]:
    aligned = xr.Dataset({"track": track_da, "ref": ref_da}).dropna(dim="time", how="any")
    n_good = int(aligned.sizes.get("time", 0))
    if n_good == 0:
        return {
            "corr": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "n_good_points": 0,
            "ref_std": np.nan,
        }

    track = aligned["track"].values
    ref = aligned["ref"].values
    diff = ref - track
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    ref_std = float(np.std(ref))
    corr = float(np.corrcoef(track, ref)[0, 1]) if n_good >= 2 else np.nan
    return {
        "corr": corr,
        "rmse": rmse,
        "bias": bias,
        "n_good_points": n_good,
        "ref_std": ref_std,
    }


def plot_metrics_bars_2x2(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("corr_u", "Correlation (u)"),
        ("rmse_u", "RMSE (u)"),
        ("bias_u", "Bias (u)"),
        ("n_good_points_u", "Matching points (u)"),
    ]
    x = np.arange(len(metrics_df))
    # x_labels = (x + 1).astype(str)
    x_labels = metrics_df["lats_inter"].apply(lambda x: f"{x:.2f}").to_numpy()
    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(24, 12), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (column, title) in zip(axes_flat, metrics):
        values = metrics_df[column].to_numpy()
        if column == "rmse_u":
            ref_std = metrics_df["std_msla_u"].to_numpy()
            ax.bar(x - width / 2, values, width=width, label="RMSE (u)")
            ax.bar(x + width / 2, ref_std, width=width, label="std(msla_u)")
        else:
            ax.bar(x, values, width=0.7, label=column)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_xlabel("Intersection index")
        ax.grid(axis="y", alpha=0.3)
        if column == "corr_u":
            ax.set_ylim(.0, 1.0)
        ax.legend()

    fig.savefig(output_path, dpi=200)
    plt.show()
    plt.close(fig)


def main() -> None:
    inter_csv = GUV_ROOT / f"tracks_intersections_{SAT}_1.csv"
    inter_df = pd.read_csv(inter_csv)
    # sort inter_df by lats_inter
    # inter_df = inter_df.sort_values(by="lats_inter")
    rows: list[dict[str, float | str]] = []

    for _, row in inter_df.iterrows():
        sat = row["sat"]
        track_self = str(row["track_self"])
        track_other = str(row["track_other"])
        lats_inter = float(row["lats_inter"])
        unique_str = f"{sat}_{track_self}_{track_other}"
        print(f"Processing {unique_str}")

        msla_path = MSLA_DIR / f"{unique_str}_msla_uv_at_intersections.nc"
        guv_path = GUV_DIR / f"guv_at_intersection_{sat}_{track_self}_{track_other}_{METHOD}_{PARA}.nc"

        if not msla_path.exists() or not guv_path.exists():
            print(f"Skipping {unique_str}: missing file(s)")
            continue

        with xr.open_dataset(msla_path) as ds_msla, xr.open_dataset(guv_path) as ds_guv:
            gu_monthly = ds_guv["gu"].resample(time=MONTHLY_RULE).mean()
            gv_monthly = ds_guv["gv"].resample(time=MONTHLY_RULE).mean()

            u_ref = ds_msla["u_at"].interp(time=gu_monthly.time)
            v_ref = ds_msla["v_at"].interp(time=gv_monthly.time)

            u_metrics = compute_metrics(gu_monthly, u_ref)
            v_metrics = compute_metrics(gv_monthly, v_ref)

        rows.append(
            {
                "unique_str": unique_str,
                "corr_u": u_metrics["corr"],
                "rmse_u": u_metrics["rmse"],
                "bias_u": u_metrics["bias"],
                "n_good_points_u": u_metrics["n_good_points"],
                "std_msla_u": u_metrics["ref_std"],
                "corr_v": v_metrics["corr"],
                "rmse_v": v_metrics["rmse"],
                "bias_v": v_metrics["bias"],
                "n_good_points_v": v_metrics["n_good_points"],
                "std_msla_v": v_metrics["ref_std"],
                "lats_inter": lats_inter,
            }
        )

    # metrics_df = pd.DataFrame(rows).sort_values("unique_str").reset_index(drop=True)
    metrics_df = pd.DataFrame(rows).sort_values("lats_inter", ascending=False).reset_index(drop=True)
    out_csv = Path(__file__).resolve().with_name(f"metrics_msla_uvVSguv_{SAT}_{METHOD}_{PARA}.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"Saved metrics CSV: {out_csv}")

    out_plot = Path(__file__).resolve().with_name(f"metrics_msla_uvVSguv_{SAT}_{METHOD}_{PARA}_2x2.png")
    plot_metrics_bars_2x2(metrics_df, out_plot)
    print(f"Saved metrics plot: {out_plot}")


if __name__ == "__main__":
    main()
