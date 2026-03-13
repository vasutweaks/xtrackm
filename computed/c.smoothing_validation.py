#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs

# codex resume 019ce593-bb33-7dd1-b30f-450e711fac85

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from namesm import sats_new
from tools_xtrackm import (
    compute_geostrophy_gc,
    cor_p,
    geostrophic_components,
    grav,
    index_at_lat,
    local_angle,
    mad_filter_1d_xr,
    slope_smooth_optimal_filter_x,
    track_dist_time_asn,
    track_smooth_box_x,
    track_smooth_gaussian_x,
    track_smooth_loess_statsmodels_x,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
GUV_DIR = ROOT_DIR / "guv"
MSLA_INTERSECTION_DIR = GUV_DIR / "msla_uv_at_intersections"
OUTPUT_ROOT = Path(__file__).resolve().parent / "smoothing_validation"
TEMP_GUV_DIR = OUTPUT_ROOT / "temp_guv"
METRICS_DIR = OUTPUT_ROOT / "metrics"
PLOTS_DIR = OUTPUT_ROOT / "plots"

DEFAULT_MONTHLY_RULE = "1ME"
DEFAULT_IOFF = 3
DEFAULT_MIN_LATITUDE = 2.0


@dataclass(frozen=True)
class SmoothingConfig:
    method: str
    params: tuple[tuple[str, object], ...]

    @property
    def params_dict(self) -> dict[str, object]:
        return dict(self.params)

    @property
    def params_json(self) -> str:
        return json.dumps(self.params_dict, sort_keys=True)

    @property
    def config_id(self) -> str:
        parts = [self.method]
        for key, value in self.params:
            value_str = str(value).replace(".", "p").replace("-", "m")
            parts.append(f"{key}{value_str}")
        return "_".join(parts)


def ensure_output_dirs() -> None:
    for path in (OUTPUT_ROOT, TEMP_GUV_DIR, METRICS_DIR, PLOTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def parse_optim_pair(value: str) -> tuple[int, int]:
    left, right = value.split(":")
    p_val = int(left)
    q_val = int(right)
    if p_val >= 0 or q_val <= 0:
        raise argparse.ArgumentTypeError(
            "Optimal filter pairs must be given as negative:positive, e.g. -7:8"
        )
    return p_val, q_val


def build_configs(args: argparse.Namespace) -> list[SmoothingConfig]:
    methods = set(args.methods)
    if "all" in methods:
        methods = {"gaussian", "box", "loess", "optim"}
    configs: list[SmoothingConfig] = []

    if "gaussian" in methods:
        for sigma in args.gaussian_sigmas:
            configs.append(SmoothingConfig("gaussian", (("sigma", float(sigma)),)))

    if "box" in methods:
        for smooth in args.box_windows:
            min_periods = args.box_min_periods
            if min_periods is None:
                min_periods = max(1, int(math.ceil(smooth / 2.0)))
            configs.append(
                SmoothingConfig(
                    "box",
                    (
                        ("smooth", int(smooth)),
                        ("min_periods", int(min_periods)),
                    ),
                )
            )

    if "loess" in methods:
        for frac in args.loess_fracs:
            configs.append(SmoothingConfig("loess", (("frac", float(frac)),)))

    if "optim" in methods:
        for p_val, q_val in args.optim_pairs:
            configs.append(
                SmoothingConfig(
                    "optim",
                    (
                        ("p", int(p_val)),
                        ("q", int(q_val)),
                    ),
                )
            )

    if not configs:
        raise ValueError("No smoothing configurations were built.")
    return configs


def load_all_intersections(satellites: list[str], limit: int | None) -> pd.DataFrame:
    frames = []
    for sat in satellites:
        path = GUV_DIR / f"tracks_intersections_{sat}_1.csv"
        if not path.exists():
            print(f"Skipping missing intersection CSV: {path}")
            continue
        frame = pd.read_csv(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No intersection CSV files were found.")
    df = pd.concat(frames, ignore_index=True)
    if limit is not None:
        df = df.head(limit).copy()
    return df


@lru_cache(maxsize=64)
def load_track_context(sat: str, track_number: str) -> tuple[xr.Dataset, xr.DataArray]:
    track_path = DATA_DIR / sat / f"ctoh.sla.ref.{sat}.nindian.{track_number.zfill(3)}.nc"
    if not track_path.exists():
        raise FileNotFoundError(f"Track file not found: {track_path}")
    with xr.open_dataset(track_path, engine="netcdf4", decode_times=False) as ds_in:
        ds = ds_in.load()
    sla_track = track_dist_time_asn(ds, var_str="sla", units_in="m")
    return ds, sla_track


def smooth_track(sla_track: xr.DataArray, method: str, params: dict[str, object]) -> tuple[xr.DataArray, str]:
    if method == "gaussian":
        return track_smooth_gaussian_x(sla_track, sigma=float(params["sigma"])), "sla"
    if method == "box":
        return (
            track_smooth_box_x(
                sla_track,
                smooth=int(params["smooth"]),
                min_periods=int(params["min_periods"]),
            ),
            "sla",
        )
    if method == "loess":
        return track_smooth_loess_statsmodels_x(sla_track, frac=float(params["frac"])), "sla"
    if method == "optim":
        slope = sla_track.differentiate("x")
        return (
            slope_smooth_optimal_filter_x(
                slope,
                p=int(params["p"]),
                q=int(params["q"]),
            ),
            "slope",
        )
    raise ValueError(f"Unsupported smoothing method: {method}")


def compute_geostrophy_gc_from_slope(ds: xr.Dataset, slope_smooth: xr.DataArray) -> xr.DataArray:
    lats_track_rev = ds.lat.values[::-1]
    if len(lats_track_rev) < 3:
        raise ValueError("Insufficient points for slope-based GC calculation")
    coriolis = cor_p(lats_track_rev).values
    coriolis = np.where(lats_track_rev > DEFAULT_MIN_LATITUDE, coriolis, np.nan)
    gravity = grav(lats_track_rev).values
    ratio = gravity / coriolis
    gc = ratio[:, np.newaxis] * slope_smooth
    gc.attrs["units"] = "m/s"
    gc.attrs["long_name"] = "Cross-track geostrophic current"
    gc.attrs["description"] = "Computed from slope-smoothed SLA"
    return gc


@lru_cache(maxsize=256)
def compute_track_gc(sat: str, track_number: str, method: str, params_json: str) -> xr.DataArray:
    ds, sla_track = load_track_context(sat, track_number)
    params = json.loads(params_json)
    smoothed, smooth_kind = smooth_track(sla_track, method, params)
    if smooth_kind == "slope":
        gc = compute_geostrophy_gc_from_slope(ds, smoothed)
    else:
        gc = compute_geostrophy_gc(ds, smoothed)
    if gc is None:
        raise ValueError(f"GC calculation returned None for {sat} track {track_number}")
    return gc


@lru_cache(maxsize=512)
def load_msla_intersection(unique_str: str) -> xr.Dataset:
    path = MSLA_INTERSECTION_DIR / f"{unique_str}_msla_uv_at_intersections.nc"
    if not path.exists():
        raise FileNotFoundError(f"MSLA intersection file not found: {path}")
    with xr.open_dataset(path) as ds_in:
        ds = ds_in.load()
    return ds


def filter_current_timeseries(da: xr.DataArray) -> xr.DataArray:
    da_clip = da.where((da <= 3.5) & (da >= -3.5))
    return mad_filter_1d_xr(da_clip, 3.5, dim="time")


def align_and_score(track_da: xr.DataArray, ref_da: xr.DataArray) -> dict[str, float]:
    aligned = xr.Dataset({"track": track_da, "ref": ref_da}).dropna(dim="time", how="any")
    n_valid = int(aligned.sizes.get("time", 0))
    if n_valid == 0:
        return {
            "corr": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "n_valid": 0,
            "valid_pct": 0.0,
        }

    bias = (aligned["ref"] - aligned["track"]).mean(dim="time").item()
    rmse = xs.rmse(aligned["track"], aligned["ref"], dim="time", skipna=True).item()
    if n_valid < 2:
        corr = np.nan
    else:
        corr = xs.pearson_r(
            aligned["track"],
            aligned["ref"],
            dim="time",
            skipna=True,
        ).item()

    total = int(track_da.sizes.get("time", 0))
    valid_pct = 100.0 * n_valid / total if total else np.nan
    return {
        "corr": corr,
        "rmse": rmse,
        "bias": bias,
        "n_valid": n_valid,
        "valid_pct": valid_pct,
    }


def save_intersection_dataset(
    ds_out: xr.Dataset,
    config: SmoothingConfig,
    unique_str: str,
    save_temp: bool,
) -> str | None:
    if not save_temp:
        return None
    config_dir = TEMP_GUV_DIR / config.config_id
    config_dir.mkdir(parents=True, exist_ok=True)
    out_path = config_dir / f"guv_at_intersection_{unique_str}_{config.config_id}.nc"
    ds_out.to_netcdf(out_path)
    return str(out_path)


def reconstruct_intersection_currents(
    row: pd.Series,
    config: SmoothingConfig,
    save_temp: bool,
) -> tuple[xr.Dataset, str | None]:
    sat = row["sat"]
    track_self = str(row["track_self"])
    track_other = str(row["track_other"])
    x_self = float(row["x_from_coast_self"])
    x_other = float(row["x_from_coast_other"])
    lat_inter = float(row["lats_inter"])

    ds_self, _ = load_track_context(sat, track_self)
    ds_other, _ = load_track_context(sat, track_other)
    gc_self = compute_track_gc(sat, track_self, config.method, config.params_json)
    gc_other = compute_track_gc(sat, track_other, config.method, config.params_json)

    ii_self = index_at_lat(ds_self, lat_inter)
    ii_other = index_at_lat(ds_other, lat_inter)
    a1 = local_angle(ds_self.lon.values, ds_self.lat.values, ii_self, ioff=DEFAULT_IOFF)
    a2 = local_angle(ds_other.lon.values, ds_other.lat.values, ii_other, ioff=DEFAULT_IOFF)

    gc_self_at = gc_self.sel(x=x_self, method="nearest", drop=True)
    gc_other_at = gc_other.sel(x=x_other, method="nearest", drop=True)
    gc_other_at = gc_other_at.interp(time=gc_self_at.time)

    gc_self_at_filt = mad_filter_1d_xr(gc_self_at, 3.5, dim="time")
    gc_other_at_filt = mad_filter_1d_xr(gc_other_at, 3.5, dim="time")

    gu, gv = geostrophic_components(gc_self_at_filt, gc_other_at_filt, a1, a2)
    gu_filtered = filter_current_timeseries(gu)
    gv_filtered = filter_current_timeseries(gv)

    ds_out = xr.Dataset(
        {
            "gu": gu_filtered,
            "gv": gv_filtered,
            "gc_self": gc_self_at_filt,
            "gc_other": gc_other_at_filt,
        }
    )
    ds_out.attrs["method"] = config.method
    ds_out.attrs["params_json"] = config.params_json

    unique_str = f"{sat}_{track_self}_{track_other}"
    out_path = save_intersection_dataset(ds_out, config, unique_str, save_temp)
    return ds_out, out_path


def compare_with_msla(
    reconstructed: xr.Dataset,
    unique_str: str,
    monthly_rule: str,
) -> dict[str, float]:
    msla_ds = load_msla_intersection(unique_str)

    gu_monthly = reconstructed["gu"].resample(time=monthly_rule).mean()
    gv_monthly = reconstructed["gv"].resample(time=monthly_rule).mean()

    gu_ngp = 100.0 * gu_monthly.count(dim="time").item() / len(gu_monthly.time)
    gv_ngp = 100.0 * gv_monthly.count(dim="time").item() / len(gv_monthly.time)

    u_ref = msla_ds["u_at"].interp(time=gu_monthly.time)
    v_ref = msla_ds["v_at"].interp(time=gv_monthly.time)

    u_limit = float(np.nanmax(np.abs(u_ref.values))) if np.isfinite(u_ref.values).any() else np.nan
    v_limit = float(np.nanmax(np.abs(v_ref.values))) if np.isfinite(v_ref.values).any() else np.nan
    if np.isfinite(u_limit):
        gu_monthly = gu_monthly.where(np.abs(gu_monthly) <= u_limit)
    if np.isfinite(v_limit):
        gv_monthly = gv_monthly.where(np.abs(gv_monthly) <= v_limit)

    u_scores = align_and_score(gu_monthly, u_ref)
    v_scores = align_and_score(gv_monthly, v_ref)

    return {
        "corr_u": u_scores["corr"],
        "rmse_u": u_scores["rmse"],
        "bias_u": u_scores["bias"],
        "n_valid_u": u_scores["n_valid"],
        "valid_pct_u": u_scores["valid_pct"],
        "ngp_u": gu_ngp,
        "corr_v": v_scores["corr"],
        "rmse_v": v_scores["rmse"],
        "bias_v": v_scores["bias"],
        "n_valid_v": v_scores["n_valid"],
        "valid_pct_v": v_scores["valid_pct"],
        "ngp_v": gv_ngp,
    }


def run_validation(args: argparse.Namespace) -> tuple[Path, Path]:
    ensure_output_dirs()
    configs = build_configs(args)
    intersections = load_all_intersections(args.sats, args.limit_intersections)
    rows: list[dict[str, object]] = []

    print(
        f"Running smoothing validation for {len(configs)} configurations "
        f"across {len(intersections)} intersections."
    )

    for config in configs:
        print(f"Configuration: {config.config_id}")
        for _, row in intersections.iterrows():
            lat_inter = float(row["lats_inter"])
            track_self = str(row["track_self"])
            track_other = str(row["track_other"])
            unique_str = f"{row['sat']}_{track_self}_{track_other}"
            base_record: dict[str, object] = {
                "config_id": config.config_id,
                "method": config.method,
                "params_json": config.params_json,
                "sat": row["sat"],
                "track_self": track_self,
                "track_other": track_other,
                "unique_str": unique_str,
                "lons_inter": float(row["lons_inter"]),
                "lats_inter": lat_inter,
                "x_from_coast_self": float(row["x_from_coast_self"]),
                "x_from_coast_other": float(row["x_from_coast_other"]),
                "angle_acute": float(row["angle_acute"]),
                "angle_obtuse": float(row["angle_obtuse"]),
                "status": "ok",
                "error": "",
                "temp_path": "",
            }

            if abs(lat_inter) < args.min_latitude:
                base_record["status"] = "skipped_low_latitude"
                rows.append(base_record)
                continue

            try:
                ds_out, temp_path = reconstruct_intersection_currents(
                    row=row,
                    config=config,
                    save_temp=args.save_temp,
                )
                metrics = compare_with_msla(
                    reconstructed=ds_out,
                    unique_str=unique_str,
                    monthly_rule=args.monthly_rule,
                )
                base_record.update(metrics)
                base_record["temp_path"] = temp_path or ""
            except Exception as exc:
                base_record["status"] = "failed"
                base_record["error"] = str(exc)
            rows.append(base_record)

    detail_df = pd.DataFrame(rows)
    detail_path = METRICS_DIR / args.detail_csv_name
    detail_df.to_csv(detail_path, index=False)

    summary_path = summarise_metrics(detail_df, METRICS_DIR / args.summary_csv_name)
    plot_path = make_bar_plot(
        summary_csv=summary_path,
        out_path=PLOTS_DIR / args.plot_name,
        ranking_metric=args.ranking_metric,
        top_n=args.top_n,
    )

    print(f"Saved detail metrics: {detail_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(f"Saved ranking plot: {plot_path}")
    return detail_path, summary_path


def summarise_metrics(detail_df: pd.DataFrame, out_path: Path) -> Path:
    if detail_df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return out_path

    requested = (
        detail_df.groupby(["config_id", "method", "params_json"], dropna=False)
        .size()
        .rename("requested_intersections")
        .reset_index()
    )
    success_df = detail_df.loc[detail_df["status"] == "ok"].copy()
    if success_df.empty:
        requested.to_csv(out_path, index=False)
        return out_path

    summary = (
        success_df.groupby(["config_id", "method", "params_json"], dropna=False)
        .agg(
            successful_intersections=("unique_str", "size"),
            corr_u_mean=("corr_u", "mean"),
            corr_u_median=("corr_u", "median"),
            corr_v_mean=("corr_v", "mean"),
            corr_v_median=("corr_v", "median"),
            rmse_u_mean=("rmse_u", "mean"),
            rmse_v_mean=("rmse_v", "mean"),
            bias_u_mean=("bias_u", "mean"),
            bias_v_mean=("bias_v", "mean"),
            abs_bias_u_mean=("bias_u", lambda x: np.nanmean(np.abs(x))),
            abs_bias_v_mean=("bias_v", lambda x: np.nanmean(np.abs(x))),
            ngp_u_mean=("ngp_u", "mean"),
            ngp_v_mean=("ngp_v", "mean"),
            n_valid_u_mean=("n_valid_u", "mean"),
            n_valid_v_mean=("n_valid_v", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(
        requested,
        on=["config_id", "method", "params_json"],
        how="left",
    )
    summary["score_corr_mean"] = summary[["corr_u_mean", "corr_v_mean"]].mean(axis=1, skipna=True)
    summary["score_rmse_mean"] = summary[["rmse_u_mean", "rmse_v_mean"]].mean(axis=1, skipna=True)
    summary["score_abs_bias_mean"] = summary[["abs_bias_u_mean", "abs_bias_v_mean"]].mean(axis=1, skipna=True)
    summary = summary.sort_values(
        by=["score_corr_mean", "score_rmse_mean"],
        ascending=[False, True],
    )
    summary.to_csv(out_path, index=False)
    return out_path


def make_bar_plot(
    summary_csv: Path,
    out_path: Path,
    ranking_metric: str,
    top_n: int,
) -> Path:
    summary = pd.read_csv(summary_csv)
    if summary.empty:
        print(f"Summary CSV is empty, skipping plot: {summary_csv}")
        return out_path

    ascending = ranking_metric in {"score_rmse_mean", "score_abs_bias_mean"}
    plot_df = summary.copy()
    plot_df = plot_df.sort_values(by=ranking_metric, ascending=ascending).head(top_n).copy()
    plot_df = plot_df.iloc[::-1]

    method_order = list(dict.fromkeys(plot_df["method"]))
    cmap = plt.get_cmap("tab10")
    colors = {method: cmap(idx % 10) for idx, method in enumerate(method_order)}
    y_pos = np.arange(len(plot_df))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(18, max(6, 0.45 * len(plot_df))),
        sharey=True,
        constrained_layout=True,
    )

    columns = [
        ("corr_u_mean", "Mean corr(u)"),
        ("corr_v_mean", "Mean corr(v)"),
        ("rmse_u_mean", "Mean RMSE(u)"),
        ("rmse_v_mean", "Mean RMSE(v)"),
    ]
    bar_colors = [colors[method] for method in plot_df["method"]]

    for axis, (column, title) in zip(axes, columns):
        axis.barh(y_pos, plot_df[column], color=bar_colors)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.3)
        if column.startswith("corr"):
            axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
        if column.startswith("rmse"):
            axis.set_xlim(left=0.0)

    labels = [f"{cfg} ({method})" for cfg, method in zip(plot_df["config_id"], plot_df["method"])]
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels)
    axes[0].set_ylabel("Smoothing configuration")
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)

    handles = [
        plt.Line2D([0], [0], color=colors[method], linewidth=6, label=method)
        for method in method_order
    ]
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)))
    fig.suptitle("Smoothing validation against MSLA currents", y=1.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate smoothing methods for intersection-derived geostrophic currents."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run smoothing sweep and metric extraction.")
    run_parser.add_argument(
        "--sats",
        nargs="+",
        default=list(sats_new),
        help="Satellite missions to include.",
    )
    run_parser.add_argument(
        "--methods",
        nargs="+",
        choices=["all", "gaussian", "box", "loess", "optim"],
        default=["all"],
        help="Smoothing methods to sweep.",
    )
    run_parser.add_argument(
        "--gaussian-sigmas",
        nargs="+",
        type=float,
        default=[2.0, 3.0, 6.0],
    )
    run_parser.add_argument(
        "--box-windows",
        nargs="+",
        type=int,
        default=[6, 12, 18],
    )
    run_parser.add_argument(
        "--box-min-periods",
        type=int,
        default=None,
        help="If omitted, uses ceil(window / 2) for each box configuration.",
    )
    run_parser.add_argument(
        "--loess-fracs",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20],
    )
    run_parser.add_argument(
        "--optim-pairs",
        nargs="+",
        type=parse_optim_pair,
        default=[(-5, 6), (-7, 8)],
        help="Pairs in p:q format, e.g. -7:8",
    )
    run_parser.add_argument("--limit-intersections", type=int, default=None)
    run_parser.add_argument("--monthly-rule", default=DEFAULT_MONTHLY_RULE)
    run_parser.add_argument("--min-latitude", type=float, default=DEFAULT_MIN_LATITUDE)
    run_parser.add_argument(
        "--no-save-temp",
        action="store_false",
        dest="save_temp",
        help="Disable saving reconstructed gu/gv/gc datasets under computed/smoothing_validation/temp_guv.",
    )
    run_parser.set_defaults(save_temp=True)
    run_parser.add_argument(
        "--detail-csv-name",
        default="smoothing_validation_metrics_detail.csv",
    )
    run_parser.add_argument(
        "--summary-csv-name",
        default="smoothing_validation_metrics_summary.csv",
    )
    run_parser.add_argument(
        "--plot-name",
        default="smoothing_validation_ranking.png",
    )
    run_parser.add_argument(
        "--ranking-metric",
        default="score_corr_mean",
        choices=[
            "score_corr_mean",
            "corr_u_mean",
            "corr_v_mean",
            "score_rmse_mean",
            "score_abs_bias_mean",
        ],
    )
    run_parser.add_argument("--top-n", type=int, default=15)

    plot_parser = subparsers.add_parser("plot", help="Regenerate the ranking plot from summary CSV.")
    plot_parser.add_argument(
        "--summary-csv",
        type=Path,
        default=METRICS_DIR / "smoothing_validation_metrics_summary.csv",
    )
    plot_parser.add_argument(
        "--out-path",
        type=Path,
        default=PLOTS_DIR / "smoothing_validation_ranking.png",
    )
    plot_parser.add_argument(
        "--ranking-metric",
        default="score_corr_mean",
        choices=[
            "score_corr_mean",
            "corr_u_mean",
            "corr_v_mean",
            "score_rmse_mean",
            "score_abs_bias_mean",
        ],
    )
    plot_parser.add_argument("--top-n", type=int, default=15)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ensure_output_dirs()

    if args.command == "run":
        run_validation(args)
        return

    if args.command == "plot":
        out_path = make_bar_plot(
            summary_csv=args.summary_csv,
            out_path=args.out_path,
            ranking_metric=args.ranking_metric,
            top_n=args.top_n,
        )
        print(f"Saved ranking plot: {out_path}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
