import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

SAT = "TPN+J1N+J2N+J3N"
SAT = "TP+J1+J2+J3+S6A"
CSV_NO_EK = f"drifter_ve_vs_gu_{SAT}.csv"
CSV_W_EK = f"drifter_ve_vs_gu_ek_{SAT}.csv"
OUT_PNG = f"scatter_drifter_ve_vs_gu_ww_ek_{SAT}.png"
COAST_DIST_PATH = "~/allData/topo/GMT_intermediate_coast_distance_01d_track_reg.nc"
PROXIMITY_DIST_M = 100_000.0
MAX_ABS_TIME_DIFF_DAYS = None


def print_r2_ekman_gain(csv_path, obs_col, label):
    # df = filter_by_distance(pd.read_csv(csv_path))
    df = pd.read_csv(csv_path)
    gu = df["gu"].to_numpy(dtype=float)
    u_ek = df["u_ek"].to_numpy(dtype=float)
    obs = df[obs_col].to_numpy(dtype=float)

    valid = np.isfinite(gu) & np.isfinite(u_ek) & np.isfinite(obs)
    gu = gu[valid]
    u_ek = u_ek[valid]
    obs = obs[valid]

    if gu.size < 3:
        print(f"{label} R² gain: insufficient points")
        return

    r2_gu = np.corrcoef(gu, obs)[0, 1] ** 2
    r2_gu_ek = np.corrcoef(gu + u_ek, obs)[0, 1] ** 2
    delta_pct = (r2_gu_ek - r2_gu) * 100.0

    print(f"\n{label} R² gain from adding u_ek (N={gu.size})")
    print(f"  R²(gu only)      = {r2_gu:.4f}  ({r2_gu * 100:.2f}%)")
    print(f"  R²(gu + u_ek)    = {r2_gu_ek:.4f}  ({r2_gu_ek * 100:.2f}%)")
    print(f"  Variance explained increase = {delta_pct:+.2f} percentage points")


def get_coast_distance_km(lons, lats):
    with xr.open_dataset(os.path.expanduser(COAST_DIST_PATH)) as ds_coast:
        lon_da = xr.DataArray(np.asarray(lons, dtype=float), dims="point")
        lat_da = xr.DataArray(np.asarray(lats, dtype=float), dims="point")
        coast_km = ds_coast["coast_distance"].sel(lon=lon_da, lat=lat_da, method="nearest")
    return coast_km.values.astype(float)


def compute_time_diff_days(df):
    drifter_t = pd.to_datetime(df["drifter_time"], errors="coerce")
    gu_t = pd.to_datetime(df["gu_time"], errors="coerce")
    return (drifter_t - gu_t).dt.total_seconds() / 86400.0


def summarize_time_diff_days(dt_days, label):
    dt_days = dt_days[np.isfinite(dt_days.to_numpy(dtype=float))]
    if dt_days.size == 0:
        print(f"[{label}] No valid drifter_time/gu_time pairs for temporal-difference stats.")
        return
    dt_abs = np.abs(dt_days)
    mode_signed = pd.Series(np.round(dt_days, 6)).mode().to_numpy()
    mode_abs = pd.Series(np.round(dt_abs, 6)).mode().to_numpy()
    mode_signed_str = ", ".join([f"{v:.6f}" for v in mode_signed[:5]])
    mode_abs_str = ", ".join([f"{v:.6f}" for v in mode_abs[:5]])
    print(
        f"[{label}] Time diff days (drifter_time - gu_time): "
        f"min={dt_days.min():.6f}, max={dt_days.max():.6f}, "
        f"mean={dt_days.mean():.6f}, median={dt_days.median():.6f}, mode={mode_signed_str}"
    )
    print(
        f"[{label}] Abs time diff days: "
        f"min={dt_abs.min():.6f}, max={dt_abs.max():.6f}, "
        f"mean={dt_abs.mean():.6f}, median={dt_abs.median():.6f}, mode={mode_abs_str}"
    )


def load_filtered_df(csv_path, label):
    df = pd.read_csv(csv_path)
    coast_km = get_coast_distance_km(df["lon_inter"].values, df["lat_inter"].values)
    proximity_km = PROXIMITY_DIST_M / 1000.0
    keep = np.isfinite(coast_km) & (np.abs(coast_km) <= proximity_km)
    df = df.loc[keep].copy()

    dt_days = compute_time_diff_days(df)
    if MAX_ABS_TIME_DIFF_DAYS is not None:
        keep_time = np.isfinite(dt_days.to_numpy(dtype=float)) & (
            np.abs(dt_days.to_numpy(dtype=float)) <= float(MAX_ABS_TIME_DIFF_DAYS)
        )
        df = df.loc[keep_time].copy()
        dt_days = dt_days.loc[df.index]

    summarize_time_diff_days(dt_days, label)
    return df


def fit_and_stats(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r = np.corrcoef(x, y)[0, 1]
    stderr = np.sqrt(np.sum((y - yhat) ** 2) / (x.size - 2))
    return slope, intercept, r, stderr


def prepare_xy(df, x_col):
    x = df[x_col].to_numpy(dtype=float)
    y = df["drifter_ve"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def prepare_xy_with_ek(df):
    x = df["gu"].to_numpy(dtype=float) + df["u_ek"].to_numpy(dtype=float)
    y = df["drifter_ve"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def print_rmse_and_variance_stats(df_no_ek, df_w_ek):
    x_no_ek, y_no_ek = prepare_xy(df_no_ek, "gu")
    x_w_ek, y_w_ek = prepare_xy_with_ek(df_w_ek)

    rmse_no_ek = np.sqrt(np.mean((y_no_ek - x_no_ek) ** 2))
    rmse_w_ek = np.sqrt(np.mean((y_w_ek - x_w_ek) ** 2))
    print(f"RMSE before Ekman correction (drifter_ve vs gu): {rmse_no_ek:.6f} m/s")
    print(f"RMSE after Ekman correction  (drifter_ve vs gu+u_ek): {rmse_w_ek:.6f} m/s")

    gu = df_w_ek["gu"].to_numpy(dtype=float)
    u_ek = df_w_ek["u_ek"].to_numpy(dtype=float)
    drifter = df_w_ek["drifter_ve"].to_numpy(dtype=float)
    valid = np.isfinite(gu) & np.isfinite(u_ek) & np.isfinite(drifter)
    gu = gu[valid]
    u_ek = u_ek[valid]
    drifter = drifter[valid]

    var_drifter = np.var(drifter, ddof=0)
    if var_drifter <= 0.0:
        print("Variance stats skipped: drifter_ve variance is zero or invalid.")
        return

    var_gu = np.var(gu, ddof=0)
    var_u_ek = np.var(u_ek, ddof=0)
    var_gu_plus_uek = np.var(gu + u_ek, ddof=0)

    print(
        "Relative variance out of drifter variance: "
        f"gu={100.0 * var_gu / var_drifter:.2f}%, "
        f"u_ek={100.0 * var_u_ek / var_drifter:.2f}%, "
        f"(gu+u_ek)={100.0 * var_gu_plus_uek / var_drifter:.2f}%"
    )


def plot_panel(ax, x, y, panel_title, x_label, y_label):
    slope, intercept, r, stderr = fit_and_stats(x, y)
    hb = ax.hexbin(x, y, gridsize=55, cmap="viridis", mincnt=1)

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
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.55, edgecolor="gray"),
        color="k",
    )

    ax.text(
        0.5,
        0.97,
        panel_title,
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="White", alpha=0.35, edgecolor="gray"),
        color="black",
    )
    ax.set_xlabel(x_label)
    # ax.set_ylabel("drifter ve (m/s)")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    return hb


def main():
    none = ""
    df_no_ek = load_filtered_df(CSV_NO_EK, "without_ek")
    df_w_ek = load_filtered_df(CSV_W_EK, "with_ek")
    print_r2_ekman_gain(CSV_W_EK, "drifter_ve", "drifter")

    x_no_ek, y_no_ek = prepare_xy(df_no_ek, "gu")
    x_w_ek, y_w_ek = prepare_xy_with_ek(df_w_ek)

    if x_no_ek.size < 3 or x_w_ek.size < 3:
        print(
            "Insufficient valid points after filtering.\n"
            f"without_ek: {x_no_ek.size} points, with_ek: {x_w_ek.size} points.\n"
            f"Current filters: PROXIMITY_DIST_M={PROXIMITY_DIST_M}, "
            f"MAX_ABS_TIME_DIFF_DAYS={MAX_ABS_TIME_DIFF_DAYS}"
        )
        return

    print_rmse_and_variance_stats(df_no_ek, df_w_ek)

    all_vals = np.concatenate([x_no_ek, y_no_ek, x_w_ek, y_w_ek])
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    hb0 = plot_panel(
        axes[0],
        x_no_ek,
        y_no_ek,
        "Without Ekman",
        "Altimetry gu (m/s)",
        "Drifter ve (m/s)"
    )
    hb1 = plot_panel(
        axes[1],
        x_w_ek,
        y_w_ek,
        "With Ekman",
        "Altimetry gu + u_ek (m/s)",
        none
    )

    for ax in axes:
        # ax.set_xlim(vmin, vmax)
        # ax.set_ylim(vmin, vmax)
        ax.set_xlim(vmin, 1.5)
        ax.set_ylim(vmin, 1.5)

    cmax = max(hb0.get_array().max(), hb1.get_array().max())
    hb0.set_clim(1, cmax)
    hb1.set_clim(1, cmax)

    # fig.suptitle(
    #     f"{SAT}: Drifter ve comparison | coast <= {PROXIMITY_DIST_M:.0f} m, "
    #     f"|dt| <= {MAX_ABS_TIME_DIFF_DAYS if MAX_ABS_TIME_DIFF_DAYS is not None else 'all'} d"
    # )
    fig.tight_layout(rect=[0.0, 0.0, 0.89, 0.95])
    cax = fig.add_axes([0.91, 0.13, 0.015, 0.75])
    cbar = fig.colorbar(hb1, cax=cax)
    cbar.set_label("Count")
    fig.savefig(OUT_PNG, dpi=180)
    plt.show()
    plt.close(fig)

    print(f"Rows after filters (without_ek): {len(df_no_ek)}")
    print(f"Rows after filters (with_ek): {len(df_w_ek)}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
