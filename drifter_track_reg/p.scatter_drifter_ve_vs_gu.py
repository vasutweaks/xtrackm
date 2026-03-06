import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os

# codex resume 019cc485-7aec-7e61-95d4-af803cd14133

SAT = "TP+J1+J2+J3+S6A"
CSV_PATH = f"drifter_ve_vs_gu_{SAT}.csv"
OUT_PNG = f"scatter_drifter_ve_vs_gu_{SAT}.png"
COAST_DIST_PATH = "~/allData/topo/GMT_intermediate_coast_distance_01d_track_reg.nc"
PROXIMITY_DIST_M = 100000.0 # in meters
MAX_ABS_TIME_DIFF_DAYS = None  # e.g., 2.0 to keep rows with |drifter_time - gu_time| <= 2 days
MAX_ABS_TIME_DIFF_DAYS = 3  # e.g., 2.0 to keep rows with |drifter_time - gu_time| <= 2 days

# MAX_ABS_TIME_DIFF_DAYS = 10  # e.g., 2.0 to keep rows with |drifter_time - gu_time| <= 2 days

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


def summarize_time_diff_days(dt_days):
    dt_days = dt_days[np.isfinite(dt_days.to_numpy(dtype=float))]
    if dt_days.size == 0:
        print("No valid drifter_time/gu_time pairs for temporal-difference stats.")
        return
    dt_abs = np.abs(dt_days)
    mode_signed = pd.Series(np.round(dt_days, 6)).mode().to_numpy()
    mode_abs = pd.Series(np.round(dt_abs, 6)).mode().to_numpy()
    mode_signed_str = ", ".join([f"{v:.6f}" for v in mode_signed[:5]])
    mode_abs_str = ", ".join([f"{v:.6f}" for v in mode_abs[:5]])
    print(
        "Time diff days (drifter_time - gu_time): "
        f"min={dt_days.min():.6f}, max={dt_days.max():.6f}, "
        f"mean={dt_days.mean():.6f}, median={dt_days.median():.6f}, mode={mode_signed_str}"
    )
    print(
        "Abs time diff days: "
        f"min={dt_abs.min():.6f}, max={dt_abs.max():.6f}, "
        f"mean={dt_abs.mean():.6f}, median={dt_abs.median():.6f}, mode={mode_abs_str}"
    )


def main():
    df = pd.read_csv(CSV_PATH)
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
    summarize_time_diff_days(dt_days)

    x = df["gu"].to_numpy(dtype=float)
    y = df["drifter_ve"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 3:
        print(
            f"Only {x.size} valid points after filters "
            f"(PROXIMITY_DIST_M={PROXIMITY_DIST_M}, MAX_ABS_TIME_DIFF_DAYS={MAX_ABS_TIME_DIFF_DAYS}). "
            "Increase threshold to plot."
        )
        return

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r = np.corrcoef(x, y)[0, 1]
    stderr = np.sqrt(np.sum((y - yhat) ** 2) / (x.size - 2))

    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(x, y, gridsize=55, cmap="viridis", mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Count")

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
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.55, edgecolor="white"),
        color="white",
    )

    ax.set_title(
        f"{SAT}: Drifter ve vs Altimetry gu | coast <= {PROXIMITY_DIST_M:.0f} m, "
        f"|dt| <= {MAX_ABS_TIME_DIFF_DAYS if MAX_ABS_TIME_DIFF_DAYS is not None else 'all'} d"
    )
    ax.set_xlabel("Altimetry gu (m/s)")
    ax.set_ylabel("Drifter ve (m/s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.show()
    plt.close(fig)
    print(f"Rows after filters: {len(df)}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
