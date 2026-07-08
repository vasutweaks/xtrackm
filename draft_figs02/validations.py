import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


script = Path(sys.argv[0])
png_file = script.with_suffix(".png")

RAMA_EK_CSV = "rama_guv_c_nearest_global_matches.ek.csv"
OMNI_EK_CSV = "omni_guv_c_nearest_global_matches.ek.csv"

SAT = "TP+J1+J2+J3+S6A"
DRIFTER_CSV_NO_EK = f"drifter_ve_vs_gu_{SAT}.csv"
DRIFTER_CSV_W_EK = f"drifter_ve_vs_gu_ek_{SAT}.csv"
COAST_DIST_PATH = "~/allData/topo/GMT_intermediate_coast_distance_01d_track_reg.nc"

MIN_DISTANCE_KM = None
MAX_DISTANCE_KM = 250.0
PROXIMITY_DIST_M = 100_000.0
MAX_ABS_TIME_DIFF_DAYS = None


def distance_label():
    min_lbl = MIN_DISTANCE_KM if MIN_DISTANCE_KM is not None else "-inf"
    max_lbl = MAX_DISTANCE_KM if MAX_DISTANCE_KM is not None else "+inf"
    return f"distance_km in [{min_lbl}, {max_lbl}]"


def load_mooring_xy(csv_path, y_col, with_ekman=False):
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
    return x[valid], y[valid], len(df)


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


def load_drifter_df(csv_path, label):
    df = pd.read_csv(csv_path)
    coast_km = get_coast_distance_km(df["lon_inter"].values, df["lat_inter"].values)
    proximity_km = PROXIMITY_DIST_M / 1000.0
    keep = np.isfinite(coast_km) & (np.abs(coast_km) <= proximity_km)
    df = df.loc[keep].copy()

    dt_days = compute_time_diff_days(df)
    if MAX_ABS_TIME_DIFF_DAYS is not None:
        dt_values = dt_days.to_numpy(dtype=float)
        keep_time = np.isfinite(dt_values) & (np.abs(dt_values) <= float(MAX_ABS_TIME_DIFF_DAYS))
        df = df.loc[keep_time].copy()
        dt_days = dt_days.loc[df.index]

    summarize_time_diff_days(dt_days, label)
    return df


def prepare_drifter_xy(df, with_ekman=False):
    x = df["gu"].to_numpy(dtype=float)
    if with_ekman:
        x = x + df["u_ek"].to_numpy(dtype=float)
    y = df["drifter_ve"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def fit_and_stats(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r = np.corrcoef(x, y)[0, 1]
    stderr = np.sqrt(np.sum((y - yhat) ** 2) / (x.size - 2))
    return slope, intercept, r, stderr


def add_panel(ax, x, y, title, ylabel, xlabel, panel_label, extent=None, limits=None):
    if x.size < 3:
        ax.text(
            0.5,
            0.97,
            f"{title}\nInsufficient points",
            transform=ax.transAxes,
            va="top",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.65, edgecolor="gray"),
            color="black",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        add_panel_label(ax, panel_label)
        return None

    slope, intercept, r, stderr = fit_and_stats(x, y)
    hb = ax.hexbin(x, y, gridsize=55, cmap="viridis", mincnt=1, extent=extent)

    if limits is None:
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        xline = np.linspace(min(xmin, ymin), max(xmax, ymax), 200)
    else:
        xline = np.linspace(limits[0], limits[1], 200)

    ax.plot(xline, slope * xline + intercept, color="crimson", linewidth=2.0, label="Linear fit")
    ax.plot(xline, xline, color="white", linestyle="--", linewidth=1.2, label="1:1 line")

    text = (
        f"N = {x.size}\n"
        f"r = {r:.3f}\n"
        f"Std error = {stderr:.3f} m/s\n"
        f"y = {slope:.3f}x + {intercept:.3f}"
    )
    ax.text(
        0.97,
        0.03,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.58, edgecolor="gray"),
        color="black",
    )
    ax.text(
        0.58,
        0.97,
        title,
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.45, edgecolor="gray"),
        color="black",
    )
    add_panel_label(ax, panel_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    if limits is not None:
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    return hb


def observation_title(observation_name, comparison_text, filter_text):
    return f"Observation: {observation_name}\n{comparison_text}\n{filter_text}"


def add_panel_label(ax, panel_label):
    ax.text(
        0.03,
        0.03,
        panel_label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="black",
    )


def set_row_colorbar(fig, axes, hbs):
    hbs = [hb for hb in hbs if hb is not None]
    if not hbs:
        return

    vmax = max(float(np.nanmax(hb.get_array())) for hb in hbs)
    for hb in hbs:
        hb.set_clim(1.0, vmax)
    cbar = fig.colorbar(hbs[0], ax=axes, location="right", fraction=0.03, pad=0.02)
    cbar.set_label("Count")


def finite_extent(*arrays):
    vals = np.concatenate([np.asarray(a, dtype=float) for a in arrays if len(a) > 0])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    return vmin, vmax, vmin, vmax


def main():
    none = ""
    x_label_no_ek = "Altimetry gu (m/s)"
    x_label_ek = "Altimetry gu + u_ek (m/s)"

    x_rama_no, y_rama_no, n_rama_no = load_mooring_xy(RAMA_EK_CSV, "rama_u", with_ekman=False)
    x_rama_ek, y_rama_ek, n_rama_ek = load_mooring_xy(RAMA_EK_CSV, "rama_u", with_ekman=True)
    x_omni_no, y_omni_no, n_omni_no = load_mooring_xy(OMNI_EK_CSV, "omni_u", with_ekman=False)
    x_omni_ek, y_omni_ek, n_omni_ek = load_mooring_xy(OMNI_EK_CSV, "omni_u", with_ekman=True)

    df_drifter_no = load_drifter_df(DRIFTER_CSV_NO_EK, "drifter_without_ek")
    df_drifter_ek = load_drifter_df(DRIFTER_CSV_W_EK, "drifter_with_ek")
    x_drifter_no, y_drifter_no = prepare_drifter_xy(df_drifter_no, with_ekman=False)
    x_drifter_ek, y_drifter_ek = prepare_drifter_xy(df_drifter_ek, with_ekman=True)

    mooring_extent = finite_extent(
        x_rama_no,
        y_rama_no,
        x_rama_ek,
        y_rama_ek,
        x_omni_no,
        y_omni_no,
        x_omni_ek,
        y_omni_ek,
    )
    drifter_extent = finite_extent(x_drifter_no, y_drifter_no, x_drifter_ek, y_drifter_ek)
    drifter_min = drifter_extent[0] if drifter_extent is not None else -1.0

    fig, axes = plt.subplots(3, 2, figsize=(9, 13), sharex=False, sharey=False, constrained_layout=True)

    rama_hbs = [
        add_panel(
            axes[0, 0],
            x_rama_no,
            y_rama_no,
            observation_title("RAMA buoy", "u vs altimetry gu (no Ekman)", distance_label()),
            "RAMA u (m/s)",
            none,
            "a)",
            extent=mooring_extent,
            limits=(-1.0, 1.0),
        ),
        add_panel(
            axes[0, 1],
            x_rama_ek,
            y_rama_ek,
            observation_title("RAMA buoy", "u vs altimetry gu+u_ek (with Ekman)", distance_label()),
            none,
            none,
            "b)",
            extent=mooring_extent,
            limits=(-1.0, 1.0),
        ),
    ]
    omni_hbs = [
        add_panel(
            axes[1, 0],
            x_omni_no,
            y_omni_no,
            observation_title("OMNI", "u vs altimetry gu (no Ekman)", distance_label()),
            "OMNI u (m/s)",
            none,
            "c)",
            extent=mooring_extent,
            limits=(-1.0, 1.0),
        ),
        add_panel(
            axes[1, 1],
            x_omni_ek,
            y_omni_ek,
            observation_title("OMNI", "u vs altimetry gu+u_ek (with Ekman)", distance_label()),
            none,
            none,
            "d)",
            extent=mooring_extent,
            limits=(-1.0, 1.0),
        ),
    ]
    drifter_hbs = [
        add_panel(
            axes[2, 0],
            x_drifter_no,
            y_drifter_no,
            observation_title(
                "Drifter",
                "ve vs altimetry gu (no Ekman)",
                f"coast <= {PROXIMITY_DIST_M / 1000.0:.0f} km",
            ),
            "Drifter ve (m/s)",
            x_label_no_ek,
            "e)",
            extent=drifter_extent,
            limits=(drifter_min, 1.5),
        ),
        add_panel(
            axes[2, 1],
            x_drifter_ek,
            y_drifter_ek,
            observation_title(
                "Drifter",
                "ve vs altimetry gu+u_ek (with Ekman)",
                f"coast <= {PROXIMITY_DIST_M / 1000.0:.0f} km",
            ),
            none,
            x_label_ek,
            "f)",
            extent=drifter_extent,
            limits=(drifter_min, 1.5),
        ),
    ]

    set_row_colorbar(fig, axes[0, :], rama_hbs)
    set_row_colorbar(fig, axes[1, :], omni_hbs)
    set_row_colorbar(fig, axes[2, :], drifter_hbs)

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Rows after distance filter | RAMA no-ek: {n_rama_no}, RAMA ek: {n_rama_ek}")
    print(f"Rows after distance filter | OMNI no-ek: {n_omni_no}, OMNI ek: {n_omni_ek}")
    print(f"Rows after filters | Drifter no-ek: {len(df_drifter_no)}, Drifter ek: {len(df_drifter_ek)}")
    print(f"Saved {png_file}")


if __name__ == "__main__":
    main()
