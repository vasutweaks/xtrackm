#!/usr/bin/env python3

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc_context
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from pathlib import Path
import sys

# codex resume 019d633e-790a-7633-9992-2651d98a8294

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")

# Input files
CCMP_CURL_FILE = "/home/srinivasu/allData/ccmp31/ccmp_v31_daily_iod_windstress_curl_mon.nc"
CCMP_U_FILE = "/home/srinivasu/allData/ccmp31/ccmp_v31_daily_iod_u_mon.nc"
CCMP_V_FILE = "/home/srinivasu/allData/ccmp31/ccmp_v31_daily_iod_v_mon.nc"
AVISO_SLA_FILE = "/home/srinivasu/allData/aviso/allyear_dt_global_allsat_msla_h_anom.nc"
MASK_FILE = "/home/srinivasu/allData/aviso/bob_nov_gyre_mask.nc"

# Output
OUT_FIG = "indian_ocean_regions_4timeseries.png"
LOWPASS_CUTOFF_DAYS = 500.0

# Region bounds
BOB_LAT_BOUNDS = (8.0, 23.0)
BOB_LON_BOUNDS = (78.0, 99.0)
EQ_LAT_BOUNDS = (-5.0, 5.0)
EQ_LON_BOUNDS = (40.0, 100.0)
MAP_EXTENT = [40.0, 102.0, -10.0, 25.0]

# Windstress constants
RHO_AIR = 1.22
DRAG_COEFF = 1.3e-3


def subset_lat_lon(da, lat_bounds, lon_bounds):
    return da.sel(
        latitude=slice(lat_bounds[0], lat_bounds[1]),
        longitude=slice(lon_bounds[0], lon_bounds[1]),
    )


def interp_mask_to_target(mask_2d, target_da):
    return mask_2d.interp(
        latitude=target_da.latitude,
        longitude=target_da.longitude,
        method="nearest",
    )


def area_weighted_mean(da_3d, mask_2d=None):
    if mask_2d is not None:
        valid = np.isfinite(mask_2d) & (mask_2d > 0)
        da_3d = da_3d.where(valid)

    weights_1d = xr.DataArray(
        np.cos(np.deg2rad(da_3d.latitude)),
        coords={"latitude": da_3d.latitude},
        dims=("latitude",),
    )
    return da_3d.weighted(weights_1d).mean(("latitude", "longitude"), skipna=True)


def remove_monthly_climatology(ts):
    clim = ts.groupby("time.month").mean("time")
    return ts.groupby("time.month") - clim


def lowpass_filter_days(ts, cutoff_days=500.0, order=4):
    t = ts.time.values
    y = ts.values.astype(float)
    m = np.isfinite(y)

    if m.sum() < (order * 3):
        raise ValueError("Not enough valid samples for low-pass filter.")

    dt_days = float(np.median(np.diff(t[m]) / np.timedelta64(1, "D")))
    fs = 1.0 / dt_days
    cutoff_cpd = 1.0 / cutoff_days
    wn = cutoff_cpd / (0.5 * fs)

    if not (0.0 < wn < 1.0):
        raise ValueError(
            f"Invalid normalized cutoff {wn:.3f}. Check data sampling and cutoff_days."
        )

    b, a = butter(order, wn, btype="low")
    yf = np.full_like(y, np.nan, dtype=float)
    yf[m] = filtfilt(b, a, y[m])

    return xr.DataArray(yf, coords=ts.coords, dims=ts.dims), dt_days


def linear_trend_stats(ts):
    t = ts.time.values
    y = ts.values.astype(float)
    m = np.isfinite(y)

    if m.sum() < 3:
        raise ValueError("Not enough valid samples for linear regression.")

    x_years = (t[m] - t[m][0]) / np.timedelta64(1, "D") / 365.2425
    fit = linregress(x_years, y[m])

    x_all_years = (t - t[m][0]) / np.timedelta64(1, "D") / 365.2425
    yhat = fit.intercept + fit.slope * x_all_years
    return fit, yhat


def compute_windstress_magnitude(u, v):
    speed = np.hypot(u, v)
    tau_x = RHO_AIR * DRAG_COEFF * speed * u
    tau_y = RHO_AIR * DRAG_COEFF * speed * v
    return np.hypot(tau_x, tau_y)


def build_series(da, mask_2d=None, remove_clim=True):
    ts = area_weighted_mean(da, mask_2d=mask_2d)
    if remove_clim:
        ts = remove_monthly_climatology(ts)
    ts_f, dt_days = lowpass_filter_days(ts, cutoff_days=LOWPASS_CUTOFF_DAYS)
    fit, yhat = linear_trend_stats(ts_f)
    return ts_f, yhat, fit, dt_days


def plot_region_map(ax, gyre_mask, along_mask):
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Equatorial band: 5S to 5N over selected Indian Ocean longitudes
    eq_lons = np.linspace(EQ_LON_BOUNDS[0], EQ_LON_BOUNDS[1], 300)
    eq_lats = np.linspace(EQ_LAT_BOUNDS[0], EQ_LAT_BOUNDS[1], 80)
    eq_z = np.ones((eq_lats.size, eq_lons.size))
    with rc_context({"hatch.color": "green", "hatch.linewidth": 0.9}):
        ax.contourf(
            eq_lons,
            eq_lats,
            eq_z,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["xx"],
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

    gyre_z = np.where(np.isfinite(gyre_mask.values) & (gyre_mask.values > 0), 1.0, np.nan)
    with rc_context({"hatch.color": "blue", "hatch.linewidth": 0.9}):
        ax.contourf(
            gyre_mask.longitude,
            gyre_mask.latitude,
            gyre_z,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["xx"],
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    along_z = np.where(np.isfinite(along_mask.values) & (along_mask.values > 0), 1.0, np.nan)
    with rc_context({"hatch.color": "red", "hatch.linewidth": 0.9}):
        ax.contourf(
            along_mask.longitude,
            along_mask.latitude,
            along_z,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["xx"],
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    legend_handles = [
        Patch(facecolor="none", edgecolor="green", hatch="xx", label="Equatorial band (5S-5N)"),
        Patch(facecolor="none", edgecolor="blue", hatch="xx", label="BoB gyre region"),
        Patch(facecolor="none", edgecolor="red", hatch="xx", label="BoB along-shore region"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True)
    ax.set_title("Indian Ocean Regions")


def plot_ts_panel(ax, ts_f, yhat, fit, ylabel, line_color="k"):
    ax.plot(ts_f.time.values, ts_f.values, color=line_color, lw=1.3, label="500-day low-pass")
    ax.plot(ts_f.time.values, yhat, "k--", lw=1.8, label="Linear trend")
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylabel(ylabel, fontsize=8)
    # ax.legend(loc="upper right", fontsize=8)


def main():
    ds_mask = xr.open_dataset(MASK_FILE)
    gyre_mask = ds_mask["gyre_mask"]
    along_mask = ds_mask["along_shore_mask"]

    # 1) Windstress curl over BoB gyre region
    ds_curl = xr.open_dataset(CCMP_CURL_FILE)
    curl_var = "wind_stress_curl" if "wind_stress_curl" in ds_curl.data_vars else "windstress_curl"
    curl_bob = subset_lat_lon(ds_curl[curl_var], BOB_LAT_BOUNDS, BOB_LON_BOUNDS)
    gyre_on_curl = interp_mask_to_target(subset_lat_lon(gyre_mask, BOB_LAT_BOUNDS, BOB_LON_BOUNDS), curl_bob)
    ts1_f, yhat1, fit1, _ = build_series(curl_bob, mask_2d=gyre_on_curl, remove_clim=True)

    # 2) AVISO SLA over BoB gyre region
    ds_sla = xr.open_dataset(AVISO_SLA_FILE)
    sla = ds_sla["sla"]
    sla_bob = subset_lat_lon(sla, BOB_LAT_BOUNDS, BOB_LON_BOUNDS)
    gyre_on_sla = interp_mask_to_target(subset_lat_lon(gyre_mask, BOB_LAT_BOUNDS, BOB_LON_BOUNDS), sla_bob)
    ts2_f, yhat2, fit2, _ = build_series(sla_bob, mask_2d=gyre_on_sla, remove_clim=False)

    # 3) Windstress over BoB along-shore mask
    ds_u = xr.open_dataset(CCMP_U_FILE)
    ds_v = xr.open_dataset(CCMP_V_FILE)
    u_bob = subset_lat_lon(ds_u["uwnd"], BOB_LAT_BOUNDS, BOB_LON_BOUNDS)
    v_bob = subset_lat_lon(ds_v["vwnd"], BOB_LAT_BOUNDS, BOB_LON_BOUNDS)
    tau_bob = compute_windstress_magnitude(u_bob, v_bob)
    along_on_tau = interp_mask_to_target(subset_lat_lon(along_mask, BOB_LAT_BOUNDS, BOB_LON_BOUNDS), tau_bob)
    ts3_f, yhat3, fit3, _ = build_series(tau_bob, mask_2d=along_on_tau, remove_clim=True)

    # 4) Windstress over equatorial band (5S-5N, 40E-100E), ocean points only
    u_eq = subset_lat_lon(ds_u["uwnd"], EQ_LAT_BOUNDS, EQ_LON_BOUNDS)
    v_eq = subset_lat_lon(ds_v["vwnd"], EQ_LAT_BOUNDS, EQ_LON_BOUNDS)
    tau_eq = compute_windstress_magnitude(u_eq, v_eq)

    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = land.mask(tau_eq.longitude, tau_eq.latitude, wrap_lon=True)
    ocean_mask = xr.where(land_mask.isnull(), 1.0, np.nan)
    ts4_f, yhat4, fit4, _ = build_series(tau_eq, mask_2d=ocean_mask, remove_clim=True)

    # Figure layout: left map, right 4-row time series
    fig = plt.figure(figsize=(16, 10), dpi=160, constrained_layout=False)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.05, 1.7], wspace=0.18, hspace=0.0)

    ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax4 = fig.add_subplot(gs[3, 1], sharex=ax1)

    plot_region_map(ax_map, gyre_mask, along_mask)
    plot_ts_panel(
        ax1,
        ts1_f,
        yhat1,
        fit1,
        ylabel="Windstress curl (N m$^{-3}$)",
        line_color="tab:blue",
    )
    plot_ts_panel(
        ax2,
        ts2_f,
        yhat2,
        fit2,
        ylabel="SLA (m)",
        line_color="tab:blue",
    )
    plot_ts_panel(
        ax3,
        ts3_f,
        yhat3,
        fit3,
        ylabel="Windstress (N m$^{-2}$)",
        line_color="tab:red",
    )
    plot_ts_panel(
        ax4,
        ts4_f,
        yhat4,
        fit4,
        ylabel="Windstress (N m$^{-2}$)",
        line_color="tab:green",
    )
    ax1.text(0.01, 0.96, "a)", transform=ax1.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax2.text(0.01, 0.96, "b)", transform=ax2.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax3.text(0.01, 0.96, "c)", transform=ax3.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax4.text(0.01, 0.96, "d)", transform=ax4.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="x", labelbottom=False)
    ax3.tick_params(axis="x", labelbottom=False)
    ax4.set_xlabel("Time")
    fig.subplots_adjust(hspace=0.0)

    # fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.savefig(f"{png_file}", dpi=300, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

    print(f"Saved figure: {OUT_FIG}")
    print("Trend statistics:")
    print(f"(a) stderr={fit1.stderr:.6e}, p-value={fit1.pvalue:.6e}")
    print(f"(b) stderr={fit2.stderr:.6e}, p-value={fit2.pvalue:.6e}")
    print(f"(c) stderr={fit3.stderr:.6e}, p-value={fit3.pvalue:.6e}")
    print(f"(d) stderr={fit4.stderr:.6e}, p-value={fit4.pvalue:.6e}")


if __name__ == "__main__":
    main()
