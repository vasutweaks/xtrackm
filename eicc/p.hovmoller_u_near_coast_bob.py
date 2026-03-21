#!/usr/bin/env python3
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# copilot --resume=c0c59aeb-32fd-409b-a69f-a03a902a3b60

DEFAULT_SAT = "ERS1+ERS2+ENV+SRL"
DEFAULT_METHOD = "loess"
DEFAULT_FRAC = "0.05"
DEFAULT_VAR = "gu"  # requested "u" maps to gu for this dataset family
DEFAULT_INPUT_CSV = "intersections_near_coast_BOB_ERS1+ERS2+ENV+SRL.csv"
DEFAULT_GUV_DIR = "../guv/guv_at_intersections_loess"
DEFAULT_OUT_NC = "hovmoller_u_near_coast_BOB_ERS1+ERS2+ENV+SRL.nc"
DEFAULT_OUT_PNG = "hovmoller_u_near_coast_BOB_ERS1+ERS2+ENV+SRL.png"
BOB_EXTENT = (78.0, 100.0, 4.0, 24.0)  # lon_min, lon_max, lat_min, lat_max


def resolve_guv_file(guv_dir: Path, sat: str, track_self: int, track_other: int, method: str, frac: str) -> Path:
    t1 = str(int(track_self))
    t2 = str(int(track_other))
    candidates = [
        guv_dir / f"guv_at_intersection_{sat}_{t1}_{t2}_{method}_{frac}.nc",
        guv_dir / f"guv_at_intersection_{sat}_{t2}_{t1}_{method}_{frac}.nc",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No guv file found for tracks "
        f"({t1}, {t2}). Tried: " + ", ".join(str(x) for x in candidates)
    )

# create pandas date range between 1992-11-11,2016-03-21 with timeperiod 35 days
date_range = pd.date_range(start="1992-11-11", end="2016-03-21", freq="35D")
def build_transect_hovmoller(input_csv: Path, guv_dir: Path, sat: str, method: str, frac: str, var: str):
    df = pd.read_csv(input_csv).copy()
    if "lats_inter" not in df or "lons_inter" not in df:
        raise ValueError("Input CSV must contain lats_inter and lons_inter columns.")
    # Step 4: selected points sorted by latitude -> transect order
    df = df.sort_values("lats_inter").reset_index(drop=True)
    series = []
    kept_rows = []
    for i, row in df.iterrows():
        track_self = int(row["track_self"])
        track_other = int(row["track_other"])
        p = resolve_guv_file(guv_dir, sat, track_self, track_other, method, frac)
        with xr.open_dataset(p) as ds:
            if var not in ds.data_vars:
                raise KeyError(f"Variable '{var}' not found in {p}. Available: {list(ds.data_vars)}")
            # Step 3: append requested u-like variable from each guv file
            da = ds[var].load().rename("u")
        da = da.interp(time=date_range, method="nearest", kwargs={"fill_value": "extrapolate"})
        da = da.expand_dims(transect=[i])
        da = da.assign_coords(
            latitude=("transect", [float(row["lats_inter"])]),
            longitude=("transect", [float(row["lons_inter"])]),
            track_self=("transect", [track_self]),
            track_other=("transect", [track_other]),
        )
        series.append(da)
        kept_rows.append(row)
    if not series:
        raise ValueError("No intersection points were loaded.")
    # Step 5: concat array of u dataarrays with transect dimension
    u_transect = xr.concat(series, dim="transect", join="outer").sortby("latitude")
    return u_transect, pd.DataFrame(kept_rows)


def plot_bob_transect_map(ax, df_used: pd.DataFrame):
    proj = ccrs.PlateCarree()
    df_plot = df_used.sort_values("lats_inter").reset_index(drop=True)
    ax.set_extent(BOB_EXTENT, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.coastlines(resolution="50m", linewidth=1.0, zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.plot(
        df_plot["lons_inter"],
        df_plot["lats_inter"],
        color="tab:blue",
        linewidth=1.5,
        marker="o",
        markersize=4,
        transform=proj,
        zorder=3,
    )
    for i, row in df_plot.iterrows():
        lon = float(row["lons_inter"])
        lat = float(row["lats_inter"])
        dx = 0.18 if i % 2 == 0 else -0.95
        dy = 0.18 if i % 3 == 0 else -0.18
        ax.text(
            lon + dx,
            lat + dy,
            f"({lon:.2f}, {lat:.2f})",
            fontsize=8,
            transform=proj,
            zorder=4,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0),
        )
    ax.set_title("BOB transect intersections")


def plot_hovmoller(u_transect: xr.DataArray, df_used: pd.DataFrame, out_png: Path, title: str):
    hov_plot = u_transect.copy()
    vmax = float(np.nanpercentile(np.abs(hov_plot.values), 98))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.8
    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 2.4], wspace=0.05)
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_hov = fig.add_subplot(gs[0, 1])
    plot_bob_transect_map(ax_map, df_used)
    vmax = 1.2
    hov_plot.plot.contourf(
        ax=ax_hov,
        x="time",
        y="latitude",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        cbar_kwargs={"label": "u (m/s)"},
    )
    # add some contour lines for better visibility
    hov_plot.plot.contour(
        ax=ax_hov,
        x="time",
        y="latitude",
        colors="k",
        linewidths=0.5,
        levels=10,
        alpha=0.7,
        )
    ax_hov.set_xlabel("Time")
    ax_hov.set_ylabel("Latitude")
    ax_hov.set_title(title + " (monthly interpolated)")
    fig.savefig(out_png, dpi=180)
    plt.show()
    plt.close(fig)


parser = argparse.ArgumentParser(
    description=(
        "Build transect-time array from selected intersections and plot Hovmoller diagram "
        "(latitude vs time)."
    )
)
parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Selected intersections CSV")
parser.add_argument("--sat", default=DEFAULT_SAT, help="Satellite family string in guv filenames")
parser.add_argument("--method", default=DEFAULT_METHOD, help="Method token in guv filenames (e.g., loess)")
parser.add_argument("--frac", default=DEFAULT_FRAC, help="Frac token in guv filenames (e.g., 0.05)")
parser.add_argument("--var", default=DEFAULT_VAR, help="Variable name inside guv files (default: gu)")
parser.add_argument("--guv-dir", default=DEFAULT_GUV_DIR, help="Directory containing guv intersection netCDF files")
parser.add_argument("--out-nc", default=DEFAULT_OUT_NC, help="Output netCDF for concatenated transect array")
parser.add_argument("--out-png", default=DEFAULT_OUT_PNG, help="Output PNG for Hovmoller diagram")
args = parser.parse_args()

input_csv = Path(args.input_csv)
guv_dir = Path(args.guv_dir)
out_nc = Path(args.out_nc)
out_png = Path(args.out_png)

u_transect, df_used = build_transect_hovmoller(
    input_csv=input_csv,
    guv_dir=guv_dir,
    sat=args.sat,
    method=args.method,
    frac=args.frac,
    var=args.var,
)

ds_out = u_transect.to_dataset(name="u")
ds_out.to_netcdf(out_nc)

title = f"Hovmoller: {args.var}->u at intersections ({args.sat})"
plot_hovmoller(u_transect, df_used, out_png, title)

print(f"Loaded points: {len(df_used)}")
print(f"Transect dimension: {u_transect.sizes.get('transect', 0)}")
print(f"Time dimension: {u_transect.sizes.get('time', 0)}")
print(f"Saved concatenated array: {out_nc}")
print(f"Saved Hovmoller diagram: {out_png}")
