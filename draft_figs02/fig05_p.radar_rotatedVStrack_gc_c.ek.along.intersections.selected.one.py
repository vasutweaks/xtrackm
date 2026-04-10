import os

import xarray as xr
import xskillscore as xs
from tools_xtrackm import *
from pathlib import Path
import sys

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")

# codex resume 019d7247-c5e4-7662-bdec-18a975f1646b
METHOD = "loess"
FRAC1 = "0.05"
RADARS = ("TN", "AP", "OR")
GC_DIR = f"../guv/gc_along_{METHOD}"
EXTRACTED_DIR = "../radar/radar_along_intersections"
EKMAN_DIR = "../radar/ekman_along_intersections"
OUT_DIR = "../radar/radar_rotatedVStrack_gc_along_intersections"
BOB_REG = (78.0, 95.0, 4.0, 24.0)


def rotate_to_cross(u_field, v_field, lons_track, lats_track):
    ln = len(lons_track)
    ioff = max(1, min(5, ln // 10))
    angles = []
    for ii in range(ln):
        theta_tangent = local_angle(lons_track, lats_track, ii, ioff=ioff, degrees=False)
        angles.append(theta_tangent + (np.pi / 2.0))
    theta_cross = xr.DataArray(np.array(angles), coords={"x": u_field.x.values}, dims=["x"])
    cross = (np.cos(theta_cross) * u_field) + (np.sin(theta_cross) * v_field)
    cross.name = "cross"
    return cross


def local_angle_series_deg(xvals, lons_track, lats_track):
    ln = len(lons_track)
    ioff = max(1, min(5, ln // 10))
    angles_deg = [
        local_angle(lons_track, lats_track, ii, ioff=ioff, degrees=True)
        for ii in range(ln)
    ]
    return xr.DataArray(np.array(angles_deg), coords={"x": xvals}, dims=["x"])


def plot_2x2(
    gc_ek_valid,
    radar_ek_valid,
    corr_x_masked,
    corr_ek_x_masked,
    rmse_x_masked,
    rmse_ek_x_masked,
    radar_str,
    sat,
    tno,
    polygon,
    lons_track,
    lats_track,
    lon_x1,
    lat_x1,
    lon_x2,
    lat_x2,
    f_plot,
):
    sat_tsta, sat_tend = get_time_limits_o(sat)
    radar_tsta, radar_tend = radar_times[radar_str]
    overlap_start, overlap_end = overlap_dates(radar_tsta, radar_tend, sat_tsta, sat_tend)

    gc_plot = gc_ek_valid.assign_coords(x=gc_ek_valid.x / 1000.0)
    radar_plot = radar_ek_valid.assign_coords(x=radar_ek_valid.x / 1000.0)
    corr_plot = corr_x_masked.assign_coords(x=corr_x_masked.x / 1000.0)
    corr_ek_plot = corr_ek_x_masked.assign_coords(x=corr_ek_x_masked.x / 1000.0)
    rmse_plot = rmse_x_masked.assign_coords(x=rmse_x_masked.x / 1000.0)
    rmse_ek_plot = rmse_ek_x_masked.assign_coords(x=rmse_ek_x_masked.x / 1000.0)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.8], hspace=0.22, wspace=0.2)

    # ax_blank_top = fig.add_subplot(gs[0, 0])
    # ax_blank_bot = fig.add_subplot(gs[2, 0])
    ax0 = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    # ax_blank_top.axis("off")
    # ax_blank_bot.axis("off")

    decorate_axis(ax0, "", *BOB_REG, step=4)
    ax0.grid(alpha=0.3, linestyle="--")
    ax0.add_geometries(
        [polygon],
        ccrs.PlateCarree(),
        facecolor="tab:orange",
        edgecolor="tab:red",
        alpha=0.25,
        linewidth=1.5,
    )
    ax0.plot(lons_track, lats_track, color="tab:blue", linewidth=1.2, transform=ccrs.PlateCarree())

    poly_pt = polygon.representative_point()
    ax0.text(
        poly_pt.x,
        poly_pt.y,
        "OR",
        fontsize=11,
        fontweight="bold",
        color="tab:red",
        transform=ccrs.PlateCarree(),
        ha="center",
        va="center",
    )

    i_mid = len(lons_track) // 2
    ioff = max(1, min(5, len(lons_track) // 10))
    angle_deg = local_angle(lons_track, lats_track, i_mid, ioff=ioff, degrees=True)
    ax0.text(
        lons_track[i_mid],
        lats_track[i_mid],
        str(tno),
        fontsize=10,
        color="k",
        rotation=angle_deg,
        transform=ccrs.PlateCarree(),
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
    )

    ax0.plot([lon_x1, lon_x2], [lat_x1, lat_x2], "x", color="k", markersize=5, transform=ccrs.PlateCarree())
    # ax0.text(lon_x1, lat_x1, "x1", fontsize=9, color="magenta", transform=ccrs.PlateCarree())
    # ax0.text(lon_x2, lat_x2, "x2", fontsize=9, color="magenta", transform=ccrs.PlateCarree())

    cf1 = gc_plot.plot(
        ax=ax1,
        x="x",
        y="time",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        add_labels=False,
        add_colorbar=False,
    )
    ax1.set_ylabel("time")
    ax1.set_xlabel("")
    ax1.grid(alpha=0.3, linestyle="--")
    if overlap_start is not None and overlap_end is not None:
        ax1.set_ylim(overlap_start, overlap_end)

    radar_plot.plot(
        ax=ax2,
        x="x",
        y="time",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        add_labels=False,
        add_colorbar=False,
    )
    ax2.set_ylabel("time")
    ax2.set_xlabel("")
    ax2.grid(alpha=0.3, linestyle="--")
    if overlap_start is not None and overlap_end is not None:
        ax2.set_ylim(overlap_start, overlap_end)

    cax = ax1.inset_axes([0.0, 1.05, 1.0, 0.06])
    cb = fig.colorbar(cf1, cax=cax, orientation="horizontal")
    cb.set_label("m/s")
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")

    corr_plot.plot(ax=ax3, color="c", linewidth=1.5, label="no ekman", marker='x')
    corr_ek_plot.plot(ax=ax3, color="b", linewidth=1.5, label="with ekman")
    ax3.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax3.set_ylim(-1.0, 1.0)
    ax3.set_xlabel("x from coast (km)")
    ax3.set_ylabel("corr(time)")
    ax3.grid(alpha=0.3, linestyle="--")

    ax3_r = ax3.twinx()
    rmse_plot.plot(ax=ax3_r, color="orange", linewidth=1.3, label="rmse no ekman", marker="x")
    rmse_ek_plot.plot(ax=ax3_r, color="red", linewidth=1.3, label="rmse with ekman")
    ax3_r.set_ylabel("rmse (m/s)")

    ax3.spines["left"].set_color("tab:blue")
    ax3.spines["left"].set_linewidth(2.5)
    ax3.yaxis.label.set_color("tab:blue")
    ax3.tick_params(axis="y", colors="tab:blue", width=2.0)

    ax3_r.spines["right"].set_color("tab:red")
    ax3_r.spines["right"].set_linewidth(2.5)
    ax3_r.yaxis.label.set_color("tab:red")
    ax3_r.tick_params(axis="y", colors="tab:red", width=2.0)

    lines_l, labels_l = ax3.get_legend_handles_labels()
    lines_r, labels_r = ax3_r.get_legend_handles_labels()
    ax3.legend(lines_l + lines_r, labels_l + labels_r, loc="lower right")

    ax0.text(0.02, 0.98, "a)", transform=ax0.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    ax1.text(0.02, 0.98, "b)", transform=ax1.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    ax2.text(0.02, 0.98, "c)", transform=ax2.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    ax3.text(0.02, 0.98, "d)", transform=ax3.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(f_plot, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


radar_str = "OR"
sat = "S3A"
tno = "294"

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Using GC_DIR: {GC_DIR}")
print(f"Using EXTRACTED_DIR: {EXTRACTED_DIR}")
print(f"Using EKMAN_DIR: {EKMAN_DIR}")
print(f"Selection: radar={radar_str}, sat={sat}, track={tno}")

polygon = gpd.read_file(f"polygon_shapefile_around_{radar_str}.shp").geometry.iloc[0]
plot_dir = f"{OUT_DIR}/plots_{radar_str}_ek"
metric_dir = f"{OUT_DIR}/metrics_{radar_str}_ek"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)

f_ex = f"{EXTRACTED_DIR}/radar_along_{radar_str}_{sat}_{tno}.nc"
f_ek = f"{EKMAN_DIR}/ekman_along_{radar_str}_{sat}_{tno}.nc"
f_gc = f"{GC_DIR}/gc_along_{sat}_{tno}_{METHOD}_{FRAC1}.nc"
for f_in in (f_ex, f_ek, f_gc):
    if not os.path.exists(f_in):
        raise FileNotFoundError(f"missing input file: {f_in}")

with xr.open_dataset(f_ex) as ds_ex:
    radar_u = ds_ex.radar_u.load()
    radar_v = ds_ex.radar_v.load()
    dist_to_coast_track = ds_ex.dist_to_coast_gshhg_track.load()
    lons_track = ds_ex.lon_track.values
    lats_track = ds_ex.lat_track.values
    lon_x1 = float(ds_ex.attrs["lon_x1"])
    lat_x1 = float(ds_ex.attrs["lat_x1"])
    lon_x2 = float(ds_ex.attrs["lon_x2"])
    lat_x2 = float(ds_ex.attrs["lat_x2"])
    x1_m = float(ds_ex.attrs["x1_m"])
    x2_m = float(ds_ex.attrs["x2_m"])

with xr.open_dataset(f_ek) as ds_ek:
    ek_u = ds_ek.ekman_u.load()
    ek_v = ds_ek.ekman_v.load()

with xr.open_dataset(f_gc) as ds_gc:
    gc = ds_gc.gc.load()

radar_cross = rotate_to_cross(radar_u, radar_v, lons_track, lats_track)
ekman_cross = rotate_to_cross(ek_u, ek_v, lons_track, lats_track)

x_lo = min(x1_m, x2_m)
x_hi = max(x1_m, x2_m)

gc_cut = gc.sel(x=slice(x_lo, x_hi))
radar_cut = radar_cross.interp(x=gc_cut.x, method="nearest")
ekman_cut = ekman_cross.interp(x=gc_cut.x, method="nearest")
gc_ek_cut = gc_cut + ekman_cut

dist_to_coast_cut = dist_to_coast_track.interp(x=gc_cut.x, method="nearest")
angle_deg_cut = local_angle_series_deg(
    radar_u.x.values, lons_track, lats_track
).interp(x=gc_cut.x, method="nearest")

radar_cut = radar_cut.reindex(
    time=gc_cut.time,
    method="nearest",
    tolerance=np.timedelta64(24, "h"),
)
gc_ek_cut = gc_ek_cut.reindex(time=gc_cut.time)

valid = np.isfinite(gc_cut) & np.isfinite(radar_cut)
gc_valid = gc_cut.where(valid)
radar_valid = radar_cut.where(valid)

corr_x = xs.pearson_r(radar_valid, gc_valid, dim="time", skipna=True)
pval_x = xs.pearson_r_p_value(radar_valid, gc_valid, dim="time", skipna=True)
rmse_x = xs.rmse(radar_valid, gc_valid, dim="time", skipna=True)

sig_mask = pval_x < 0.05
corr_x_masked = corr_x.where(sig_mask)
rmse_x_masked = rmse_x.where(sig_mask)

valid_ek = np.isfinite(gc_ek_cut) & np.isfinite(radar_cut)
gc_ek_valid = gc_ek_cut.where(valid_ek)
radar_ek_valid = radar_cut.where(valid_ek)

corr_ek_x = xs.pearson_r(radar_ek_valid, gc_ek_valid, dim="time", skipna=True)
pval_ek_x = xs.pearson_r_p_value(radar_ek_valid, gc_ek_valid, dim="time", skipna=True)
rmse_ek_x = xs.rmse(radar_ek_valid, gc_ek_valid, dim="time", skipna=True)
bias_ek_x = (radar_ek_valid - gc_ek_valid).mean(dim="time", skipna=True)

sig_ek_mask = pval_ek_x < 0.05
corr_ek_x_masked = corr_ek_x.where(sig_ek_mask)
rmse_ek_x_masked = rmse_ek_x.where(sig_ek_mask)

f_plot = png_file
plot_2x2(
    gc_ek_valid,
    radar_ek_valid,
    corr_x_masked,
    corr_ek_x_masked,
    rmse_x_masked,
    rmse_ek_x_masked,
    radar_str,
    sat,
    tno,
    polygon,
    lons_track,
    lats_track,
    lon_x1,
    lat_x1,
    lon_x2,
    lat_x2,
    f_plot,
)
print(f"saved {f_plot}")
