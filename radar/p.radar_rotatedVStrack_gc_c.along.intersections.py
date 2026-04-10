import os
import pickle

import xarray as xr
import xskillscore as xs
from tools_xtrackm import *

# codex resume 019d7247-c5e4-7662-bdec-18a975f1646b

METHOD = "loess"
FRAC1 = "0.05"
RADARS = ("TN", "AP", "OR")
GC_DIR = f"../guv/gc_along_{METHOD}"
EXTRACTED_DIR = "radar_along_intersections"
OUT_DIR = "radar_rotatedVStrack_gc_along_intersections"
BOB_REG = (78.0, 95.0, 4.0, 24.0)


def rotate_radar_to_cross(radar_u, radar_v, lons_track, lats_track):
    ln = len(lons_track)
    ioff = max(1, min(5, ln // 10))
    angles = []
    for ii in range(ln):
        theta_tangent = local_angle(lons_track, lats_track, ii, ioff=ioff, degrees=False)
        angles.append(theta_tangent + (np.pi / 2.0))
    theta_cross = xr.DataArray(np.array(angles), coords={"x": radar_u.x.values}, dims=["x"])
    radar_cross = (np.cos(theta_cross) * radar_u) + (np.sin(theta_cross) * radar_v)
    radar_cross.name = "radar_cross"
    return radar_cross


def local_angle_series_deg(xvals, lons_track, lats_track):
    ln = len(lons_track)
    ioff = max(1, min(5, ln // 10))
    angles_deg = [
        local_angle(lons_track, lats_track, ii, ioff=ioff, degrees=True)
        for ii in range(ln)
    ]
    return xr.DataArray(np.array(angles_deg), coords={"x": xvals}, dims=["x"])


def plot_2x2(
    gc_valid,
    radar_valid,
    corr_x_masked,
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

    gc_plot = gc_valid.assign_coords(x=gc_valid.x / 1000.0)
    radar_plot = radar_valid.assign_coords(x=radar_valid.x / 1000.0)
    corr_plot = corr_x_masked.assign_coords(x=corr_x_masked.x / 1000.0)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.8], hspace=0.22, wspace=0.2)

    # Keep map at first-column second-row (middle-left), per request.
    ax_blank_top = fig.add_subplot(gs[0, 0])
    ax_blank_bot = fig.add_subplot(gs[2, 0])
    ax0 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax_blank_top.axis("off")
    ax_blank_bot.axis("off")

    decorate_axis(ax0, f"{radar_str} in Bay of Bengal", *BOB_REG, step=2)
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

    ax0.plot([lon_x1, lon_x2], [lat_x1, lat_x2], ".", color="magenta", markersize=5, transform=ccrs.PlateCarree())
    ax0.text(lon_x1, lat_x1, "x1", fontsize=9, color="magenta", transform=ccrs.PlateCarree())
    ax0.text(lon_x2, lat_x2, "x2", fontsize=9, color="magenta", transform=ccrs.PlateCarree())

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
    ax1.set_title("GC(time,x)")
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
    ax2.set_title("Radar cross(time,x)")
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

    corr_plot.plot(ax=ax3, color="k", linewidth=1.5)
    ax3.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax3.set_ylim(-1.0, 1.0)
    ax3.set_xlabel("x from coast (km)")
    ax3.set_ylabel("corr(time)")
    ax3.set_title("corr vs x")
    ax3.grid(alpha=0.3, linestyle="--")

    fig.suptitle(f"{radar_str} | {sat} track {tno}")
    fig.tight_layout()
    fig.savefig(f_plot, dpi=170, bbox_inches="tight")
    # plt.show()
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Using GC_DIR: {GC_DIR}")
    print(f"Using EXTRACTED_DIR: {EXTRACTED_DIR}")

    for radar_str in RADARS:
        polygon = gpd.read_file(f"polygon_shapefile_around_{radar_str}.shp").geometry.iloc[0]

        f_tracks = f"tracks_passing_{radar_str}.pkl"
        if not os.path.exists(f_tracks):
            print(f"missing pickle: {f_tracks}")
            continue
        with open(f_tracks, "rb") as f:
            tracks_dict = pickle.load(f)

        plot_dir = f"{OUT_DIR}/plots_{radar_str}"
        metric_dir = f"{OUT_DIR}/metrics_{radar_str}"
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(metric_dir, exist_ok=True)

        rows = []
        for sat, files in sorted(tracks_dict.items()):
            for f_track in files:
                if not os.path.exists(f_track):
                    continue

                with xr.open_dataset(f_track, decode_times=False) as ds_track:
                    tno = str(int(ds_track.Pass))

                f_ex = f"{EXTRACTED_DIR}/radar_along_{radar_str}_{sat}_{tno}.nc"
                if not os.path.exists(f_ex):
                    print(f"missing extracted radar file: {f_ex}")
                    continue

                f_gc = f"{GC_DIR}/gc_along_{sat}_{tno}_{METHOD}_{FRAC1}.nc"
                if not os.path.exists(f_gc):
                    print(f"missing gc file: {f_gc}")
                    continue

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

                with xr.open_dataset(f_gc) as ds_gc:
                    gc = ds_gc.gc.load()

                radar_cross = rotate_radar_to_cross(radar_u, radar_v, lons_track, lats_track)

                x_lo = min(x1_m, x2_m)
                x_hi = max(x1_m, x2_m)

                gc_cut = gc.sel(x=slice(x_lo, x_hi))
                radar_cut = radar_cross.interp(x=gc_cut.x, method="nearest")
                dist_to_coast_cut = dist_to_coast_track.interp(x=gc_cut.x, method="nearest")
                angle_deg_cut = local_angle_series_deg(
                    radar_u.x.values, lons_track, lats_track
                ).interp(x=gc_cut.x, method="nearest")
                radar_cut = radar_cut.reindex(
                    time=gc_cut.time,
                    method="nearest",
                    tolerance=np.timedelta64(24, "h"),
                )
                valid = np.isfinite(gc_cut) & np.isfinite(radar_cut)
                gc_valid = gc_cut.where(valid)
                radar_valid = radar_cut.where(valid)

                corr_x = xs.pearson_r(radar_valid, gc_valid, dim="time", skipna=True)
                pval_x = xs.pearson_r_p_value(radar_valid, gc_valid, dim="time", skipna=True)
                rmse_x = xs.rmse(radar_valid, gc_valid, dim="time", skipna=True)
                bias_x = (radar_valid - gc_valid).mean(dim="time", skipna=True)

                sig_mask = pval_x < 0.05
                corr_x_masked = corr_x.where(sig_mask)
                pval_x_masked = pval_x.where(sig_mask)
                rmse_x_masked = rmse_x.where(sig_mask)
                bias_x_masked = bias_x.where(sig_mask)

                n_sig_x = int(sig_mask.sum().item())
                keep = (
                    np.isfinite(pval_x.values)
                    & (pval_x.values < 0.05)
                    & np.isfinite(corr_x.values)
                    & np.isfinite(rmse_x.values)
                    & np.isfinite(bias_x.values)
                )
                for xx, dd, aa, cc, rr, bb in zip(
                    gc_cut.x.values[keep],
                    dist_to_coast_cut.values[keep],
                    angle_deg_cut.values[keep],
                    corr_x.values[keep],
                    rmse_x.values[keep],
                    bias_x.values[keep],
                ):
                    rows.append(
                        {
                            "radar_str": radar_str,
                            "sat": sat,
                            "track_number": tno,
                            "x": xx,
                            "dist_to_coast": dd,
                            "local_angle_deg": aa,
                            "corr_at_x": cc,
                            "rmse_at_x": rr,
                            "bias_x": bb,
                        }
                    )

                n_corr_valid = int(np.isfinite(corr_x_masked.values).sum())
                pct_corr_valid = 100.0 * n_corr_valid / corr_x_masked.size
                if pct_corr_valid >= 5.0:
                    f_plot = f"{plot_dir}/gc_vs_radar_xt_{sat}_{tno}.png"
                    plot_2x2(
                        gc_valid,
                        radar_valid,
                        corr_x_masked,
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
                else:
                    print(
                        f"{radar_str} {sat} {tno}: skip plot, "
                        f"valid corr_x={pct_corr_valid:.2f}% < 5%"
                    )
                if n_sig_x > 0:
                    corr_sig = corr_x.where(sig_mask).dropna(dim="x")
                    pval_sig = pval_x.where(sig_mask).dropna(dim="x")
                    x_sig_km = (corr_sig.x.values / 1000.0).tolist()
                    c_sig = corr_sig.values.tolist()
                    p_sig = pval_sig.values.tolist()
                    for xx, cc, pp in zip(x_sig_km, c_sig, p_sig):
                        print(
                            f"{radar_str} {sat} {tno}: "
                            f"x={xx:.1f} km corr={cc:.3f} p={pp:.3f}"
                        )
                else:
                    print(f"{radar_str} {sat} {tno}: no x points with p<0.05")

        df_out = pd.DataFrame(
            rows,
            columns=[
                "radar_str",
                "sat",
                "track_number",
                "x",
                "dist_to_coast",
                "local_angle_deg",
                "corr_at_x",
                "rmse_at_x",
                "bias_x",
            ],
        )
        f_sum = f"{OUT_DIR}/radar_rotatedVStrack_gc_along_summary_{radar_str}.csv"
        df_out.to_csv(f_sum, index=False)
        print(f"saved {f_sum} rows={len(df_out)}")


if __name__ == "__main__":
    main()
