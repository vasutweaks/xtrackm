import os
import pickle
import re

import xarray as xr
import xskillscore as xs
from tools_xtrackm import *

# codex resume 019d7247-c5e4-7662-bdec-18a975f1646b

METHOD = "loess"
FRAC1 = "0.05"
RADARS = ("TN", "AP", "OR")
GC_DIR = f"../guv/gc_along_intersections_{METHOD}"
GC_DIR_FALLBACK = f"../guv/gc_along_{METHOD}"
OUT_DIR = "radar_rotatedVStrack_gc_along_intersections"

# Omit correlation outputs where overlap along time is too low.
MIN_OVERLAP_TIME_PER_X = 8
MIN_X_WITH_GOOD_OVERLAP = 5


def extract_radar_cross_2d(u_radar, v_radar, polygon, lons_coast_to_off, lats_coast_to_off, xvals):
    ln = len(lons_coast_to_off)
    ioff = max(1, min(5, ln // 10))

    cross_ts_list = []
    in_poly = []
    for ii, (lon, lat) in enumerate(zip(lons_coast_to_off, lats_coast_to_off)):
        if not (np.isfinite(lon) and np.isfinite(lat)):
            in_poly.append(False)
            continue

        inside = polygon.covers(Point(lon, lat))
        in_poly.append(bool(inside))

        theta_tangent = local_angle(
            lons_coast_to_off,
            lats_coast_to_off,
            ii,
            ioff=ioff,
            degrees=False,
        )
        theta_cross = theta_tangent + (np.pi / 2.0)

        ur = u_radar.sel(longitude=lon, latitude=lat, method="nearest", drop=True)
        vr = v_radar.sel(longitude=lon, latitude=lat, method="nearest", drop=True)
        rc = (np.cos(theta_cross) * ur) + (np.sin(theta_cross) * vr)
        rc.name = "radar_cross"
        cross_ts_list.append(rc)

    if not cross_ts_list:
        return None, None

    radar_cross = xr.concat(cross_ts_list, dim="x")
    radar_cross = radar_cross.assign_coords(x=xvals)

    mask_x = xr.DataArray(np.array(in_poly, dtype=bool), coords=[xvals], dims=["x"])
    radar_cross = radar_cross.where(mask_x)
    return radar_cross, mask_x


def plot_xt_and_corr(gc_valid, radar_valid, corr_x_masked, radar_str, sat, tno, f_plot):
    gc_plot = gc_valid.assign_coords(x=gc_valid.x / 1000.0)
    radar_plot = radar_valid.assign_coords(x=radar_valid.x / 1000.0)
    corr_plot = corr_x_masked.assign_coords(x=corr_x_masked.x / 1000.0)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    gc_plot.plot(
        ax=ax1,
        x="time",
        y="x",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        add_labels=False,
    )
    ax1.set_title("GC(x,time)")
    ax1.set_ylabel("x from coast (km)")
    ax1.set_xlabel("time")

    radar_plot.plot(
        ax=ax2,
        x="time",
        y="x",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        add_labels=False,
    )
    ax2.set_title("Radar cross(x,time)")
    ax2.set_ylabel("")
    ax2.set_xlabel("time")

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
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gc_dir_use = GC_DIR if os.path.isdir(GC_DIR) else GC_DIR_FALLBACK
    print(f"Using GC_DIR: {gc_dir_use}")

    for radar_str in RADARS:
        f_radar = f"/home/srinivasu/allData/radar/{radar_str}Codar_daily.nc"
        if not os.path.exists(f_radar):
            print(f"missing radar file: {f_radar}")
            continue

        dsr = xr.open_dataset(f_radar).rename(
            {"XAXS": "longitude", "YAXS": "latitude", "ZAXS": "lev", "TAXIS1D": "time"}
        )
        u_radar = 0.01 * dsr.U_RADAR.isel(lev=0, drop=True)
        v_radar = 0.01 * dsr.V_RADAR.isel(lev=0, drop=True)

        gdf = gpd.read_file(f"polygon_shapefile_around_{radar_str}.shp")
        polygon = gdf.geometry.iloc[0]

        f_tracks = f"tracks_passing_{radar_str}.pkl"
        if not os.path.exists(f_tracks):
            print(f"missing pickle: {f_tracks}")
            dsr.close()
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
                    tno = track_number_from_track(ds_track, f_track)
                    if tno == "NA":
                        continue
                    sla_asn = track_dist_time_asn(ds_track, var_str="sla", units_in="m")
                    lons_coast_to_off = ds_track.lon.values[::-1]
                    lats_coast_to_off = ds_track.lat.values[::-1]

                f_gc = f"{gc_dir_use}/gc_along_{sat}_{tno}_{METHOD}_{FRAC1}.nc"
                if not os.path.exists(f_gc):
                    print(f"missing gc file: {f_gc}")
                    continue

                try:
                    gc = xr.open_dataset(f_gc).gc
                except ValueError as e:
                    print(e)
                    continue

                radar_cross, mask_x = extract_radar_cross_2d(
                    u_radar,
                    v_radar,
                    polygon,
                    lons_coast_to_off,
                    lats_coast_to_off,
                    sla_asn.x.values,
                )
                if radar_cross is None:
                    continue

                # Align x and time to gc grid.
                radar_on_gc = radar_cross.interp(x=gc.x, method="nearest")
                radar_on_gc = radar_on_gc.reindex(
                    time=gc.time,
                    method="nearest",
                    tolerance=np.timedelta64(24, "h"),
                )

                valid = np.isfinite(gc) & np.isfinite(radar_on_gc)
                gc_valid = gc.where(valid)
                radar_valid = radar_on_gc.where(valid)

                n_valid_x = valid.sum(dim="time")
                corr_x = xs.pearson_r(radar_valid, gc_valid, dim="time", skipna=True)
                rmse_x = xs.rmse(radar_valid, gc_valid, dim="time", skipna=True)
                bias_x = (radar_valid - gc_valid).mean(dim="time", skipna=True)

                # Omit correlation outputs where overlap is too low.
                corr_x_masked = corr_x.where(n_valid_x >= MIN_OVERLAP_TIME_PER_X)
                rmse_x_masked = rmse_x.where(n_valid_x >= MIN_OVERLAP_TIME_PER_X)
                bias_x_masked = bias_x.where(n_valid_x >= MIN_OVERLAP_TIME_PER_X)

                n_good_x = int((n_valid_x >= MIN_OVERLAP_TIME_PER_X).sum().item())
                if n_good_x >= MIN_X_WITH_GOOD_OVERLAP:
                    corr_x_mean = corr_x_masked.mean(dim="x", skipna=True).item()
                    rmse_x_mean = rmse_x_masked.mean(dim="x", skipna=True).item()
                    bias_x_mean = bias_x_masked.mean(dim="x", skipna=True).item()
                else:
                    corr_x_mean = np.nan
                    rmse_x_mean = np.nan
                    bias_x_mean = np.nan

                n_inside = int(mask_x.sum().item())
                rows.append(
                    {
                        "radar": radar_str,
                        "sat": sat,
                        "track_number": tno,
                        "n_x_inside_polygon": n_inside,
                        "n_x_good_overlap": n_good_x,
                        "corr_x_mean": corr_x_mean,
                        "rmse_x_mean": rmse_x_mean,
                        "bias_x_mean": bias_x_mean,
                    }
                )

                print(
                    f"{radar_str} {sat} {tno}: nx_inside={n_inside}, "
                    f"nx_good={n_good_x}, corr_x_mean={corr_x_mean}"
                )

                ds_out = xr.Dataset(
                    {
                        "gc": gc_valid,
                        "radar_cross": radar_valid,
                        "corr_x": corr_x_masked,
                        "rmse_x": rmse_x_masked,
                        "bias_x": bias_x_masked,
                        "n_valid_x": n_valid_x,
                    }
                )
                ds_out.to_netcdf(f"{metric_dir}/radar_gc_xstats_{sat}_{tno}.nc")

                f_plot = f"{plot_dir}/gc_vs_radar_xt_{sat}_{tno}.png"
                plot_xt_and_corr(gc_valid, radar_valid, corr_x_masked, radar_str, sat, tno, f_plot)

        df_out = pd.DataFrame(rows)
        f_sum = f"{OUT_DIR}/radar_rotatedVStrack_gc_along_summary_{radar_str}.csv"
        df_out.to_csv(f_sum, index=False)
        print(f"saved {f_sum} rows={len(df_out)}")

        dsr.close()


if __name__ == "__main__":
    main()
