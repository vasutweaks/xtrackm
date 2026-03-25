import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tools_xtrackm import *
from tools_shapely import *
from scipy.signal import butter, filtfilt
from scipy.signal import periodogram as scipy_periodogram
from scipy.stats import linregress
# copilot --resume=6b563fd7-e036-45d8-95d0-bd5af130d84f

method = "loess"
cmap1 = cmo.cm.diff
d = 1.5
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset

sat_val = "TP+J1+J2+J3+S6A"
sat_val = "ERS1+ERS2+ENV+SRL"
sat_val = "S3A"

df = pd.read_csv(f"intersections_near_east_coast_{sat_val}.csv")
df_out = df.copy()

# Optional input: Gaussian smoothing sigma
sigma = 3
# Optional input: clipping bbox tuple
# Default used by the function is (70, 5, 95, 30)
region_bbox_tuple = (70, 5, 95, 30)

frac1 = "0.05"
for i, rows in df_out.iterrows():
    sat1 = rows["sat"]
    track_number_self = str(rows["track_self"])
    track_number_other = str(rows["track_other"])
    lons_inter1 = rows["lons_inter"]
    lats_inter1 = rows["lats_inter"]
    lon1 = lons_inter1
    lat1 = lats_inter1
    query_point = (lon1, lat1)
    result = coastline_gaussian_tangent(
        query_point=query_point,
        sigma=sigma,
        region_bbox_tuple=region_bbox_tuple,
    )
    print("heading_angle_deg:", result["heading_angle_deg"])
    print("tangent_vector:", result["tangent_vector"])
    print("nearest_point:", result["nearest_point"])
    print("distance:", result["distance"])
    coast_heading_deg = result["heading_angle_deg"]
    print(
        sat1,
        track_number_self,
        track_number_other,
        lons_inter1,
        lats_inter1,
    )
    f_self = f"../data/{sat1}/ctoh.sla.ref.{sat1}.nindian.{track_number_self.zfill(3)}.nc"
    f_other = f"../data/{sat1}/ctoh.sla.ref.{sat1}.nindian.{track_number_other.zfill(3)}.nc"
    f_guv = (
        f"../guv/guv_at_intersections_{method}/"
        f"guv_at_intersection_{sat1}_{track_number_self}_{track_number_other}_{method}_{frac1}.nc"
    )
    ds_self = xr.open_dataset(f_self, engine="netcdf4", decode_times=False)
    ds_other = xr.open_dataset(f_other, engine="netcdf4", decode_times=False)
    ds_guv = xr.open_dataset(f_guv)
    lons_track_self = ds_self.lon.values
    lats_track_self = ds_self.lat.values
    lon_coast_self = lons_track_self[-1]  # this on coast
    lat_coast_self = lats_track_self[-1]  # this on coast
    lon_equat_self = lons_track_self[0]  # this on equator
    lat_equat_self = lats_track_self[0]  # this on equator
    lons_track_other = ds_other.lon.values
    lats_track_other = ds_other.lat.values
    lon_equat_other = lons_track_other[0]
    lat_equat_other = lats_track_other[0]
    lon_coast_other = lons_track_other[-1]
    lat_coast_other = lats_track_other[-1]
    iself = index_at_lat(ds_self, lat1)
    a1_d = local_angle(
        lons_track_self, lats_track_self, iself, ioff=3, degrees=True
    )
    iothr = index_at_lat(ds_other, lat1)
    a2_d = local_angle(
        lons_track_other, lats_track_other, iothr, ioff=3, degrees=True
    )
    print(f"heading angle {a1_d:.2f} {a2_d:.2f}")
    gu = ds_guv.gu
    gv = ds_guv.gv
    gs = np.sqrt(gu**2 + gv**2)
    times = pd.to_datetime(gs.time.values)
    gs_vals_raw = gs.values.astype(float)

    # 500-day low-pass filter for long-term variability
    dt_days = np.median(
        np.diff(times).astype("timedelta64[s]").astype(float) / 86400.0
    )
    fs = 1.0 / dt_days  # cycles/day
    nyq = fs / 2.0
    cutoff = 1.0 / 500.0  # cycles/day
    b_filt, a_filt = butter(4, cutoff / nyq, btype="low")
    nan_mask = np.isnan(gs_vals_raw)
    gs_interp = pd.Series(gs_vals_raw).interpolate(limit_direction="both").values
    if len(gs_interp) > 3 * max(len(a_filt), len(b_filt)):
        gs_lowpass_vals = filtfilt(b_filt, a_filt, gs_interp)
    else:
        # Fallback if the record is too short for filtfilt padding constraints
        samples_500d = max(3, int(round(500.0 / dt_days)))
        gs_lowpass_vals = (
            pd.Series(gs_interp)
            .rolling(window=samples_500d, center=True, min_periods=1)
            .mean()
            .values
        )
    gs_lowpass_vals[nan_mask] = np.nan
    gs_lowpass = xr.DataArray(
        gs_lowpass_vals, coords=gs.coords, dims=gs.dims, name="gs_lowpass"
    )

    # Linear trend on low-pass series with standard error and significance
    t_days = (times - times[0]).days.astype(float)
    valid = np.isfinite(gs_lowpass_vals)
    trend_res = linregress(t_days[valid], gs_lowpass_vals[valid])
    fit_vals = trend_res.intercept + trend_res.slope * t_days
    trend_fit = xr.DataArray(fit_vals, coords=gs.coords, dims=gs.dims, name="trend_fit")
    slope_per_year = trend_res.slope * 365.25
    stderr_per_year = trend_res.stderr * 365.25
    significant = trend_res.pvalue < 0.05

    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(2, 1, 2)
    gs.plot(ax=ax0, label="current speed", color="lightcoral", linewidth=1.2, alpha=0.7)
    gs_lowpass.plot(ax=ax0, label="500-day low-pass", color="r", linewidth=2.0)
    trend_fit.plot(ax=ax0, label="linear trend fit", color="k", linewidth=2.0, linestyle="--")
    ax0.set_ylabel("current speed m/s")
    info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat1} {track_number_self} {track_number_other}"
    plt.text(
        0.15,
        0.90,
        info,
        transform=ax0.transAxes,
        fontsize=14,
        fontweight="bold",
    )

    # --- Seasonal shading and year markers on (ax0) ---
    year_min, year_max = times.year.min(), times.year.max()
    for yr in range(year_min, year_max + 1):
        # Feb-May: yellow
        ax0.axvspan(
            pd.Timestamp(yr, 2, 1),
            pd.Timestamp(yr, 5, 31),
            color="yellow",
            alpha=0.15,
            zorder=0,
        )
        # Jun-Sep: cyan
        ax0.axvspan(
            pd.Timestamp(yr, 6, 1),
            pd.Timestamp(yr, 9, 30),
            color="cyan",
            alpha=0.15,
            zorder=0,
        )
        # Oct-Dec: gray
        ax0.axvspan(
            pd.Timestamp(yr, 10, 1),
            pd.Timestamp(yr, 12, 31),
            color="gray",
            alpha=0.15,
            zorder=0,
        )
        # Year boundary vertical line on Jan 1
        ax0.axvline(
            pd.Timestamp(yr, 1, 1),
            color="k",
            linewidth=0.5,
            linestyle="--",
            alpha=0.5,
        )

    sig_txt = "significant (p<0.05)" if significant else "not significant (p>=0.05)"
    trend_txt = (
        f"Trend={slope_per_year:.4f} m/s/yr ± {stderr_per_year:.4f} (SE), "
        f"p={trend_res.pvalue:.3g}, {sig_txt}"
    )
    ax0.text(
        0.02,
        0.03,
        trend_txt,
        transform=ax0.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    seasonal_handles = [
        Patch(facecolor="yellow", alpha=0.15, edgecolor="none", label="Feb-May"),
        Patch(facecolor="cyan", alpha=0.15, edgecolor="none", label="Jun-Sep"),
        Patch(facecolor="gray", alpha=0.15, edgecolor="none", label="Oct-Dec"),
    ]
    line_handles = [
        Line2D([0], [0], color="lightcoral", lw=1.2, label="current speed"),
        Line2D([0], [0], color="r", lw=2.0, label="500-day low-pass"),
        Line2D([0], [0], color="k", lw=2.0, linestyle="--", label="linear trend fit"),
    ]
    ax0.legend(handles=line_handles + seasonal_handles, loc="upper right")

    ax2 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
        ETOPO05_Y=slice(*TRACKS_REG[2:])
    )
    rose_cut.plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
    decorate_axis(ax2, "", *TRACKS_REG, step=5)
    ax2.grid()
    ax2.plot(
        lon1,
        lat1,
        c="r",
        marker="o",
        markersize=6,
        markerfacecolor="red",
        markeredgecolor="black",
        markeredgewidth=2,
    )
    ax2.scatter(
        lons_track_self,
        lats_track_self,
        marker=".",
        color="c",
        s=4,
    )
    ax2.scatter(
        lons_track_other,
        lats_track_other,
        marker=".",
        color="c",
        s=4,
    )
    lonm, latm = get_point_at_distance(
        lon_equat_self, lat_equat_self, lon_coast_self, lat_coast_self, d
    )
    if is_within_region(lonm, latm, *TRACKS_REG):
        plt.text(
            lonm,
            latm,
            s=track_number_self,
            fontsize=14,
            rotation=a1_d,
            color="k",
        )
    lonm, latm = get_point_at_distance(
        lon_equat_other, lat_equat_other, lon_coast_other, lat_coast_other, d
    )
    if is_within_region(lonm, latm, *TRACKS_REG):
        plt.text(
            lonm,
            latm,
            s=track_number_other,
            fontsize=14,
            rotation=a2_d,
            color="k",
        )
    plt.savefig(
        f"speed_timeseries_{sat1}_{track_number_self}_{track_number_other}.png",
    )
    plt.show()
    plt.close("all")
