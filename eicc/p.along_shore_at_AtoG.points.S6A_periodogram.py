import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from astropy.timeseries import LombScargle
from tools_xtrackm import *
from tools_shapely import *
# codex resume 019d5ed6-75c3-7ec3-b74c-b7b7403d77f7


def build_unique_str(sat, row):
    track_self = track_to_str(row["track_self"])
    track_other = track_to_str(row["track_other"])
    return f"{sat}_{track_self}_{track_other}"


def track_to_str(track):
    track_f = float(track)
    if track_f.is_integer():
        return str(int(track_f))
    return str(track)


sat_val = "TP+J1+J2+J3+S6A"

df = pd.read_csv(f"intersections_near_east_coast_{sat_val}.csv")

# Skip the first point (lowest latitude)
df_out = df.sort_values("lats_inter").iloc[1:].reset_index(drop=True)
EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd/"
EKMAN_DEPTH = "0"

N_PTS = len(df_out)
assert len(df_out) == N_PTS, (
    f"Expected {N_PTS} points after skipping first, got {len(df_out)}"
)

POINT_COLORS = [plt.cm.tab10(i) for i in range(N_PTS)]
POINT_LABELS = list("ABCDEFG")

# Map extent: 10N–22N, 78E–88E
MAP_EXTENT = (78.5, 88.0, 10.0, 22.0)

def along_across_shore_components(gu, gv, coast_heading_deg):
    """
    Resolve vector (gu, gv) into along-shore and across-shore components.
    Sign convention:
    - coast_heading_deg is measured CCW from +x (East).
    - along_shore is positive in the heading direction.
    - across_shore is positive 90 deg CCW from heading (left-hand side of coast).
      If you need right-hand positive, negate across_shore.
    """
    theta = np.deg2rad(coast_heading_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    # Elementwise projection keeps scalar or array/xarray shape.
    along_shore = gu * c + gv * s
    across_shore = -gu * s + gv * c  # +90 deg CCW
    return along_shore, across_shore


method = "loess"
frac1 = "0.05"
cmap1 = cmo.cm.diff
d = 1.5
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)

important_periods = [
    (365, "365d"), (180, "180d"), (120, "120d"), (90, "90d"), (60, "60d"),
]
MIN_PERIOD_DAYS = 90.0
MAX_PERIOD_DAYS = 1000.0
SIGNIFICANCE_FAP = 0.05
sigma = 3
region_bbox_tuple = (70, 5, 95, 30)

for i, row in df_out.iterrows():
    point_label = POINT_LABELS[i]
    sat1 = row["sat"]
    track_number_self = str(row["track_self"])
    track_number_other = str(row["track_other"])
    lons_inter1 = row["lons_inter"]
    lats_inter1 = row["lats_inter"]
    coast_distance_km_nearest = row["coast_distance_km_nearest"]
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
    unique_str = build_unique_str(sat1, row)
    f_ek = f"{EKMAN_DIR}/{unique_str}_ekman_{EKMAN_DEPTH}m_at_intersections_cd.nc"
    ds_ek = xr.open_dataset(f_ek)
    u_ek = ds_ek['u_at']
    v_ek = ds_ek['v_at']
    gu = gu + u_ek
    gv = gv + v_ek
    # s_ek = np.sqrt(u_ek**2 + v_ek**2)
    al, ac = along_across_shore_components(gu, gv, coast_heading_deg)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.28, wspace=0.28, height_ratios=[0.85, 1, 1])

    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax_al = fig.add_subplot(gs[1, :2])
    ax_al_psd = fig.add_subplot(gs[1, 2])
    ax_ac = fig.add_subplot(gs[2, :2])
    ax_ac_psd = fig.add_subplot(gs[2, 2])

    # --- Along-shore time series ---
    al.plot(ax=ax_al, label="alongshore", color="r", linewidth=2)
    ax_al.axhline(y=0, color="k", linewidth=0.8, linestyle="--")
    ax_al.grid(True, linestyle="--", alpha=0.5)
    ax_al.set_ylabel("along current m/s")
    info = (
        f"{point_label} ({coast_distance_km_nearest:.1f} km) "
        f"intersection ({lon1:.2f}, {lat1:.2f}) {sat1} {track_number_self} {track_number_other}"
    )
    ax_al.text(0.15, 0.90, info, transform=ax_al.transAxes, fontsize=14, fontweight="bold")

    # --- Cross-shore time series ---
    ac.plot(ax=ax_ac, label="crossshore", color="b", linewidth=2)
    ax_ac.axhline(y=0, color="k", linewidth=0.8, linestyle="--")
    ax_ac.grid(True, linestyle="--", alpha=0.5)
    ax_ac.set_ylabel("cross current m/s")

    # Seasonal shading and year markers on both time series
    times_pd = pd.to_datetime(al.time.values)
    year_min, year_max = times_pd.min().year, times_pd.max().year
    for yr in range(year_min, year_max + 1):
        for ax_ts in [ax_al, ax_ac]:
            ax_ts.axvline(pd.Timestamp(yr, 1, 1), color="k", linewidth=0.5, alpha=0.5)
            ax_ts.axvspan(
                pd.Timestamp(yr, 2, 1), pd.Timestamp(yr, 6, 1),
                color="yellow", alpha=0.3, zorder=0,
            )
            ax_ts.axvspan(
                pd.Timestamp(yr, 6, 1), pd.Timestamp(yr, 10, 1),
                color="cyan", alpha=0.3, zorder=0,
            )
            ax_ts.axvspan(
                pd.Timestamp(yr, 10, 1), pd.Timestamp(yr + 1, 1, 1),
                color="gray", alpha=0.2, zorder=0,
            )
    ax_al.legend(loc="upper right")
    ax_ac.legend(loc="upper right")

    # --- Lomb-Scargle periodograms ---
    t_days = (times_pd - times_pd[0]).total_seconds().to_numpy() / 86400.0
    for signal, ax_psd, title in [
        (al, ax_al_psd, "Along-shore"),
        (ac, ax_ac_psd, "Cross-shore"),
    ]:
        vals = signal.values
        mask = ~np.isnan(vals)
        ls = LombScargle(t_days[mask], vals[mask])
        frequency, power = ls.autopower(
            minimum_frequency=1.0 / MAX_PERIOD_DAYS,
            maximum_frequency=1.0 / MIN_PERIOD_DAYS,
        )
        periods = 1.0 / frequency
        significance_power = ls.false_alarm_level(SIGNIFICANCE_FAP)
        sort_idx = np.argsort(periods)
        periods_sorted = periods[sort_idx]
        power_sorted = power[sort_idx]
        ax_psd.plot(periods_sorted, power_sorted, color="b", linewidth=1, label="Lomb power")
        ax_psd.plot(
            periods_sorted,
            np.full_like(periods_sorted, significance_power),
            color="magenta",
            linewidth=1.2,
            linestyle="--",
            label=f"{int((1.0 - SIGNIFICANCE_FAP) * 100)}% significance",
        )
        ax_psd.set_xscale("log")
        ax_psd.set_xlabel("Period (days)")
        ax_psd.set_ylabel("Power")
        ax_psd.set_title(f"Periodogram — {title}")
        ax_psd.set_xlim(MIN_PERIOD_DAYS, MAX_PERIOD_DAYS)
        ax_psd.grid(True, linestyle="--", alpha=0.5, which="both")
        ymax = ax_psd.get_ylim()[1]
        for pval, label in important_periods:
            if not (MIN_PERIOD_DAYS <= pval <= MAX_PERIOD_DAYS):
                continue
            ax_psd.axvline(pval, color="r", linewidth=0.8, linestyle="--", alpha=0.7)
            ax_psd.text(
                pval, ymax * 0.92, label, rotation=90, va="top", ha="right",
                fontsize=9, color="r", fontweight="bold",
            )
        ax_psd.legend(loc="upper right", fontsize=8)

    # --- Map subplot ---
    rose_cut.plot(ax=ax_map, add_colorbar=False, add_labels=False, cmap=cmap1)
    decorate_axis(ax_map, "", *TRACKS_REG, step=5)
    ax_map.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax_map.grid()
    ax_map.plot(
        lon1, lat1, c="r", marker="o", markersize=6,
        markerfacecolor="red", markeredgecolor="black", markeredgewidth=2,
    )
    ax_map.text(
        lon1 + 0.10,
        lat1 + 0.10,
        f"{point_label} ({coast_distance_km_nearest:.1f} km)",
        fontsize=10,
        fontweight="bold",
        color="k",
        transform=ccrs.PlateCarree(),
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
        zorder=10,
    )
    ax_map.scatter(lons_track_self, lats_track_self, marker=".", color="c", s=4)
    ax_map.scatter(lons_track_other, lats_track_other, marker=".", color="c", s=4)
    lonm, latm = get_point_at_distance(
        lon_equat_self, lat_equat_self, lon_coast_self, lat_coast_self, d
    )
    if is_within_region(lonm, latm, *TRACKS_REG):
        plt.text(
            lonm, latm, s=track_number_self,
            fontsize=14, rotation=a1_d, color="k",
        )
    lonm, latm = get_point_at_distance(
        lon_equat_other, lat_equat_other, lon_coast_other, lat_coast_other, d
    )
    if is_within_region(lonm, latm, *TRACKS_REG):
        plt.text(
            lonm, latm, s=track_number_other,
            fontsize=14, rotation=a2_d, color="k",
        )
    fig.subplots_adjust(top=0.96, bottom=0.07, left=0.07, right=0.98)
    plt.savefig(
            f"alongshore_periodogram_{track_number_self}_{track_number_other}_{sat1}.png",
            bbox_inches="tight",
            pad_inches=0.03,
    )
    plt.show()
    plt.close("all")
