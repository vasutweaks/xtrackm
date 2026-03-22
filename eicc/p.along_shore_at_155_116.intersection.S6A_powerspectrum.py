import cmocean as cmo
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
from tools_xtrackm import *
from tools_shapely import *
# codex resume 019d16b4-4223-7451-ba2c-f9b9d2bac698

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


def shade_shared_periods(ax, time_index):
    start_year = time_index.min().year
    end_year = time_index.max().year
    seasonal_windows = [
        ((2, 1), (5, 31), "yellow", 0.18),
        ((6, 1), (9, 30), "cyan", 0.16),
        ((10, 1), (12, 31), "gray", 0.12),
    ]
    for year in range(start_year, end_year + 1):
        for start_md, end_md, color, alpha in seasonal_windows:
            ax.axvspan(
                pd.Timestamp(year=year, month=start_md[0], day=start_md[1]),
                pd.Timestamp(year=year, month=end_md[0], day=end_md[1]),
                color=color,
                alpha=alpha,
                zorder=0,
            )
        ax.axvline(
            pd.Timestamp(year=year, month=1, day=1),
            color="k",
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )
    ax.axvline(
        pd.Timestamp(year=end_year + 1, month=1, day=1),
        color="k",
        linewidth=0.8,
        alpha=0.35,
        zorder=1,
    )


def compute_power_spectrum_days(da):
    series = pd.Series(da.values, index=pd.to_datetime(da.time.values)).sort_index()
    series = series[~series.index.duplicated(keep="first")].dropna()
    dt_days = np.diff(series.index.values).astype("timedelta64[s]").astype(float) / 86400.0
    sample_days = int(round(np.nanmedian(dt_days)))
    regular_index = pd.date_range(
        series.index.min(), series.index.max(), freq=f"{sample_days}D"
    )
    regular_series = series.reindex(regular_index).interpolate("time").dropna()
    regular_signal = regular_series.to_numpy(dtype=float)
    freq, power = signal.welch(
        regular_signal,
        fs=1.0 / sample_days,
        window="hann",
        detrend="linear",
        scaling="spectrum",
        nperseg=min(256, regular_signal.size),
    )
    valid = freq > 0
    freq = freq[valid]
    power = power[valid]
    order = np.argsort(freq)
    return freq[order], power[order], sample_days


method = "loess"
cmap1 = cmo.cm.diff
d = 1.5
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset

sat_val = "TP+J1+J2+J3+S6A"
track_self_val = 155
track_other_val = 116

df = pd.read_csv(f"tracks_intersections_{sat_val}_1.csv")
df_out = df.copy()

rows = df_out.loc[
    (df_out["sat"] == sat_val)
    & (df_out["track_self"] == track_self_val)
    & (df_out["track_other"] == track_other_val)
]
# Optional input: Gaussian smoothing sigma
sigma = 3
# Optional input: clipping bbox tuple
# Default used by the function is (70, 5, 95, 30)
region_bbox_tuple = (70, 5, 95, 30)

frac1 = "0.05"
i = 0
sat1 = rows["sat"].values[i]
# track_number_self = rows["track_self"].apply(lambda x: str(x)).values[i]
# track_number_other = rows["track_other"].apply(lambda x: str(x)).values[i]
track_number_self = str(rows["track_self"].values[i])
track_number_other = str(rows["track_other"].values[i])
lons_inter1 = rows["lons_inter"].values[i]
lats_inter1 = rows["lats_inter"].values[i]
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
al, ac = along_across_shore_components(gu, gv, coast_heading_deg)
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(3, 1, 3)
ax0 = fig.add_subplot(3, 1, 2)
al.plot(ax=ax0, label="alongshore", color="r", linewidth=2)
time_index = pd.to_datetime(al.time.values)
shade_shared_periods(ax0, time_index)
ax0.legend()
ax0.set_ylabel("along current m/s")
ax0.set_xlim(time_index.min(), time_index.max())
ax0.xaxis.set_major_locator(mdates.YearLocator(base=2))
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax0.grid(axis="x", alpha=0.25)
freq_cpd, power, sample_days = compute_power_spectrum_days(al)
min_freq_cpd = 1.0 / 500.0
max_freq_cpd = 1.0 / 20.0
display_mask = (freq_cpd >= min_freq_cpd) & (freq_cpd <= max_freq_cpd)
ax1.loglog(
    freq_cpd[display_mask],
    power[display_mask],
    color="r",
    linewidth=2,
    label="alongshore spectrum",
)
important_periods = [365, 180, 120, 90, 60, 30]
for period in important_periods:
    freq_marker = 1.0 / period
    if (
        display_mask.any()
        and freq_cpd[display_mask].min() <= freq_marker <= freq_cpd[display_mask].max()
    ):
        ax1.axvline(freq_marker, color="k", linestyle="--", linewidth=0.9, alpha=0.5)
        ax1.text(
            freq_marker,
            ax1.get_ylim()[1] * 0.95,
            f"{period}d",
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
            color="k",
        )
info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat1} {track_number_self} {track_number_other}"
plt.text(
    0.15,
    0.90,
    info,
    transform=ax0.transAxes,
    fontsize=14,
    fontweight="bold",
)
ax1.legend()
ax1.set_xlabel("frequency (cycles/day, log scale)")
ax1.set_ylabel("power spectrum (log scale)")
ax1.set_title(f"Alongshore current power spectrum ({sample_days}-day sampling)")
ax1.set_xlim(0.001, 1.0)
ax1.set_xticks([0.001, 0.01, 0.1, 1.0])
ax1.set_xticklabels(["0.001", "0.01", "0.1", "1"])
ax1.grid(alpha=0.3)
ax2 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
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
    f"along_shore_timeseries_{sat1}_{track_number_self}_{track_number_other}.png",
)
plt.show()
plt.close("all")
