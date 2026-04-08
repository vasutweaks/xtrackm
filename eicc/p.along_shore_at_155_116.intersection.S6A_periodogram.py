import cmocean as cmo
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
from astropy.timeseries import LombScargle
from scipy.signal import detrend
from tools_xtrackm import *
from tools_shapely import *
import sys


def _load_isobath_heading_func():
    module_path = Path(__file__).with_name("c.isobath_based_slopes.one_func.py")
    spec = importlib.util.spec_from_file_location("isobath_one_func", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.isobath_heading_at_point


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
topo_ds = xr.open_dataset("/home/srinivasu/allData/topo/etopo_60s_io.nc")
topo_reg = topo_ds.rose.sel(lon=slice(62, 92)).sel(lat=slice(0, 30))
isobath_heading_at_point = _load_isobath_heading_func()

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
print(len(rows), "length of rows ...")

i = 0
row = rows.iloc[i]
sat1 = row["sat"]
track_number_self = str(row["track_self"])
track_number_other = str(row["track_other"])
lons_inter1 = row["lons_inter"]
lats_inter1 = row["lats_inter"]
lon1 = lons_inter1
lat1 = lats_inter1
query_point = (lon1, lat1)
result = isobath_heading_at_point(
    topo=topo_reg,
    level=-1500,
    query_point=query_point,
    method="gaussian",
    sigma=10.0,
    step=15,
)
print(lon1, lat1)
print("heading_deg:", result["heading_deg"])
print("nearest_point:", result["nearest_point"])

coast_heading_deg = result["heading_deg"]
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
# Low-pass Butterworth filter: remove periodicities < 60 days (for FFT only)
al_filt = butter_lowpass_1d(al, cutoff_days=60, order=4)
ac_filt = butter_lowpass_1d(ac, cutoff_days=60, order=4)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

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
info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat1} {track_number_self} {track_number_other}"
ax_al.text(0.15, 0.90, info, transform=ax_al.transAxes, fontsize=14, fontweight="bold")

# --- Cross-shore time series ---
ac.plot(ax=ax_ac, label="acrossshore", color="b", linewidth=2)
ax_ac.axhline(y=0, color="k", linewidth=0.8, linestyle="--")
ax_ac.grid(True, linestyle="--", alpha=0.5)
ax_ac.set_ylabel("across current m/s")

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

important_periods = [
    (365, "365d"), (180, "180d"), (120, "120d"), (90, "90d"),
]
period_grid = np.linspace(90, 600, 2000)  # days
freq_grid = 1.0 / period_grid             # cycles per day
false_alarm_prob = 0.01  # 1% significance level

for signal, ax_psd, title in [
    (al, ax_al_psd, "Along-shore"),
    (ac, ax_ac_psd, "Cross-shore"),
]:
    vals = signal.values
    mask = ~np.isnan(vals)
    vals_clean = vals[mask]
    vals_clean = detrend(vals_clean - np.mean(vals_clean))
    t_days_arr = (times_pd[mask] - times_pd[mask][0]).total_seconds().to_numpy() / 86400.0

    ls = LombScargle(t_days_arr, vals_clean)
    ls_power = ls.power(freq_grid)
    # Significance threshold for the given false-alarm probability
    sig_level = ls.false_alarm_level(false_alarm_prob, method="baluev")

    ax_psd.plot(period_grid, ls_power, color="b", linewidth=1.2)
    ax_psd.axhline(sig_level, color="green", linewidth=1.2, linestyle="--",
                    label=f"{100*(1 - false_alarm_prob):.0f}% significance")
    ax_psd.set_xlim(90, 600)
    ax_psd.set_xscale("log")
    ax_psd.set_xlabel("Period (days)")
    ax_psd.set_ylabel("Lomb-Scargle Power")
    ax_psd.set_title(f"Lomb-Scargle — {title}")
    ax_psd.legend(loc="upper right", fontsize=8)
    ax_psd.grid(True, linestyle="--", alpha=0.5, which="both")
    ymax = ax_psd.get_ylim()[1]
    for pval, label in important_periods:
        ax_psd.axvline(pval, color="r", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_psd.text(
            pval, ymax * 0.92, label, rotation=90, va="top", ha="right",
            fontsize=9, color="r", fontweight="bold",
        )


# --- Map subplot ---
rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)
rose_cut.plot(ax=ax_map, add_colorbar=False, add_labels=False, cmap=cmap1)
decorate_axis(ax_map, "", *TRACKS_REG, step=5)
ax_map.grid()
ax_map.plot(
    lon1, lat1, c="r", marker="o", markersize=6,
    markerfacecolor="red", markeredgecolor="black", markeredgewidth=2,
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
plt.savefig(
    f"along_shore_timeseries_{sat1}_{track_number_self}_{track_number_other}.png",
)
plt.show()
plt.close("all")
