import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *
from tools_shapely import *
# copilot --resume=6b563fd7-e036-45d8-95d0-bd5af130d84f


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

# Bandpass filter: remove periods shorter than 45 days (high-freq noise)
from scipy.signal import butter, filtfilt

al_vals_raw = al.values.copy()
dt_days_bp = np.median(np.diff(pd.to_datetime(al.time.values)).astype("timedelta64[D]").astype(float))
fs_bp = 1.0 / dt_days_bp  # cycles per day
# Low-pass cutoff: 1/45 cycles per day (keep periods > 45 days)
cutoff = 1.0 / 60.0
# Nyquist frequency
nyq = fs_bp / 2.0
b_filt, a_filt = butter(4, cutoff / nyq, btype="low")
# Interpolate NaNs for filtering, then restore NaN positions
nan_mask = np.isnan(al_vals_raw)
al_interp = pd.Series(al_vals_raw).interpolate(limit_direction="both").values
al_filtered = filtfilt(b_filt, a_filt, al_interp)
al_filtered[nan_mask] = np.nan
al = al.copy(data=al_filtered)

fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(3, 1, 3)
ax0 = fig.add_subplot(3, 1, 2)
al.plot(ax=ax0, label="alongshore", color="r", linewidth=2)
ax0.legend()
ax0.set_ylabel("along current m/s")
info = f"intersection point ({lon1:.2f}, {lat1:.2f}) {sat1} {track_number_self} {track_number_other}"
plt.text(
    0.15,
    0.90,
    info,
    transform=ax0.transAxes,
    fontsize=14,
    fontweight="bold",
)

# --- Seasonal shading and year markers on along-shore plot (ax0) ---
times = pd.to_datetime(al.time.values)
year_min, year_max = times.year.min(), times.year.max()
for yr in range(year_min, year_max + 1):
    # Feb-May: yellow
    ax0.axvspan(pd.Timestamp(yr, 2, 1), pd.Timestamp(yr, 5, 31), color="yellow", alpha=0.15, zorder=0)
    # Jun-Sep: cyan
    ax0.axvspan(pd.Timestamp(yr, 6, 1), pd.Timestamp(yr, 9, 30), color="cyan", alpha=0.15, zorder=0)
    # Oct-Dec: gray
    ax0.axvspan(pd.Timestamp(yr, 10, 1), pd.Timestamp(yr, 12, 31), color="gray", alpha=0.15, zorder=0)
    # Year boundary vertical line on Jan 1
    ax0.axvline(pd.Timestamp(yr, 1, 1), color="k", linewidth=0.5, linestyle="--", alpha=0.5)

# --- Periodogram of along-shore current ---
from scipy.signal import periodogram as scipy_periodogram

al_vals = al.values
al_times = pd.to_datetime(al.time.values)
# Compute median sampling interval in days
dt_days = np.median(np.diff(al_times).astype("timedelta64[D]").astype(float))
fs = 1.0 / dt_days  # sampling frequency in cycles/day

# Remove NaNs for periodogram
mask = np.isfinite(al_vals)
al_clean = al_vals[mask]
al_clean = al_clean - np.nanmean(al_clean)

freqs, psd = scipy_periodogram(al_clean, fs=fs, scaling="density")
# Convert frequency to period in days (skip freq=0)
freqs = freqs[1:]
psd = psd[1:]
periods = 1.0 / freqs

ax1.semilogy(periods, psd, color="b", linewidth=1)
ax1.set_xlabel("Period (days)")
ax1.set_ylabel("PSD (m²/s²/cpd)")
ax1.set_title("Periodogram of along-shore current")
ax1.set_xlim(0, 600)

# Mark important periodicities
important_periods = {"365 d": 365, "180 d": 180, "120 d": 120, "90 d": 90, "60 d": 60}
for label, p in important_periods.items():
    ax1.axvline(p, color="r", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.text(p, ax1.get_ylim()[1] * 0.5, label, rotation=90, va="top", ha="right",
             fontsize=9, color="r", alpha=0.9)
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
