import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import regionmask
from scipy.stats import linregress
from scipy.signal import butter, filtfilt

in_file = "/home/srinivasu/allData/ccmp31/ccmp_v31_daily_iod_windstress_curl_mon.nc"
mask_file = "/home/srinivasu/allData/topo/bob_nov_gyre_mask.nc"
out_plot = "bob_windstress_curl_trend_500day.png"
lat_bounds = (8.0, 23.0)
lon_bounds = (78.0, 99.0)

ds = xr.open_dataset(in_file)
curl_name = "wind_stress_curl" if "wind_stress_curl" in ds.data_vars else "windstress_curl"
da = ds[curl_name].sel(latitude=slice(*lat_bounds), longitude=slice(*lon_bounds))
gyre_mask = xr.open_dataset(mask_file)["gyre_mask"].sel(
    latitude=slice(*lat_bounds), longitude=slice(*lon_bounds)
)
gyre_mask = gyre_mask.interp(latitude=da.latitude, longitude=da.longitude, method="nearest")
da = da.where(gyre_mask == 1)
# land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
# land_mask = land.mask(da.longitude, da.latitude, wrap_lon=True)
# da = da.where(land_mask.isnull())
w = xr.DataArray(np.cos(np.deg2rad(da.latitude)), coords={"latitude": da.latitude}, dims=("latitude",))
regional = da.weighted(w).mean(("latitude", "longitude"), skipna=True)

clim = regional.groupby("time.month").mean("time")
anomaly = regional.groupby("time.month") - clim
dt_days = float(np.median(np.diff(anomaly.time.values) / np.timedelta64(1, "D")))
nyq = 0.5 / dt_days
cutoff = 1.0 / 500.0
b_filt, a_filt = butter(4, cutoff / nyq, btype="low")
y = anomaly.values
m = np.isfinite(y)
y_f = np.full_like(y, np.nan, dtype=float)
y_f[m] = filtfilt(b_filt, a_filt, y[m])
filtered = xr.DataArray(y_f, coords=anomaly.coords, dims=anomaly.dims)

t_years = (filtered.time.values - filtered.time.values[0]) / np.timedelta64(1, "D") / 365.25
y2 = filtered.values
m2 = np.isfinite(y2)
res = linregress(t_years[m2], y2[m2])
yhat = res.slope * t_years + res.intercept

plt.figure(figsize=(10, 4.6))
plt.plot(filtered.time.values, filtered.values, "k", lw=1.4, label="BoB curl anomaly (500-day low-pass)")
plt.plot(filtered.time.values, yhat, "r--", lw=2, label=f"Linear trend: {res.slope:.3e} per year")
plt.grid(alpha=0.3)
plt.legend()
plt.title("Bay of Bengal Windstress Curl Anomaly: Low-pass and Linear Trend")
plt.xlabel("Time")
plt.ylabel("Windstress curl (N m-3)")
plt.tight_layout()
plt.savefig(out_plot, dpi=170)
plt.show()

print(f"Sampling interval (days): {dt_days:.3f}")
print("Removed monthly climatology before filtering")
print("Applied Butterworth low-pass with 500-day cutoff")
print(f"Slope (per year): {res.slope:.6e}")
print(f"Standard error:   {res.stderr:.6e}")
print(f"p-value:          {res.pvalue:.6e}")
print("Trend is significant at alpha=0.05" if res.pvalue < 0.05 else "Trend is not significant at alpha=0.05")
