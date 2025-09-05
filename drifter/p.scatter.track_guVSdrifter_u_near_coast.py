import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

import numpy as np
import pandas as pd
import xarray as xr


def clean_outliers(gu, ve, max_speed=3.0, mad_thresh=3.5):
    """
    Remove outliers from paired arrays (gu, ve).
    Parameters
    ----------
    gu, ve : array-like
        Arrays of satellite geostrophic u and drifter ve.
    max_speed : float
        Max absolute speed (m/s) allowed before removing.
    mad_thresh : float
        Threshold (in MAD units) for robust outlier removal.
    Returns
    -------
    gu_clean, ve_clean : ndarray
        Filtered arrays with outliers removed.
    mask : ndarray (bool)
        Boolean mask of valid points.
    """
    gu = np.asarray(gu)
    ve = np.asarray(ve)
    # Step 1: remove NaNs and infinities
    mask = np.isfinite(gu) & np.isfinite(ve)
    # Step 2: physical limit
    mask &= (np.abs(gu) <= max_speed) & (np.abs(ve) <= max_speed)
    # Step 3: robust MAD filter (applied separately to gu and ve)
    def mad_based_outlier(points, thresh=mad_thresh):
        median = np.median(points)
        mad = np.median(np.abs(points - median))
        if mad == 0:
            return np.zeros_like(points, dtype=bool)
        z = 0.6745 * (points - median) / mad
        return np.abs(z) > thresh

    mask &= ~mad_based_outlier(gu, mad_thresh)
    mask &= ~mad_based_outlier(ve, mad_thresh)

    return gu[mask], ve[mask], mask

sat_here = sys.argv[1]
df1 = pd.read_csv(f"track_guvVSnearby_drifters_at_intersection_points_{sat_here}_2.csv")

topo_dir = "/home/srinivasu/allData/topo/"
dsc = xr.open_dataset(f"{topo_dir}"
                      f"GMT_intermediate_coast_distance_01d_track_reg.nc")
print(dsc)
dist_to_coast = dsc.coast_dist


def is_within_coast(dist_to_coast, lon_inter, lat_inter, threshold=100):
    """
    Checks if a coordinate pair is within a certain distance from the coast.
    """
    # Using try-except to handle coordinates that may be outside the grid bounds
    try:
        dist1 = dist_to_coast.sel(lon=lon_inter, lat=lat_inter, method='nearest').item()
        if dist1 < threshold:
            return True
    except (KeyError, ValueError):
        # This will catch errors if lon/lat are not in the 'dist_to_coast' dataset
        return False
    return False


print(f"Original number of rows: {len(df1)}")

# Create a boolean mask by applying the function to each row
coastal_mask = df1.apply(
    lambda row: is_within_coast(dist_to_coast, row['lon_inters'], row['lat_inters'], threshold=1000),
    axis=1
)

# Use the mask to create a new DataFrame containing only coastal points
df_coastal = df1[coastal_mask].copy()

print(f"Number of rows after filtering for coast proximity: {len(df_coastal)}")

# --- Continue with your analysis on the filtered DataFrame ---
# Now use df_coastal instead of df1
gu_clean, ve_clean, mask = clean_outliers(df_coastal["gu"], df_coastal["ve"])

df = pd.DataFrame({
    "gu_mask": gu_clean,
    "ve_mask": ve_clean
})
# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df["gu_mask"], df["ve_mask"], alpha=0.5)
plt.title(f"Scatter Plot of gu vs ve for {sat_here}")
plt.xlabel("track gu")
plt.ylabel("drifter ve")
plt.grid(True)

# Calculate the correlation coefficient
correlation = df["gu_mask"].corr(df["ve_mask"])

# Place the correlation coefficient value on the plot
plt.text(0.05, 0.85, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

# Calculate the trend line (line of best fit)
m, b = np.polyfit(df['gu_mask'], df['ve_mask'], 1)
x_sorted = np.sort(df['gu_mask'])

# import statsmodels.api as sm
# X = sm.add_constant(df['gu_mask'])
# model = sm.RLM(df['ve_mask'], X, M=sm.robust.norms.HuberT()).fit()
# b, m = model.params
# x_sorted = np.sort(df['gu_mask'])

# Add the trend line to the plot
plt.plot(x_sorted, m*x_sorted + b, color='red', linewidth=2, label='Trend Line')
plt.legend()
# Save the plot
plt.savefig(f'scatter_plot_{sat_here}_track_uVSnearby_drifter_u_coast.png')
plt.show()

print(f"Correlation coefficient: {correlation}")
