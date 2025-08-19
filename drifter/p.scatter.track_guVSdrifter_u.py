import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

import numpy as np
import pandas as pd


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
    # gu = pd.to_numeric(gu, errors='coerce')
    # ve = pd.to_numeric(ve, errors='coerce')
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
df1 = pd.read_csv(f"track_guvVSnearby_drifters_at_intersection_points_{sat_here}.csv")

gu_clean, ve_clean, mask = clean_outliers(df1["gu"], df1["ve"])

df = pd.DataFrame({
    "gu_mask": gu_clean,
    "ve_mask": ve_clean
})
# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df["gu_mask"], df["ve_mask"], alpha=0.5)
plt.title("Scatter Plot of gu vs ve")
plt.xlabel("track gu")
plt.ylabel("drifter ve")
plt.grid(True)

# Calculate the correlation coefficient
correlation = df["gu_mask"].corr(df["ve_mask"])

# Place the correlation coefficient value on the plot
plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12,
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
plt.savefig(f'scatter_plot_{sat_here}_track_uVSnearby_drifter_u.png')
plt.show()

print(f"Correlation coefficient: {correlation}")
