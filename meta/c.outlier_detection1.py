import numpy as np
import xarray as xr
from scipy.stats import median_abs_deviation
import glob

def mad_outlier_detection_fixed(data, threshold=3.5, dim="cycles_numbers"):
    """
    MAD outlier detection using your actual dimension names.
    Parameters:
    -----------
    data : xarray.DataArray
        Input SLA data
    threshold : float
        MAD threshold (3.5 is commonly used in X-TRACK)
    dim : str
        Dimension along which to detect outliers ('cycles_numbers' or 'points_numbers')
    Returns:
    --------
    xarray.DataArray : Boolean mask where True indicates outliers
    """
    def mad_filter_1d(x, thresh):
        """Apply MAD filter to 1D array"""
        if len(x) < 3 or np.all(np.isnan(x)):
            return np.zeros_like(x, dtype=bool)
        # Remove NaNs for calculation
        valid_mask = ~np.isnan(x)
        if np.sum(valid_mask) < 3:
            return np.zeros_like(x, dtype=bool)
        valid_data = x[valid_mask]
        median = np.median(valid_data)
        mad = median_abs_deviation(valid_data, scale='normal')
        if mad == 0:
            # If MAD is 0, use modified approach
            mad = np.median(np.abs(valid_data - median)) * 1.4826
            if mad == 0:
                return np.zeros_like(x, dtype=bool)
        # Calculate modified Z-score
        modified_z_scores = 0.6745 * (x - median) / mad
        outliers = np.abs(modified_z_scores) > thresh
        return outliers
    # Apply MAD filter along specified dimension
    outlier_mask = xr.apply_ufunc(
        lambda x: mad_filter_1d(x, threshold),
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[bool]
    )
    return outlier_mask

sat = "GFO"
passn = "008"
file1 = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{passn}.nc"
ds = xr.open_dataset(file1,decode_times=False)
# print(ds)
sla = ds["sla"]  # Assuming 'sla' is the variable name
# print(f"SLA data shape: {sla.shape}")
# print(f"SLA dimensions: {sla.dims}")
# Detect outliers along time dimension (cycles_numbers)
outlier_mask = mad_outlier_detection_fixed(sla, threshold=3.5, dim='cycles_numbers')
# Remove outliers
sla_filtered = sla.where(~outlier_mask)
# Print summary statistics
total_points = sla.size
valid_original = sla.count().values
outliers_detected = outlier_mask.sum().values
valid_filtered = sla_filtered.count().values
# if outliers_detected != 0:
if True:
    print(f"\n=== Outlier Detection Summary ===")
    print(f"Total data points: {total_points}")
    print(f"Originally valid: {valid_original} ({valid_original/total_points*100:.1f}%)")
    print(f"Outliers detected: {outliers_detected} ({outliers_detected/total_points*100:.1f}%)")
    print(f"Final valid points: {valid_filtered} ({valid_filtered/total_points*100:.1f}%)")
