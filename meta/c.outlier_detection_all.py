import numpy as np
import xarray as xr
from scipy.stats import median_abs_deviation
import glob
from namesm import sats_new


def mad_outlier_detection_fixed(data, threshold=3.5, dim="cycles_numbers"):
    """
    MAD outlier detection using your actual dimension names.
    """
    def mad_filter_1d(x, thresh):
        if len(x) < 3 or np.all(np.isnan(x)):
            return np.zeros_like(x, dtype=bool)
        valid_mask = ~np.isnan(x)
        if np.sum(valid_mask) < 3:
            return np.zeros_like(x, dtype=bool)
        # size of valid data might be less than x depending on the
        # number of NaNs
        valid_data = x[valid_mask]
        median = np.median(valid_data)
        mad = median_abs_deviation(valid_data, scale='normal')
        if mad == 0:
            mad = np.median(np.abs(valid_data - median)) * 1.4826
            if mad == 0:
                return np.zeros_like(x, dtype=bool)
        modified_z_scores = 0.6745 * (x - median) / mad
        outliers = np.abs(modified_z_scores) > thresh
        # size of outliers is same as the size of x
        return outliers

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


for sat in sats_new:
    total_size = 0      # Total size (n_orig1)
    total_initial_nans = 0
    total_initial_valid = 0
    # Outliers are numerical values and counted over valid values
    total_outliers = 0
    file_count = 0
    file_list = sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc"))
    for f in file_list:
        ds = xr.open_dataset(f)
        sla = ds["sla"]
        outlier_mask = mad_outlier_detection_fixed(sla, threshold=3.5, dim='cycles_numbers')
        n_orig1 = sla.size                         # Total size
        n_valid = int(sla.count().values)          # Non-NaNs before filtering
        n_nans = n_orig1 - n_valid                 # Initial NaNs
        n_outliers = int(outlier_mask.sum().values) # Outliers found in n_valid
        # 3. Accumulate
        total_size += n_orig1
        total_initial_nans += n_nans
        total_initial_valid += n_valid
        total_outliers += n_outliers
        file_count += 1
        ds.close() # Good practice for large batches
    # --- Final Statistics ---
    print(f"Satellite: {sat} ----------------")
    if total_size > 0:
        avg_nan_perc = (total_initial_nans / total_size) * 100
        # Outlier percentage is typically calculated relative to the valid data points
        avg_outlier_perc = (total_outliers / total_initial_valid) * 100 if total_initial_valid > 0 else 0
        print(f"Processed {file_count} files across for {sat}.")
        print(f"Average NaN Percentage: {avg_nan_perc:.5f}%")
        print(f"Average Outlier Percentage (relative to valid data): {avg_outlier_perc:.5f}%")
    else:
        print("No files found.")
