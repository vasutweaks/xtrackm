import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
import pandas as pd
import math
import os
from tools_xtrackm import *


def mad_filter_1d(x, thresh=3.5):
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


for sat in sats_new:
    # read guv_at_intersections_HY2B_0.2.npz file
    arr = np.load(f"guv_at_intersections_{sat}_0.2.npz")
    arr1 = arr['gu_all']
    # remove nans for arr
    arr_nonan = arr1[~np.isnan(arr1)]
    # filter if arr is greater than 3 or less than -3
    arr_filter1 = arr_nonan[(arr_nonan > -3.5) & (arr_nonan < 3.5)]
    outliers = mad_filter_1d(arr_filter1, thresh=3.5)
    arr_filter = arr_filter1[~outliers]
    mean = np.nanmean(arr_filter)
    mx = np.nanmax(arr_filter)
    mn = np.nanmin(arr_filter)
    md = np.nanmedian(arr_filter)
    # plot histograms
    plt.hist(arr_filter, bins=100)
    # add text of mean, mx, min on plot
    plt.text(0.9, 0.9, f"sat={sat}\nmedian={md:.3f}\nmean={mean:.3f}\nmax={mx:.3f}\nmin={mn:.3f}", transform=plt.gca().transAxes, ha='right', va='top', fontsize=10)
    plt.show()
