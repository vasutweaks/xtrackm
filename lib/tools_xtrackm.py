import functools
from joblib import Parallel, delayed
import glob
import time
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry as sg
import statsmodels.api as sm
import xarray as xr
import xskillscore as xs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from dateutil.relativedelta import relativedelta
from geopy import distance
from loess.loess_1d import loess_1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from shapely.geometry import Point, Polygon
from namesm import *
import math


def read_drifter_data(file_path):
    # Define column names
    col_names = [
        "id",
        "mm",
        "dd",
        "yy",
        "lat",
        "lon",
        "temp",
        "ve",
        "vn",
        "spd",
        "var_lat",
        "var_lon",
        "var_tmp",
    ]
    # Read the file into a pandas dataframe
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        header=None,
        names=col_names,
        na_values=999.999,
    )
    # Separate the integer and fractional parts of the day
    df["day_fraction"], df["dd"] = np.modf(df["dd"])
    # Convert the integer part of the day to integer type
    df["dd"] = df["dd"].astype(int)
    # Calculate the hours from the fractional part of the day
    df["hours"] = (df["day_fraction"] * 24).round().astype(int)
    # Combine the date and time components into a single datetime column
    df["datetime1"] = pd.to_datetime(df[["yy", "mm", "dd"]].astype(str).agg(
        "-".join, axis=1)) + pd.to_timedelta(df["hours"], unit="h")
    # Drop the intermediary columns used for calculations
    df = df.drop(columns=["mm", "dd", "yy", "day_fraction", "hours"])
    df.set_index("datetime1", inplace=True)
    return df


def read_drifter_data_simple(fname):
    """Read NOAA Global Drifter Program text data into pandas format.
    Parameters
    ----------
    fname : str
    Returns
    -------
    df : datafram
    """
    def _parse_date(mm, dd, yy):
        dd = float(dd)
        mm = int(mm)
        yy = int(yy)
        day = int(dd)
        hour = int(24 * (dd - day))
        return datetime(yy, mm, day, hour)
    return pd.read_csv(
        fname,
        names=col_names,
        sep=r"\s+",
        header=None,
        na_values=999.999,
        parse_dates={"time": [1, 2, 3]},
        date_parser=_parse_date,
    )


def drifter_time_asn(ds_d, var_str="ve"):
    time1 = ds_d.time.isel(traj=0).values
    var1 = ds_d[var_str].isel(traj=0).values
    var_asn = xr.DataArray(var1,
                           coords=[time1],
                           dims=["time"],
                           attrs=ds_d[var_str].attrs)
    var_asn = var_asn.resample(time="1D").mean()
    var_asn.name = var_str
    return var_asn


def geostrophic_current_components(g1, g2, m1, m2):
    """
    Calculates the zonal (u) and meridional (v) components and the speed of
    the geostrophic current.
    Parameters:
    - g1: Cross-track geostrophic current along track T1 (float)
    - g2: Cross-track geostrophic current along track T2 (float)
    - m1: Slope of track T1 (float)
    - m2: Slope of track T2 (float)
    Returns:
    - u: Zonal component of the geostrophic current (float)
    - v: Meridional component of the geostrophic current (float)
    - speed: Speed (magnitude) of the geostrophic current (float)
    """
    # Convert slopes to angles in radians
    theta1 = math.atan(m1)
    theta2 = math.atan(m2)
    print(theta1, theta2)
    sin_theta1 = math.sin(theta1)
    cos_theta1 = math.cos(theta1)
    sin_theta2 = math.sin(theta2)
    cos_theta2 = math.cos(theta2)
    sin_theta_diff = math.sin(theta2 - theta1)
    if sin_theta_diff == 0:
        raise ValueError("The tracks are parallel or the angle difference is \
                         zero, cannot compute components.")
    u = (cos_theta2 * g1 - cos_theta1 * g2.interp(time=g1.time)) / sin_theta_diff
    v = (sin_theta2 * g1 - sin_theta1 * g2.interp(time=g1.time)) / sin_theta_diff
    speed = (u**2 + v**2)**0.5
    u1 = u
    v1 = v
    speed1 = speed
    return u1, v1, speed1


def angle_between_lines(m1, m2):
    angle_r = np.arctan(np.abs((m2 - m1) / (1 + m1*m2))) 
    # angle_d = np.rad2deg(angle_r)
    return angle_r


def geostrophic_current_components_1(ga, gd, meridian_angle_r):
    """
    Calculates the zonal (u) and meridional (v) components and the speed
    of the geostrophic current.
    Parameters:
    - g1: Cross-track geostrophic current along track T1 (float)
    - g2: Cross-track geostrophic current along track T2 (float)
    - angle_inter: Acute angle of intersection between track T1 and T2
    Returns:
    - u: Zonal component of the geostrophic current (float)
    - v: Meridional component of the geostrophic current (float)
    - speed: Speed (magnitude) of the geostrophic current (float)
    """
    # Convert slopes to angles in radians
    sin_theta = math.sin(meridian_angle_r)
    cos_theta = math.cos(meridian_angle_r)
    u = (gd + ga.interp(time=gd.time)) / 2*cos_theta
    v = (gd - ga.interp(time=gd.time)) / 2*sin_theta
    speed = (u**2 + v**2)**0.5
    u1 = u
    v1 = v
    speed1 = speed
    return u1, v1, speed1


def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Designs and applies a Butterworth low-pass filter.
    Parameters:
    - data: 1D numpy array, the time series data.
    - cutoff: float, the cutoff frequency in cycles per day.
    - fs: float, the sampling frequency in samples per day.
    - order: int, the order of the filter.
    Returns:
    - y: 1D numpy array, the filtered data.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_2d(da, cutoff_days=3, order=4):
    """
    Applies a Butterworth low-pass filter along the 'time' dimension.
    Parameters:
    - da: xarray DataArray, input data with dimensions ('x', 'time').
    - cutoff_days: float, the cutoff period in days.
    - order: int, the order of the Butterworth filter.
    Returns:
    - filtered_da: xarray DataArray, the filtered data.
    """
    fs = 1  # Sampling frequency: 1 sample per day
    cutoff = 1 / cutoff_days  # Cutoff frequency in cycles per day
    def filter_func(ts):
        """
        Filters a 1D time series, handling missing values.
        Parameters:
        - ts: 1D numpy array, the time series data.
        Returns:
        - y: 1D numpy array, the filtered time series.
        """
        # Find indices where data is not NaN
        not_nan = ~np.isnan(ts)
        if np.sum(not_nan) < 2:
            # Not enough data to interpolate
            return ts
        else:
            # Interpolate missing values
            x = np.arange(len(ts))
            ts_interp = np.interp(x, x[not_nan], ts[not_nan])
            # Apply the Butterworth filter
            y = butter_lowpass_filter(ts_interp, cutoff, fs, order)
            # Restore NaN values in their original positions
            y[~not_nan] = np.nan
            return y
    # Apply the filter function along the 'time' dimension
    filtered_da = xr.apply_ufunc(
        filter_func,
        da,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',  # If using Dask for parallel computation
        output_dtypes=[da.dtype]
    )
    return filtered_da


def butter_lowpass_1d(da, cutoff_days=3, order=4):
    """
    Applies a Butterworth low-pass filter to a 1D xarray DataArray along the
    'time' dimension,
    smoothing out signals with periods shorter than the specified cutoff.
    Parameters:
    - da: xarray DataArray with dimension 'time'.
    - cutoff_days: float, the cutoff period in days (default is 3 days).
    - order: int, the order of the Butterworth filter (default is 4).
    Returns:
    - filtered_da: xarray DataArray, the filtered data.
    """
    fs = 1  # Sampling frequency: 1 sample per day
    cutoff = 1 / cutoff_days  # Cutoff frequency in cycles per day
    # Extract the data as a numpy array
    ts = da.values
    # Identify indices with valid (non-NaN) data
    not_nan = ~np.isnan(ts)
    if np.sum(not_nan) < 2:
        # Not enough data to perform filtering
        filtered_values = ts.copy()
    else:
        # Interpolate missing values for filtering
        x = np.arange(len(ts))
        ts_interp = np.interp(x, x[not_nan], ts[not_nan])
        # Design the Butterworth filter
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq  # Normalized cutoff frequency
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        # Apply the filter using filtfilt for zero-phase filtering
        # y = filtfilt(b, a, ts_interp)
        if len(ts_interp) <= 15:
            y = ts_interp.copy()  # Skip filtering or handle appropriately
        else:
            y = filtfilt(b, a, ts_interp)
        # Restore NaN values at their original positions
        y[~not_nan] = np.nan
        filtered_values = y
    # Create a new DataArray with the filtered values, preserving the original coordinates and attributes
    filtered_da = xr.DataArray(filtered_values, coords=da.coords, dims=da.dims, attrs=da.attrs)
    return filtered_da


def outer_labels_k(m, n, k, ax, xticks1=[], yticks1=[]):
    """
    In multi m x n subplots puts y axis labels and ticks only along left side of
    leftmost vertical columns and puts x axis labels
    and ticks only along bottom side of bottommost horizontal rows
    """
    i1 = k // n # row number
    j1 = k % n # column number
    # all other top right subplots
    if i1 < m - 1 and j1 > 0:
        print(k, i1,j1,"some top right subplot")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        return False, False
    # left most column
    elif j1 == 0 and i1 != m - 1:
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_yticks(yticks1)
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        print(k, i1,j1,"leftmost but not corner bottom left subplot")
        return False, True
    # bottom most row
    elif i1 == m - 1 and j1 != 0:
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xticks(xticks1)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        print(k, i1,j1,"bottommost but not corner bottom left subplot")
        return True, False
        # p,"hor")
    # single bottom left subplots
    # only subplot to have all ticks and labels
    else:
        print(k, i1,j1,"corner bottom left subplot")
        ax.set_xticks(xticks1)
        ax.set_yticks(yticks1)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        return True, True


def outer_labels_k_bool(m, n, k):
    """
    In multi m x n subplots puts y axis labels and ticks only along left side of
    leftmost vertical columns and puts x axis labels
    and ticks only along bottom side of bottommost horizontal rows
    """
    i1 = k // n # row number
    j1 = k % n # column number
    print(f"{m}x{n}  {k}th suplot")
    # all other top right subplots
    if i1 < m - 1 and j1 > 0:
        print(k, i1, j1, "some top right subplot")
        return False, False
    # left most column
    elif j1 == 0 and i1 != m - 1:
        print(k, i1, j1, "leftmost but not corner bottom left subplot")
        return False, True
    # bottom most row
    elif i1 == m - 1 and j1 != 0:
        print(k, i1, j1, "bottommost but not corner bottom left subplot")
        return True, False
    # single bottom left subplots
    # only subplot to have all ticks and labels
    else:
        print(k, i1, j1, "corner bottom left subplot")
        return True, True


def print_stats(da):
    """
    Prints comprehensive statistics for a given xarray.DataArray.
    Args:
        da (xarray.DataArray): The DataArray to analyze.
    Returns:
        None
    """
    # Handle missing values
    valid_data = da  # .dropna(dim='time')
    # Calculate statistics
    min_val = valid_data.min().item()
    max_val = valid_data.max().item()
    mean_val = valid_data.mean().item()
    std_val = valid_data.std().item()
    num_valid_points = valid_data.count().item()
    # Print formatted results
    print(f"DataArray statistics:")
    print(f"  * Minimum: {min_val}")
    print(f"  * Maximum: {max_val}")
    print(f"  * Mean: {mean_val}")
    print(f"  * Standard deviation: {std_val}")
    print(f"  * Number of valid points: {num_valid_points}")


def is_monotonic(arr):
    return np.all(np.diff(arr) >= 0)


def is_ocean(lon, lat):
    land = gpd.read_file("~/allData/topo/ne_10m_land.shp")
    point = Point(lon, lat)
    return not any(land.contains(point))


def is_land(lon, lat):
    land = gpd.read_file("~/allData/topo/ne_10m_land.shp")
    point = Point(lon, lat)
    return any(land.contains(point))


def is_within_region(lon, lat, lon_min, lon_max, lat_min, lat_max):
    return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)


# write a function to check if all points described by lons and lat
# are within the region defined by lon_min, lon_max, lat_min, lat_max
def is_within_region_arr(lons, lats, lon_min, lon_max, lat_min, lat_max):
    lons_within_range = np.logical_and(lons >= lon_min, lons <= lon_max)
    # Check if all latitudes are within the range
    lats_within_range = np.logical_and(lats >= lat_min, lats <= lat_max)
    # Combine both conditions
    all_within_region = np.logical_and(lons_within_range, lats_within_range)
    # Check if all points are within the region
    if np.all(all_within_region):
        print("All points are within the region.")
        return True
    else:
        print("Not all points are within the region.")
        return False


# def is_within_region_arr(lons, lats, lon_min, lon_max, lat_min, lat_max):
#     lons_within_range = np.logical_and(lons >= lon_min, lons <= lon_max)
#     # Check if all latitudes are within the range
#     lats_within_range = np.logical_and(lats >= lat_min, lats <= lat_max)
#     # Combine both conditions
#     all_within_region = np.logical_and(lons_within_range, lats_within_range)
#     # Check if all points are within the region
#     if np.all(all_within_region):
#         print("All points are within the region.")
#         return True
#     else:
#         print("Not all points are within the region.")
#         return False
#

def timeit_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Execution time for {func.__name__}: {end_time - start_time} seconds"
        )
        return result

    return wrapper


# CARTESIAN GEOMETRY FUNCTIONS
def get_point_at_distance(x1, y1, x2, y2, d):
    # point on line (x1, y1) to (x2, y2) at distance d
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x = x1 + (d / dist) * (x2 - x1)
    y = y1 + (d / dist) * (y2 - y1)
    return x, y


def filter_between(valc, vale, buoy, val1, val2):
    """
    valc is control
    vale is experimental for comparison
    """
    cond = valc.between(val1, val2) & vale.between(val1, val2)
    valc1 = valc[cond]
    vale1 = vale[cond]
    buoy1 = buoy[cond]
    return valc1, vale1, buoy1


def index_at_x(x, x1):
    i1 = np.abs(x - x1).argmin()
    return i1


def lonlat_at_x(ds, x, x1):
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    i1 = index_at_x(x, x1)
    lon1 = lons_track_rev[i1]
    lat1 = lats_track_rev[i1]
    return lon1, lat1


def index_at_lon(ds, lon1):
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    i1 = np.abs(lons_track_rev - lon1).argmin()
    return i1


def index_at_lat(ds, lat1):
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    i1 = np.abs(lats_track_rev - lat1).argmin()
    return i1


def sel_along_lon(ds, da, lon1, lon2):
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    i1 = np.abs(lons_track_rev - lon1).argmin()
    i2 = np.abs(lons_track_rev - lon2).argmin()
    da_out = da.isel(x=slice(i1, i2 + 1))
    return da_out


def sel_along_lat(ds, da, lat1, lat2):
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    i1 = np.abs(lats_track_rev - lat1).argmin()
    i2 = np.abs(lats_track_rev - lat2).argmin()
    da_out = da.isel(x=slice(i1, i2 + 1))
    return da_out


def rename_xax(ds):
    for d in ds.dims:
        if "XAX" in d:
            xdim = d
    ds = ds.rename({xdim: "x"})
    return ds


def flip_n_asn(ds, da):
    x = ds.x.values
    times = da.time.values
    da_rev = da.isel(x=slice(None, None, -1)).values
    da_asn = xr.DataArray(da_rev, coords={"time": times, "x": x})
    return da_asn


def to_datetime(date):
    timestamp = (date - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(
        1, "s"
    )
    return datetime.utcfromtimestamp(timestamp)


def n64todatetime(d):
    if np.isnat(d):
        return None
    return datetime.strptime(
        np.datetime_as_string(d, unit="s"), "%Y-%m-%dT%H:%M:%S"
    )


def append_metrics(ts1, ts2, df, sat, psmsl_id, track_no, lon1, lat1, jj, mindist):
    # print(df)
    std1 = ts1.std(dim="time").item()
    std2 = ts2.std(dim="time").item()
    bias = ts1 - ts2
    ln = len(bias)
    bias1 = bias.mean(dim="time").item()
    corr = xs.pearson_r(ts1, ts2, dim="time", skipna=True)
    corr1 = corr.item()
    p = xs.pearson_r_p_value(ts1, ts2, dim="time", skipna=True)
    p1 = p.item()
    rmse = xs.rmse(ts1, ts2, dim="time", skipna=True)
    rmse1 = rmse.item()
    vc_p = 100 * (bias.count(dim="time").item()) / ln
    row = [sat, psmsl_id, track_no, lon1, lat1, bias1, corr1, rmse1, p1, vc_p, mindist, std1, std2]
    df.loc[jj] = row
    # print(sat, psmsl_id, track_no, lon1, lat1, bias1, corr1, rmse1, p1, vc_p)
    return df


def append_metrics_s(ts1, ts2, df, sat, psmsl_id, track_no, lon1, lat1, jj):
    # print(df)
    bias = ts1 - ts2
    ln = len(bias)
    bias1 = bias.mean(dim="time").item()
    std1 = ts1.std(dim="time").item()
    std2 = ts2.std(dim="time").item()
    corr = xs.pearson_r(ts1, ts2, dim="time", skipna=True)
    corr1 = corr.item()
    p = xs.pearson_r_p_value(ts1, ts2, dim="time", skipna=True)
    p1 = p.item()
    rmse = xs.rmse(ts1, ts2, dim="time", skipna=True)
    rmse1 = rmse.item()
    vc_p = 100 * (bias.count(dim="time").item()) / ln
    row = [
        sat,
        psmsl_id,
        track_no,
        lon1,
        lat1,
        bias1,
        corr1,
        rmse1,
        p1,
        vc_p,
        std1,
        std2,
    ]
    df.loc[jj] = row
    return df


def overlap_dates(t1_start, t1_end, t2_start, t2_end):
    # Check for overlap
    if t1_start is None or t1_end is None or t2_start is None or t2_end is None:
        return None, None  # Handle the case where None is found
    if t1_end < t2_start or t1_start > t2_end:
        # No overlap, the periods are disjoint
        overlap_start = None
        overlap_end = None
    else:
        # Overlap exists, calculate the overlap
        overlap_start = max(t1_start, t2_start)
        overlap_end = min(t1_end, t2_end)
    # Now, the overlap_start and overlap_end represent the overlapping time period, if any.
    return overlap_start, overlap_end


# GEOSTROPHIC FUNCTIONS
def earth_radius(lat):
    # https://en.wikipedia.org/wiki/Earth_radius
    """
    Calculate the Earth's radius at a given latitude based on the WGS-84 ellipsoid model
    Parameters:
    lat (float): Latitude in degrees.
    Returns:
    float: Earth's radius at the given latitude in kilometers.
    """
    # Convert latitude from degrees to radians
    lat_rad = np.radians(lat)
    # WGS-84 ellipsoid parameters
    R_e = 6378.137  # Equatorial radius in kilometers
    R_p = 6356.752  # Polar radius in kilometers
    # Calculate Earth's radius at the given latitude
    numerator = (R_e**2 * np.cos(lat_rad)) ** 2 + (
        R_p**2 * np.sin(lat_rad)
    ) ** 2
    denominator = (R_e * np.cos(lat_rad)) ** 2 + (R_p * np.sin(lat_rad)) ** 2
    return np.sqrt(numerator / denominator)


def degrees2kilometers(degrees, lat):
    """
    :type degrees: float
    :param degrees: Distance in (great circle) degrees
    :type radius: float, optional
    :param radius: Radius of the Earth used for the calculation.
    :rtype: float
    :return: Distance in kilometers as a floating point number.
    """
    radius = earth_radius(lat)
    return degrees * (2.0 * radius * np.pi / 360.0)


def cor_p(lat):
    # write a test case for this function
    """
    Coriolis parameter in 1/s for latitude in degrees.
    """
    omega = 7.292115e-5  # (1/s)   (Groten, 2004).
    f = 2 * omega * np.sin(np.radians(lat))
    da = xr.DataArray(f, coords={"lat": lat}, dims="lat")
    return da


def grav(lat):
    X = np.sin(lat * (np.pi) / 180.0)
    sin2 = X**2
    grav = 9.780327 * (1.0 + (5.2792e-3 + (2.32e-5 * sin2)) * sin2)
    # international gravity formula
    # https://en.wikipedia.org/wiki/Gravity_of_Earth#Latitude
    da = xr.DataArray(grav, coords={"lat": lat}, dims="lat")
    return da


def compute_dist(ds, units="m"):
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lon_coast = lons_track[-1]  # this on coast
    lat_coast = lats_track[-1]  # this on coast
    if units == "km":
        dists = [
            distance.distance((y, x), (lat_coast, lon_coast)).km
            for x, y in zip(lons_track, lats_track)
        ]
    elif units == "m":
        dists = [
            distance.distance((y, x), (lat_coast, lon_coast)).m
            for x, y in zip(lons_track, lats_track)
        ]
    return dists


def compute_geostrophy(ds, sla_smooth, current="zonal"):
    lons_track_rev = ds.lon.values[::-1]
    lats_track_rev = ds.lat.values[::-1]
    if len(lons_track_rev) == 0:
        return
    m = (lats_track_rev[-1] - lats_track_rev[0]) / (
        lons_track_rev[-1] - lons_track_rev[0]
    )
    angle_r = np.arctan(m)
    angle = np.rad2deg(angle_r)
    f1 = cor_p(lats_track_rev)
    f2 = f1.values
    f3 = np.where(lats_track_rev > 2, f2, np.nan)
    # 2s-2n band equatorial band is masked according to Sudre 2013
    g = grav(lats_track_rev)  # ms-1
    g1 = g.values
    ratio = g1 / f3
    ratio1 = ratio[:, np.newaxis]
    dhbydx = sla_smooth.differentiate("x")
    gc = -1 * ratio1 * dhbydx
    if current == "zonal":
        gu = -1 * gc * np.cos(np.deg2rad(90.0 - abs(angle)))
        return gu
    elif current == "meridional":
        gu = gc * np.sin(np.deg2rad(90.0 - abs(angle)))
        return gu
    elif current == "speed":
        guu = -1 * gc * np.cos(np.deg2rad(90.0 - abs(angle)))
        guv = gc * np.sin(np.deg2rad(90.0 - abs(angle)))
        gu = (guu**2 + guv**2) ** 0.5
        return gu


def compute_geostrophy_atan2(ds, sla_smooth, current="zonal"):
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = ds.lon.values[::-1]
    lats_track_rev = ds.lat.values[::-1]
    lon_coast = lons_track[-1]  # this on coast
    lat_coast = lats_track[-1]  # this on coast
    lon_equator = lons_track[0]  # this on equator
    lat_equator = lats_track[0]  # this on equator
    if len(lons_track) == 0:
        return
    angle_r2 = np.arctan2(lat_coast - lat_equator, lon_coast - lon_equator)
    angle2 = np.rad2deg(angle_r2)
    f1 = cor_p(lats_track_rev)
    f1 = f1.where(lats_track_rev > 2, np.nan)
    # 2s-2n band equatorial band is masked according to Sudre 2013
    g1 = grav(lats_track_rev)  # ms-1
    ratio = g1 / f1
    ratio1 = ratio.values
    ratio2 = ratio1[:, np.newaxis]
    dhbydx = sla_smooth.differentiate("x")
    gc = -1 * ratio2 * dhbydx
    if current == "zonal":
        gu = -1 * gc * np.sin(angle_r2)
        return gu
    elif current == "meridional":
        gv = gc * np.cos(angle_r2)
        return gv
    elif current == "speed":
        gu = -1 * gc * np.sin(angle_r2)
        gv = gc * np.cos(angle_r2)
        gs = (gu**2 + gv**2) ** 0.5
        return gs
    elif current == "gc":
        gc = ratio2 * dhbydx
        return gc


def compute_geostrophy_v(ds, sla_smooth, current="meridional"):
    lons_track_rev = ds.lon.values[::-1]
    lats_track_rev = ds.lat.values[::-1]
    if len(lons_track_rev) == 0:
        return
    # Compute the angle using arctan2 for correct sign and quadrant
    delta_lat = lats_track_rev[-1] - lats_track_rev[0]
    delta_lon = lons_track_rev[-1] - lons_track_rev[0]
    angle_r = np.arctan2(delta_lat, delta_lon)  # Angle in radians

    # Compute Coriolis parameter and gravity
    f = cor_p(lats_track_rev).values
    f = np.where(np.abs(lats_track_rev) > 2, f, np.nan)  # Mask equatorial band
    g = grav(lats_track_rev).values
    ratio = (g / f)[:, np.newaxis]

    # Gradient of SLA along the track (s-direction)
    dhbyds = sla_smooth.differentiate("x")  # "x" is the distance along the track

    # Geostrophic current magnitude along the track
    gc = ratio * dhbyds

    # Compute zonal and meridional components using correct trigonometry
    if current == "zonal":
        u = -gc * np.sin(angle_r)
        return u
    elif current == "meridional":
        # v = gc * np.cos(angle_r)
        u = -gc * np.sin(angle_r)
        v = (gc + u*np.sin(angle_r))/np.cos(angle_r)
        return v
    elif current == "speed":
        u = gc * np.sin(angle_r)
        v = -gc * np.cos(angle_r)
        speed = np.sqrt(u**2 + v**2)
        return speed


def compute_geostrophy_gc(ds, sla_smooth):
    lons_track_rev = ds.lon.values[::-1]
    lats_track_rev = ds.lat.values[::-1]
    if len(lons_track_rev) == 0:
        return None
    # Add minimum length check
    if len(lons_track_rev) < 3:
        print("Warning: Insufficient data points for gradient calculation")
        return None
    f1 = cor_p(lats_track_rev)
    f2 = f1.values
    f3 = np.where(lats_track_rev > 2, f2, np.nan)
    # 2s-2n band equatorial band is masked according to Sudre 2013
    g = grav(lats_track_rev)  # ms-1
    g1 = g.values
    ratio = g1 / f3
    ratio1 = ratio[:, np.newaxis]
    dhbydx = sla_smooth.differentiate("x")
    gc = ratio1 * dhbydx
    # Add metadata
    gc.attrs["units"] = "m/s"
    gc.attrs["long_name"] = "Cross-track geostrophic current"
    gc.attrs["description"] = "Geostrophic current perpendicular to satellite track"
    return gc


def geostrophic_components(gc1, gc2, a1_deg, a2_deg):
    """
    Compute zonal and meridional geostrophic current components from 
    cross-track velocities measured by two intersecting satellite tracks.
    Parameters:
    -----------
    gc1 : float or array-like
        Cross-track geostrophic velocity from track 1 (m/s)
    gc2 : float or array-like  
        Cross-track geostrophic velocity from track 2 (m/s)
    a1_deg : float
        Azimuth of track 1 in degrees (measured counter-clockwise from east)
    a2_deg : float
        Azimuth of track 2 in degrees (measured counter-clockwise from east)
    Returns:
    --------
    u : float or array-like
        Zonal (eastward) velocity component (m/s)
    v : float or array-like
        Meridional (northward) velocity component (m/s)
    Raises:
    -------
    ValueError: If tracks are parallel (intersection angle ≈ 0°)
    """
    # Convert azimuths from degrees to radians
    a1 = a1 #math.radians(a1_deg)
    a2 = a2 #math.radians(a2_deg)
    # Calculate intersection angle
    theta = a2 - a1
    sin_theta = math.sin(theta)
    # Check for parallel tracks
    if abs(sin_theta) < 1e-6:
        raise ValueError(f"Tracks are nearly parallel"
                         "(angle = {math.degrees(theta):.2f}°). "
                        "Cannot resolve both velocity components.")
    # Calculate trigonometric values
    cos_a1 = math.cos(a1)
    cos_a2 = math.cos(a2)
    sin_a1 = math.sin(a1)
    sin_a2 = math.sin(a2)
    # Solve for velocity components using the derived formulas:
    # u = (gc1 * cos(a2) - gc2 * cos(a1)) / sin(a2 - a1)
    # v = (gc2 * sin(a1) - gc1 * sin(a2)) / sin(a2 - a1)
    u = (gc1 * cos_a2 - gc2 * cos_a1) / sin_theta
    v = (gc2 * sin_a1 - gc1 * sin_a2) / sin_theta
    return u, v


def geostrophic_components_from_a(gc1, gc2, a1, a2):
    """
    Compute zonal and meridional geostrophic current components from
    cross-track velocities and track slopes of two intersecting satellite tracks.
    Parameters:
    -----------
    gc1 : float or array-like
        Cross-track geostrophic velocity from track 1 (m/s)
    gc2 : float or array-like
        Cross-track geostrophic velocity from track 2 (m/s)
    slope1 : float
        Slope of track 1 (dy/dx = rise/run)
    slope2 : float
        Slope of track 2 (dy/dx = rise/run)
    Returns:
    --------
    u : float or array-like
        Zonal (eastward) velocity component (m/s)
    v : float or array-like
        Meridional (northward) velocity component (m/s)
    Raises:
    -------
    ValueError: If tracks are parallel (same slope)
    """
    # Calculate intersection angle
    theta = a2 - a1
    sin_theta = math.sin(theta)
    # Additional check using sine of intersection angle
    if abs(sin_theta) < 1e-6:
        raise ValueError(
            f"Tracks are nearly parallel (intersection angle = {math.degrees(theta):.2f}°). "
            "Cannot resolve both velocity components.")
    # Calculate trigonometric values
    cos_a1 = math.cos(a1)
    cos_a2 = math.cos(a2)
    sin_a1 = math.sin(a1)
    sin_a2 = math.sin(a2)
    # Solve for velocity components using the derived formulas:
    # u = (gc1 * cos(a2) - gc2 * cos(a1)) / sin(a2 - a1)
    # v = (gc2 * sin(a1) - gc1 * sin(a2)) / sin(a2 - a1)
    u = (gc1 * cos_a2 - gc2 * cos_a1) / sin_theta
    v = (gc2 * sin_a1 - gc1 * sin_a2) / sin_theta
    u = -1 * u
    return u, v


def compute_geostrophy_from_slope(ds, slope_smooth, current="zonal"):
    lons_track_rev = ds.lon.values[::-1]
    lats_track_rev = ds.lat.values[::-1]
    if len(lons_track_rev) == 0:
        return
    m = (lats_track_rev[-1] - lats_track_rev[0]) / (
        lons_track_rev[-1] - lons_track_rev[0]
    )
    angle_r = np.arctan(m)
    angle = np.rad2deg(angle_r)
    f1 = cor_p(lats_track_rev)
    f2 = f1.values
    f3 = np.where(lats_track_rev > 2, f2, np.nan)
    g = grav(lats_track_rev)  # ms-1
    g1 = g.values
    ratio = g1 / f3
    ratio1 = ratio[:, np.newaxis]
    gc = -1 * ratio1 * slope_smooth
    if current == "zonal":
        gu = -1 * gc * np.cos(np.deg2rad(90.0 - abs(angle)))
        return gu
    elif current == "meridional":
        gu = gc * np.sin(np.deg2rad(90.0 - abs(angle)))
        return gu
    elif current == "speed":
        guu = -1 * gc * np.cos(np.deg2rad(90.0 - abs(angle)))
        guv = gc * np.sin(np.deg2rad(90.0 - abs(angle)))
        gu = (guu**2 + guv**2) ** 0.5
        return gu


def compute_ekman(ds_wind):
    uwnd1 = ds_wind.uwnd
    vwnd1 = ds_wind.vwnd
    speed1 = (uwnd1**2.0 + vwnd1**2.0) ** 0.5
    rho_air = Constants.rho_air
    rho_sea = Constants.rho_sea  # kg/m3
    cd = Constants.cd
    Av = Constants.Av
    D = Constants.D
    wstress = cd * rho_air * speed1 * speed1
    wstress_u = cd * rho_air * speed1 * uwnd1
    wstress_v = cd * rho_air * speed1 * vwnd1
    wstress_tm = wstress.mean(dim="time")
    wstress_v_tm = wstress_v.mean(dim="time")
    wstress_u_tm = wstress_u.mean(dim="time")
    lat_w = ds_wind.latitude.values
    g = grav(lat_w)  # ms-1
    f = cor_p(lat_w)
    f1 = f.where((f.lat <= -2) | (f.lat >= 2))
    f2 = f1.values[np.newaxis, :, np.newaxis]
    D1 = np.sqrt(np.abs((2 * Av) / f1)) # # see equation 7.17
    D2 = D1.values[np.newaxis, :, np.newaxis]
    # Citation: Talley L.D., Pickard G.L., Emery W.J., Swift J.H., 2011. Descriptive Physical Oceanography: An Introduction (Sixth Edition), Elsevier, Boston, 560 pp.
    ekman_u = wstress_v / (f2 * rho_sea * D2)  # see equation 7.19a and 7.19b
    ekman_v = -1 * wstress_u / (f2 * rho_sea * D2)
    return ekman_u, ekman_v



# Function to calculate u_e and v_e
def compute_ekman_currents_wenbo_wang(ds_wind, lat):
    rho = 1.02 * 10**3  # kg/m^3 (water density)
    r = 2.15 * 10**-4   # m/s (friction coefficient)
    h_md = 32.5         # m (mixed layer depth)
# Function to calculate B and theta
    def compute_B_theta(lat):
        """Calculate B and theta for a given latitude."""
        f = cor_p(lat)
        B = 1 / (rho * np.sqrt(r**2 + (f * h_md)**2))
        theta = np.arctan(f * h_md / r)
        return B, theta
    """Calculate east (u_e) and north (v_e) components of Ekman currents."""
    uwnd1 = ds_wind.uwnd
    vwnd1 = ds_wind.vwnd
    speed1 = (uwnd1**2.0 + vwnd1**2.0) ** 0.5
    rho_air = Constants.rho_air
    rho_sea = Constants.rho_sea  # kg/m3
    cd = Constants.cd
    Av = Constants.Av
    D = Constants.D
    wstress = cd * rho_air * speed1 * speed1
    wstress_u = cd * rho_air * speed1 * uwnd1
    wstress_v = cd * rho_air * speed1 * vwnd1
    B, theta = compute_B_theta(lat)
    tau_complex = tau_x + 1j * tau_y  # tau_x + i*tau_y
    ue_ve = B * np.exp(1j * theta) * tau_complex
    return np.real(ue_ve), np.imag(ue_ve)


# SMOOTHING FUNCTIONS
def genweights(p, q, dt=1):
    """
    Given p and q, return the vector of cpn's: the optimal weighting
    coefficients and the noise reduction factor of h.
    p: Number of points before the point of interest (can be negative)
    q: Number of points after the point of interest (can be negative)
    dt: Sampling period (defaults to 1 s)
    Returns:
        tuple: (cn, noise_reduction_factor)
    """
    # Do some verification (if needed, allow negative p and q)
    if p > 0 or q < 0:
        raise ValueError(
            "p is supposed to be negative and q is supposed to be positive"
        )
    # Build the matrices
    N = abs(p) + abs(q)
    # T is the window length
    T = N + 1
    A = np.zeros((T, T))
    ones = np.ones(N)  # Assign ones to the last row
    # last row is made ones except for the last element
    A[T - 1, :] = np.append(ones, 0)  # and last elements is 0
    n = np.arange(p, q + 1)
    n = n[n != 0]  # Remove the zero
    # make a numpy array from list n
    n = np.array(n, dtype=float)
    for i in range(len(n)):
        # from the equation (11) in the paper Powell and Leben (2004)
        # right side term
        first_term = (-n[i] / 2) * np.reciprocal(n)
        # left side term
        second_term = (n[i] ** 2) * (dt**2) / 4
        A[i, :] = np.append(first_term, second_term)
        A[i, i] = -1
    B = np.zeros(T)
    B[-1] = 1  # Set the last element of B to 1
    # Compute the coefficients
    cn = np.linalg.solve(A, B)
    cn = cn[:-1]  # Remove the last element
    # Compute the error (noise reduction factor)
    # from the equation (8) in the paper Powell and Leben (2004)
    summ = np.sum(cn / (n * dt))
    squared = (cn / (n * dt)) ** 2
    noise_reduction_factor = np.sqrt(np.sum(squared) + summ**2)
    return cn, noise_reduction_factor


@timeit_decorator
def track_smooth_box_x(da, smooth=1, min_periods=1):
    var1 = da.rolling(
        x=smooth,
        center=True,
        min_periods=min_periods,
    ).mean()
    return var1


@timeit_decorator
def track_smooth_gaussian_x(da, sigma=3):
    # vals = np.empty((len(dists_rev), len(times)))
    dists_rev = da.x.values
    x_size = da.x.size
    t_size = da.time.size
    vals = np.empty((x_size, t_size))
    vals[:] = np.nan
    if t_size == 1:
        times = [da.time.item()]
    else:
        times = da.time.values
    for l in range(t_size):
        if t_size == 1:
            da1 = da.values
        else:
            da1 = da.isel(time=l).values
        da1_f = gaussian_filter1d(da1, sigma=sigma)
        vals[:, l] = da1_f
    var1 = xr.DataArray(
        vals,
        coords=[dists_rev, times],
        dims=["x", "time"],
    )
    return var1


@timeit_decorator
def slope_smooth_optimal_filter_x(da, p=-5, q=6):
    window_len = abs(p) + abs(q)
    dists_rev = da.x.values
    x_size = da.x.size
    if x_size < window_len:
        print(f"returning raw data ---------------------")
        return da
    t_size = da.time.size
    vals = np.empty((x_size, t_size))
    of_weights, error = genweights(p, q, dt=1)
    vals[:] = np.nan
    if t_size == 1:
        times = [da.time.item()]
    else:
        times = da.time.values
    for l in range(t_size):
        if t_size == 1:
            da1 = da.values
        else:
            da1 = da.isel(time=l).values
        da1_f = np.convolve(da1, of_weights, mode="same")
        vals[:, l] = da1_f
    var1 = xr.DataArray(
        vals,
        coords=[dists_rev, times],
        dims=["x", "time"],
    )
    return var1


@timeit_decorator
def track_smooth_loess_x(da, frac=0.2):
    dists_rev = da.x.values
    x_size = da.x.size
    t_size = da.time.size
    if t_size == 1:
        times = [da.time.item()]
    else:
        times = da.time.values
    da_f = da.interpolate_na(
        dim="x", method="linear", fill_value="extrapolate"
    )
    vals = np.empty((x_size, t_size))
    # vals = np.empty((len(dists_rev), len(times)))
    vals[:] = np.nan
    for l in range(t_size):
        if t_size == 1:
            da1 = da_f.values
        else:
            da1 = da_f.isel(time=l).values
        xout, yout, wout = loess_1d(
            dists_rev,
            da1,
            xnew=dists_rev,
            degree=1,
            frac=frac,
            npoints=None,
            rotate=False,
            sigy=None,
        )
        vals[:, l] = yout
    # print(vals.shape)
    var1 = xr.DataArray(
        vals,
        coords=[dists_rev, times],
        dims=["x", "time"],
    )
    return var1


@timeit_decorator
# @functools.lru_cache(maxsize=128)
# @njit
# @jit(nopython=True)
def track_smooth_loess_statsmodels_x(da, frac=0.2):
    """
    Smooths along-track sea level anomaly data using LOESS.

    Parameters:
    da (xr.DataArray): Input data array with dimensions 'x' and 'time'.
    frac (float, optional): Fraction of data points to use for smoothing. Defaults to 0.02.

    Returns:
    xr.DataArray: Smoothed data array.
    """
    dists_rev = da.x.values
    x_size = da.x.size
    t_size = da.time.size
    if t_size == 1:
        times = [da.time.item()]
    else:
        times = da.time.values
    da_f = da.interpolate_na(
        dim="x", method="linear", fill_value="extrapolate"
    )
    vals = np.empty((x_size, t_size))
    vals[:] = np.nan
    for l in range(t_size):
        if t_size == 1:
            da1 = da_f.values
        else:
            da1 = da_f.isel(time=l).values
        yout = sm.nonparametric.lowess(
            exog=dists_rev, endog=da1, frac=frac, return_sorted=False
        )
        vals[:, l] = yout
    var1 = xr.DataArray(
        vals,
        coords=[dists_rev, times],
        dims=["x", "time"],
        attrs=da.attrs
    )
    return var1


@timeit_decorator
def track_smooth_loess_statsmodels_x_parallel(da, frac=0.2, n_jobs=-1):
    """
    Smooths along-track sea level anomaly data using LOESS.

    Parameters:
    da (xr.DataArray): Input data array with dimensions 'x' and 'time'.
    frac (float, optional): Fraction of data points to use for smoothing. Defaults to 0.02.

    Returns:
    xr.DataArray: Smoothed data array.
    """
    def loess_smoothing(dists_rev, da1, frac):
        """Apply LOESS smoothing to a 1D array."""
        return sm.nonparametric.lowess(
            exog=dists_rev, endog=da1, frac=frac, return_sorted=False
        )
    dists_rev = da.x.values
    x_size = da.x.size
    t_size = da.time.size
    if t_size == 1:
        times = [da.time.item()]
    else:
        times = da.time.values
    da_f = da.interpolate_na(
        dim="x", method="linear", fill_value="extrapolate"
    )
    vals = np.empty((x_size, t_size))
    vals[:] = np.nan

    def process_time_step(l):
        if t_size == 1:
            da1 = da_f.values
        else:
            da1 = da_f.isel(time=l).values
        return loess_smoothing(dists_rev, da1, frac)

    # Parallel processing using joblib
    smoothed_results = Parallel(n_jobs=n_jobs)(
        delayed(process_time_step)(l) for l in range(t_size)
    )

    # Combine the results back into the vals array
    for l, yout in enumerate(smoothed_results):
        vals[:, l] = yout

    var1 = xr.DataArray(
        vals,
        coords=[dists_rev, times],
        dims=["x", "time"],
        attrs=da.attrs
    )
    return var1


def track_dist_time_asn(ds, var_str="sla", units_in="m"):
    # dists = compute_dist(ds, units="km")  # to convert to meters
    dists = compute_dist(ds, units=units_in)  # to convert to meters
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    dt0 = datetime.strptime("1950-01-01", "%Y-%m-%d")
    # these dists start from equator
    dists_rev = dists[::-1]
    # these dists start from coast
    ln = ds[var_str].sizes["points_numbers"]
    ln2 = ln // 2
    dvals = ds.time.isel(points_numbers=ln2).values
    try:
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    except ValueError:
        # dvals = pd.Series(dvals).ffill().bfill()
        dvals = pd.Series(dvals).ffill()
        dvals = pd.Series(dvals).bfill()
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    var1 = ds[var_str]
    # now sla values oriented from coast
    var1_rev = var1.isel(points_numbers=slice(None, None, -1)).values
    var2 = xr.DataArray(
        var1_rev,
        coords=[dists_rev, dates],
        dims=["x", "time"],
    )
    if units_in == "m":
        var2.coords["x"].attrs["units"] = "m"
    elif units_in == "km":
        var2.coords["x"].attrs["units"] = "km"
    return var2


def virtual_dist_time_asn(ds):
    # dists = compute_dist(ds, units="km")  # to convert to meters
    dists = ds.distance_to_coast
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    dt0 = datetime.strptime("1950-01-01", "%Y-%m-%d")
    # these dists start from equator
    dists_rev = dists[::-1]
    # these dists start from coast
    ln = ds[var_str].sizes["points_numbers"]
    ln2 = ln // 2
    dvals = ds.time.isel(points_numbers=ln2).values
    try:
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    except ValueError:
        # dvals = pd.Series(dvals).ffill().bfill()
        dvals = pd.Series(dvals).ffill()
        dvals = pd.Series(dvals).bfill()
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    var1 = ds[var_str]
    # now sla values oriented from coast
    var1_rev = var1.isel(points_numbers=slice(None, None, -1)).values
    var2 = xr.DataArray(
        var1_rev,
        coords=[dists_rev, dates],
        dims=["x", "time"],
    )
    if units_in == "m":
        var2.coords["x"].attrs["units"] = "m"
    elif units_in == "km":
        var2.coords["x"].attrs["units"] = "km"
    return var2


def track_dist_time_asn_gshhg(ds, var_str="sla", units_in="m"):
    dt0 = datetime.strptime("1950-01-01", "%Y-%m-%d")
    # these dists start from equator
    if units_in == "m":
        dists = ds.dist_to_coast_gshhg.values
    elif units_in == "km":
        dists = 0.001 * ds.dist_to_coast_gshhg.values
    dists_rev = dists[::-1]
    # these dists start from coast
    ln = ds[var_str].sizes["points_numbers"]
    ln2 = ln // 2
    dvals = ds.time.isel(points_numbers=ln2).values
    try:
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    except ValueError:
        # dvals = pd.Series(dvals).ffill().bfill()
        dvals = pd.Series(dvals).ffill()
        dvals = pd.Series(dvals).bfill()
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    var1 = ds[var_str]
    # now sla values oriented from coast
    var1_rev = var1.isel(points_numbers=slice(None, None, -1)).values
    var2 = xr.DataArray(
        var1_rev,
        coords=[dists_rev, dates],
        dims=["x", "time"],
    )
    if units_in == "m":
        var2.coords["x"].attrs["units"] = "m"
    elif units_in == "km":
        var2.coords["x"].attrs["units"] = "km"
    return var2


def track_dist_time_asn_midx(ds, var_str="sla", units_in="m"):
    # dists = compute_dist(ds, units="km")  # to convert to meters
    dists = compute_dist(ds, units=units_in)  # to convert to meters
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    dt0 = datetime.strptime("1950-01-01", "%Y-%m-%d")
    # these dists start from equator
    dists_rev = dists[::-1]
    # these dists start from coast
    ln = ds[var_str].sizes["points_numbers"]
    ln2 = ln // 2
    dvals = ds.time.isel(points_numbers=ln2).values
    try:
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    except ValueError:
        dvals = pd.Series(dvals).ffill().bfill()
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    var1 = ds[var_str]
    # now sla values oriented from coast
    var1_rev = var1.isel(points_numbers=slice(None, None, -1)).values
    var2 = xr.DataArray(
        var1_rev,
        dims=["x", "time"],
        coords={"x":dists_rev, "time":dates, "lon":("x", lons_track_rev), "lat":("x", lats_track_rev)}
    )
    if units_in == "m":
        var2.coords["x"].attrs["units"] = "m"
    elif units_in == "km":
        var2.coords["x"].attrs["units"] = "km"
    return var2


def track_dist_time_asn_mssh(ds, var_str="mssh", units_in="m"):
    dists = compute_dist(ds, units=units_in)  # to convert to meters
    dists_rev = dists[::-1]
    var1 = ds[var_str]  # .count(dim="cycles_numbers")
    var1_rev = var1.isel(points_numbers=slice(None, None, -1)).values
    var2 = xr.DataArray(
        var1_rev,
        coords=[dists_rev],
        dims=["x"],
    )
    if units_in == "m":
        var2.coords["x"].attrs["units"] = "m"
    elif units_in == "km":
        var2.coords["x"].attrs["units"] = "km"
    return var2


def track_dist_time_asn_mdt(ds, var_str="mdt_cnes_cls22", units_in="m"):
    dists = compute_dist(ds, units=units_in)  # to convert to meters
    dists_rev = dists[::-1]
    var1 = ds[var_str]  # .count(dim="cycles_numbers")
    var1_rev = var1.isel(points_numbers=slice(None, None, -1)).values
    var2 = xr.DataArray(
        var1_rev,
        coords=[dists_rev],
        dims=["x"],
    )
    if units_in == "m":
        var2.coords["x"].attrs["units"] = "m"
    elif units_in == "km":
        var2.coords["x"].attrs["units"] = "km"
    return var2


# META INFO FUNCTIONS
def get_first_file(sat):
    return sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    )[0]


def get_first_file_a(sat):
    return sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/extracted_aviso/aviso.sla.ref.{sat}.nindian.*.nc"
        )
    )[0]


def get_total_tracks(sat):
    return len(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    )


def get_time_limits(sat):
    f = get_first_file(sat)
    ds = xr.open_dataset(f, engine="h5netcdf")
    # ds = xr.open_dataset(f)
    time = ds.time.isel(points_numbers=0).values
    first_time = time[0]
    last_time = time[-1]
    # Convert the first and last time stamps to string format
    first_time_str = first_time.strftime("%Y-%m-%d")
    last_time_str = last_time.strftime("%Y-%m-%d")
    return first_time_str, last_time_str


def get_time_limits_o(sat):
    f = get_first_file(sat)
    ds = xr.open_dataset(f, engine="h5netcdf")
    # ds = xr.open_dataset(f)
    time = ds.time.isel(points_numbers=0).values
    # these are cftime objects which are not very useful
    first_time = time[0]
    last_time = time[-1]
    # print(type(first_time), type(last_time))
    # Convert the first and last time stamps to string format
    first_time_str = first_time.strftime("%Y-%m-%d")
    last_time_str = last_time.strftime("%Y-%m-%d")
    # first_time_str = datetime.strptime(first_time, "%Y-%m-%d")
    # last_time_str = datetime.strptime(last_time, "%Y-%m-%d")
    # print(type(first_time_str), type(last_time_str))
    track_tsta_o = datetime.strptime(first_time_str, "%Y-%m-%d")
    track_tend_o = datetime.strptime(last_time_str, "%Y-%m-%d")
    return track_tsta_o, track_tend_o


def change_time(ds1, time_str):
    dvals = ds1[time_str].values
    splitted = ds1[time_str].units.split(" ")
    if len(splitted) == 3:
        frmt = "%Y-%m-%d"
        # rftime is reference time like since 2004-01-01
        rftime = splitted[2]
    elif len(splitted) == 4:
        frmt = "%Y-%m-%d %H:%M:%S"
        rftime = splitted[2] + " " + splitted[3]
    # dt0 is refence date object
    dt0 = datetime.strptime(rftime, frmt)
    units1 = splitted[0]
    if units1 == "months":
        dates = [dt0 + relativedelta(months=int(d)) for d in dvals]
    elif units1 == "days":
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    elif units1 == "hours":
        dates = [dt0 + timedelta(hours=int(d)) for d in dvals]
    elif units1 == "minutes":
        dates = [dt0 + timedelta(minutes=int(d)) for d in dvals]
    elif units1 == "seconds":
        dates = [dt0 + timedelta(seconds=int(d)) for d in dvals]
    ds1[time_str] = dates
    ds1 = ds1.rename({time_str: "time"})
    return ds1


# TIDE GAUGE FUNCTIONS
# @timeit_decorator
def read_tide_meta(data_loc="/home/srinivasu/slnew/psmsl/rlr_monthly"):
    df = pd.read_csv(
        f"{data_loc}/filelist.txt",
        sep=";",
        names=[
            "id",
            "latitude",
            "longitude",
            "name",
            "coastline",
            "stationcode",
            "stationflag",
        ],
        header=None,
    )
    df = df.set_index("id")
    return df


# Filter DataFrame based on bounds
# @timeit_decorator
def region_selected(df, lon_min, lon_max, lat_min, lat_max):
    filtered_df = df[
        (df["latitude"] >= lat_min)
        & (df["latitude"] <= lat_max)
        & (df["longitude"] >= lon_min)
        & (df["longitude"] <= lon_max)
    ]
    return filtered_df


def get_lonlat_psmsl(PSMSL_ID, df_nio):
    row = df_nio.loc[PSMSL_ID]
    lon_psmsl = row["longitude"]
    lat_psmsl = row["latitude"]
    name_psmsl = row["name"]
    return lon_psmsl, lat_psmsl, name_psmsl


def read_id(id1, data_loc="/home/srinivasu/slnew/psmsl/rlr_monthly/data/"):
    def convert_partial_year(number):
        # https://stackoverflow.com/questions/19305991/convert-fractional-years-to-a-real-date-in-python
        # TODO this function need a revisit
        year = int(float(number))
        d = timedelta(days=(float(number) - year) * 365.24)
        day_one = datetime(year, 1, 1)
        date = d + day_one
        return date

    df = pd.read_csv(
        f"{data_loc}/{id1}.rlrdata",
        delimiter=";",
        names=["time", "height", "a1", "a2"],
        na_values=-99999,
        parse_dates=[0],
        date_parser=convert_partial_year,
    )
    # df.set_index("time", inplace=True)
    df = df.set_index("time")
    df = df.to_xarray()
    return df


def read_id1(id1):
    # def convert_to_datetime(year_dec):
    #     # https://github.com/astg606/py_materials/blob/master/pandas/introduction_pandas.ipynb
    #     year_int = int(float(year_dec))
    #     base = datetime(year_int, 1, 1)
    #     rem = float(year_dec) - year_int
    #     result = base + timedelta(
    #         seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
    #     )
    #     # print(result.strftime("%y-%m-%d"))
    #     return result
    def convert_to_datetime(year_dec):
        # https://github.com/astg606/py_materials/blob/master/pandas/introduction_pandas.ipynb
        year_int = int(float(year_dec))
        base = pd.Timestamp(year_int, 1, 1)
        days = (float(year_dec) - year_int) * 365.24
        result = base + pd.Timedelta(days=days)
        return result

    df = pd.read_csv(
        f"/home/srinivasu/slnew/psmsl/rlr_monthly/data/{id1}.rlrdata",
        delimiter=";",
        names=["time", "height", "a1", "a2"],
        na_values=-99999,
    )
    df["time"] = df["time"].apply(convert_to_datetime)
    df.set_index("time", inplace=True)
    df = df.to_xarray()
    return df


def read_id_ai(id1, data_loc="/home/srinivasu/slnew/psmsl/rlr_monthly/data/"):
    def convert_partial_year(number):
        try:
            number = float(number)
            year = int(number)
            frac = number - year
            # Derive month (1-12) from fraction
            month = int(frac * 12) + 1
            # Clamp month to 1-12 in case of floating-point edge cases
            month = max(1, min(12, month))
            # Get days in that month (handles leap years)
            _, days_in_month = calendar.monthrange(year, month)
            # Calculate mid-month day (e.g., 15 for Feb, 16 for Jan/Mar)
            mid_day = (days_in_month // 2) + 1
            # Create datetime (will raise ValueError if invalid, e.g., Feb 30)
            return datetime(year, month, mid_day)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid fractional year '{number}': {e}")

    df = pd.read_csv(
        f"{data_loc}/{id1}.rlrdata",
        delimiter=";",
        names=["time", "height", "a1", "a2"],
        na_values=-99999,
        parse_dates=[0],
        date_parser=convert_partial_year,
    )
    df = df.set_index("time")  # Avoid inplace=True
    df = df.to_xarray()
    return df


def create_topo_map(dse,
                    xsta=30,
                    xend=120,
                    ysta=-30,
                    yend=30,
                    title1="",
                    step=5):
    lon_range = xend - xsta
    lat_range = yend - ysta
    aspect_ratio = lon_range / lat_range if lat_range != 0 else 1
    base_height = 8
    fig_width = round(aspect_ratio * base_height)
    fig_height = round(base_height)
    fig, ax1 = plt.subplots(
        1,
        1,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(fig_width, fig_height),
    )
    fig.tight_layout(pad=1.7, h_pad=1.7, w_pad=1.2)
    dse.rose.sel(lon=slice(xsta, xend)).sel(lat=slice(ysta, yend)).plot(
        ax=ax1, add_colorbar=False, add_labels=False)
    ax1.set_extent([xsta, xend, ysta, yend], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE)
    xticks = list(range(int(xsta), int(xend), step))
    yticks = list(range(int(ysta), int(yend), step))
    ax1.axhline(y=0.0, linestyle="--", color="k")
    ax1.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax1.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax1.set_title(title1)
    ax1.xaxis.set_major_formatter(LongitudeFormatter())
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    return fig, ax1


def plot_etopo_subplot(fig, position, tracks_reg, cmap, title="", step=2,
                       data_path="~/allData/topo/etopo5.cdf"):
    """
    Create a Cartopy subplot with ETOPO bathymetry data.
    Parameters:
        fig        : matplotlib Figure object
        position   : subplot position index (e.g., 311, 312, 313)
        tracks_reg : (xsta, xend, ysta, yend) region tuple
        cmap       : matplotlib colormap
        title      : subplot title
        step       : tick step interval
        data_path  : path to the ETOPO NetCDF file
    """
    # Load ETOPO dataset
    dse = xr.open_dataset(data_path)
    da = dse.ROSE
    # Extract region
    xsta, xend, ysta, yend = tracks_reg
    da_reg = da.sel(ETOPO05_X=slice(xsta, xend), ETOPO05_Y=slice(ysta, yend))
    # Create subplot with Cartopy
    ax = fig.add_subplot(position, projection=ccrs.PlateCarree())
    # Plot data
    da_reg.plot(ax=ax, cmap=cmap, add_colorbar=False, add_labels=False)
    # Decorate map
    ax.add_feature(cfeature.COASTLINE, linewidth=2.0)
    xticks = list(range(int(xsta), int(xend), step))
    yticks = list(range(int(ysta), int(yend), step))
    ax.axhline(y=0.0, linestyle="--", color="k")
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_title(title)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent([xsta, xend, ysta, yend], crs=ccrs.PlateCarree())
    return ax

def decorate_axis(ax, title1="", xsta=60, xend=100, ysta=-5, yend=25, step=2):
    ax.add_feature(cfeature.COASTLINE, linewidth=2.0)
    xticks = list(range(int(xsta), int(xend), step))
    yticks = list(range(int(ysta), int(yend), step))
    ax.axhline(y=0.0, linestyle="--", color="k")
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_title(title1)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent([xsta, xend, ysta, yend], crs=ccrs.PlateCarree())


@timeit_decorator
def omni_nearby_track(omni_id, sat):
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    dists = []
    index_of_omni = omni_buoys.index(omni_id)
    x0, y0 = lons_omni[index_of_omni], lats_omni[index_of_omni]
    for f in sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False, engine="h5netcdf")
        # ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
        c = (lats_track[0]) - m * (lons_track[0])
        denom = np.sqrt(m**2 + 1)
        val = m * x0 - y0 + c
        # denom = (
        #     (lon_track_0 - lon_track_m1) ** 2
        #     + (lat_track_0 - lat_track_m1) ** 2
        # ) ** 0.5
        # val = (lon_track_0 - lon_track_m1) * (lat_track_m1 - y0) - (
        #     lat_track_0 - lat_track_m1
        # ) * (lon_track_m1 - x0)
        dist_deg = abs(val) / denom
        dist = degrees2kilometers(dist_deg, y0)
        if dist > 200:
            continue
        x1 = (x0 + m * y0 - m * c) / (1 + m**2)
        y1 = m * x1 + c
        dists.append((track_number, x1, y1, dist))
    trackn1, x2, y2, mindist = min(dists, key=lambda x: x[3])
    return trackn1, x2, y2, mindist


@timeit_decorator
def rama_nearby_track(rama_id, sat):
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    dists = []
    x0, y0 = rama_d[rama_id]
    for f in sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False, engine="h5netcdf")
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
        c = (lats_track[0]) - m * (lons_track[0])
        denom = np.sqrt(m**2 + 1)
        val = m * x0 - y0 + c
        dist_deg = abs(val) / denom
        dist = degrees2kilometers(dist_deg, y0)
        if dist > 200:
            continue
        x1 = (x0 + m * y0 - m * c) / (1 + m**2)
        y1 = m * x1 + c
        dists.append((track_number, x1, y1, dist))
    if len(dists) == 0:
        return None, None, None, None
    trackn1, x2, y2, mindist = min(dists, key=lambda x: x[3])
    return trackn1, x2, y2, mindist


@timeit_decorator
def rama_nearby_track_box(rama_id, sat, box_size=2.0):
    x0, y0 = rama_d[rama_id]
    polygon = create_box_patch(x0, y0, box_size)
    list_tuples = []
    for f in sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False, engine="h5netcdf")
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        line_start = (lons_track[0], lats_track[0])
        line_end = (lons_track[-1], lats_track[-1])
        track_path = sg.LineString([
            (lon, lat) for lon, lat in zip(lons_track, lats_track)
        ])
        if polygon.intersects(track_path):
            # print(track_number)
            distances = [
                distance.distance((y0, x0), (y, x)).km
                for y, x in zip(lats_track, lons_track)
            ]
            # print(distances)
            minindex = np.nanargmin(distances)
            x1 = lons_track[minindex]
            y1 = lats_track[minindex]
            mindist = distances[minindex]
            if mindist > 200:
                continue
            # print(rama_id, sat, track_number, x1, y1, mindist)
            list_tuples.append((sat, track_number, x1, y1, mindist, rama_id))
    if len(list_tuples) == 0:
        return [(None, None, None, None, None, None)]
    sorted_list = sorted(list_tuples, key=lambda x: x[4])
    return sorted_list


@timeit_decorator
def coastal_nearby_track_box(coastal_id, sat, box_size=2.0):
    x0, y0 = coastal_d[coastal_id]
    polygon = create_box_patch(x0, y0, box_size)
    list_tuples = []
    for f in sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False, engine="h5netcdf")
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        line_start = (lons_track[0], lats_track[0])
        line_end = (lons_track[-1], lats_track[-1])
        track_path = sg.LineString([
            (lon, lat) for lon, lat in zip(lons_track, lats_track)
        ])
        if polygon.intersects(track_path):
            # print(track_number)
            distances = [
                distance.distance((y0, x0), (y, x)).km
                for y, x in zip(lats_track, lons_track)
            ]
            # print(distances)
            minindex = np.nanargmin(distances)
            x1 = lons_track[minindex]
            y1 = lats_track[minindex]
            mindist = distances[minindex]
            if mindist > 100:
                continue
            # print(rama_id, sat, track_number, x1, y1, mindist)
            list_tuples.append((sat, track_number, x1, y1, mindist, coastal_id))
    if len(list_tuples) == 0:
        return [(None, None, None, None, None, None)]
    sorted_list = sorted(list_tuples, key=lambda x: x[4])
    return sorted_list

@timeit_decorator
def omni_nearby_track_box(omni_id, sat, box_size=2.0):
    index_of_omni = omni_buoys.index(omni_id)
    x0, y0 = lons_omni[index_of_omni], lats_omni[index_of_omni]
    polygon = create_box_patch(x0, y0, box_size)
    list_tuples = []
    for f in sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False, engine="h5netcdf")
        # ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        line_start = (lons_track[0], lats_track[0])
        line_end = (lons_track[-1], lats_track[-1])
        track_path = sg.LineString([line_start, line_end])
        track_path = sg.LineString([
            (lon, lat) for lon, lat in zip(lons_track, lats_track)
        ])
        if polygon.intersects(track_path):
            # print(track_number)
            distances = [
                distance.distance((y0, x0), (y, x)).km
                for y, x in zip(lats_track, lons_track)
            ]
            # print(distances)
            minindex = np.nanargmin(distances)
            x1 = lons_track[minindex]
            y1 = lats_track[minindex]
            mindist = distances[minindex]
            if mindist > 200:
                continue
            # print(omni_id, sat, track_number, x1, y1, mindist)
            list_tuples.append((sat, track_number, x1, y1, mindist, omni_id))
    if len(list_tuples) == 0:
        return [(None, None, None, None, None, None)]
    sorted_list = sorted(list_tuples, key=lambda x: x[4])
    return sorted_list


def create_box_patch(lon0, lat0, half_width):
    # Define the coordinates of the corners of the box
    lon_min = lon0 - half_width
    lon_max = lon0 + half_width
    lat_min = lat0 - half_width
    lat_max = lat0 + half_width
    # Create a Polygon using these coordinates
    box = Polygon(
        [
            (lon_min, lat_min),
            (lon_min, lat_max),
            (lon_max, lat_max),
            (lon_max, lat_min),
            (lon_min, lat_min),
        ]
    )
    return box


@timeit_decorator
def cb_nearby_track_box(cb_id, sat, box_size=2.0):
    x0, y0 = cb_d[cb_id]
    polygon = create_box_patch(x0, y0, box_size)
    list_tuples = []
    for f in sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False, engine="h5netcdf")
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        line_start = (lons_track[0], lats_track[0])
        line_end = (lons_track[-1], lats_track[-1])
        track_path = sg.LineString([
            (lon, lat) for lon, lat in zip(lons_track, lats_track)
        ])
        if polygon.intersects(track_path):
            # print(track_number)
            distances = [
                distance.distance((y0, x0), (y, x)).km
                for y, x in zip(lats_track, lons_track)
            ]
            # print(distances)
            minindex = np.nanargmin(distances)
            x1 = lons_track[minindex]
            y1 = lats_track[minindex]
            # print(minindex, x1, y1)
            # break
            mindist = distances[minindex]
            if mindist < 200:
                list_tuples.append((sat, track_number, x1, y1, mindist, cb_id))
    sorted_list = sorted(list_tuples, key=lambda x: x[3])
    return sorted_list


@timeit_decorator
def psmsl_nearby_track_box(psmsl_id, df_nio, sat, box_size=2.0):
    lon_psmsl = df_nio.loc[psmsl_id]["longitude"]
    lat_psmsl = df_nio.loc[psmsl_id]["latitude"]
    polygon = create_box_patch(lon_psmsl, lat_psmsl, box_size)
    list_tuples = []
    for f in sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False, engine="h5netcdf")
        # ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        line_start = (lons_track[0], lats_track[0])
        line_end = (lons_track[-1], lats_track[-1])
        track_path = sg.LineString([line_start, line_end])
        if polygon.intersects(track_path):
            # print(track_number)
            distances = [
                distance.distance((lat_psmsl, lon_psmsl), (y, x)).km
                for y, x in zip(lats_track, lons_track)
            ]
            # print(distances)
            minindex = np.nanargmin(distances)
            x1 = lons_track[minindex]
            y1 = lats_track[minindex]
            mindist = distances[minindex]
            # print(minindex, x1, y1)
            # break
            if mindist > 100:
                continue
            # print(psmsl_id, sat, track_number, x1, y1, mindist)
            list_tuples.append((sat, track_number, x1, y1, mindist, psmsl_id))
    if len(list_tuples) == 0:
        return [(None, None, None, None, None, None)]
    sorted_list = sorted(list_tuples, key=lambda x: x[4])
    return sorted_list


@timeit_decorator
def psmsl_nearby_track(psmsl_id, df_nio, sat):
    lon_psmsl = df_nio.loc[psmsl_id]["longitude"]
    lat_psmsl = df_nio.loc[psmsl_id]["latitude"]
    dists = []
    for f in sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False, engine="h5netcdf")
        # ds = xr.open_dataset(f, decode_times=False)
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        # distance between
        dist = distance.distance(
            (lat_psmsl, lon_psmsl), (lat_coast, lon_coast)
        ).km
        if dist > 100.0:
            continue
        dists.append((track_number, lon_coast, lat_coast, dist))
    # return None if len of dists is zero
    if len(dists) == 0:
        return None, None, None, None
    trackn1, lon_coast1, lat_coast1, mindist = min(dists, key=lambda x: x[3])
    return trackn1, lon_coast1, lat_coast1, mindist


@timeit_decorator
def cb_nearby_track_new(cb_id, sat):
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    dists = []
    x0, y0 = cb_d[cb_id]
    for f in sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False, engine="h5netcdf")
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        if len(lons_track) == 0:
            continue
        m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
        c = (lats_track[0]) - m * (lons_track[0])
        # distp = distance.distance((y0, x0), (lat_coast, lon_coast)).km
        # distn = degrees2kilometers(dist_deg)
        # dist1 = min(distn, distp)
        x1 = (x0 + m * y0 - m * c) / (1 + m**2)
        y1 = m * x1 + c
        if is_land(x1, y1):
            x1 = lon_coast
            y1 = lat_coast
            dist = distance.distance((y0, x0), (lat_coast, lon_coast)).km
        else:
            denom = np.sqrt(m**2 + 1)
            val = m * x0 - y0 + c
            dist_deg = abs(val) / denom
            dist = degrees2kilometers(dist_deg, y0)
        if dist > 200.0:
            continue
        dists.append((track_number, x1, y1, dist))
    if len(dists) == 0:
        return None, None, None, None, None, None, None, None
    if len(dists) == 1:
        return *dists[0], None, None, None, None
    sorted_dists = sorted(dists, key=lambda x: x[3])
    (trackn1, x2, y2, mindist1), (trackn2, x3, y3, mindist2) = sorted_dists[:2]
    return (trackn1, x2, y2, mindist1), (trackn2, x3, y3, mindist2)


@timeit_decorator
def nearby_tide_guage_normal(ds, df_nio):
    lons_psmsl = df_nio["longitude"].to_list()
    lats_psmsl = df_nio["latitude"].to_list()
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
    c = (lats_track[0]) - m * (lons_track[0])
    denom = np.sqrt(m**2 + 1)
    dists = []
    ln = len(df_nio)
    for i in range(ln):
        x0, y0 = lons_psmsl[i], lats_psmsl[i]
        id1 = df_nio["id"].iloc[i]
        val = m * x0 - y0 + c
        dist_deg = abs(val) / denom
        dist = degrees2kilometers(dist_deg, y0)
        if dist > 200:
            continue
        x1 = (x0 + m * y0 - m * c) / (1 + m**2)
        y1 = m * x1 + c
        dists.append((id1, x1, y1, dist))
    # id2, x2, y2, mindist = min(dists, key=lambda x: x[3])
    sorted_list = sorted(dists, key=lambda x: x[3])
    return sorted_list


@timeit_decorator
def cb_nearby_track_normal(cb_id, sat):
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    dists = []
    x0, y0 = cb_d[cb_id]
    for f in sorted(
        glob.glob(
            f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        ds = xr.open_dataset(f, decode_times=False, engine="h5netcdf")
        track_number = ds.Pass
        lons_track = ds.lon.values
        lats_track = ds.lat.values
        if len(lons_track) == 0:
            continue
        m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
        c = (lats_track[0]) - m * (lons_track[0])
        denom = np.sqrt(m**2 + 1)
        val = m * x0 - y0 + c
        dist = abs(val) / denom
        if dist > 2.0:
            continue
        x1 = (x0 + m * y0 - m * c) / (1 + m**2)
        y1 = m * x1 + c
        dists.append((track_number, x1, y1, dist))
    if len(dists) == 0:
        return None, None, None, None
    trackn1, x2, y2, mindist = min(dists, key=lambda x: x[3])
    return trackn1, x2, y2, mindist


@timeit_decorator
def nearby_tide_guage(ds, df_nio):
    """
    To find the nearest tide gauge id given the ds of a single track
    """
    lons_psmsl = df_nio["longitude"].to_list()
    lats_psmsl = df_nio["latitude"].to_list()
    ids = df_nio["id"].to_list()
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lon_coast = lons_track[-1]  # this on coast
    lat_coast = lats_track[-1]  # this on coast
    dists = []
    ln = len(df_nio)
    for i in range(ln):
        id1 = ids[i]
        lon_psmsl, lat_psmsl = lons_psmsl[i], lats_psmsl[i]
        dist = distance.distance(
            (lat_psmsl, lon_psmsl), (lat_coast, lon_coast)
        ).km
        if dist > 200.0:
            continue
        dists.append((id1, lon_psmsl, lat_psmsl, dist))
    id2, x1, y1, mindist = min(dists, key=lambda x: x[3])
    return id2, x1, y1, mindist
