import math
import os
from itertools import zip_longest
from numbers import Number

import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *

try:
    import nc_time_axis  # noqa: F401
except ImportError:
    nc_time_axis = None

# codex resume 019cf1aa-6bd9-7d61-8b47-51d2d4635320


def track_time_asn(ds, var_str="sla"):
    if "time" not in ds:
        return ds

    # time_var is also a two d variable, its dimension are 
    # (points_numbers, cycles_numbers)
    time_var = ds["time"]
    cycle_dim = (
        "cycles_numbers"
        if "cycles_numbers" in time_var.dims
        else time_var.dims[-1]
    )

    # time is stored as (points_numbers, cycles_numbers); collapse to per-cycle.
    if "points_numbers" in time_var.dims:
        p_mid = time_var.sizes["points_numbers"] // 2
        time_1d = time_var.isel(points_numbers=p_mid)
    else:
        time_1d = time_var

    vals = pd.Series(time_1d.values).ffill().bfill().to_numpy()
    units = time_1d.attrs.get("units")
    calendar = time_1d.attrs.get("calendar", "standard")

    if np.issubdtype(np.asarray(vals).dtype, np.number) and units:
        decoded = xr.coding.times.decode_cf_datetime(
            np.asarray(vals), units, calendar=calendar
        )
    else:
        decoded = pd.to_datetime(vals, errors="coerce").to_numpy()

    ds = ds.assign_coords(time=(cycle_dim, decoded))
    if cycle_dim != "time":
        ds = ds.swap_dims({cycle_dim: "time"})
    return ds


def _to_list(values):
    if hasattr(values, "values"):
        values = values.values
    if hasattr(values, "ravel"):
        values = values.ravel()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list):
        return values
    return [values]


def _format_timestamp(value, units=None, calendar="standard"):
    if value == "":
        return ""

    if isinstance(value, np.datetime64):
        return pd.to_datetime(value).strftime("%Y-%m-%dT%H:%M:%S")

    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%dT%H:%M:%S")
        except (TypeError, ValueError):
            pass

    if isinstance(value, Number) and units:
        try:
            decoded = xr.coding.times.decode_cf_datetime(
                np.array([value]), units, calendar=calendar
            )[0]
            if hasattr(decoded, "strftime"):
                return decoded.strftime("%Y-%m-%dT%H:%M:%S")
            return pd.to_datetime(decoded).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            return str(value)

    if isinstance(value, str):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%dT%H:%M:%S")
        return value

    return str(value)


def print_timestamps_side_by_side(
    time_self1,
    time_other1,
    required_length=None,
    col_width=24,
    diff_col_width=12,
):
    attrs_self = getattr(time_self1, "attrs", {})
    attrs_other = getattr(time_other1, "attrs", {})
    units_self = attrs_self.get("units")
    units_other = attrs_other.get("units")
    cal_self = attrs_self.get("calendar", "standard")
    cal_other = attrs_other.get("calendar", "standard")

    left = [
        _format_timestamp(v, units=units_self, calendar=cal_self)
        for v in _to_list(time_self1)
    ]
    right = [
        _format_timestamp(v, units=units_other, calendar=cal_other)
        for v in _to_list(time_other1)
    ]

    if required_length is None:
        required_length = max(len(left), len(right))
    else:
        required_length = int(required_length)

    left = left[:required_length]
    right = right[:required_length]
    if len(left) < required_length:
        left.extend([""] * (required_length - len(left)))
    if len(right) < required_length:
        right.extend([""] * (required_length - len(right)))

    diffs = []
    for t_self, t_other in zip_longest(left, right, fillvalue=""):
        if not t_self or not t_other:
            diffs.append("")
            continue
        dt_self = pd.to_datetime(t_self, errors="coerce")
        dt_other = pd.to_datetime(t_other, errors="coerce")
        if pd.isna(dt_self) or pd.isna(dt_other):
            diffs.append("")
            continue
        diff_days = (dt_other - dt_self).total_seconds() / 86400.0
        diffs.append(f"{diff_days:.2f}")

    print(
        f"{'time_self1':<{col_width}} | "
        f"{'time_other1':<{col_width}} | "
        f"{'diff_days':>{diff_col_width}}"
    )
    print(
        f"{'-' * col_width}-+-{'-' * col_width}-+-{'-' * diff_col_width}"
    )
    for t_self, t_other, diff in zip_longest(
        left, right, diffs, fillvalue=""
    ):
        print(
            f"{t_self:<{col_width}} | {t_other:<{col_width}} | "
            f"{diff:>{diff_col_width}}"
        )


ioff = 3
frac1 = "0.05"
method = "loess"
out_dir = f"/home/srinivasu/xtrackm/guv/guv_at_intersections_{method}/"
os.makedirs(out_dir, exist_ok=True)

sat = "TP+J1+J2+J3+S6A"
df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
# sort by track_self (convert it to int if it's not already)
if df["track_self"].dtype == object:
    df["track_self"] = df["track_self"].astype(int)
df = df.sort_values("track_self")
for i, r in df.iterrows():
    sat1 = r["sat"]
    track_number_self = str(r["track_self"])
    track_number_other = str(r["track_other"])
    lons_inter1 = r["lons_inter"]
    lats_inter1 = r["lats_inter"]
    x_from_coast_self1 = r["x_from_coast_self"]
    x_from_coast_other1 = r["x_from_coast_other"]
    angle_acute = r["angle_acute"]
    angle_obtuse = r["angle_obtuse"]
    lon1 = lons_inter1
    lat1 = lats_inter1
    if abs(lat1) < 2:
        continue
    print(
        sat,
        track_number_self,
        track_number_other,
        "----------------------------------------",
    )
    f_self = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
    f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
    ds_self = xr.open_dataset(f_self, engine="netcdf4", decode_times=False)
    ds_other = xr.open_dataset(
        f_other, engine="netcdf4", decode_times=False
    )
    ds_self = track_time_asn(ds_self)
    ds_other = track_time_asn(ds_other)
    sla_self = ds_self.sla
    sla_other = ds_other.sla
    ii_self = index_at_lat(ds_self, lats_inter1)
    ii_other = index_at_lat(ds_other, lats_inter1)
    # times_self = ds_self.time
    # times_other = ds_other.time
    sla_self1 = sla_self.isel(points_numbers=ii_self, drop=True)
    sla_other1 = sla_other.isel(points_numbers=ii_other, drop=True)
    time_target = sla_self.isel(points_numbers=ii_self, drop=True).time.values
    time_self1 = sla_self.isel(points_numbers=ii_self).time
    time_other1 = sla_other.isel(points_numbers=ii_other).time
    sla_other_re = sla_other.reindex(
        time=time_target,
        method="nearest",
        tolerance=pd.Timedelta(days=10),
    )
    sla_other_sel = sla_other.sel(
        time=time_target,
        method="nearest",
        tolerance=pd.Timedelta(days=10),
    )
    sla_other_re1 = sla_other_re.isel(points_numbers=ii_other, drop=True)
    sla_other_sel1 = sla_other_sel.isel(points_numbers=ii_other, drop=True)
    time_other_re = sla_other_re.isel(points_numbers=ii_other).time
    time_other_sel = sla_other_sel.isel(points_numbers=ii_other).time
    print("original:")
    print_timestamps_side_by_side(time_self1, time_other1, required_length=2)
    print("reindex (relabeled to self time):")
    print_timestamps_side_by_side(time_self1, time_other_re, required_length=2)
    print("sel nearest (actual matched other time):")
    print_timestamps_side_by_side(time_self1, time_other_sel, required_length=2)
    sla_self1.plot(label="self", color="blue")
    sla_other_re1.plot(label="other (reindexed)", color="orange")
    sla_other_sel1.plot(label="other (sel nearest)", color="green")
    plt.legend()
    plt.show()
    if i > 10:
        break
