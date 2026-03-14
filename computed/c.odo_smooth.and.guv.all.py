import math
import os

import numpy as np
import pandas as pd
import xarray as xr
from tools_xtrackm import *

method = "odo"
p = -7
q = 8
para = f"{abs(p)}{abs(q)}"
ioff = 3
out_dir = f"/home/srinivasu/xtrackm/guv/guv_at_intersections_{method}/"
os.makedirs(out_dir, exist_ok=True)


def compute_geostrophy_gc_from_slope(ds, slope_smooth):
    lons_track_rev = ds.lon.values[::-1]
    lats_track_rev = ds.lat.values[::-1]
    if len(lons_track_rev) < 3:
        return None
    f1 = cor_p(lats_track_rev)
    f2 = f1.values
    f3 = np.where(np.abs(lats_track_rev) > 2, f2, np.nan)
    g = grav(lats_track_rev).values
    ratio = g / f3
    # to orient slope coast to equator as is the convention
    slope_smooth = -1 * slope_smooth
    gc = ratio[:, np.newaxis] * slope_smooth
    gc.attrs["units"] = "m/s"
    gc.attrs["long_name"] = "Cross-track geostrophic current"
    gc.attrs["description"] = "Geostrophic current perpendicular to satellite track"
    return gc


def track_dist_time_asn_midx_da(ds, da_smooth, units_in="m"):
    dists = compute_dist(ds, units=units_in)
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    dt0 = datetime.strptime("1950-01-01", "%Y-%m-%d")
    dists_rev = dists[::-1]
    ln = ds["sla"].sizes["points_numbers"]
    ln2 = ln // 2
    dvals = ds.time.isel(points_numbers=ln2).values
    try:
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    except ValueError:
        dvals = pd.Series(dvals).ffill().bfill()
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    var1_rev = da_smooth.load().isel(points_numbers=slice(None, None, -1)).values
    var2 = xr.DataArray(
        var1_rev,
        dims=["x", "time"],
        coords={
            "x": dists_rev,
            "time": dates,
            "lon": ("x", lons_track_rev),
            "lat": ("x", lats_track_rev),
        },
    )
    if units_in == "m":
        var2.coords["x"].attrs["units"] = "m"
    elif units_in == "km":
        var2.coords["x"].attrs["units"] = "km"
    return var2


def apply_optimal_operator(ds, p=-5, q=6, dt=1.0):
    def get_optimal_weights(p, q, dt):
        p_abs, q_abs = abs(p), abs(q)
        n = np.array([i for i in range(-p_abs, q_abs + 1) if i != 0])
        N = p_abs + q_abs
        T = N + 1
        A = np.zeros((T, T))
        A[N, :N] = 1.0
        for i in range(N):
            A[i, :N] = (1.0 / n) * (-n[i] / 2.0)
            A[i, N] = (n[i] ** 2 * dt**2) / 4.0
            A[i, i] = -1.0
        B = np.zeros(T)
        B[N] = 1.0
        results = np.linalg.solve(A, B)
        return results[:N], n

    cn, n_indices = get_optimal_weights(p, q, dt)
    sla = ds.sla.values
    points_dim = sla.shape[0]
    slopes = np.full(sla.shape, np.nan)
    p_abs = abs(p)
    q_abs = abs(q)
    # slopes here are oriented equator to coast
    for i in range(p_abs, points_dim - q_abs):
        h_i = sla[i, :]
        h_rel = sla[i + n_indices, :]
        window_slopes = (h_rel - h_i) / (n_indices[:, np.newaxis] * dt)
        valid = ~np.isnan(window_slopes)
        w = cn[:, np.newaxis] * valid
        w_sum = np.sum(np.abs(w), axis=0)
        w_sum[w_sum == 0] = np.nan
        cn_sum = np.sum(np.abs(cn))
        slopes[i, :] = np.nansum(cn[:, np.newaxis] * window_slopes, axis=0) * (cn_sum / w_sum)
    return xr.DataArray(slopes, coords=ds.sla.coords, dims=ds.sla.dims, name="slope")


def _dx_from_track(ds):
    dists_m = np.asarray(compute_dist(ds, units="m"), dtype=float)
    if dists_m.size < 2:
        return np.nan
    dx = np.diff(dists_m)
    dx = np.abs(dx[np.isfinite(dx)])
    dx = dx[dx > 0]
    if dx.size == 0:
        return np.nan
    return float(np.nanmedian(dx))


def _compute_gc_odo(ds, p, q):
    dx_m = _dx_from_track(ds)
    print(dx_m, ds.Pass)
    if not np.isfinite(dx_m):
        return None
    slope_odo = apply_optimal_operator(ds, p=p, q=q, dt=dx_m)
    slope_odo_asn = track_dist_time_asn_midx_da(ds, slope_odo, units_in="m")
    return compute_geostrophy_gc_from_slope(ds, slope_odo_asn)


def _load_track_data_cached(sat, track_number, cache):
    key = (sat, str(track_number))
    if key in cache:
        return cache[key]
    f = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{str(track_number).zfill(3)}.nc"
    if not os.path.exists(f):
        cache[key] = None
        return None
    ds = xr.open_dataset(f, engine="netcdf4", decode_times=False)
    if len(ds.lon.values) < (abs(p) + abs(q) + 1):
        cache[key] = None
        return None
    gc = _compute_gc_odo(ds, p=p, q=q)
    if gc is None:
        cache[key] = None
        return None
    cache[key] = {"ds": ds, "gc": gc}
    return cache[key]


def main():
    sats_new = ["TP+J1+J2+J3+S6A"]
    print(sats_new)
    cache = {}
    for sat in sats_new[:]:
        inter_path = f"tracks_intersections_{sat}_1.csv"
        if not os.path.exists(inter_path):
            # print(f"Skipping {sat}: missing {inter_path}")
            continue
        df = pd.read_csv(inter_path)
        for _, r in df.iterrows():
            sat1 = r["sat"]
            track_number_self = str(r["track_self"])
            track_number_other = str(r["track_other"])
            lats_inter1 = r["lats_inter"]
            x_from_coast_self1 = r["x_from_coast_self"]
            x_from_coast_other1 = r["x_from_coast_other"]
            if abs(lats_inter1) < 2:
                continue

            self_data = _load_track_data_cached(sat, track_number_self, cache)
            other_data = _load_track_data_cached(sat, track_number_other, cache)
            if self_data is None or other_data is None:
                print(f"Skipping {sat} {track_number_self}-{track_number_other}: missing/invalid track data")
                continue

            ds_self = self_data["ds"]
            ds_other = other_data["ds"]
            gc_self = self_data["gc"]
            gc_other = other_data["gc"]

            ii_self = index_at_lat(ds_self, lats_inter1)
            ii_other = index_at_lat(ds_other, lats_inter1)
            a1 = local_angle(ds_self.lon.values, ds_self.lat.values, ii_self, ioff=ioff)
            a2 = local_angle(ds_other.lon.values, ds_other.lat.values, ii_other, ioff=ioff)
            # print(f"heading angle {sat}: a1 = {math.degrees(a1):.2f}, a2 = {math.degrees(a2):.2f}")

            gc_self_at = gc_self.sel(x=x_from_coast_self1, method="nearest", drop=True)
            gc_other_at = gc_other.sel(x=x_from_coast_other1, method="nearest", drop=True)
            gc_other_at = gc_other_at.interp(time=gc_self_at.time)

            gc_self_at_filt = mad_filter_1d_xr(gc_self_at, 3.5, dim="time")
            gc_other_at_filt = mad_filter_1d_xr(gc_other_at, 3.5, dim="time")
            try:
                gu, gv = geostrophic_components(gc_self_at_filt, gc_other_at_filt, a1, a2)
            except ValueError as err:
                print(f"Skipping {sat} {track_number_self}-{track_number_other}: {err}")
                continue

            gu_filt = gu.where((gu <= 3.5) & (gu >= -3.5))
            gv_filt = gv.where((gv <= 3.5) & (gv >= -3.5))
            gu_filtered = mad_filter_1d_xr(gu_filt, 3.5)
            gv_filtered = mad_filter_1d_xr(gv_filt, 3.5)

            ds_out = xr.Dataset({"gu": gu_filtered, "gv": gv_filtered})
            ds_out_self = xr.Dataset({"gc_self": gc_self_at_filt})
            ds_out_other = xr.Dataset({"gc_other": gc_other_at_filt})

            filename = f"{out_dir}/guv_at_intersection_{sat1}_{track_number_self}_{track_number_other}_{method}_{para}.nc"
            # print(f"saving to {filename}")
            ds_out.to_netcdf(filename)
            ds_out_self.to_netcdf(f"{out_dir}/gc_at_{sat1}_{track_number_self}_{method}_{para}.nc")
            ds_out_other.to_netcdf(f"{out_dir}/gc_at_{sat1}_{track_number_other}_{method}_{para}.nc")


if __name__ == "__main__":
    main()
