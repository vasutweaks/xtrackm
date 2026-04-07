#!/usr/bin/env python3

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt


SAT_VAL = "ERS1+ERS2+ENV+SRL"
METHOD = "loess"
FRAC1 = "0.05"

ROOT = Path(__file__).resolve().parent
POINTS_CSV = ROOT / f"intersections_near_east_coast_{SAT_VAL}.csv"
GUV_DIR = (ROOT / "../guv" / f"guv_at_intersections_{METHOD}").resolve()
PREDICTOR_SCRIPT = ROOT / "plot_indian_ocean_regions_and_timeseries_pumping.py"


def load_module_from_path(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def lowpass_500day(values: np.ndarray, time_index: pd.DatetimeIndex, order: int = 4) -> np.ndarray:
    dt_days = np.median(np.diff(time_index).astype("timedelta64[s]").astype(float) / 86400.0)
    nyq = 0.5 / dt_days
    cutoff = 1.0 / 500.0
    b_filt, a_filt = butter(order, cutoff / nyq, btype="low")

    nan_mask = np.isnan(values)
    interp_vals = pd.Series(values).interpolate(limit_direction="both").values

    if len(interp_vals) > 3 * max(len(a_filt), len(b_filt)):
        lp = filtfilt(b_filt, a_filt, interp_vals)
    else:
        samples_500d = max(3, int(round(500.0 / dt_days)))
        lp = (
            pd.Series(interp_vals)
            .rolling(window=samples_500d, center=True, min_periods=1)
            .mean()
            .values
        )

    lp[nan_mask] = np.nan
    return lp


def load_speed_point_series():
    df = pd.read_csv(POINTS_CSV)
    df_out = df.sort_values("lats_inter").iloc[1:].reset_index(drop=True)

    series = []
    for idx, row in df_out.iterrows():
        sat1 = row["sat"]
        tno_s = str(int(row["track_self"]))
        tno_o = str(int(row["track_other"]))

        f_guv = GUV_DIR / f"guv_at_intersection_{sat1}_{tno_s}_{tno_o}_{METHOD}_{FRAC1}.nc"
        ds_guv = xr.open_dataset(f_guv)
        gs = np.hypot(ds_guv.gu, ds_guv.gv)

        time_idx = pd.to_datetime(gs.time.values)
        gs_vals = gs.values.astype(float)
        gs_lp = lowpass_500day(gs_vals, time_idx)

        series.append(
            {
                "label": chr(ord("A") + idx),
                "lon": float(row["lons_inter"]),
                "lat": float(row["lats_inter"]),
                "time": time_idx,
                "speed_lp": gs_lp,
            }
        )

    return series


def load_predictor_series():
    p = load_module_from_path(PREDICTOR_SCRIPT, "regional_predictors")

    ds_mask = xr.open_dataset(p.MASK_FILE)
    gyre_mask = ds_mask["gyre_mask"]
    along_mask = ds_mask["along_shore_mask"]

    ds_curl = xr.open_dataset(p.CCMP_CURL_FILE)
    curl_var = "wind_stress_curl" if "wind_stress_curl" in ds_curl.data_vars else "windstress_curl"
    curl_bob = p.subset_lat_lon(ds_curl[curl_var], p.BOB_LAT_BOUNDS, p.BOB_LON_BOUNDS)
    ekman_bob = p.compute_ekman_pumping_from_curl(curl_bob)
    gyre_on_curl = p.interp_mask_to_target(
        p.subset_lat_lon(gyre_mask, p.BOB_LAT_BOUNDS, p.BOB_LON_BOUNDS),
        curl_bob,
    )
    ts1, _, _, _ = p.build_series(ekman_bob, mask_2d=gyre_on_curl, remove_clim=True)

    ds_sla = xr.open_dataset(p.AVISO_SLA_FILE)
    sla_bob = p.subset_lat_lon(ds_sla["sla"], p.BOB_LAT_BOUNDS, p.BOB_LON_BOUNDS)
    gyre_on_sla = p.interp_mask_to_target(
        p.subset_lat_lon(gyre_mask, p.BOB_LAT_BOUNDS, p.BOB_LON_BOUNDS),
        sla_bob,
    )
    ts2, _, _, _ = p.build_series(sla_bob, mask_2d=gyre_on_sla, remove_clim=False)

    ds_u = xr.open_dataset(p.CCMP_U_FILE)
    ds_v = xr.open_dataset(p.CCMP_V_FILE)

    u_bob = p.subset_lat_lon(ds_u["uwnd"], p.BOB_LAT_BOUNDS, p.BOB_LON_BOUNDS)
    v_bob = p.subset_lat_lon(ds_v["vwnd"], p.BOB_LAT_BOUNDS, p.BOB_LON_BOUNDS)
    tau_bob = p.compute_windstress_magnitude(u_bob, v_bob)
    along_on_tau = p.interp_mask_to_target(
        p.subset_lat_lon(along_mask, p.BOB_LAT_BOUNDS, p.BOB_LON_BOUNDS),
        tau_bob,
    )
    ts3, _, _, _ = p.build_series(tau_bob, mask_2d=along_on_tau, remove_clim=True)

    u_eq = p.subset_lat_lon(ds_u["uwnd"], p.EQ_LAT_BOUNDS, p.EQ_LON_BOUNDS)
    v_eq = p.subset_lat_lon(ds_v["vwnd"], p.EQ_LAT_BOUNDS, p.EQ_LON_BOUNDS)
    tau_eq = p.compute_windstress_magnitude(u_eq, v_eq)
    land = p.regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = land.mask(tau_eq.longitude, tau_eq.latitude, wrap_lon=True)
    ocean_mask = xr.where(land_mask.isnull(), 1.0, np.nan)
    ts4, _, _, _ = p.build_series(tau_eq, mask_2d=ocean_mask, remove_clim=True)

    s1 = pd.Series(ts1.values, index=pd.to_datetime(ts1.time.values), name="ekman_bob")
    s2 = pd.Series(ts2.values, index=pd.to_datetime(ts2.time.values), name="sla_bob")
    s3 = pd.Series(ts3.values, index=pd.to_datetime(ts3.time.values), name="tau_along")
    s4 = pd.Series(ts4.values, index=pd.to_datetime(ts4.time.values), name="tau_eq")
    return pd.concat([s1, s2, s3, s4], axis=1, join="outer").sort_index()


def interpolate_predictors_to_times(df_pred: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
    t_src = df_pred.index.view("int64").astype(float)
    t_tgt = target_times.view("int64").astype(float)

    out = {}
    for col in df_pred.columns:
        y = df_pred[col].values.astype(float)
        m = np.isfinite(y)
        if m.sum() < 2:
            out[col] = np.full(target_times.size, np.nan, dtype=float)
            continue
        out[col] = np.interp(t_tgt, t_src[m], y[m], left=np.nan, right=np.nan)

    return pd.DataFrame(out, index=target_times)


def fit_multilinear(y: np.ndarray, X: np.ndarray):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, yhat, r2


def standardized_betas(y: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    y_std = (y - np.mean(y)) / np.std(y, ddof=0)
    x_mean = np.mean(X_pred, axis=0)
    x_std = np.std(X_pred, axis=0, ddof=0)
    x_std[x_std == 0] = np.nan
    Xz = (X_pred - x_mean) / x_std
    valid = np.isfinite(y_std) & np.all(np.isfinite(Xz), axis=1)
    bz, _, _, _ = np.linalg.lstsq(Xz[valid], y_std[valid], rcond=None)
    return bz


def pretty_equation(
    label: str, lon: float, lat: float, n: int, beta: np.ndarray, beta_std: np.ndarray, r2: float
) -> str:
    s1, s2, s3 = beta_std
    return (
        f"Point {label} ({lon:.2f}E, {lat:.2f}N), N={n}, R^2={r2:.3f}\n"
        f"  z(speed_lp) = {s1:+.2f}*z(ekman_bob)"
        f" {s2:+.2f}*z(tau_along)"
        f" {s3:+.2f}*z(tau_eq)"
    )


def main():
    point_series = load_speed_point_series()
    df_pred = load_predictor_series()

    print("Multilinear regression at each coastal point")
    print("Model: speed_lp = b0 + b1*ekman_bob + b2*tau_along + b3*tau_eq")
    print("-" * 96)

    for p in point_series:
        t = p["time"]
        y = p["speed_lp"].astype(float)

        x_df = interpolate_predictors_to_times(df_pred, t)
        X = np.column_stack(
            [
                np.ones(t.size, dtype=float),
                x_df["ekman_bob"].values,
                x_df["tau_along"].values,
                x_df["tau_eq"].values,
            ]
        )

        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        yv = y[valid]
        Xv = X[valid, :]

        if yv.size < 10:
            print(f"Point {p['label']} ({p['lon']:.2f}E, {p['lat']:.2f}N): insufficient valid samples ({yv.size})")
            print("-" * 96)
            continue

        beta, _, r2 = fit_multilinear(yv, Xv)
        beta_std = standardized_betas(yv, Xv[:, 1:])
        print(pretty_equation(p["label"], p["lon"], p["lat"], yv.size, beta, beta_std, r2))
        print("-" * 96)


if __name__ == "__main__":
    main()
