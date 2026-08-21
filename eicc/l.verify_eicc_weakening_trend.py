"""Diagnostics for the multidecadal EICC weakening claim (draft_v02.2a, Fig. 6).

Runs five independent checks on the track-derived speed at points A-G:
  1. signed components gu, gv vs the positive-definite speed |V|
  2. within-instrument-era trends vs the trend across the ERS-2/ENVISAT boundary
  3. mean speed against the cycle-to-cycle noise level (Rice rectification)
  4. signal-to-noise of the two-track inversion at the ~25 deg crossing angle
  5. AVISO ugosa/vgosa at the same points over the same period
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress
from tools_xtrackm import *

method = "loess"
frac1 = "0.05"
sat_val = "ERS1+ERS2+ENV+SRL"
GUV_DIR = f"../guv/guv_at_intersections_{method}"
AVISO_UV_DIR = "/home/srinivasu/allData/aviso/msla_uv"

# ERS-1/ERS-2 are Ku-band on the 35-day orbit; ENVISAT (from mid-2002) and
# SARAL/AltiKa (Ka-band, from 2013) fly the same ground track but are quieter.
ERA_CUT = pd.Timestamp("2003-07-01")
REC_END = pd.Timestamp("2016-05-01")

POINT_LABELS = list("ABCDEFG")


def track_to_str(track):
    track_f = float(track)
    if track_f.is_integer():
        return str(int(track_f))
    return str(track)


def trend_per_year(times, y):
    """OLS slope per year, ignoring NaNs."""
    m = np.isfinite(y)
    if m.sum() < 10:
        return np.nan, np.nan
    t_days = (times - times[0]).days.astype(float)
    res = linregress(t_days[m], y[m])
    return res.slope * 365.25, res.pvalue


def hf_noise(x):
    """Per-sample scatter from successive differences (signal-blind estimator)."""
    d = np.diff(x)
    d = d[np.isfinite(d)]
    return np.std(d) / np.sqrt(2) if len(d) > 5 else np.nan


df = pd.read_csv(f"intersections_near_east_coast_{sat_val}.csv")
df_out = df.sort_values("lats_inter").iloc[1:].reset_index(drop=True)

store = {}
for i, row in df_out.iterrows():
    tno_s = track_to_str(row["track_self"])
    tno_o = track_to_str(row["track_other"])
    ds_guv = xr.open_dataset(
        f"{GUV_DIR}/guv_at_intersection_{sat_val}_{tno_s}_{tno_o}_"
        f"{method}_{frac1}.nc"
    )
    store[POINT_LABELS[i]] = (
        pd.to_datetime(ds_guv.time.values),
        ds_guv.gu.values,
        ds_guv.gv.values,
        float(row["angle_acute"]),
        float(row["lons_inter"]),
        float(row["lats_inter"]),
    )

print("=" * 78)
print("1. signed components vs speed")
print("=" * 78)
print(f"{'pt':>3} {'trend |V|':>10} {'p':>7} | {'trend gu':>9} {'p':>7} | "
      f"{'trend gv':>9} {'p':>7}")
for L in POINT_LABELS:
    t, gu, gv = store[L][:3]
    a, pa = trend_per_year(t, np.hypot(gu, gv))
    b, pb = trend_per_year(t, gu)
    c, pc = trend_per_year(t, gv)
    print(f"{L:>3} {a:+10.4f} {pa:7.3f} | {b:+9.4f} {pb:7.3f} | "
          f"{c:+9.4f} {pc:7.3f}")

print()
print("=" * 78)
print("2. within-era trend vs step at the ERS-2/ENVISAT boundary")
print("=" * 78)
print(f"{'pt':>3} {'full record':>12} {'ERS-1+2':>9} {'ENV+SRL':>9} "
      f"{'mean early':>11} {'mean late':>10} {'step':>7}")
full, within, steps = [], [], []
for L in POINT_LABELS:
    t, gu, gv = store[L][:3]
    sp = np.hypot(gu, gv)
    m1 = t < ERA_CUT
    m2 = (t >= ERA_CUT) & (t < REC_END)
    a, _ = trend_per_year(t, sp)
    b, _ = trend_per_year(t[m1], sp[m1])
    c, _ = trend_per_year(t[m2], sp[m2])
    e1, e2 = np.nanmean(sp[m1]), np.nanmean(sp[m2])
    full.append(a)
    within.append((b * m1.sum() + c * m2.sum()) / (m1.sum() + m2.sum()))
    steps.append(e2 - e1)
    print(f"{L:>3} {a:+12.4f} {b:+9.4f} {c:+9.4f} {e1:11.3f} {e2:10.3f} "
          f"{e2 - e1:+7.3f}")
print(f"    mean full-record trend {np.mean(full):+.4f} m/s/yr | "
      f"mean within-era trend {np.mean(within):+.4f} m/s/yr | "
      f"mean step {np.mean(steps):+.3f} m/s")

print()
print("=" * 78)
print("3. does mean speed track the noise level? (Rice rectification)")
print("=" * 78)
print(f"{'pt':>3} {'sigma early':>12} {'sigma late':>11} {'ratio sigma':>12} "
      f"{'ratio |V|':>10}")
rs, rv = [], []
for L in POINT_LABELS:
    t, gu, gv = store[L][:3]
    sp = np.hypot(gu, gv)
    m1 = t < ERA_CUT
    m2 = (t >= ERA_CUT) & (t < REC_END)
    s1 = np.hypot(hf_noise(gu[m1]), hf_noise(gv[m1]))
    s2 = np.hypot(hf_noise(gu[m2]), hf_noise(gv[m2]))
    rs.append(s2 / s1)
    rv.append(np.nanmean(sp[m2]) / np.nanmean(sp[m1]))
    print(f"{L:>3} {s1:12.3f} {s2:11.3f} {s2 / s1:12.3f} {rv[-1]:10.3f}")
print(f"    mean noise ratio {np.mean(rs):.3f} vs mean speed ratio "
      f"{np.mean(rv):.3f} | corr = {np.corrcoef(rs, rv)[0, 1]:.3f}")

print()
print("=" * 78)
print("4. conditioning and signal-to-noise of the two-track inversion")
print("=" * 78)
print(f"{'pt':>3} {'angle':>7} {'std gu':>8} {'std gv':>8} {'rms |V|':>9} "
      f"{'noise':>7} {'signal':>7} {'SNR':>6}")
for L in POINT_LABELS:
    t, gu, gv, ang = store[L][:4]
    ms = np.nanmean(gu**2 + gv**2)
    n2 = hf_noise(gu) ** 2 + hf_noise(gv) ** 2
    sig = np.sqrt(max(ms - n2, 0.0))
    print(f"{L:>3} {np.degrees(ang):6.1f}d {np.nanstd(gu):8.3f} "
          f"{np.nanstd(gv):8.3f} {np.sqrt(ms):9.3f} {np.sqrt(n2):7.3f} "
          f"{sig:7.3f} {sig / np.sqrt(n2):6.2f}")

print()
print("=" * 78)
print("5. AVISO geostrophic anomaly speed, same points, same period")
print("=" * 78)
ds_aviso = xr.merge(
    [xr.open_dataset(f"{AVISO_UV_DIR}/allyear_dt_global_allsat_msla_u.nc")["u"],
     xr.open_dataset(f"{AVISO_UV_DIR}/allyear_dt_global_allsat_msla_v.nc")["v"]]
).sel(longitude=slice(NIO[0], NIO[1]), latitude=slice(NIO[2], NIO[3]))
print(f"{'pt':>3} {'trend 93-16':>12} {'p':>7} {'mean early':>11} "
      f"{'mean late':>10} {'step':>7}")
av_tr, av_step = [], []
for L in POINT_LABELS:
    lon1, lat1 = store[L][4], store[L][5]
    u = ds_aviso.u.sel(longitude=lon1, latitude=lat1, method="nearest")
    v = ds_aviso.v.sel(longitude=lon1, latitude=lat1, method="nearest")
    sp = np.hypot(u.values, v.values)
    t = pd.to_datetime(u.time.values)
    m = t < REC_END
    a, pa = trend_per_year(t[m], sp[m])
    e1 = np.nanmean(sp[t < ERA_CUT])
    e2 = np.nanmean(sp[(t >= ERA_CUT) & (t < REC_END)])
    av_tr.append(a)
    av_step.append(e2 - e1)
    print(f"{L:>3} {a:+12.4f} {pa:7.3f} {e1:11.3f} {e2:10.3f} {e2 - e1:+7.3f}")
print(f"    AVISO   mean trend {np.mean(av_tr):+.4f} m/s/yr | "
      f"mean step {np.mean(av_step):+.3f} m/s")
print(f"    X-TRACK mean trend {np.mean(full):+.4f} m/s/yr | "
      f"mean step {np.mean(steps):+.3f} m/s")
ds_aviso.close()
