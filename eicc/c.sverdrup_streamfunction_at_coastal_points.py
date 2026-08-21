"""Sverdrup streamfunction at the EICC coastal points A-G, from CCMP wind stress curl.

    beta * V = curl_z(tau) / rho0        V = integral of v dz   [m2 s-1]
    psi(x,y) = -(1/(rho0 beta)) * integral_x^xE curl_z(tau) dx'  [m3 s-1]

psi is referenced to zero at the eastern boundary, so psi evaluated at a
western-boundary point is the transport the boundary current must carry to
close the interior Sverdrup flow at that latitude.

Sign convention (V = d(psi)/dx, U = -d(psi)/dy):
    psi < 0  cyclonic gyre     -> northward interior flow, SOUTHWARD (equatorward) WBC
    psi > 0  anticyclonic gyre -> southward interior flow, NORTHWARD (poleward)   WBC

Writes sverdrup_psi_at_coastal_points.csv and a diagnostic png.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from pathlib import Path
import sys

from tools_xtrackm import *

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")
csv_file = "sverdrup_psi_at_coastal_points.csv"

CURL_FILE = ("/home/srinivasu/allData/ccmp31/"
             "ccmp_v31_daily_iod_windstress_curl_mon.nc")
sat_val = "ERS1+ERS2+ENV+SRL"

RHO_WATER = 1025.0
OMEGA = 7.2921e-5
R_EARTH = 6371000.0
LOWPASS_CUTOFF_DAYS = 500.0

# the along-track record that Fig. 6 is built from ends 2016-04
REC_END = pd.Timestamp("2016-05-01")

POINT_LABELS = list("ABCDEFG")


def beta_at(lat_deg):
    return 2.0 * OMEGA * np.cos(np.deg2rad(lat_deg)) / R_EARTH


def lowpass_days(y, dt_days, cutoff_days=LOWPASS_CUTOFF_DAYS, order=4):
    nyq = 0.5 / dt_days
    b, a = butter(order, (1.0 / cutoff_days) / nyq, btype="low")
    m = np.isnan(y)
    yi = pd.Series(y).interpolate(limit_direction="both").values
    out = filtfilt(b, a, yi)
    out[m] = np.nan
    return out


def trend_per_year(times, y):
    m = np.isfinite(y)
    if m.sum() < 12:
        return np.nan, np.nan
    t_days = (times - times[0]).days.astype(float)
    res = linregress(t_days[m], y[m])
    return res.slope * 365.25, res.pvalue


def psi_at_point(curl, lon0, lat0, cross_islands=False):
    """Zonally integrate curl eastward from lon0 to the eastern boundary."""
    row = curl.sel(latitude=lat0, method="nearest")
    lons = row.longitude.values
    j0 = int(np.argmin(np.abs(lons - lon0)))
    ocean = np.isfinite(row.isel(time=0).values)
    if cross_islands:
        # treat island gaps as leaky: integrate to the last ocean point
        jE = np.where(ocean)[0][-1] + 1
    else:
        east_land = np.where(~ocean[j0:])[0]
        jE = j0 + east_land[0] if len(east_land) else len(lons)
    seg = row.isel(longitude=slice(j0, jE))
    dx = R_EARTH * np.cos(np.deg2rad(lat0)) * np.deg2rad(
        np.gradient(seg.longitude.values)
    )
    w = xr.DataArray(dx, coords={"longitude": seg.longitude},
                     dims=("longitude",))
    integral = (seg * w).sum("longitude", skipna=True)
    psi = -integral / (RHO_WATER * beta_at(lat0))     # m3 s-1
    return psi / 1.0e6, float(seg.longitude.values[-1]), seg.longitude.size


df = pd.read_csv(f"intersections_near_east_coast_{sat_val}.csv")
df_out = df.sort_values("lats_inter").iloc[1:].reset_index(drop=True)

ds = xr.open_dataset(CURL_FILE)
curl = ds.windstress_curl.sel(latitude=slice(5, 24), longitude=slice(78, 100))
curl = curl.dropna("time", how="all")
times = pd.to_datetime(curl.time.values)
dt_days = float(np.median(np.diff(times) / np.timedelta64(1, "D")))
print(f"CCMP wind stress curl: {times[0].date()} -> {times[-1].date()}  "
      f"dt = {dt_days:.1f} d\n")

rows = []
series = {}
for i, r in df_out.iterrows():
    L = POINT_LABELS[i]
    lon0, lat0 = float(r["lons_inter"]), float(r["lats_inter"])
    psi, lon_e, npts = psi_at_point(curl, lon0, lat0)
    psi_isl, lon_e2, _ = psi_at_point(curl, lon0, lat0, cross_islands=True)
    y = psi.values.astype(float)
    y_lp = lowpass_days(y, dt_days)
    m16 = times < REC_END

    a_all, p_all = trend_per_year(times, y_lp)
    a_16, p_16 = trend_per_year(times[m16], y_lp[m16])
    a_isl, _ = trend_per_year(times[m16],
                              lowpass_days(psi_isl.values.astype(float),
                                           dt_days)[m16])
    series[L] = (times, y, y_lp)
    rows.append(dict(
        point=L, lon=lon0, lat=lat0, lon_east=lon_e, n_grid=npts,
        psi_mean_Sv=np.nanmean(y),
        psi_mean_1993_2016_Sv=np.nanmean(y[m16]),
        trend_full_Sv_per_yr=a_all, p_full=p_all,
        trend_1993_2016_Sv_per_yr=a_16, p_1993_2016=p_16,
        trend_1993_2016_islands_leaky=a_isl,
    ))

res = pd.DataFrame(rows)
res.to_csv(csv_file, index=False)

print("Sverdrup streamfunction at the western boundary, psi(x_pt) in Sv")
print("psi < 0 : cyclonic gyre, closed by a SOUTHWARD (equatorward) EICC\n")
print(f"{'pt':>3} {'lat':>6} {'east bdy':>9} {'mean psi':>9} "
      f"{'mean 93-16':>11} | {'trend 93-16':>12} {'p':>7} | "
      f"{'trend full':>11} {'p':>7}")
for _, q in res.iterrows():
    print(f"{q.point:>3} {q.lat:6.2f} {q.lon_east:9.2f} "
          f"{q.psi_mean_Sv:9.2f} {q.psi_mean_1993_2016_Sv:11.2f} | "
          f"{q.trend_1993_2016_Sv_per_yr:+12.4f} {q.p_1993_2016:7.3f} | "
          f"{q.trend_full_Sv_per_yr:+11.4f} {q.p_full:7.3f}")

res["pct_over_record"] = (100 * res.trend_1993_2016_Sv_per_yr * 23.4
                          / res.psi_mean_1993_2016_Sv.abs())
res.to_csv(csv_file, index=False)
print("\nchange over the 23.4-yr X-TRACK record, per point:")
for _, q in res.iterrows():
    print(f"    {q.point}  {q.trend_1993_2016_Sv_per_yr * 23.4:+.2f} Sv "
          f"on a mean of {q.psi_mean_1993_2016_Sv:+.2f} Sv "
          f"({q.pct_over_record:+.0f}%)")
mt = res.trend_1993_2016_Sv_per_yr.mean()
print(f"\n    mean psi   1993-2016     : {res.psi_mean_1993_2016_Sv.mean():+.2f} Sv"
      f"   (range {res.psi_mean_1993_2016_Sv.min():+.2f} to "
      f"{res.psi_mean_1993_2016_Sv.max():+.2f})")
print(f"    mean trend 1993-2016     : {mt:+.4f} Sv/yr "
      f"({res.trend_1993_2016_Sv_per_yr.min():+.4f} to "
      f"{res.trend_1993_2016_Sv_per_yr.max():+.4f}, signs mixed)")
print(f"    same, Andaman ridge leaky: "
      f"{res.trend_1993_2016_islands_leaky.mean():+.4f} Sv/yr")

# --- seasonal phases: the annual mean is a small residual of large,
# --- opposing monsoon-phase transports, so report the phases separately.
print("\nmonsoon-phase psi (Sv) and its 1993-2016 trend (Sv/yr)")
print(f"{'pt':>3} | {'SW Jun-Sep':>11} {'trend':>8} {'p':>6} | "
      f"{'NE Nov-Feb':>11} {'trend':>8} {'p':>6}")
for L in POINT_LABELS:
    t, y, _ = series[L]
    out = []
    for months in ((6, 7, 8, 9), (11, 12, 1, 2)):
        sel = np.isin(t.month, months)
        ts = pd.Series(y[sel], index=t[sel]).resample("YE").mean()
        ts = ts[ts.index < REC_END]
        a, pv = trend_per_year(ts.index, ts.values)
        out.append((np.nanmean(ts.values), a, pv))
    print(f"{L:>3} | {out[0][0]:11.2f} {out[0][1]:+8.4f} {out[0][2]:6.3f} | "
          f"{out[1][0]:11.2f} {out[1][1]:+8.4f} {out[1][2]:6.3f}")

# --- is the claimed EICC weakening within the Sverdrup budget? -----------
# Convert the observed surface-speed trend to a transport trend:
#   dT/dt = (surface trend) * (depth-avg / surface) * width * depth
SPEED_TREND = -0.0098          # m/s/yr, measured from the X-TRACK record
SURF_TO_MEAN = 0.45            # first-baroclinic vertical structure
print("\ncan the wind supply the claimed weakening?")
print(f"{'width':>7} {'depth':>7} {'implied dT/dt':>15} {'Sverdrup dT/dt':>16} "
      f"{'ratio':>7}")
for W_km in (100.0, 175.0, 250.0):
    for H_m in (100.0, 200.0):
        dT = SPEED_TREND * SURF_TO_MEAN * (W_km * 1e3) * H_m / 1e6
        print(f"{W_km:6.0f}k {H_m:6.0f}m {dT:+15.4f} {mt:+16.4f} "
              f"{abs(dT / mt):7.1f}x")

# --- figure -------------------------------------------------------------
fig, axes = plt.subplots(len(POINT_LABELS), 1, figsize=(11, 13), sharex=True)
colors = [plt.cm.tab10(i) for i in range(len(POINT_LABELS))]
for k, L in enumerate(POINT_LABELS):
    ax = axes[len(POINT_LABELS) - 1 - k]
    t, y, y_lp = series[L]
    q = res[res.point == L].iloc[0]
    ax.plot(t, y, color="0.75", lw=0.6)
    ax.plot(t, y_lp, color=colors[k], lw=2.0)
    m16 = t < REC_END
    td = np.asarray((t[m16] - t[0]).days, dtype=float)
    ax.plot(t[m16], np.nanmean(y_lp[m16]) + q.trend_1993_2016_Sv_per_yr
            * (td - td.mean()) / 365.25, "k--", lw=1.4)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(REC_END, color="r", lw=0.8, ls=":")
    ax.text(0.005, 0.5, L, transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=colors[k], va="center")
    ax.text(0.99, 0.06,
            f"{q.trend_1993_2016_Sv_per_yr:+.3f} Sv/yr (p={q.p_1993_2016:.2f})",
            transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_ylabel("Sv", fontsize=10)
axes[0].set_title("Sverdrup streamfunction at the western boundary, "
                  "points A-G (CCMP wind stress curl)\n"
                  "grey: monthly   colour: 500-day low-pass   "
                  "dashed: 1993-2016 trend   red: end of the X-TRACK record",
                  fontsize=10)
plt.tight_layout()
plt.savefig(png_file, dpi=300, bbox_inches="tight")
plt.close("all")
print(f"\nwrote {csv_file} and {png_file}")
