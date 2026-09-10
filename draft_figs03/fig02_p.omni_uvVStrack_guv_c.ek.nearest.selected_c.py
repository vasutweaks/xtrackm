"""Figure 2 -- OMNI buoy zonal current against track geostrophic + Ekman current
at the nearest S3A track intersection.

Rewrite of draft_figs02/fig02_p.omni_uvVStrack_guv_c.ek.nearest.selected.py
against tools_xtrackm_c.  Changes that affect the figure, not just the code:

  1. EKMAN DEPTH MATCHED TO THE BUOY.  The original compared OMNI current
     interpolated to 15 m (dpth = 15) against Ekman evaluated at the SURFACE
     (EKMAN_DEPTH = "0"), although 15 m Ekman files exist alongside the 0 m
     ones.  The geostrophic part is near depth-independent in the upper layer,
     but the Ekman spiral is not, so the depths must agree.  With the stored
     guv product: corr(gu+ek, omni) = 0.725 at 0 m, 0.735 at 15 m.
  2. The correlations were computed, printed to stdout, and never drawn.  They
     are now on the panel, with RMSE and n, for gu and gu+ekman.
  3. The AVISO msla reference line was the full record interpolated at the
     intersection and merely hidden by set_xlim.  It is now put on the same
     timestamps and scored like the others.  Note that this msla_uv file is
     MONTHLY (31 d step, mid-month stamps), not daily: nearest-within-24 h keeps
     only 8 of the 62 cycles, so it is time-interpolated and labelled monthly.
     It is a monthly-mean reference against instantaneous track samples, so its
     skill is not comparable like for like with OMNI.
  4. `raise("...")` raises a str, which is TypeError: exceptions must derive
     from BaseException -- all three guard paths died with the wrong error.
  5. Datasets closed; the duplicated dse / ds_omni / script / cmap1 / d
     assignments and the unused N removed.

UPSTREAM CAVEATS -- the stored guv product carries these; not fixable here:
  - guv/c.guv_at_intersection_points_loess_0.05.py computes the track heading
    with index_at_lat() + local_angle(), the index-order mismatch of summary
    section 7(b).  For this pair it took the heading at 5.03N and 3.73N instead
    of the intersection latitude 16.52N.  S3A tracks are nearly straight, so the
    angle error is only 0.22 deg / 0.25 deg and the effect on the product is
    0.1% of rms for gu and 1.3% for gv -- negligible here by luck, not design.
  - that producer also uses mad_filter_1d_xr (0.6745x too lenient, section 9.1)
    and track_smooth_loess_statsmodels_x (fabricates data, section 9.3).
  Set RECOMPUTE_GUV = True to rebuild gu/gv here from the raw tracks with
  smooth_loess + cross_track_geostrophy + resolve_uv and corrected headings.

DO NOT USE guv/guv_at_intersections_loess/gc_at_*.nc.  Those names carry no
intersection identifier, but a track is "self" in many pairs (track 522 in 10,
at x = 126.6 ... 2210.4 km), so each pair overwrites the last.  The file on disk
for track 522 matches recomputed gc at x = 1263.7 km (corr +0.9968), not the
571.5 km of the 522 x 279 pair.  No script reads them, so it is a latent trap
rather than an active bug -- but it defeats any attempt to rebuild guv from them.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path

import cmocean as cmo
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from geopy import distance

from tools_xtrackm import *  # decorate_axis, is_within_region, TRACKS_REG, omni_d, plt, ccrs
from tools_xtrackm_c import (cross_track_geostrophy, mad_filter, resolve_uv,
                             smooth_loess, track_angle_at, track_dist_time_asn_midx)

SELECT_SAT = "S3A"
SELECT_OMNI_ID = "BD10"
SELECT_TRACK_SELF = "522"
SELECT_TRACK_OTHER = "279"

OMNI_DEPTH = 15            # m, the OMNI current bin used
EKMAN_DEPTH = OMNI_DEPTH   # matched to it -- this is change (1)
METHOD, FRAC = "loess", "0.05"
MATCH_TOL = np.timedelta64(24, "h")
RECOMPUTE_GUV = False      # True -> rebuild gu/gv here with the corrected chain

TOPO = "~/allData/topo/etopo5.cdf"
AVISO_DIR = "/home/srinivasu/allData/aviso/msla_uv"
OMNI_DIR = "/home/srinivasu/allData/MB_cleaned/raju_omni_clean"
GUV_DIR = f"/home/srinivasu/xtrackm/guv/guv_at_intersections_{METHOD}"
EKMAN_DIR = "/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd"
CMAP = cmo.cm.diff

script = Path(sys.argv[0])
png_file = script.with_suffix(".png")
nc_file = script.with_name(script.stem + ".nc")


@lru_cache(maxsize=None)
def track_geometry(sat, track_number, lat_inter):
    """Track lon/lat plus the geodesic heading at the intersection latitude."""
    fn = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{str(track_number).zfill(3)}.nc"
    with xr.open_dataset(fn, engine="netcdf4", decode_times=False) as ds:
        lons, lats = ds.lon.values.copy(), ds.lat.values.copy()
        angle_deg, _ = track_angle_at(ds, lat=lat_inter, degrees=True)
        angle_rad, _ = track_angle_at(ds, lat=lat_inter, order="coast")
    return lons, lats, angle_deg, angle_rad


def match(series, times):
    """Nearest-in-time sample of `series` at `times`, within MATCH_TOL.

    Scalar coordinates left behind by .interp() (an AVISO latitude, an Ekman
    lon/lat) are dropped -- otherwise assembling the output Dataset raises
    MergeError: conflicting values for variable 'latitude'.
    """
    out = series.reindex(time=times, method="nearest", tolerance=MATCH_TOL)
    return out.reset_coords(drop=True)


def skill(pred, obs):
    good = np.isfinite(pred) & np.isfinite(obs)
    n = int(good.sum())
    if n < 3:
        return np.nan, np.nan, n
    r = float(xs.pearson_r(pred.where(good), obs.where(good), dim="time", skipna=True))
    rmse = float(xs.rmse(pred.where(good), obs.where(good), dim="time", skipna=True))
    return r, rmse, n


# ── locate the intersection nearest the buoy ────────────────────────────────
lon_omni, lat_omni = omni_d[SELECT_OMNI_ID]
if abs(lat_omni) < 2:
    raise ValueError(f"{SELECT_OMNI_ID} is inside the 2S-2N band; f is unusable")
omni_file = f"{OMNI_DIR}/{SELECT_OMNI_ID}_current_sst.nc"
if not os.path.exists(omni_file):
    raise FileNotFoundError(omni_file)

df_all = pd.concat([pd.read_csv(f"../draft_figs02/tracks_intersections_{s}_1.csv")
                    for s in sats_new], ignore_index=True)
sel = df_all[(df_all["sat"] == SELECT_SAT)
             & (df_all["track_self"].astype(str) == SELECT_TRACK_SELF)
             & (df_all["track_other"].astype(str) == SELECT_TRACK_OTHER)].copy()
if sel.empty:
    raise ValueError(f"{SELECT_SAT} {SELECT_TRACK_SELF}x{SELECT_TRACK_OTHER} "
                     "not in the intersections file")
sel["distance_km"] = [distance.distance((la, lo), (lat_omni, lon_omni)).km
                      for la, lo in zip(sel["lats_inter"], sel["lons_inter"])]
row = sel.nsmallest(1, "distance_km").iloc[0]

sat = row["sat"]
t_self, t_other = str(row["track_self"]), str(row["track_other"])
lon1, lat1, distance_km = row["lons_inter"], row["lats_inter"], row["distance_km"]
unique_str = f"{sat}_{t_self}_{t_other}"
print(f"{sat} {SELECT_OMNI_ID} {t_self} {t_other} "
      f"({lon1:.3f}E, {lat1:.3f}N) {distance_km:.2f} km from the buoy")

lons_self, lats_self, ang_self_deg, ang_self = track_geometry(sat, t_self, lat1)
lons_other, lats_other, ang_other_deg, ang_other = track_geometry(sat, t_other, lat1)

# ── the track current ───────────────────────────────────────────────────────
if RECOMPUTE_GUV:
    def gc_at(track, x_at):
        fn = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{str(track).zfill(3)}.nc"
        with xr.open_dataset(fn, decode_times=False) as ds:
            sla = track_dist_time_asn_midx(ds, "sla", units_in="m")
        gc = cross_track_geostrophy(smooth_loess(sla, frac=0.05))
        return gc.sel(x=x_at, method="nearest").reset_coords(drop=True)

    gc_self = gc_at(t_self, row["x_from_coast_self"])
    gc_other = gc_at(t_other, row["x_from_coast_other"]).interp(time=gc_self.time)
    gu, gv = resolve_uv(mad_filter(gc_self, 3.5, dim="time"),
                        mad_filter(gc_other, 3.5, dim="time"), ang_self, ang_other)
    source = "recomputed here (smooth_loess + cross_track_geostrophy + resolve_uv)"
else:
    f_guv = f"{GUV_DIR}/guv_at_intersection_{sat}_{t_self}_{t_other}_{METHOD}_{FRAC}.nc"
    if not os.path.exists(f_guv):
        raise FileNotFoundError(f_guv)
    with xr.open_dataset(f_guv) as ds_guv:
        gu, gv = ds_guv.gu.load(), ds_guv.gv.load()
    source = f"stored product {Path(f_guv).name}"
print(f"gu source: {source}")

# ── observations and references, all on the gu timestamps ───────────────────
with xr.open_dataset(omni_file) as ds_omni:
    omni_u = ds_omni.u_current.interp(current_depth=OMNI_DEPTH).resample(time="1D").mean()

f_ek = f"{EKMAN_DIR}/{unique_str}_ekman_{EKMAN_DEPTH}m_at_intersections_cd.nc"
if not os.path.exists(f_ek):
    raise FileNotFoundError(f"no {EKMAN_DEPTH} m Ekman file: {f_ek}")
with xr.open_dataset(f_ek) as ds_ek:
    u_ek = ds_ek["u_at"].load()

with xr.open_dataset(f"{AVISO_DIR}/allyear_dt_global_allsat_msla_u.nc") as ds_av:
    msla_u = ds_av.u.interp(longitude=lon1, latitude=lat1).load()

omni_m = match(omni_u, gu.time)
ek_m = match(u_ek, gu.time)
# AVISO msla_uv here is MONTHLY (31 d step, mid-month stamps). Nearest-within-
# 24 h keeps only 8 of 62 cycles, so the monthly field is interpolated instead
# and labelled as the monthly reference it is.
msla_m = msla_u.interp(time=gu.time).reset_coords(drop=True)

gu = gu.reset_coords(drop=True)
gv = gv.reset_coords(drop=True)
valid = np.isfinite(gu) & np.isfinite(omni_m)
gu_p, omni_p, ek_p, msla_p = (gu.where(valid), omni_m.where(valid),
                              ek_m.where(valid), msla_m.where(valid))
gu_ek_p = gu_p + ek_p

metrics = {name: skill(pred, omni_p) for name, pred in
           (("track", gu_p), (f"track+ekman{EKMAN_DEPTH}m", gu_ek_p), ("aviso_msla_monthly", msla_p))}
for name, (r, rmse, n) in metrics.items():
    print(f"  {name:<20} r = {r:+.3f}   rmse = {rmse:.3f} m/s   n = {n}")

xr.Dataset(
    {"gu": gu_p, "gv": gv.where(valid), "omni_u": omni_p,
     "u_ekman": ek_p, "gu_plus_ekman": gu_ek_p, "aviso_msla_u": msla_p},
    attrs={"sat": sat, "omni_id": SELECT_OMNI_ID, "track_self": t_self,
           "track_other": t_other, "lon_inter": lon1, "lat_inter": lat1,
           "distance_to_buoy_km": float(distance_km), "omni_depth_m": OMNI_DEPTH,
           "ekman_depth_m": EKMAN_DEPTH, "gu_source": source,
           "heading_self_deg": float(ang_self_deg), "heading_other_deg": float(ang_other_deg),
           **{f"{k}_r": v[0] for k, v in metrics.items()},
           **{f"{k}_rmse": v[1] for k, v in metrics.items()}},
).to_netcdf(nc_file)
print(f"wrote {nc_file}")

# ── figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax1 = fig.add_subplot(2, 1, 2)

with xr.open_dataset(TOPO) as dse:
    dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2]),
                 ETOPO05_Y=slice(*TRACKS_REG[2:])).plot(
        ax=ax2, add_colorbar=False, add_labels=False, cmap=CMAP)
decorate_axis(ax2, "", *TRACKS_REG, step=10)
ax2.grid()
ax2.scatter(lons_self, lats_self, marker=".", color="c", s=4)
ax2.scatter(lons_other, lats_other, marker=".", color="c", s=4)
ax2.plot(lon_omni, lat_omni, marker="o", ms=6, mfc="red", mec="red")
ax2.plot(lon1, lat1, marker=".", ms=10, mfc="black", mec="black")
ax2.text(lon_omni - 0.5, lat_omni, SELECT_OMNI_ID, transform=ccrs.PlateCarree(),
         fontsize=10, color="r", ha="right")
# label the two tracks at different fractions along their length, otherwise
# they collide with each other and with the buoy marker at the intersection
for lons, lats, ang, number, frac in ((lons_self, lats_self, ang_self_deg, t_self, 0.45),
                                      (lons_other, lats_other, ang_other_deg, t_other, 0.80)):
    i = int(frac * (len(lons) - 1))
    if is_within_region(lons[i], lats[i], *TRACKS_REG):
        ax2.text(lons[i], lats[i], s=number, fontsize=13, rotation=ang, color="w",
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.45))
ax2.text(0.49, 0.92, f"{sat} {t_self} {t_other} {distance_km:.2f} km",
         transform=ax2.transAxes, fontsize=10, fontweight="bold")
ax2.text(0.05, 0.80, "a)", transform=ax2.transAxes, fontsize=14,
         fontweight="bold", color="k")

omni_p.plot(ax=ax1, color="r", lw=2, label="OMNI")
gu_p.plot(ax=ax1, color="k", lw=2, label="track")
gu_ek_p.plot(ax=ax1, color="b", lw=2, label=f"track+ekman {EKMAN_DEPTH} m")
msla_p.plot(ax=ax1, color="c", lw=2, label="AVISO msla u (monthly)")
ax1.axhline(0, color="k", ls="--", lw=0.5)
ax1.set_xlim(omni_p.time.min(), omni_p.time.max())
ax1.set_ylabel("zonal current (m/s)")
ax1.set_xlabel("")
ax1.text(0.25, 0.90, f"intersection ({lon1:.2f}, {lat1:.2f}) {sat} "
                     f"{distance_km:.2f} km  |  OMNI {OMNI_DEPTH} m",
         transform=ax1.transAxes, fontsize=11, fontweight="bold")
ax1.text(0.02, 0.80, "b)", transform=ax1.transAxes, fontsize=14,
         fontweight="bold", color="k")
ax1.text(0.02, 0.03, "\n".join(
    f"{n}: r={r:+.2f}  rmse={e:.3f}  n={k}" for n, (r, e, k) in metrics.items()),
    transform=ax1.transAxes, fontsize=9, va="bottom",
    bbox=dict(boxstyle="round", fc="w", alpha=0.75))
ax1.legend(loc="upper right", fontsize=9)

fig.savefig(png_file, dpi=300, bbox_inches="tight")
plt.show()
plt.close("all")
print(f"wrote {png_file}")
