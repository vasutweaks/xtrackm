from glob import glob

import xarray as xr
from tools_xtrackm import *

# Checks:
# 1. Whether lat[0] is the equator-end and lat[-1] is the coast-end
# 2. MultiPoint / non-Point intersection geometries

multipoint_cases = []  # (sat, f_self, f_other, geom_type, wkt_snippet)
order_issues = []      # (sat, file, lat[0], lat[-1], note)

for sat in sats_new:
    for f_self in sorted(
            glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")):
        print(f_self)
        ds_self = xr.open_dataset(f_self, decode_times=False, engine="netcdf4")
        lons_track_self = ds_self.lon.values
        lats_track_self = ds_self.lat.values

        if len(lons_track_self) == 0:
            continue

        # Check 1: equator/coast ordering
        # Assumption: lat[0] is closer to equator (smaller |lat|)
        #             lat[-1] is closer to coast   (larger  |lat|)
        lat_first = float(lats_track_self[0])
        lat_last  = float(lats_track_self[-1])
        if abs(lat_first) >= abs(lat_last):
            order_issues.append((
                sat, f_self,
                round(lat_first, 4), round(lat_last, 4),
                "lat[0] is NOT closer to equator than lat[-1]"
            ))

        slope_self = (lat_last - lat_first) / (
            float(lons_track_self[-1]) - float(lons_track_self[0]))
        if slope_self < 0:
            continue

        track_path_self = sg.LineString(zip(lons_track_self, lats_track_self))

        for f_other in sorted(
                glob.glob(f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")):
            ds_other = xr.open_dataset(f_other, engine="netcdf4",
                                       decode_times=False)
            lons_track_other = ds_other.lon.values
            lats_track_other = ds_other.lat.values

            if len(lons_track_other) == 0:
                continue

            slope_other = (
                (float(lats_track_other[-1]) - float(lats_track_other[0])) /
                (float(lons_track_other[-1]) - float(lons_track_other[0])))
            if slope_other > 0:
                continue

            track_path_other = sg.LineString(
                zip(lons_track_other, lats_track_other))

            if track_path_self.intersects(track_path_other):
                geom = track_path_self.intersection(track_path_other)

                # Check 2: expect a single Point; flag anything else
                if geom.geom_type != "Point":
                    multipoint_cases.append((
                        sat,
                        f_self,
                        f_other,
                        geom.geom_type,
                        str(geom)[:150],
                    ))

# ── Report ─────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"CHECK 1 — EQUATOR/COAST ORDER ISSUES  ({len(order_issues)} found)")
print("=" * 70)
for sat, f, la0, la1, note in order_issues:
    print(f"  sat={sat}  lat[0]={la0:8.4f}  lat[-1]={la1:8.4f}")
    print(f"  note : {note}")
    print(f"  file : {f}")
    print()

print("=" * 70)
print(f"CHECK 2 — NON-POINT INTERSECTIONS  ({len(multipoint_cases)} found)")
print("=" * 70)
for sat, fs, fo, gtype, wkt in multipoint_cases:
    print(f"  sat={sat}  geom_type={gtype}")
    print(f"  self : {fs}")
    print(f"  other: {fo}")
    print(f"  geom : {wkt}")
    print()

print("Done.")
