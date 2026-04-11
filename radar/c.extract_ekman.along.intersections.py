import os
import pickle

import xarray as xr
from geopy import distance
from tools_xtrackm import *

RADARS = ("TN", "AP", "OR")
OUT_DIR = "ekman_along_intersections"

ekman_PATH_PRIMARY = "/media/srinivasu/bkup1tb/allData_archive/ccmp31/ekman_velocities_0m_all_years_daily_cd.nc"
ekman_PATH_FALLBACK = "~/allData/ccmp31/ekman_velocities_0m_all_years_daily_cd.nc"


def extract_ekman_uv_2d(uwnd, vwnd, lons_coast_to_off, lats_coast_to_off, xvals):
    u_list = []
    v_list = []
    for lon, lat in zip(lons_coast_to_off, lats_coast_to_off):
        u_at = uwnd.sel(longitude=lon, latitude=lat, method="nearest", drop=True)
        v_at = vwnd.sel(longitude=lon, latitude=lat, method="nearest", drop=True)
        u_list.append(u_at)
        v_list.append(v_at)
    u_2d = xr.concat(u_list, dim="x").assign_coords(x=xvals)
    v_2d = xr.concat(v_list, dim="x").assign_coords(x=xvals)
    return u_2d, v_2d


def get_intersection_endpoints(lons_coast_to_off, lats_coast_to_off, polygon, lon_coast, lat_coast):
    linestring = sg.LineString(list(zip(lons_coast_to_off, lats_coast_to_off)))
    inter = linestring.intersection(polygon)
    if inter.is_empty:
        return None
    if hasattr(inter, "geoms"):
        line_geoms = [g for g in inter.geoms if g.geom_type in ("LineString", "LinearRing")]
        if not line_geoms:
            return None
        inter_line = max(line_geoms, key=lambda g: g.length)
    elif inter.geom_type in ("LineString", "LinearRing"):
        inter_line = inter
    else:
        return None

    lon_x1, lat_x1 = inter_line.coords[0]
    lon_x2, lat_x2 = inter_line.coords[-1]
    x1_m = distance.distance((lat_coast, lon_coast), (lat_x1, lon_x1)).m
    x2_m = distance.distance((lat_coast, lon_coast), (lat_x2, lon_x2)).m
    return lon_x1, lat_x1, lon_x2, lat_x2, x1_m, x2_m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ekman_path = ekman_PATH_PRIMARY if os.path.exists(ekman_PATH_PRIMARY) else ekman_PATH_FALLBACK
    if not os.path.exists(ekman_path):
        raise FileNotFoundError(
            f"ekman file not found at {ekman_PATH_PRIMARY} or {ekman_PATH_FALLBACK}"
        )

    with xr.open_dataset(ekman_path) as ds_ekman:
        uwnd = ds_ekman.u_ekman.load()
        vwnd = ds_ekman.v_ekman.load()

    for radar_str in RADARS:
        polygon = gpd.read_file(f"polygon_shapefile_around_{radar_str}.shp").geometry.iloc[0]

        f_tracks = f"tracks_passing_{radar_str}.pkl"
        if not os.path.exists(f_tracks):
            print(f"missing pickle: {f_tracks}")
            continue

        with open(f_tracks, "rb") as f:
            tracks_dict = pickle.load(f)

        n_saved = 0
        for sat, files in sorted(tracks_dict.items()):
            for f_track in files:
                if not os.path.exists(f_track):
                    continue

                with xr.open_dataset(f_track, decode_times=False) as ds_track:
                    tno = str(int(ds_track.Pass))
                    sla_asn = track_dist_time_asn(ds_track, var_str="sla", units_in="m")
                    xvals = sla_asn.x.values
                    lons_coast_to_off = ds_track.lon.values[::-1]
                    lats_coast_to_off = ds_track.lat.values[::-1]
                    dist_to_coast_coast_to_off = ds_track.dist_to_coast_gshhg.values[::-1]

                if len(lons_coast_to_off) < 2:
                    continue

                lon_coast = lons_coast_to_off[0]
                lat_coast = lats_coast_to_off[0]
                inter_data = get_intersection_endpoints(
                    lons_coast_to_off, lats_coast_to_off, polygon, lon_coast, lat_coast
                )
                if inter_data is None:
                    continue
                lon_x1, lat_x1, lon_x2, lat_x2, x1_m, x2_m = inter_data

                u_2d, v_2d = extract_ekman_uv_2d(
                    uwnd, vwnd, lons_coast_to_off, lats_coast_to_off, xvals
                )

                mask_in_polygon = xr.DataArray(
                    np.array(
                        [polygon.covers(Point(lo, la)) for lo, la in zip(lons_coast_to_off, lats_coast_to_off)],
                        dtype=bool,
                    ),
                    coords={"x": xvals},
                    dims=["x"],
                    name="mask_in_polygon",
                )

                ds_out = xr.Dataset(
                    {
                        "ekman_u": u_2d,
                        "ekman_v": v_2d,
                        "mask_in_polygon": mask_in_polygon,
                        "lon_track": xr.DataArray(lons_coast_to_off, coords={"x": xvals}, dims=["x"]),
                        "lat_track": xr.DataArray(lats_coast_to_off, coords={"x": xvals}, dims=["x"]),
                        "dist_to_coast_gshhg_track": xr.DataArray(
                            dist_to_coast_coast_to_off,
                            coords={"x": xvals},
                            dims=["x"],
                            attrs={"description": "Distance to coast along track (coast to offshore order)"},
                        ),
                    }
                )
                ds_out.attrs.update(
                    {
                        "source_file": ekman_path,
                        "radar_str": radar_str,
                        "sat": sat,
                        "track_number": tno,
                        "lon_x1": float(lon_x1),
                        "lat_x1": float(lat_x1),
                        "lon_x2": float(lon_x2),
                        "lat_x2": float(lat_x2),
                        "x1_m": float(x1_m),
                        "x2_m": float(x2_m),
                    }
                )

                f_out = f"{OUT_DIR}/ekman_along_{radar_str}_{sat}_{tno}.nc"
                ds_out.to_netcdf(f_out)
                n_saved += 1
                print(f"saved {f_out}")

        print(f"{radar_str}: saved {n_saved} ekman extracted files")


if __name__ == "__main__":
    main()
