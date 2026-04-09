import xarray as xr
from tools_xtrackm import *
import pickle

sat_time_limits = {}
for sat in sats_new:
    try:
        sat_time_limits[sat] = get_time_limits_o(sat)
    except Exception as e:
        sat_time_limits[sat] = (None, None)
        print(f"warning: could not read time limits for {sat}: {e}")

for radar_str in ["TN", "AP", "OR"]:
    REG = radar_extents[radar_str + "_BLOCK"]
    radar_tsta, radar_tend = radar_times[radar_str]
    gdf = gpd.read_file(f"polygon_shapefile_around_{radar_str}.shp")
    polygon = gdf.geometry.iloc[0]
    tracks_dict = {}
    for sat in sats_new:
        sat_tsta, sat_tend = sat_time_limits[sat]
        overlap_start, overlap_end = overlap_dates(
            radar_tsta, radar_tend, sat_tsta, sat_tend
        )
        if overlap_start is None or overlap_end is None:
            print(
                f"omitting {sat} for {radar_str}: "
                f"no time overlap ({sat_tsta} to {sat_tend}) vs "
                f"({radar_tsta} to {radar_tend})"
            )
            continue

        files = sorted(
            glob.glob(
                f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc"
            )
        )
        tracks_passing = []
        for f in files:
            with xr.open_dataset(f, decode_times=False) as ds:
                track_number = ds.Pass
                lons_track = ds.lon.values
                lats_track = ds.lat.values
            if len(lons_track) == 0:
                continue
            track_path = sg.LineString(
                [(lon, lat) for lon, lat in zip(lons_track, lats_track)]
            )
            if polygon.intersects(track_path):
                tracks_passing.append(f)
                print(f"{sat} track {track_number} is passing {radar_str}")
        tracks_dict[sat] = tracks_passing

    with open(f"tracks_passing_{radar_str}.pkl", "wb") as f:
        pickle.dump(tracks_dict, f)
