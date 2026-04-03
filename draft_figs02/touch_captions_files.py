from pathlib import Path

base_names = {
    "p.aviso_only_uvVStrack_guv.at.intersections_single.py": 0,
    "p.metrics.track_guvVSoscar_ww_ekman.py": 0,
    "p.vskp.tseries_new_v2.2.py": 0,
    "f.combined_map_std_valid.S3A.py": 1,
    "p.compare.std_mean.l2pVSxtrack_map.py": 1,
    "p.omni_uvVStrack_guv_c.ek.nearest.selected.py": 1,
    "p.scatter_omniVSrama_uVSgu_ww_ek.py": 1,
    "p.scatter_drifter_ve_vs_gu_ww_ek.py": 1,
    "p.radar_uvVStrack_guv.at.intersections_single.py": 1,
    "p.gc.clim.cross_structure_x_arrows_map_ERS.py": 0,
    "p.gc.clim.lat_wise.along_xwidth.py": 0,
    "p.along_shore_at_155_116.intersection.S6A.py": 1,
    "p.speed_at_coastal_points.eicc.trends.all_in_one.ERS.py": 1,
    "satellite_mission_time_spans_new.py": 2
}

for k, v in base_names.items():
    # add captions dir before each file
    k1 = "captions/" + k
    p = Path(k1)
    # check if p exists
    if p.with_suffix(".txt").exists():
        print(f"File already exists: {p}")
    else:
        print(f"touching file: {p.with_suffix('.txt')}")
        # p.with_suffix(".txt").touch()
