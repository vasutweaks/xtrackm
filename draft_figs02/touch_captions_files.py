from pathlib import Path

long_name_list = [
    "p.compare.std_mean.l2pVSxtrack_map.py",
    "p.omni_uvVStrack_guv_c.ek.nearest.selected.py",
    "p.scatter_omniVSrama_uVSgu_ww_ek.py",
    "p.scatter_drifter_ve_vs_gu_ww_ek.py",
    "p.radar_uvVStrack_guv.at.intersections_single.py",
    "p.gc.clim.cross_structure_x_arrows_map_ERS.py",
    "p.gc.clim.lat_wise.along_xwidth.py",
    "p.along_shore_at_155_116.intersection.S6A.py",
]

for f in long_name_list:
    p = Path(f)
    p.with_suffix(".txt").touch()
