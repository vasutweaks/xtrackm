import shutil
from pathlib import Path
import os
from glob import glob

links_dict = {
    "p.compare.std_mean.l2pVSxtrack_map.py":"fig01.py",
    "p.omni_uvVStrack_guv_c.ek.nearest.selected.py": "fig02.py",
    "p.scatter_omniVSrama_uVSgu_ww_ek.py": "fig03.py",
    "p.scatter_drifter_ve_vs_gu_ww_ek.py": "fig04.py",
    "p.radar_uvVStrack_guv.at.intersections_single.py": "fig05.py",
    "p.gc.clim.cross_structure_x_arrows_map_ERS.py": "fig06.py",
    "p.gc.clim.lat_wise.along_xwidth.py": "fig07.py",
    "p.along_shore_at_155_116.intersection.S6A.py": "fig08.py",
    "p.speed_at_coastal_points.eicc.trends.all_in_one.ERS.py": "fig09.py",
    "satellite_mission_time_spans_new.py": "fig_sup_01.py"
}

# archive_dir = Path("archive")
# for full_path, fig_path in links_dict.items():
#     src = Path(full_path)
#     dst = Path(fig_path)
#     stem = dst.stem
#     new_name = f"{stem}_{src.name}.py"
#     # check if src exists
#     if src.exists():
#         print(f"{src.name} --> {new_name}")
#         os.rename(src, new_name)
#     else:
#         print(f"File {src} does not exist ---------------------------")

def get_fig_glob(f):
    GLOB = f"fig*{f}"
    files = glob(GLOB)
    return files[0]


for f in links_dict.keys():
    figf = get_fig_glob(f)
    if Path(figf).exists():
        print(f"{figf} exists")
    else:
        print(f"{figf} does not exist ---------------------------")












