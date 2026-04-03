import os
import re
from glob import glob

base_names = {
    "f.combined_map_std_valid.S3A.py": 1,
    "p.omni_uvVStrack_guv_c.ek.nearest.selected.py": 1,
    "p.scatter_omniVSrama_uVSgu_ww_ek.py": 1,
    "p.scatter_drifter_ve_vs_gu_ww_ek.py": 1,
    "p.radar_uvVStrack_guv.at.intersections_single.py": 1,
    "p.along_shore_at_155_116.intersection.S6A.py": 1,
    "p.speed_at_coastal_points.eicc.trends.all_in_one.ERS.py": 1,
    "satellite_mission_time_spans_new.py": 2,
    "p.compare.std_mean.l2pVSxtrack_map.py": 0,
    "p.gc.clim.cross_structure_x_arrows_map_ERS.py": 0,
    "p.gc.clim.lat_wise.along_xwidth.py": 0,
    "p.aviso_only_uvVStrack_guv.at.intersections_single.py": 0,
    "p.metrics.track_guvVSoscar_ww_ekman.py": 0,
    "p.vskp.tseries_new_v2.2.py": 0,
}


def get_clean_name(filename):
    """Strips any existing 'figNN_' or 'supNN_' prefix."""
    return re.sub(r'^(fig|sup)\d{2}_', '', filename)


def strip_all_files(base_names):
    files = sorted(glob(f"fig??_*.py"))
    if not files:
        print("No figNN files found.")
        return
    for f in files:
        f_clean = get_clean_name(f)
        print(f"Striping {f} -> {f_clean}")
        os.rename(f, f_clean)
    files = sorted(glob(f"sup??_*.py"))
    if not files:
        print("No supNN files found.")
        return
    for f in files:
        f_clean = get_clean_name(f)
        print(f"Striping {f} -> {f_clean}")
        os.rename(f, f_clean)
        

def new_order_rename(base_names):
    strip_all_files(base_names)
    fig_i = 1
    sup_j = 1
    print(f"{'Current File':<45} | {'New Name':<45} | {'Caption'}")
    print("-" * 60)
    for original_key, status in base_names.items():
        if status == 1:
            new_name = f"fig{str(fig_i).zfill(2)}_{original_key}"
            fig_i += 1
        elif status == 2:
            new_name = f"sup{str(sup_j).zfill(2)}_{original_key}"
            sup_j += 1
        else:
            new_name = original_key

        # Check Caption
        ktxt = f"{os.path.splitext(original_key)[0]}.txt"
        kpng = f"{os.path.splitext(original_key)[0]}.png"
        # cap_path = os.path.join(cap_dir, ktxt)
        # cap_status = "✅" if os.path.exists(cap_path) else "❌"
        if os.path.exists(original_key):
            os.rename(original_key, new_name)
            print(f"{original_key:<45} -> {new_name:<45}")

new_order_rename(base_names)
# sync_filenames_and_captions(base_names)
