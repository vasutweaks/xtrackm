import shutil
from pathlib import Path

links_dict = {
    "p.compare.std_mean.l2pVSxtrack_map.py":"fig01.py",
    "p.omni_uvVStrack_guv_c.ek.nearest.selected.py": "fig02.py",
    "p.scatter_omniVSrama_uVSgu_ww_ek.py": "fig03.py",
    "p.scatter_drifter_ve_vs_gu_ww_ek.py": "fig04.py",
    "p.radar_uvVStrack_guv.at.intersections_single.py": "fig05.py",
    "p.gc.clim.cross_structure_x_arrows_map_ERS.py": "fig06.py",
    "p.gc.clim.lat_wise.along_xwidth.py": "fig07.py",
    "p.along_shore_at_155_116.intersection.S6A.py": "fig08.py",
}

archive_dir = Path("archive")
for full_path, fig_path in links_dict.items():
    src = Path(full_path)
    dst = Path(fig_path)

    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            dst.unlink()
        elif dst.is_file():
            archive_path = archive_dir / dst.name
            if archive_path.exists():
                archive_path.unlink()
            shutil.move(str(dst), str(archive_path))
        else:
            continue
    print(f"{src} -> {dst}")
    dst.hardlink_to(src)
