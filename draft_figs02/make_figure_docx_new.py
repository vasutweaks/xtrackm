import os
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from docx.shared import Pt
from pathlib import Path

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

image_list = []
caption_list = []
fig_i = 1
sup_j = 1
for original_key, status in base_names.items():
    if status == 0:
        continue
    if status == 1:
        prefix = f"fig{str(fig_i).zfill(2)}"
        image_name = Path(original_key).with_suffix(".png")
        image_name = f"{prefix}_{image_name}"
        if os.path.exists(image_name):
            # print(f"{image_name} exists")
            image_list.append(image_name)
            fig_i += 1
    elif status == 2:
        prefix = f"sup{str(sup_j).zfill(2)}"
        image_name = Path(original_key).with_suffix(".png")
        image_name = f"{prefix}_{image_name}"
        if os.path.exists(image_name):
            # print(f"{image_name} exists")
            image_list.append(image_name)
            sup_j += 1
    if status != 0:
        prefix = "captions/"
        caption_name = Path(original_key).with_suffix(".txt")
        caption_name = f"{prefix}{caption_name}"
        if os.path.exists(caption_name):
            print(f"{caption_name} exists")
            caption_list.append(caption_name)

# Caption styling options (edit these as needed).
CAPTION_FONT_NAME = "Times New Roman"
CAPTION_FONT_SIZE_PT = 12
CAPTION_ALIGNMENT = "center"  # one of: left, center, right, justify

ALIGNMENT_MAP = {
    "left": WD_ALIGN_PARAGRAPH.LEFT,
    "center": WD_ALIGN_PARAGRAPH.CENTER,
    "right": WD_ALIGN_PARAGRAPH.RIGHT,
    "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
}

doc_name = "figures_v2.1.docx"
if os.path.exists(doc_name):
    os.remove(doc_name)

doc = Document()
inserted_count = 0

for i, img in enumerate(image_list):
    if not os.path.exists(img):
        print(f"Skipping missing image: {img}")
        continue

    doc.add_picture(img, width=Inches(6))
    inserted_count += 1

    caption_text = ""
    if i < len(caption_list):
        caption_path = caption_list[i]
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                caption_text = f.read().strip()

    prefix = f"Fig{inserted_count}"
    full_caption = f"{prefix}: {caption_text}" if caption_text else prefix

    paragraph = doc.add_paragraph()
    paragraph.alignment = ALIGNMENT_MAP.get(
        CAPTION_ALIGNMENT.lower(), WD_ALIGN_PARAGRAPH.CENTER
    )
    run = paragraph.add_run(full_caption)
    run.font.name = CAPTION_FONT_NAME
    run.font.size = Pt(CAPTION_FONT_SIZE_PT)

    if i < len(image_list) - 1:
        doc.add_page_break()

doc.save(doc_name)
print(f"Inserted {inserted_count} figure(s) into {doc_name}")
# convert docx to pdf
cmd = "libreoffice --headless --convert-to pdf " + doc_name
os.system(cmd)
