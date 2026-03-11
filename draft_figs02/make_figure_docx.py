from docx import Document
from docx.shared import Inches
from pathlib import Path
import glob
import os

doc_name = "figures_v2.1.docx"
if os.path.exists(doc_name):
    os.remove(doc_name)

images = sorted(glob.glob("fig??.png"))
doc = Document()

for i, img in enumerate(images):
    doc.add_picture(img, width=Inches(6))  # adjust width if needed
    
    if i < len(images) - 1:
        doc.add_page_break()

doc.save("figures_v2.1.docx")
