#!/usr/bin/env python3
"""Create a PowerPoint with one PNG image per slide.

Default input pattern: fig*.png and sup*.png in the current directory.
"""
# codex resume 019dabb5-b4b4-75f0-b5b1-15210e62fccf
from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from pptx import Presentation
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: python-pptx. Install with: pip install python-pptx"
    ) from exc


NAT_SORT_RE = re.compile(r"(\d+)")


TOP_BOX_HEIGHT = Inches(0.7)
BOTTOM_BOX_HEIGHT = Inches(0.9)
SIDE_MARGIN = Inches(0.3)
INNER_GAP = Inches(0.12)


def natural_key(path: Path) -> list[object]:
    """Sort paths naturally so fig2 comes before fig10."""
    return [int(part) if part.isdigit() else part.lower() for part in NAT_SORT_RE.split(path.name)]


def collect_images(input_dir: Path, patterns: list[str]) -> list[Path]:
    images: list[Path] = []
    for pattern in patterns:
        images.extend(input_dir.glob(pattern))

    # Keep only files and avoid duplicates across overlapping patterns.
    unique = {p.resolve(): p for p in images if p.is_file()}
    return sorted(unique.values(), key=natural_key)


def read_caption_text(image_path: Path) -> str:
    caption_path = image_path.with_suffix(".txt")
    if not caption_path.is_file():
        return ""
    return caption_path.read_text(encoding="utf-8").strip()


def add_image_slide(prs: Presentation, image_path: Path) -> None:
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    slide_w = prs.slide_width
    slide_h = prs.slide_height

    top_box = slide.shapes.add_textbox(0, 0, slide_w, TOP_BOX_HEIGHT)
    top_box.text_frame.clear()
    top_p = top_box.text_frame.paragraphs[0]
    top_p.text = "Title:"
    top_p.alignment = PP_ALIGN.LEFT
    top_p.font.size = Pt(18)

    caption = read_caption_text(image_path)
    caption_box = slide.shapes.add_textbox(
        SIDE_MARGIN,
        slide_h - BOTTOM_BOX_HEIGHT,
        slide_w - 2 * SIDE_MARGIN,
        BOTTOM_BOX_HEIGHT,
    )
    caption_tf = caption_box.text_frame
    caption_tf.clear()
    caption_tf.word_wrap = True
    caption_p = caption_tf.paragraphs[0]
    caption_p.text = caption
    caption_p.alignment = PP_ALIGN.JUSTIFY
    caption_p.font.size = Pt(12)
    caption_p.font.bold = False

    picture = slide.shapes.add_picture(str(image_path), 0, 0)
    img_w = picture.width
    img_h = picture.height

    image_area_w = slide_w - (2 * SIDE_MARGIN)
    image_top = TOP_BOX_HEIGHT + INNER_GAP
    image_bottom = slide_h - BOTTOM_BOX_HEIGHT
    image_area_h = image_bottom - image_top

    scale = min(image_area_w / img_w, image_area_h / img_h)
    final_w = int(img_w * scale)
    final_h = int(img_h * scale)
    left = int((slide_w - final_w) / 2)
    top = int(image_top + (image_area_h - final_h) / 2)

    picture.width = final_w
    picture.height = final_h
    picture.left = left
    picture.top = top


def build_pptx(images: list[Path], output_path: Path) -> None:
    prs = Presentation()
    for image in images:
        add_image_slide(prs, image)
    prs.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PPTX with one image per slide from fig*.png/sup*.png.",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory to search for images (default: current directory)",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        action="append",
        default=["fig*.png", "sup*.png"],
        help="Glob pattern to include (repeatable). Defaults: fig*.png and sup*.png",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("coastal_alt_ppt.pptx"),
        help="Output PPTX file path (default: fig_sup_images.pptx)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    images = collect_images(input_dir, args.pattern)

    if not images:
        patterns = ", ".join(args.pattern)
        raise SystemExit(f"No images found in {input_dir} for patterns: {patterns}")

    output_path = args.output
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    build_pptx(images, output_path)
    print(f"Created {output_path} with {len(images)} slides")


if __name__ == "__main__":
    main()
