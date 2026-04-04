#!/usr/bin/env python3
"""
code2pdf.py — Gather all .py and .txt files and produce one combined,
syntax-highlighted PDF with a clickable Table of Contents.

Usage:
    python code2pdf.py                                  # current dir, recursive
    python code2pdf.py -o xtrackm_code.pdf              # custom output name
    python code2pdf.py --dir computed/ guv/ lib/         # specific directories
    python code2pdf.py --theme github-dark --font-size 8
    python code2pdf.py --exclude archive/ --exclude draft_figs02/

The script walks the directory tree, collects every .py and .txt file,
groups them by directory, and writes them into a single PDF with:
  • Table of Contents (page numbers for each file)
  • Directory separator pages
  • Syntax colouring (Pygments) for .py files
  • Line numbers, dark background, filename headers, page numbers
"""
# copilot --resume=ae78b187-8512-43ec-af6b-5225001dbe76
import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from pygments import highlight
from pygments.lexers import PythonLexer, TextLexer, get_lexer_for_filename
from pygments.formatter import Formatter
from pygments.styles import get_style_by_name, get_all_styles
from pygments.token import Token


# ── Helpers ──────────────────────────────────────────────────────────────────

def hex_to_rgb(hex_color: str):
    """Convert '#rrggbb' to (r, g, b) ints."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def choose_lexer(filepath: str):
    """Return an appropriate Pygments lexer for *filepath*."""
    try:
        return get_lexer_for_filename(filepath)
    except Exception:
        if filepath.endswith(".py"):
            return PythonLexer()
        return TextLexer()


def collect_files(roots: list[str], exclude: list[str] | None = None) -> list[str]:
    """Recursively collect .py and .txt files under *roots*, skipping *exclude* dirs."""
    extensions = {".py", ".txt"}
    exclude = set(os.path.normpath(e) for e in (exclude or []))
    files = []
    for root in roots:
        p = Path(root)
        if p.is_file() and p.suffix in extensions:
            files.append(str(p))
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            # Prune excluded and hidden directories
            rel = os.path.relpath(dirpath, ".")
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".")
                and not d.startswith("__")
                and os.path.normpath(os.path.join(rel, d)) not in exclude
            ]
            dirnames.sort()
            for fn in sorted(filenames):
                if Path(fn).suffix in extensions:
                    files.append(os.path.join(dirpath, fn))
    return files


def group_by_directory(files: list[str]) -> dict[str, list[str]]:
    """Group file paths by their parent directory (relative)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for f in files:
        d = os.path.dirname(f) or "."
        groups[d].append(f)
    return dict(groups)


# ── Custom Pygments → fpdf2 formatter ───────────────────────────────────────

class _TokenAccumulator(Formatter):
    """Collect (token_type, value) pairs from Pygments."""

    def __init__(self, **options):
        super().__init__(**options)
        self.tokens: list = []

    def format(self, tokensource, outfile):
        for ttype, value in tokensource:
            self.tokens.append((ttype, value))


# ── PDF builder ──────────────────────────────────────────────────────────────

class CodePDF(FPDF):
    """FPDF subclass that renders syntax-coloured source code."""

    MONO_FONT = "DejaVu"  # alias we register

    def __init__(self, doc_title: str, style_name: str = "monokai",
                 font_size: int = 9):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self._doc_title = doc_title
        self._file_title = ""          # current file being rendered
        self._style_name = style_name
        self._font_size = font_size

        self._style = get_style_by_name(style_name)
        bg = self._style.background_color or "#272822"
        self._bg_rgb = hex_to_rgb(bg)

        self._color_cache: dict = {}

        # TOC data: list of (level, label, page_number)
        self._toc_entries: list[tuple[int, str, int]] = []

        # Register a Unicode-capable monospace TTF font
        self._register_mono_font()

    # ── header / footer ──────────────────────────────────────────────────

    def _register_mono_font(self):
        """Register DejaVu Sans Mono (Unicode TTF) with all 4 style variants."""
        base = "/usr/share/fonts/truetype/dejavu"
        self.add_font(self.MONO_FONT, "",  f"{base}/DejaVuSansMono.ttf")
        self.add_font(self.MONO_FONT, "B", f"{base}/DejaVuSansMono-Bold.ttf")
        self.add_font(self.MONO_FONT, "I", f"{base}/DejaVuSansMono-Oblique.ttf")
        self.add_font(self.MONO_FONT, "BI", f"{base}/DejaVuSansMono-BoldOblique.ttf")

    def header(self):
        if not self._file_title:
            return
        self.set_font(self.MONO_FONT, "B", 10)
        self.set_text_color(180, 180, 180)
        self.cell(0, 6, self._file_title,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.set_draw_color(100, 100, 100)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font(self.MONO_FONT, "I", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── TOC helpers ──────────────────────────────────────────────────────

    def add_toc_entry(self, level: int, label: str):
        self._toc_entries.append((level, label, self.page_no()))

    def write_toc_pages(self):
        """Insert TOC pages at the beginning (call after all content is added).

        Since fpdf2 doesn't support inserting pages, we build the TOC as the
        first content, then render files.  So we use a two-pass approach:
        pass 1 gathers page numbers, pass 2 builds the final PDF.
        We handle this externally — see build_combined_pdf().
        """
        pass  # handled externally

    # ── directory separator ──────────────────────────────────────────────

    def add_directory_page(self, dirname: str, file_count: int):
        """Full-page divider for a new directory section."""
        self._file_title = ""  # suppress header
        self.add_page()
        self.add_toc_entry(0, dirname)

        # Centered directory name
        self.ln(60)
        self.set_font(self.MONO_FONT, "B", 22)
        self.set_text_color(*self._bg_rgb)
        # Dark box behind text
        self.set_fill_color(*self._bg_rgb)
        box_y = self.get_y() - 4
        self.rect(self.l_margin, box_y,
                  self.w - self.l_margin - self.r_margin, 20, "F")
        self.set_text_color(255, 200, 60)
        self.cell(0, 12, f"  {dirname}/", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(6)
        self.set_font(self.MONO_FONT, "", 12)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, f"  {file_count} file(s)",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── colour resolution ────────────────────────────────────────────────

    def _get_color(self, ttype):
        if ttype in self._color_cache:
            return self._color_cache[ttype]
        t = ttype
        while t:
            se = self._style.style_for_token(t)
            c = se.get("color")
            bold = se.get("bold", False)
            italic = se.get("italic", False)
            if c:
                rgb = hex_to_rgb(f"#{c}")
                self._color_cache[ttype] = (rgb, bold, italic)
                return (rgb, bold, italic)
            t = t.parent
        default = self._style.style_for_token(Token)
        fg = default.get("color") or "f8f8f2"
        rgb = hex_to_rgb(f"#{fg}")
        self._color_cache[ttype] = (rgb, False, False)
        return (rgb, False, False)

    # ── code rendering ───────────────────────────────────────────────────

    def render_file(self, filepath: str):
        """Read, tokenize, and render a single file into the PDF."""
        source = Path(filepath).read_text(errors="replace")
        if not source.strip():
            return

        rel = os.path.relpath(filepath, ".")
        self._file_title = rel
        self.add_page()
        self.add_toc_entry(1, rel)

        lexer = choose_lexer(filepath)
        formatter = _TokenAccumulator()
        highlight(source, lexer, formatter)

        line_h = self._font_size * 0.45
        # Measure one space to get the true monospace glyph width from the font
        self.set_font(self.MONO_FONT, "", self._font_size)
        char_w = self.get_string_width(" ")
        tab_w = char_w * 4
        right_limit = self.w - self.r_margin

        self._draw_bg()
        x_start = self.l_margin + 12
        x = x_start
        line_no = 1
        self._draw_line_number(line_no, line_h)

        for ttype, value in formatter.tokens:
            (r, g, b), bold, italic = self._get_color(ttype)
            st = ("B" if bold else "") + ("I" if italic else "")
            self.set_font(self.MONO_FONT, st, self._font_size)
            self.set_text_color(r, g, b)

            # Split token on newlines, render each segment as a whole string
            parts = value.split("\n")
            for pi, part in enumerate(parts):
                # Handle newline between parts (all but the first piece)
                if pi > 0:
                    line_no += 1
                    x = x_start
                    self.ln(line_h)
                    if self.get_y() > self.h - 18:
                        self.add_page()
                        self._draw_bg()
                        x = x_start
                    self._draw_line_number(line_no, line_h)

                if not part:
                    continue

                # Expand tabs to spaces for correct width measurement
                part = part.replace("\t", "    ")

                # Soft-wrap long lines
                while part:
                    avail = right_limit - x
                    seg_w = self.get_string_width(part)

                    if seg_w <= avail:
                        # Fits on the current line — write it
                        self.set_xy(x, self.get_y())
                        self.cell(seg_w, line_h, part)
                        x += seg_w
                        break
                    else:
                        # Find how many chars fit in the available space
                        fit = int(avail / char_w) if char_w > 0 else len(part)
                        fit = max(fit, 1)
                        # Fine-tune: shrink if overflows
                        while fit > 1 and self.get_string_width(part[:fit]) > avail:
                            fit -= 1

                        chunk = part[:fit]
                        part = part[fit:]

                        self.set_xy(x, self.get_y())
                        self.cell(self.get_string_width(chunk), line_h, chunk)

                        # Wrap to next line (indented to show continuation)
                        x = x_start + tab_w
                        self.ln(line_h)
                        if self.get_y() > self.h - 18:
                            self.add_page()
                            self._draw_bg()
                            x = x_start + tab_w

    def _draw_bg(self):
        self.set_fill_color(*self._bg_rgb)
        self.rect(self.l_margin, self.get_y(),
                  self.w - self.l_margin - self.r_margin,
                  self.h - self.get_y() - 14, "F")

    def _draw_line_number(self, n: int, line_h: float):
        self.set_font(self.MONO_FONT, "", self._font_size - 1)
        self.set_text_color(100, 100, 100)
        num_str = str(n).rjust(4)
        num_w = self.get_string_width(num_str)
        self.set_x(self.l_margin + 10 - num_w)
        self.cell(num_w, line_h, num_str)


# ── Two-pass PDF builder ────────────────────────────────────────────────────

def build_combined_pdf(files: list[str], output: str,
                       style: str = "monokai", font_size: int = 9):
    """
    Two-pass build:
      Pass 1 — render all files, collect TOC entries with page numbers.
      Pass 2 — build final PDF with TOC at front + all content.
    """

    groups = group_by_directory(files)
    dir_order = sorted(groups.keys())

    # ── Pass 1: render content, gather TOC data ──────────────────────────
    print("  Pass 1/2: rendering content & gathering page numbers ...")
    t0 = time.time()

    content_pdf = CodePDF(doc_title="pass1", style_name=style, font_size=font_size)
    content_pdf.alias_nb_pages()

    for d in dir_order:
        flist = groups[d]
        content_pdf.add_directory_page(d, len(flist))
        for f in flist:
            print(f"    {f}")
            content_pdf.render_file(f)

    toc_entries = list(content_pdf._toc_entries)
    content_bytes = content_pdf.output()
    t1 = time.time()
    print(f"  Pass 1 done — {content_pdf.page_no()} pages in {t1-t0:.1f}s")

    # ── Estimate how many pages the TOC itself will take ─────────────────
    lines_per_toc_page = 50
    toc_page_count = max(1, (len(toc_entries) + lines_per_toc_page - 1)
                         // lines_per_toc_page)

    # ── Pass 2: build final PDF = cover + TOC + content ──────────────────
    print("  Pass 2/2: building final PDF with Table of Contents ...")

    final = CodePDF(doc_title=output, style_name=style, font_size=font_size)
    final.alias_nb_pages()

    # ── Cover page ───────────────────────────────────────────────────────
    final._file_title = ""
    final.add_page()
    final.ln(50)
    final.set_font(final.MONO_FONT, "B", 28)
    final.set_text_color(255, 200, 60)
    final.cell(0, 14, "  xtrackm", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    final.ln(4)
    final.set_font(final.MONO_FONT, "", 14)
    final.set_text_color(180, 180, 180)
    final.cell(0, 8, "  Source Code & Notes",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    final.ln(8)
    final.set_font(final.MONO_FONT, "", 11)
    final.set_text_color(140, 140, 140)
    final.cell(0, 6, f"  {len(files)} files across {len(groups)} directories",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    final.cell(0, 6, f"  Theme: {style}",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Table of Contents ────────────────────────────────────────────────
    final._file_title = ""
    final.add_page()
    final.set_font(final.MONO_FONT, "B", 16)
    final.set_text_color(255, 200, 60)
    final.cell(0, 10, "Table of Contents",
               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    final.ln(4)

    # Page offset: cover(1) + toc pages + then content starts
    page_offset = 1 + toc_page_count

    for level, label, pg in toc_entries:
        adjusted_pg = pg + page_offset
        if level == 0:
            # Directory heading
            final.ln(2)
            final.set_font(final.MONO_FONT, "B", 10)
            final.set_text_color(255, 200, 60)
            indent = "  "
        else:
            final.set_font(final.MONO_FONT, "", 8)
            final.set_text_color(190, 190, 190)
            indent = "      "

        dots_w = 70 - len(label)
        dots = " " + "." * max(2, dots_w) + " " if dots_w > 0 else "  "
        line = f"{indent}{label}{dots}{adjusted_pg}"
        final.cell(0, 4.5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # New TOC page if needed
        if final.get_y() > final.h - 20:
            final.add_page()

    # ── Append content pages (re-render) ─────────────────────────────────
    for d in dir_order:
        flist = groups[d]
        final.add_directory_page(d, len(flist))
        for f in flist:
            final.render_file(f)

    final.output(output)
    t2 = time.time()
    total_pages = final.page_no()
    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"  Pass 2 done — {total_pages} pages, {size_mb:.1f} MB in {t2-t1:.1f}s")
    print(f"\n  ✓ {output}  ({total_pages} pages, {size_mb:.1f} MB)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gather all .py/.txt files and produce one combined, "
                    "syntax-highlighted PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available themes: " + ", ".join(sorted(get_all_styles())),
    )
    parser.add_argument("--dir", nargs="*", default=["."],
                        help="Root directories to scan (default: current dir)")
    parser.add_argument("-o", "--output", default="xtrackm_code.pdf",
                        help="Output PDF filename (default: xtrackm_code.pdf)")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Directories to skip (repeatable)")
    parser.add_argument("--theme", default="monokai",
                        help="Pygments colour theme (default: monokai)")
    parser.add_argument("--font-size", type=int, default=9,
                        help="Font size in points (default: 9)")

    args = parser.parse_args()

    # Always exclude common non-source dirs
    exclude = args.exclude + ["archive", ".git", "__pycache__", ".ipynb_checkpoints"]

    files = collect_files(args.dir, exclude=exclude)
    if not files:
        print("No .py or .txt files found.", file=sys.stderr)
        sys.exit(1)

    groups = group_by_directory(files)
    print(f"Found {len(files)} files in {len(groups)} directories  "
          f"[theme: {args.theme}]\n")

    build_combined_pdf(files, args.output, style=args.theme,
                       font_size=args.font_size)
    print("\nDone.")


if __name__ == "__main__":
    main()
