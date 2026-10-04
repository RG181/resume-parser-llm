"""PDF / DOCX / TXT text extraction with layout-aware column handling for PDFs."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO


class ExtractionError(RuntimeError):
    pass


def decode_text(raw: bytes) -> str:
    """UTF-8 (with or without BOM) first; fall back to Windows cp1252 (Notepad 'ANSI'), which keeps en dashes in date ranges."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def extract_text(source: str | Path | BinaryIO, suffix: str | None = None) -> str:
    """Extract text from a path or a file-like object (then `suffix` such as '.pdf' is required)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ExtractionError(f"File not found: {path}")
        suffix = path.suffix.lower()
        source = path
    suffix = (suffix or "").lower()
    if suffix == ".pdf":
        text = _pdf_text(source)
    elif suffix == ".docx":
        text = _docx_text(source)
    elif suffix in {".txt", ".md"}:
        text = decode_text(source.read_bytes() if isinstance(source, Path) else source.read())
    else:
        raise ExtractionError(f"Unsupported file type '{suffix}'. Use PDF, DOCX or TXT.")
    if len(text.strip()) < 30:
        raise ExtractionError("No readable text found (scanned/image-only file? OCR is not enabled).")
    return text


# ----------------------------------------------------------------- PDF
def find_gutter(words: list[dict], page_width: float, bins: int = 100) -> float | None:
    """Detect a vertical whitespace gutter (two-column layout). Returns its x position or None."""
    if len(words) < 20 or page_width <= 0:
        return None
    occupancy = [0] * bins
    for w in words:
        a = max(int(w["x0"] / page_width * bins), 0)
        b = min(int(min(w["x1"], page_width - 1e-6) / page_width * bins), bins - 1)
        for i in range(a, b + 1):
            occupancy[i] += 1
    lo, hi = int(bins * 0.25), int(bins * 0.75)
    best, run_start = (0, 0), None
    for i in range(lo, hi + 2):
        empty = i <= hi and occupancy[i] == 0
        if empty and run_start is None:
            run_start = i
        if not empty and run_start is not None:
            if i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    if best[1] - best[0] < 2:  # gutter must be >= 2% of page width
        return None
    gutter = (best[0] + best[1]) / 2 / bins * page_width
    left = sum(1 for w in words if (w["x0"] + w["x1"]) / 2 < gutter)
    right = len(words) - left
    if min(left, right) < 0.15 * len(words):
        return None
    return gutter


# Icon fonts (FontAwesome etc.) are extracted as random ASCII ('#', '+', '§', '?', 'e' ...) or '(cid:NN)'.
_ICON_FONT = re.compile(r"awesome|icon|wingding|dingbat|material|glyph", re.I)


def _is_icon_char(obj: dict) -> bool:
    return obj.get("object_type") == "char" and bool(_ICON_FONT.search(str(obj.get("fontname", ""))))


def _join_line(line: list[dict]) -> str:
    """Join words of one visual line. A gap much wider than a normal space (right-aligned dates, icon
    separators in a contact line, 'Project | tech ... Event') is kept as ' | ' so downstream splitting works."""
    line = sorted(line, key=lambda w: w["x0"])
    out = [line[0]["text"]]
    for prev, w in zip(line, line[1:]):
        height = max(float(w.get("bottom", w["top"] + 10)) - float(w["top"]), 6.0)  # tolerate minimal word dicts
        out.append(" | " if float(w["x0"]) - float(prev["x1"]) > max(12.0, 1.3 * height) else " ")
        out.append(w["text"])
    return "".join(out)


def _group_lines(words: list[dict], tol: float = 3.0) -> str:
    lines, cur, cur_top = [], [], None
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if cur and abs(w["top"] - cur_top) > tol:
            lines.append(cur)
            cur = []
        if not cur:
            cur_top = w["top"]
        cur.append(w)
    if cur:
        lines.append(cur)
    return "\n".join(_join_line(line) for line in lines)


def words_to_text(words: list[dict], page_width: float) -> str:
    """Rebuild reading order: left column fully, then right column (if a gutter exists)."""
    gutter = find_gutter(words, page_width)
    if gutter is None:
        return _group_lines(words)
    left = [w for w in words if (w["x0"] + w["x1"]) / 2 < gutter]
    right = [w for w in words if (w["x0"] + w["x1"]) / 2 >= gutter]
    return _group_lines(left) + "\n" + _group_lines(right)


def _pdf_text(source) -> str:
    try:
        import pdfplumber
    except ImportError as e:  # pragma: no cover
        raise ExtractionError("pdfplumber is not installed") from e
    pages = []
    try:
        with pdfplumber.open(source if not isinstance(source, Path) else str(source)) as pdf:
            for page in pdf.pages:
                words = page.filter(lambda o: not _is_icon_char(o)).extract_words(x_tolerance=2, y_tolerance=3)
                pages.append(words_to_text(words, float(page.width)))
    except Exception as e:
        raise ExtractionError(f"Could not read PDF: {e}") from e
    return "\n\n".join(pages)


# ----------------------------------------------------------------- DOCX
def _docx_text(source) -> str:
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:  # pragma: no cover
        raise ExtractionError("python-docx is not installed") from e
    try:
        doc = Document(str(source) if isinstance(source, Path) else source)
    except Exception as e:
        raise ExtractionError(f"Could not read DOCX: {e}") from e
    out = []
    for child in doc.element.body.iterchildren():  # keeps paragraphs and tables in document order
        if child.tag.endswith("}p"):
            out.append(Paragraph(child, doc).text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                cells = []
                for c in row.cells:
                    if c.text.strip() and c.text.strip() not in cells:
                        cells.append(c.text.strip())
                out.append(" | ".join(cells))
    return "\n".join(out)


def bytes_to_text(data: bytes, filename: str) -> str:
    return extract_text(io.BytesIO(data), suffix=Path(filename).suffix)
