"""Text cleaning: unicode normalisation, bullet unification, noise removal."""
import re
import unicodedata

_BULLET = re.compile(r"^\s*[•●▪■◦○▸►➢✔✓*]\s*")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_PAGE_NO = re.compile(r"(?i)page \d+( of \d+)?")
_CID = re.compile(r"\(cid:\d+\)")  # unmapped glyphs from pdfplumber
_PUA = re.compile(r"[\ue000-\uf8ff]")  # private-use (icon-font) code points


def restore_line_breaks(text: str) -> str:
    """Indeed-style exports flatten line breaks into runs of 2+ spaces (and inline bullets). Recover them."""
    if len(text) > 400 and text.count("\n") < 3 and text.count("  ") >= 5:
        text = re.sub(r" {2,}", "\n", text)
        text = re.sub(r"\s+([•●▪■◦○▸►➢✔✓])\s*", r"\n\1 ", text)
    return text


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = _ZERO_WIDTH.sub("", text)
    text = _PUA.sub(" ", _CID.sub(" ", text))
    text = restore_line_breaks(text)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)  # de-hyphenate line-wrapped words
    lines, pending_bullet = [], False
    for line in text.split("\n"):
        line = _BULLET.sub("- ", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if _PAGE_NO.fullmatch(line):
            continue
        if line == "-":  # PDF/Word exports often put the bullet glyph on its own line
            pending_bullet = True
            continue
        if line and pending_bullet and not line.startswith("- "):
            line = "- " + line
        if line:
            pending_bullet = False
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
