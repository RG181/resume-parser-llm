import io

import pytest

from resume_parser.text_extraction import ExtractionError, extract_text, find_gutter, words_to_text


def _word(text, x0, x1, top):
    return {"text": text, "x0": x0, "x1": x1, "top": top}


def test_two_column_reading_order():
    # left column: x 40-250, right column: x 350-560, page width 600
    left = [_word(f"L{i}w{j}", 40 + j * 60, 90 + j * 60, 100 + i * 15) for i in range(6) for j in range(3)]
    right = [_word(f"R{i}w{j}", 350 + j * 60, 400 + j * 60, 100 + i * 15) for i in range(6) for j in range(3)]
    assert find_gutter(left + right, 600) is not None
    text = words_to_text(left + right, 600)
    assert text.index("L5w0") < text.index("R0w0")  # whole left column precedes right column
    assert text.splitlines()[0] == "L0w0 L0w1 L0w2"


def test_single_column_not_split():
    words = [_word(f"w{i}", 40 + (i % 8) * 65, 95 + (i % 8) * 65, 100 + (i // 8) * 15) for i in range(64)]
    assert find_gutter(words, 600) is None


def test_txt_and_errors(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("Some reasonably long resume text goes here for the test.")
    assert "resume text" in extract_text(f)
    with pytest.raises(ExtractionError):
        extract_text(tmp_path / "missing.pdf")
    (tmp_path / "e.txt").write_text("x")
    with pytest.raises(ExtractionError):
        extract_text(tmp_path / "e.txt")  # too little text
    with pytest.raises(ExtractionError):
        extract_text(io.BytesIO(b"abc"), suffix=".exe")


def test_docx_roundtrip(tmp_path):
    from docx import Document

    d = Document()
    d.add_paragraph("Jane Doe")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Python", "SQL"
    d.add_paragraph("EXPERIENCE and some other padding text for length")
    p = tmp_path / "r.docx"
    d.save(p)
    text = extract_text(p)
    assert text.index("Jane Doe") < text.index("Python | SQL") < text.index("EXPERIENCE")


def test_real_two_column_pdf_end_to_end():
    from resume_parser import ResumeParser
    from tests.helpers import SAMPLES

    r = ResumeParser().parse_file(SAMPLES / "resume_03.pdf", mode="rules")
    assert r.name == "Neha Verma" and r.phone == "+91 98111 22334"
    assert [e.company for e in r.experience] == ["Zenith Soft", "Kiran Labs"]
    assert r.total_experience_months == 9 and r.education[0].institute == "Pink City Institute of Technology"


def test_unverified_gold_drafts_are_skipped(tmp_path):
    import json
    from evaluation.evaluate import load_docs

    (tmp_path / "a.txt").write_text("x" * 40)
    (tmp_path / "a.gold.json").write_text(json.dumps({"_todo": "verify", "name": "A"}), encoding="utf-8")
    (tmp_path / "b.txt").write_text("x" * 40)
    (tmp_path / "b.gold.json").write_text(json.dumps({"name": "B"}), encoding="utf-8")
    assert [p.name for p, _ in load_docs([tmp_path])] == ["b.txt"]


def test_txt_encodings_keep_date_range_dashes(tmp_path):
    """Windows regression: a cp1252 ('ANSI') or BOM-prefixed .txt must still contain the en dash in 'Jan 2025 – Mar 2025'."""
    body = "Riya Sharma\nEXPERIENCE\nData Intern | Acme | Jan 2025 \u2013 Mar 2025\n- Built models and dashboards for the team\n"
    for name, raw in {"utf8": body.encode("utf-8"), "bom": b"\xef\xbb\xbf" + body.encode("utf-8"),
                      "cp1252": body.encode("cp1252")}.items():
        f = tmp_path / f"{name}.txt"
        f.write_bytes(raw)
        text = extract_text(f)
        assert "Jan 2025 \u2013 Mar 2025" in text and not text.startswith("\ufeff"), name
