"""Convert the Kaggle "ATS Scoring Dataset" (train/train_data.json) into this project's gold format.

  1. Download it from https://www.kaggle.com/datasets/mgmitesh/ats-scoring-dataset  (check its licence first)
  2. unzip so you have data/raw/ats/train/train_data.json
  3. python -m tools.import_ats_dataset data/raw/ats/train/train_data.json --n 40 --out data/public_ats

What is (and is not) usable - verified on the 220-resume file:
  * Text is FLATTENED (no newlines; line breaks became double spaces) -> tests the flat-text recovery in cleaning.py.
  * EMAIL_ADDRESS spans are Indeed profile URLs (real emails were redacted) -> email/phone are NOT scored.
  * SKILLS spans are free-text blocks ("Database (Less than 1 year), HTML ...") -> split heuristically; treat skills
    P/R as indicative only (the annotation covers just the span they chose, so precision is under-estimated).
  * DESIGNATION is mixed with degrees and YEARS_OF_EXPERIENCE is sparse -> roles and months are NOT scored.
  Scored: name, institutes (COLLEGE_NAME), companies (COMPANIES_WORKED_AT), skills (noisy).
The output folder is git-ignored: do not redistribute the dataset text unless its licence allows it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

_EXPERIENCE_TAG = re.compile(r"\(\s*(?:less than\s*)?\d+(?:\.\d+)?\+?\s*(?:years?|yrs?|months?)\s*\)", re.I)
_BAD_EDGES = " -–—,;:|.•●▪\t\n"


def clean_span(s: str, max_len: int = 70) -> str | None:
    s = re.sub(r"\s+", " ", s).strip(_BAD_EDGES)
    return s if 1 < len(s) <= max_len else None


def dedupe(items):
    seen, out = set(), []
    for x in items:
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


def split_skills(span: str) -> list[str]:
    span = _EXPERIENCE_TAG.sub("", span)
    items = []
    for part in re.split(r"[,•●▪|;\n]", span):
        part = part.strip(_BAD_EDGES)
        if ":" in part:  # "Programming language: C" -> "C"
            part = part.split(":", 1)[1].strip(_BAD_EDGES)
        if 1 <= len(part) <= 40 and len(part.split()) <= 5:
            items.append(part)
    return dedupe(items)


def convert(record: dict) -> dict | None:
    text, by_label = record["text"], {}
    for s, e, label in sorted(record["entities"]):
        by_label.setdefault(label, []).append(text[s:e])
    names = [clean_span(x, 40) for x in by_label.get("NAME", [])]
    institutes = dedupe(clean_span(x, 80) for x in by_label.get("COLLEGE_NAME", []))
    companies = dedupe(clean_span(x, 60) for x in by_label.get("COMPANIES_WORKED_AT", []))
    if not names or not names[0] or not (institutes or companies):
        return None
    skills = dedupe(sk for span in by_label.get("SKILLS", []) for sk in split_skills(span))
    return {
        "name": names[0],
        "skills": skills,
        "education": [{"institute": i} for i in institutes],
        "experience": [{"company": c} for c in companies],
        "degrees": dedupe(clean_span(x, 90) for x in by_label.get("DEGREE", [])),
        "layout": "flat_text", "source": "public_ats",
        "notes": "email/phone/months/roles intentionally not scored; skills are noisy",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_json")
    ap.add_argument("--out", default="data/public_ats")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.train_json)
    if not src.exists():
        found = [str(p) for p in Path("data").rglob("train_data.json")]
        raise SystemExit(
            f"File not found: {src}\n"
            "Put the Kaggle file at exactly this path (relative to the project folder):\n"
            "    data/raw/ats/train/train_data.json\n"
            + (f"But I found one at: {found[0]}  -> re-run with that path.\n" if found else
               "Unzip archive.zip and copy its 'train' folder into data/raw/ats/ (create the folders if needed).\n")
            + f"Current folder: {Path.cwd()} (it must be the folder that contains main.py)")
    records = json.loads(src.read_text(encoding="utf-8"))
    usable = [(i, g) for i, r in enumerate(records) if (g := convert(r))]
    random.Random(args.seed).shuffle(usable)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, gold in usable[: args.n]:
        stem = f"ats_{i:03d}"
        (out / f"{stem}.txt").write_text(records[i]["text"], encoding="utf-8")
        (out / f"{stem}.gold.json").write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(records)} records, {len(usable)} usable; wrote {min(args.n, len(usable))} to {out}")
    print("Next: spot-check ~10 gold files by hand, then  python -m evaluation.evaluate --data-dir", out)


if __name__ == "__main__":
    main()
