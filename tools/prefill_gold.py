"""Draft <name>.gold.json files for REAL resumes so you only have to correct them.

  python -m tools.prefill_gold data/real_raw --out data/real_gold --layout pdf_two_col

Drafts come from the deterministic rules mode (not the LLM) so your gold is not biased toward the LLM you are
evaluating. Every draft contains "_todo": the evaluator SKIPS files that still have it. After you have checked
EVERY field against the resume itself (add missing skills/jobs, delete wrong ones), delete the "_todo" line.
Anonymise before sharing: replace real names/emails/phones with fake ones in both the resume and the gold file.
"""
import argparse
import json
import shutil
from pathlib import Path

from resume_parser import ResumeParser

EXTS = {".pdf", ".docx", ".txt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="data/real_gold")
    ap.add_argument("--layout", default="unknown", help="txt | docx | pdf_single | pdf_two_col | other (used for RQ4)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parser = ResumeParser()
    files = [f for f in sorted(Path(args.folder).iterdir()) if f.suffix.lower() in EXTS]
    for f in files:
        gold_path = out / f"{f.stem}.gold.json"
        if gold_path.exists():
            print(f"skip {f.name} (gold already exists)")
            continue
        try:
            r = parser.parse_file(f, mode="rules")
        except Exception as e:
            print(f"FAILED {f.name}: {e}")
            continue
        draft = {
            "_todo": "VERIFY EVERY FIELD AGAINST THE RESUME, THEN DELETE THIS LINE",
            "name": r.name, "email": r.email, "phone": r.phone,
            "skills": r.skills,
            "education": [{"degree": e.degree, "institute": e.institute, "year": e.year} for e in r.education],
            "experience": [{"company": e.company, "role": e.role} for e in r.experience],
            "total_experience_months": r.total_experience_months,
            "layout": args.layout, "source": "real",
        }
        shutil.copy(f, out / f.name)
        gold_path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"drafted {gold_path}")
    print("\nNow open each .gold.json next to its resume, correct it, and delete the _todo line.")


if __name__ == "__main__":
    main()
