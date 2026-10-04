"""Benchmark rules vs LLM vs hybrid against gold annotations.

  python -m evaluation.evaluate                      # rules only (free, offline)
  python -m evaluation.evaluate --modes rules llm hybrid --ablation
  python -m evaluation.evaluate --data-dir data/gold --no-cache

Gold files: <name>.gold.json next to <name>.(pdf|docx|txt). Fields: name, email, phone, skills,
education[{degree,institute,year}], experience[{company,role}], total_experience_months.
Metrics are micro-averaged over all resumes. 'strict' = exact (normalised) match; 'partial' = one string contains the other.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import sys
import time
from pathlib import Path

from resume_parser import LLMError, ResumeParser, load_config

EXTS = (".pdf", ".docx", ".txt")


def norm(s) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", str(s or "").lower()).strip()


def digits(s) -> str:
    return re.sub(r"\D", "", str(s or ""))[-10:]


def match_count(pred: list[str], gold: list[str], partial: bool) -> int:
    remaining, tp = list(gold), 0
    for p in pred:
        for g in remaining:
            if p and g and (p == g or (partial and (p in g or g in p))):
                remaining.remove(g)
                tp += 1
                break
    return tp


class Counts:
    def __init__(self):
        self.tp = self.pred = self.gold = 0

    def add(self, tp, pred, gold):
        self.tp += tp
        self.pred += pred
        self.gold += gold

    def prf(self):
        p = self.tp / self.pred if self.pred else 0.0
        r = self.tp / self.gold if self.gold else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate_doc(parser, resume, gold, acc):
    tax = parser.taxonomy
    if "name" in gold:  # public datasets may not label every field - only score what the gold file contains
        acc["name"].append(norm(resume.name) == norm(gold["name"]))
    if "email" in gold:
        acc["email"].append(norm(resume.email) == norm(gold["email"]))
    if "phone" in gold:
        acc["phone"].append(digits(resume.phone) == digits(gold["phone"]))
    p_sk = [norm(tax.normalize(s)) for s in resume.skills]
    g_sk = [norm(tax.normalize(s)) for s in gold.get("skills", [])]
    acc["skills"].add(match_count(p_sk, g_sk, False), len(p_sk), len(g_sk))
    for key, fld, items_p, items_g in (
        ("edu", "institute", [e.institute for e in resume.education], [e.get("institute") for e in gold.get("education", [])]),
        ("exp", "company", [e.company for e in resume.experience], [e.get("company") for e in gold.get("experience", [])]),
    ):
        p, g = [norm(x) for x in items_p if x], [norm(x) for x in items_g if x]
        acc[f"{key}_strict"].add(match_count(p, g, False), len(p), len(g))
        acc[f"{key}_partial"].add(match_count(p, g, True), len(p), len(g))
    if "total_experience_months" in gold:
        acc["months_err"].append(abs(resume.total_experience_months - gold["total_experience_months"]))


def run_config(label, cfg, mode, docs, no_cache):
    c = copy.deepcopy(cfg)
    if no_cache:
        c["llm"]["cache_enabled"] = False
    parser = ResumeParser(c)
    acc = {k: Counts() for k in ("skills", "edu_strict", "edu_partial", "exp_strict", "exp_partial")}
    acc.update({k: [] for k in ("name", "email", "phone", "months_err")})
    secs, in_tok, out_tok, failures = [], 0, 0, 0
    for path, gold in docs:
        t0 = time.perf_counter()
        try:
            resume = parser.parse_file(path, mode=mode)
        except LLMError as e:
            failures += 1
            print(f"  [{label}] {path.name}: {e}", file=sys.stderr)
            continue
        secs.append(time.perf_counter() - t0)
        in_tok += resume.meta.get("input_tokens", 0)
        out_tok += resume.meta.get("output_tokens", 0)
        evaluate_doc(parser, resume, gold, acc)
    pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%" if xs else "-"
    f1 = lambda c: f"{c.prf()[2]:.2f}"
    return {
        "Method": label, "Name acc": pct(acc["name"]), "Email acc": pct(acc["email"]), "Phone acc": pct(acc["phone"]),
        "Skills P": f"{acc['skills'].prf()[0]:.2f}", "Skills R": f"{acc['skills'].prf()[1]:.2f}", "Skills F1": f1(acc["skills"]),
        "Edu F1 strict/partial": f"{f1(acc['edu_strict'])}/{f1(acc['edu_partial'])}",
        "Exp F1 strict/partial": f"{f1(acc['exp_strict'])}/{f1(acc['exp_partial'])}",
        "Months MAE": f"{statistics.mean(acc['months_err']):.1f}" if acc["months_err"] else "-",
        "Sec/resume": f"{statistics.mean(secs):.2f}" if secs else "-",
        "Tokens in/out": f"{in_tok}/{out_tok}" if mode != "rules" else "-",
        "Failures": failures,
    }


def load_docs(dirs) -> list:
    docs = []
    for d in dirs:
        for gp in sorted(Path(d).glob("*.gold.json")):
            stem = gp.name[: -len(".gold.json")]
            src = next((gp.with_name(stem + e) for e in EXTS if gp.with_name(stem + e).exists()), None)
            if not src:
                continue
            gold = json.loads(gp.read_text(encoding="utf-8"))
            if "_todo" in gold:
                print(f"WARNING: skipping {gp.name} - unverified draft (delete the _todo line after checking it)", file=sys.stderr)
                continue
            docs.append((src, gold))
    return docs


def to_markdown(rows) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def _utf8_console():
    """Windows consoles/pipes may use a legacy code page; make printing '–' or '•' safe."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main():
    _utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", nargs="+", default=["data/samples"], help="one or more folders with gold files")
    ap.add_argument("--by-layout", action="store_true", help="also report per layout (txt/docx/pdf_single/pdf_two_col) for RQ4")
    ap.add_argument("--modes", nargs="+", default=["rules"], choices=["rules", "llm", "hybrid"])
    ap.add_argument("--ablation", action="store_true", help="also run each mode WITHOUT section segmentation (RQ1)")
    ap.add_argument("--no-cache", action="store_true", help="disable response cache (honest latency numbers)")
    ap.add_argument("--out", default="results/results.md")
    args = ap.parse_args()

    docs = load_docs(args.data_dir)
    if not docs:
        sys.exit(f"No <name>.gold.json + resume pairs found in {args.data_dir}")
    print(f"Evaluating on {len(docs)} resumes from {', '.join(args.data_dir)}")
    groups = [("", docs)]
    if args.by_layout:
        layouts = sorted({g.get("layout") for _, g in docs if g.get("layout")})
        groups += [(f" [{lay}]", [(p, g) for p, g in docs if g.get("layout") == lay]) for lay in layouts]

    cfg = load_config()
    rows = []
    for mode in args.modes:
        variants = [("", True)] + ([(" (no segmentation)", False)] if args.ablation else [])
        for suffix, seg in variants:
            c = copy.deepcopy(cfg)
            c["pipeline"]["use_segmentation"] = seg
            for gsuffix, gdocs in groups:
                print(f"running {mode}{suffix}{gsuffix} ({len(gdocs)} docs) ...")
                try:
                    rows.append(run_config(mode + suffix + gsuffix, c, mode, gdocs, args.no_cache))
                except LLMError as e:
                    print(f"  skipped {mode}: {e}", file=sys.stderr)
    if not rows:
        sys.exit("nothing evaluated")
    table = to_markdown(rows)
    print("\n" + table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(f"# Results ({len(docs)} resumes, model: {cfg['llm']['model']})\n\n{table}\n", encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
