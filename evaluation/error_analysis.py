"""List and categorise the parser's mistakes (feeds your error-analysis section).

  python -m evaluation.error_analysis --data-dir data/synthetic --mode rules
  python -m evaluation.error_analysis --data-dir data/synthetic data/real_gold --mode hybrid

Writes results/errors_<mode>.md: counts per cause (overall and per layout) + examples to quote in your report.
The categories are heuristic: read the examples and rename/merge causes in your write-up.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.evaluate import digits, load_docs, norm
from resume_parser import LLMError, ResumeParser, load_config


def _partial(a: str, b: str) -> bool:
    return bool(a and b and (a in b or b in a))


def analyse(parser, mode, resume, gold):
    tax, errs = parser.taxonomy, []
    add = lambda cat, field, gold_v, pred_v: errs.append((cat, field, gold_v, pred_v))
    if "name" in gold and norm(resume.name) != norm(gold["name"]):
        add("Name wrong/missing", "name", gold["name"], resume.name)
    if "email" in gold and norm(resume.email) != norm(gold["email"]):
        add("Email wrong/missing", "email", gold["email"], resume.email)
    if "phone" in gold and digits(resume.phone) != digits(gold["phone"]):
        add("Phone wrong/missing", "phone", gold["phone"], resume.phone)

    pred_sk = {norm(tax.normalize(s)): s for s in resume.skills}
    gold_sk = {norm(tax.normalize(s)): s for s in gold.get("skills", [])}
    for k, s in gold_sk.items():
        if k not in pred_sk:
            if mode == "llm":
                add("LLM omitted a skill", "skills", s, None)
            elif not tax.is_known(s):
                add("Skill outside taxonomy (rules cannot see it)", "skills", s, None)
            else:
                add("Known skill missed (scope/segmentation/format)", "skills", s, None)
    for k, s in pred_sk.items():
        if k not in gold_sk:
            add("Spurious skill (term present but not a skill / over-extraction)", "skills", None, s)

    g_exp = gold.get("experience", [])
    p_exp = [(norm(e.company), norm(e.role), e) for e in resume.experience]
    for g in g_exp:
        gc, gr = norm(g.get("company")), norm(g.get("role"))
        if any(_partial(pc, gc) and pc and gc and pc == gc for pc, _, _ in p_exp):
            continue
        if any(_partial(pc, gc) for pc, _, _ in p_exp):
            add("Company boundary error (partial match)", "experience", g["company"], None)
        elif any(_partial(pr, gr) and not pc for pc, pr, _ in p_exp):
            add("Company missing: role and company on separate lines", "experience", g["company"], None)
        else:
            add("Experience entry missed (date format / line wrap / segmentation)", "experience", g["company"], None)
    g_comp = [norm(g.get("company")) for g in g_exp]
    for pc, pr, e in p_exp:
        if pc and not any(_partial(pc, gc) for gc in g_comp):
            add("Spurious experience entry (e.g. education dates leaking in)", "experience", None, f"{e.company} / {e.role}")

    g_inst = [norm(e.get("institute")) for e in gold.get("education", [])]
    p_inst = [norm(e.institute) for e in resume.education if e.institute]
    for g in gold.get("education", []):
        if not any(_partial(norm(g.get("institute")), p) for p in p_inst):
            add("Education institute missed", "education", g.get("institute"), None)
    for p in p_inst:
        if not any(_partial(p, g) for g in g_inst):
            add("Spurious/wrong education institute", "education", None, p)

    if "total_experience_months" in gold and resume.total_experience_months != gold["total_experience_months"]:
        add("Experience months wrong (missed range / overlap / leaked dates)", "months",
            gold["total_experience_months"], resume.total_experience_months)
    return errs


def _utf8_console():
    """Windows consoles/pipes may use a legacy code page; make printing '–' or '•' safe."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main():
    _utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", nargs="+", default=["data/synthetic"])
    ap.add_argument("--mode", choices=["rules", "llm", "hybrid"], default="rules")
    ap.add_argument("--no-segmentation", action="store_true")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_config()
    cfg["pipeline"]["use_segmentation"] = not args.no_segmentation
    parser = ResumeParser(cfg)
    docs = load_docs(args.data_dir)
    if not docs:
        sys.exit("no gold data found")

    by_cat, by_layout, examples, failed = Counter(), defaultdict(Counter), defaultdict(list), 0
    for path, gold in docs:
        try:
            resume = parser.parse_file(path, mode=args.mode)
        except LLMError as e:
            failed += 1
            print(f"{path.name}: {e}", file=sys.stderr)
            continue
        layout = gold.get("layout", "n/a")
        for cat, field, g, p in analyse(parser, args.mode, resume, gold):
            by_cat[cat] += 1
            by_layout[cat][layout] += 1
            if len(examples[cat]) < args.examples:
                examples[cat].append((path.name, layout, g, p))

    layouts = sorted({lay for c in by_layout.values() for lay in c})
    lines = [f"# Error analysis - mode `{args.mode}`{' (no segmentation)' if args.no_segmentation else ''}",
             f"{len(docs) - failed} resumes analysed, {sum(by_cat.values())} errors.\n",
             "| Cause | Total | " + " | ".join(layouts) + " |", "|---|---|" + "---|" * len(layouts)]
    for cat, n in by_cat.most_common():
        lines.append(f"| {cat} | {n} | " + " | ".join(str(by_layout[cat][lay]) for lay in layouts) + " |")
    lines.append("\n## Examples\n")
    for cat, n in by_cat.most_common():
        lines.append(f"### {cat} ({n})")
        lines += [f"- `{f}` [{lay}] gold={g!r} predicted={p!r}" for f, lay, g, p in examples[cat]]
        lines.append("")
    out = Path(args.out or f"results/errors_{args.mode}{'_noseg' if args.no_segmentation else ''}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
