# Evaluation & Error-Analysis Guide (steps 4-6)

## 0. What is in the repo already
- `data/synthetic/` - 40 generated resumes (10 each: TXT, DOCX, single-column PDF, two-column PDF) + gold labels.
  Skills outside the taxonomy, aliases, concurrent roles, 4 date formats, 4 experience layouts, wrapped lines.
  **Say clearly in your report that this set is synthetic** (it measures robustness to layout, not real-world messiness).
- `tools/prefill_gold.py` - drafts gold files for real resumes (rules-based, so the gold is not biased toward the LLM).
- `evaluation/evaluate.py` - metrics, ablation (`--ablation`), per-layout (`--by-layout`).
- `evaluation/error_analysis.py` - counts errors per cause/layout and prints examples.

## 1a. Public benchmark: Kaggle "ATS Scoring Dataset" (220 labelled Indeed resumes)
```bash
# download from https://www.kaggle.com/datasets/mgmitesh/ats-scoring-dataset (check the licence), unzip so that
# data/raw/ats/train/train_data.json exists, then:
python -m tools.import_ats_dataset data/raw/ats/train/train_data.json --n 40 --out data/public_ats
python -m evaluation.evaluate --data-dir data/public_ats --modes rules llm hybrid --out results/results_ats.md
```
What the importer does and what is **not** scored (verified on the real file):
- Text is flattened (no newlines) -> `cleaning.restore_line_breaks` recovers the structure. Measured on 40 resumes (rules mode):
  institute F1 **0.00 -> 0.67**, company F1 **0.05 -> 0.33** with the recovery switched on.
- `EMAIL_ADDRESS` spans are Indeed profile URLs (real emails were redacted) -> email/phone **not scored**.
- `SKILLS` spans are free-text blocks, `DESIGNATION` is mixed with degrees, years-of-experience is sparse ->
  skills are scored but **noisy** (the annotation only covers the span the annotators chose, so precision is under-estimated; report recall or call it indicative),
  roles and months are not scored.
- Some labels are wrong (a degree tagged as a college name). **Hand-check ~10 converted gold files** and say so in your report.
- The output folder `data/public_ats/` is git-ignored: do not publish the dataset text unless its licence allows it. Cite it in the report.

Findings already measured on this set with rules only (reproduce with `error_analysis`; compare with hybrid):
1. **Name accuracy 0%** - the first line is "Name Designation - Company" with no separator, so token-count heuristics cannot split the name. This is where the LLM should win.
2. **Skills recall 0.22** - annotated skills are long-tail (ServiceNow, COBOL, JCL, Teradata...), far outside the 81-skill taxonomy.
3. **Spurious experience entries (54)** - a location such as "Hyderabad" is mistaken for a company when a date range follows it.
4. **Flattened line breaks** - fixed by `restore_line_breaks` (see numbers above).

## 1b. Add 10-15 of your OWN real resumes (recommended, ~1 hour)
1. Collect resumes with permission (classmates, seniors, your own). Replace real names/emails/phones with fake ones.
   Public alternatives: Kaggle "Resume Dataset" (mostly text), or resume templates you fill in yourself.
2. Put them in `data/real_raw/`, then run:
   `python -m tools.prefill_gold data/real_raw --out data/real_gold --layout pdf_single`
   (run once per layout group, changing `--layout`).
3. Open every `.gold.json` beside its resume. **Check each field against the resume itself**: add missing skills/jobs,
   delete wrong ones, fix the months, then delete the `_todo` line. (Files with `_todo` are skipped on purpose.)
4. Optional: have a friend annotate 5 of them independently and report % agreement.

## 2. Run the evaluation (needs your API key for llm/hybrid)
**Budget first.** Measured on the bundled data: the synthetic run (llm + hybrid, with the segmentation ablation) needs **160 API calls**,
the public ATS run **80 calls** = **240 calls, about 24 minutes at 10 requests/min**. Check your daily quota in Google AI Studio;
if it is lower, shrink the data instead of skipping modes: `python -m tools.generate_dataset --n 20 --out data/synthetic_small`.
Keep the **cache ON** (do not pass `--no-cache`) for these runs: the per-layout rows reuse the cached answers, so re-runs and layout
breakdowns cost nothing. With `--no-cache` the layout breakdown would double the calls.
```bash
python -m evaluation.evaluate --data-dir data/synthetic --modes rules llm hybrid --ablation --by-layout --out results/results_synthetic.md
python -m evaluation.evaluate --data-dir data/public_ats --modes rules llm hybrid --out results/results_ats.md
python -m evaluation.evaluate --data-dir data/real_gold --modes rules llm hybrid --out results/results_real.md     # your own resumes
```
**Latency/cost numbers need a separate small run**, because cached rows show ~0 s and throttle waits inflate the others:
set `llm.requests_per_minute: null` temporarily in `config.yaml`, then
`python -m evaluation.evaluate --data-dir data/samples --modes llm hybrid --no-cache --out results/results_timing.md` (3 resumes, 6 calls) and
quote Sec/resume and tokens from that file only.
- Report three separate tables - synthetic (layout robustness), public ATS subset (real wording, noisy labels), your own real resumes - never one blended headline number.

## 3. Fill the README table
Copy the `rules / llm / hybrid / rules (no segmentation)` rows (the "all" rows, without `[layout]`) into the README results table.
Keep `results/results_full.md` in the repo and cite it.

## 4. Interpret honestly (templates - replace the <...> with your numbers)
- **RQ1 segmentation:** "Disabling segmentation raised experience-months error from <a> to <b> months and dropped experience F1 from <c> to <d>,
  because education date ranges were counted as jobs. Skills were unaffected (<x> vs <y> F1)."
  *(Rules-only, synthetic: MAE 0.7 -> 22.9 and Exp F1 0.81 -> 0.68 - your run should reproduce this exactly; the LLM rows are yours to measure.)*
- **RQ2/RQ3 rules vs LLM vs hybrid:** "<mode> had the best skills recall (<r>) because it recognises tools outside the taxonomy,
  while rules were <faster/cheaper> (<s> s vs <t> s per resume, $0 vs ~<n> tokens). Hybrid <did/did not> beat both: <reason>."
  If hybrid is not best on something, say so and explain (e.g. LLM-only name accuracy might equal hybrid).
- **RQ4 layouts:** "Two-column PDFs were the hardest for rules (months MAE <m>) because wrapped lines split date ranges."
- **Cost/latency:** tokens in/out and seconds per resume come straight from the table.
- **Limitations:** synthetic data, small N, one model, one prompt version, skills gold depends on your annotation rules.

## 5. Error analysis (5-10 findings)
```bash
python -m evaluation.error_analysis --data-dir data/synthetic data/real_gold --mode rules
python -m evaluation.error_analysis --data-dir data/synthetic data/real_gold --mode hybrid
python -m evaluation.error_analysis --data-dir data/synthetic --mode rules --no-segmentation
```
Open `results/errors_<mode>.md`. For each of the top 5-8 causes write one paragraph:
**what happens -> example file -> why (root cause) -> fix tried or proposed -> effect.**

### Worked examples from the rules baseline on the synthetic set (verify against your own run)
1. **Skill outside taxonomy (56 misses).** Tools such as Figma/Terraform/Looker are not in `skills.json`, so rules cannot find them.
   Fix: the hybrid merge adds LLM skills; long term, grow the taxonomy from job postings. *(Check hybrid recall to quantify.)*
2. **Company missing when role and company are on separate lines (20).** `extract_experience` takes the company from the same line as the date,
   so "Company, City / Role  dates" yields a role only. Fix: look at the previous line when only a role is found.
3. **Experience missed in two-column PDFs (6) and wrong months (4).** The narrow right column wraps the line in the middle of the date range
   ("Jun 2024 - / Aug 2024"), so `RANGE_RE` never sees the full range. Fix: join wrapped lines before date detection.
4. **Education dates counted as jobs without segmentation (24 spurious entries, months MAE 0.7 -> 22.9).** Shows why segmentation matters (RQ1).
5. **A bug found by this analysis (good viva story):** the alias `js` was stored in lowercase but matched case-sensitively, so "JS" was never found,
   and it falsely matched the ".js" in "Node.js" (a JavaScript false positive). Fixed (alias is now `JS`) and locked in with
   `test_js_alias_regression`. Errors fell from 104 to 86.
Add your own LLM/hybrid findings, e.g. hallucinated skills, date format mistakes, wrong name on resumes with a company header, JSON repairs needed
(`usage` counters / logs), and cases where the LLM and rules disagree (`review_flags`).

## 6. Prompt iteration log (bonus marks: "effectiveness of the prompt file")
Keep a short table in your report: prompt version -> change -> skills F1 / Exp F1 on the same set.
Bump `version` in `prompts/prompts.yaml` each time; it is stamped into every output (`meta.prompt_version`).

## 7. More bugs real data exposed (all fixed + regression-tested - mention in your viva)
- PDFs glued a heading to the next text (`WORK EXPERIENCESoftware Engineer`), so the whole experience section vanished -> `segmentation.glued_heading`.
- Bullet glyphs on their own line (`•` / text on the next line) -> `cleaning` merges them.
- Flattened Indeed text -> `cleaning.restore_line_breaks`.
- Alias `js` / `.js` false positive (found by the synthetic error analysis).
