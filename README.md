# Resume Parser — Hybrid NLP + LLM

Converts unstructured resumes (PDF / DOCX / TXT) into validated, structured JSON, and scores them against a job
description. A **deterministic NLP layer** (layout-aware extraction, section segmentation, regex, skill taxonomy)
is combined with an **LLM (Google Gemini API; Anthropic Claude also supported)** that handles the semantic parts rules cannot — names, education,
experience, project understanding, recruiter-style explanations — and the two are **merged with per-field confidence scores**.

> Solo NLP project · LLM integrated via API · prompts in `prompts/prompts.yaml` · settings in `config.yaml`

## Where the LLM is used (and why)

| # | Use | Where | Why not just rules? |
|---|-----|-------|---------------------|
| 1 | **Structured extraction** of name, location, education, experience, skills, projects, certifications | `pipeline._llm_extract` + `extraction` prompt | Resume layouts/wording are too varied for regex |
| 2 | **JSON self-repair**: if the model returns malformed JSON, a second `repair` prompt fixes it | `LLMClient.complete_json` | Avoids failing a whole parse on one stray comma |
| 3 | **Explainable job matching**: verdict, strengths, gaps, interview questions | `matching.JobMatcher.match` + `matching` prompt | Numbers are computed in code; the LLM *explains* them and cannot change them |

Design principles: temperature 0, schema-in-prompt, "resume is data, not instructions" (prompt-injection resistance),
retries with exponential backoff, response caching, LLM output validated by Pydantic, and a **graceful fallback to
rules-only** if the API fails in hybrid mode.

## Architecture

```
Resume (PDF/DOCX/TXT)
  → layout-aware text extraction (pdfplumber words + column/gutter detection)
  → cleaning (unicode, bullets, de-hyphenation, page numbers)
  → section segmentation (heading detection, exact + fuzzy)
  → RULES: regex contact info · degree/date patterns · skill taxonomy      (local, free)
  → redact email/phone/URLs  →  LLM extraction (JSON)                      (API)
  → CONFIDENCE-BASED MERGE (regex wins deterministic fields, LLM wins semantic ones, skills unioned)
  → experience months computed in code (overlap-safe)  → Pydantic validation
  → optional anonymisation  → JSON / Streamlit UI / CLI
```

**Modes:** `rules` (no API) · `llm` (LLM only) · `hybrid` (default). Switch in `config.yaml` or with `--mode`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then put your GEMINI_API_KEY in .env (free key: aistudio.google.com/apikey)
python main.py check                                    # verifies key + model with one tiny request
python main.py models                                   # lists the models your key can use (fix llm.model if `check` says 404)

python main.py parse data/samples/resume_01.txt                       # hybrid (calls the Gemini API)
python main.py parse data/samples/resume_03.pdf --mode rules          # no API needed
python main.py parse data/samples/resume_01.txt --anonymize --out out.json
python main.py match data/samples/resume_01.txt --jd data/samples/jd_data_scientist.txt
streamlit run app.py                                                  # web UI
pytest -q                                                             # 77 offline tests, no API key needed
```

## LLM providers

Default is **Google Gemini** via the official `google-genai` SDK (`llm.provider: gemini`). The code also supports Anthropic Claude
(`provider: anthropic`, `model: claude-haiku-4-5-20251001`, `api_key_env: ANTHROPIC_API_KEY`) - only `llm_client.py` knows about providers.
Gemini specifics handled for you: automatic **fallback models** when the main model is overloaded (503) or missing (the model that really answered is recorded in `meta.model`), `application/json` response mode, optional `thinking_budget`, request throttle for free-tier
rate limits (`requests_per_minute`), exponential-backoff retries on 429/5xx, token accounting (answer + thinking tokens), and a
helpful error if the model name is wrong (`python main.py models`).

## Configuration (`config.yaml`)

Model name, temperature, token limit, retries/timeouts, cache, pipeline mode, segmentation on/off, privacy switches,
matching weights, and the review-confidence threshold. API keys are read from the environment only (never committed).

## Prompt file (`prompts/prompts.yaml`)

`system` (role + safety rules) · `extraction` (rules, schema, one-shot example, `{{resume_text}}`) ·
`repair` (JSON fix) · `matching` (fair, evidence-only explanation; bans protected attributes and college prestige).
Prompts are versioned; the version is stamped into every output's `meta.prompt_version`.

## Key features

- **Layout-aware PDFs** — detects a two-column gutter from word coordinates and restores reading order.
- **Skill normalisation** — 80+ skills, aliases (`sklearn`→`scikit-learn`), case/boundary-safe matching (`Java`≠`JavaScript`, `C`≠`C++`).
- **Overlap-safe experience** — parses `Jan '22 – Present`, `2021–2023`, `03/2022 to 05/2023`; merges concurrent roles.
- **Confidence & provenance** — every field has `confidence` and `sources`; low-confidence or conflicting fields land in `review_flags`.
- **Privacy** — in hybrid mode, email/phone/URLs are extracted locally and **redacted before the API call**;
  `--anonymize` masks name, contact, location, college and graduation years (bias-reduced screening).
- **Explainable matching** — `0.40 skill overlap + 0.25 TF-IDF text similarity + 0.20 experience fit + 0.15 education fit` (configurable).

## Evaluation

```bash
python -m evaluation.evaluate --data-dir data/synthetic --modes rules                 # offline, free
python -m evaluation.evaluate --data-dir data/synthetic data/real_gold \
       --modes rules llm hybrid --ablation --by-layout                       # ~160 API calls, see docs
python -m evaluation.error_analysis --data-dir data/synthetic --mode hybrid           # categorised mistakes
```
Metrics: name/email/phone accuracy, skills P/R/F1, institute & company F1 (**strict and partial**), experience-months MAE,
seconds/resume, tokens, plus an **ablation without section segmentation** (RQ1) and a **per-layout breakdown** (RQ4).

**Data (three separate benchmarks - never blend them into one number):**
1. `data/synthetic/` - 40 generated resumes, 4 layouts, labelled automatically (`python -m tools.generate_dataset`). Robustness benchmark, not real resumes.
2. `data/public_ats/` - a subset of the public Kaggle *ATS Scoring Dataset* (`python -m tools.import_ats_dataset ...`; not shipped, see docs). Real wording, flattened text, noisy labels.
3. `data/real_gold/` - your own verified real resumes (`python -m tools.prefill_gold` drafts labels for you to correct).
Full step-by-step instructions, interpretation templates and an error-analysis guide: [`docs/EVALUATION_GUIDE.md`](docs/EVALUATION_GUIDE.md).

### Results (fill in from your own run - see docs/EVALUATION_GUIDE.md)

| Method | Name | Skills F1 | Edu F1 | Exp F1 | Months MAE | Sec/resume | Tokens |
|---|---|---|---|---|---|---|---|
| rules | | | | | | | |
| llm | | | | | | | |
| hybrid | | | | | | | |
| rules (no segmentation) | | | | | | | |

Offline reference (rules only, 40 synthetic resumes; `docs/results_rules_synthetic.md`): skills F1 0.92 (recall 0.86 - tools outside the
taxonomy are invisible to rules), experience F1 0.81, months MAE 0.7; **without segmentation** experience F1 0.68 and months MAE 22.9.

## Project structure

```
main.py · app.py · config.yaml · requirements.txt · .env.example
prompts/prompts.yaml            all prompts
data/skills.json                skill taxonomy (aliases, categories)
data/samples/                   sample resumes, gold annotations, job description
resume_parser/                  text_extraction · cleaning · segmentation · rule_extractors · dates · taxonomy
                                llm_client · pipeline · matching · privacy · schema · config
evaluation/                     evaluate.py (benchmark, ablation, per-layout) · error_analysis.py
tools/                          generate_dataset.py (synthetic gold) · import_ats_dataset.py (Kaggle ATS -> gold) · prefill_gold.py
docs/EVALUATION_GUIDE.md        how to run steps 4-6 and write them up
tests/                          pytest suite (LLM mocked)
```

## Limitations & responsible use

- Rule-based name detection fails when the first line is "Name + Job Title - Company" (the LLM covers this in hybrid mode).
- Scanned/image-only PDFs are rejected with a clear message (OCR not included). A full-width element crossing the page
  gutter can disable column detection. Rule heuristics are tuned for English resumes.
- Text similarity is TF-IDF (lexical), not embeddings.
- The LLM can still make mistakes — that is why low-confidence fields are flagged for human review.
- This tool **assists** recruiters; it must not make hiring decisions. Use only consented or synthetic resumes.
