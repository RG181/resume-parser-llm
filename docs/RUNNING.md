# Running the project with a Gemini API key

## 1. One-time setup
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # Windows   (Mac/Linux: cp .env.example .env)
```
Create a key at https://aistudio.google.com/apikey and paste it into `.env`:
```
GEMINI_API_KEY=your_key_here
```

## 2. Verify (do this first)
```bash
pytest -q                 # 77 passed - works offline, proves the code is healthy
python main.py check      # one tiny live request: confirms key + model
```
If `check` says **model not found (404)**: run `python main.py models`, pick a name from the list (e.g. `gemini-2.5-flash`)
and set it as `llm.model` in `config.yaml`. Model names change often, which is why this command exists.

## 3. Use it
```bash
python main.py parse data/samples/resume_01.txt                 # hybrid: rules + Gemini
python main.py parse data/samples/resume_03.pdf --mode llm
python main.py parse data/samples/resume_01.txt --anonymize
python main.py match data/samples/resume_01.txt --jd data/samples/jd_data_scientist.txt
streamlit run app.py
```

## 4. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `API key not found` | `.env` must be next to `main.py` and contain `GEMINI_API_KEY=...` (no quotes, no spaces). `GOOGLE_API_KEY` also works. |
| `API error 404 ... model not found` | Wrong model name -> `python main.py models`, update `llm.model`. |
| `API error 400/403` | Invalid/restricted key, or the API is not enabled for that Google project. Create a fresh key in AI Studio. |
| `503 UNAVAILABLE ... high demand` | Google's server is overloaded for that model (not your fault). The client retries, then switches automatically to `llm.fallback_models` (default `gemini-2.5-flash`) and logs `Answered by fallback model`. If it keeps happening, pick a steadier model from `python main.py models` as `llm.model`. |
| Many `attempt 1/4 ... 429` warnings | Free-tier rate limit. Lower `llm.requests_per_minute` (e.g. 5) or wait; the client backs off and retries automatically. |
| `Gemini returned no text (finish_reason=MAX_TOKENS)` | Thinking tokens used the budget. Raise `llm.max_tokens`, or on gemini-2.5 models set `llm.thinking_budget: 0`. |
| `finish_reason=SAFETY` / `no candidates (blocked?)` | The response was filtered; re-run, or use another model. Resumes rarely trigger this. |
| `ModuleNotFoundError: google` | Virtual env not active, or `pip install -r requirements.txt` was skipped. |
| Hybrid output looks rules-only | See `review_flags` for `LLM extraction failed ...`; run with `-v` to see the cause. |
| Corporate/college network blocks the API | Try a mobile hotspot; the API host is `generativelanguage.googleapis.com`. |

## 5. Privacy note (worth saying in your viva)
Free-tier Gemini requests may be used by Google to improve its products. This project therefore redacts email/phone/URLs before the
request in hybrid mode, and the evaluation uses synthetic or anonymised resumes only. Never upload real candidates' resumes on a free key.
