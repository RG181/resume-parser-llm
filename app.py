"""Streamlit UI:  streamlit run app.py"""
import json
from html import escape as esc

import pandas as pd
import streamlit as st

from resume_parser import JobMatcher, LLMError, ResumeParser
from resume_parser.text_extraction import ExtractionError

st.set_page_config(page_title="Resume Parser", page_icon="📄", layout="wide")

st.markdown("""<style>
.block-container{max-width:1100px;padding-top:1.8rem}
#MainMenu,footer{visibility:hidden}
.hero{padding:1.5rem 1.8rem;border-radius:18px;color:#fff;margin-bottom:1.1rem;
  background:linear-gradient(120deg,#4f46e5,#7c3aed 55%,#db2777)}
.hero h1{margin:0;font-size:1.8rem;color:#fff;padding:0}.hero p{margin:.3rem 0 0;opacity:.85}
.card{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:1.1rem 1.3rem;margin-bottom:1rem;
  background:rgba(128,128,128,.06)}
.profile{display:flex;gap:1.1rem;align-items:center}
.avatar{width:64px;height:64px;border-radius:50%;flex:none;display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:1.4rem;color:#fff;background:linear-gradient(135deg,#6366f1,#ec4899)}
.name{font-size:1.5rem;font-weight:700;line-height:1.2}
.muted{opacity:.7;font-size:.9rem}
.summary{margin-top:.9rem;padding-top:.9rem;border-top:1px solid rgba(128,128,128,.25);line-height:1.5}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:1rem}
.stat{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:.8rem 1rem;background:rgba(128,128,128,.06)}
.stat b{display:block;font-size:1.6rem;line-height:1.3}.stat span{font-size:.8rem;opacity:.65}
.chip{display:inline-block;padding:.18rem .65rem;margin:.15rem .25rem .15rem 0;border-radius:999px;font-size:.82rem;
  border:1px solid rgba(99,102,241,.4);background:rgba(99,102,241,.13);text-decoration:none;color:inherit}
.chip.low{border-color:rgba(245,158,11,.55);background:rgba(245,158,11,.15)}
.chip.tech{border-color:rgba(128,128,128,.3);background:rgba(128,128,128,.12)}
.cat{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;opacity:.55;margin:.8rem 0 .1rem}
.tl{border-left:2px solid rgba(99,102,241,.4);margin-left:.4rem;padding-left:1.2rem}
.item{position:relative;margin-bottom:1.1rem}
.item:before{content:"";position:absolute;left:-1.67rem;top:.4rem;width:10px;height:10px;border-radius:50%;background:#6366f1}
.item h4{margin:0;font-size:1rem}.when{font-size:.8rem;opacity:.6}
.item ul{margin:.4rem 0 0;padding-left:1.1rem;font-size:.92rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:.9rem}
.grid .card{margin:0}.grid h4{margin:0 0 .15rem}.grid p{font-size:.9rem;margin:.5rem 0}
@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_parser() -> ResumeParser:
    return ResumeParser()


def chips(items, cls="tech") -> str:
    return "".join(f'<span class="chip {cls}">{esc(str(i))}</span>' for i in items)


def when(a, b, months=None) -> str:
    text = " – ".join(esc(x) for x in (a, b) if x)
    return text + (f" · {months} mo" if months else "")


parser = get_parser()

with st.sidebar:
    st.subheader("Settings")
    mode = st.radio("Extraction mode", ["hybrid", "rules", "llm"],
                    help="rules = regex/heuristics only (free). llm = LLM only. hybrid = rules + LLM, merged with confidence scores.")
    anonymize = st.toggle("Anonymise output", value=False, help="Bias-reduced screening: masks institute and graduation year.")
    with st.expander("About this run"):
        st.caption(f"Model: `{parser.cfg['llm']['model']}`  \nPrompt version: `{parser.prompts.version}`")
        if mode == "hybrid":
            st.caption("🔒 Email, phone and URLs are extracted locally and never sent to the LLM.")

st.markdown('<div class="hero"><h1>📄 Resume Parser</h1>'
            '<p>Turn any resume into clean, structured data — rules + LLM, with confidence scores.</p></div>',
            unsafe_allow_html=True)

up_col, btn_col = st.columns([4, 1], vertical_alignment="bottom")
upload = up_col.file_uploader("Upload a resume", type=["pdf", "docx", "txt"], label_visibility="collapsed")
go = btn_col.button("Parse resume", type="primary", use_container_width=True, disabled=upload is None)

if upload and go:
    try:
        with st.spinner("Parsing…"):
            st.session_state["resume"] = parser.parse_bytes(upload.getvalue(), upload.name, mode=mode,
                                                           anonymize_output=anonymize)
    except (ExtractionError, LLMError) as e:
        st.session_state.pop("resume", None)
        st.error(str(e))

resume = st.session_state.get("resume")
if not resume:
    st.info("Upload a PDF, DOCX or TXT resume and press **Parse resume**.", icon="👆")
    st.stop()

# ------------------------------------------------------------------ profile + stats
name = resume.name or "Unknown candidate"
initials = "".join(w[0] for w in name.split()[:2]).upper() or "?"
contact = " · ".join(esc(x) for x in (resume.location, resume.email, resume.phone) if x)
links = ""
for k, v in resume.links.items():
    href = v if v.startswith("http") else f"https://{v}"
    links += f'<a class="chip" href="{esc(href)}" target="_blank">{esc(k.title())}</a>' if href.startswith("http") else ""
summary = f'<div class="summary">{esc(resume.summary)}</div>' if resume.summary else ""
st.markdown(f'<div class="card"><div class="profile"><div class="avatar">{esc(initials)}</div><div>'
            f'<div class="name">{esc(name)}</div><div class="muted">{contact or "No contact details found"}</div>'
            f'<div>{links}</div></div></div>{summary}</div>', unsafe_allow_html=True)

stats = [(f"{resume.total_experience_months / 12:.1f} yrs", "Experience"), (len(resume.skills), "Skills"),
         (len(resume.projects), "Projects"), (f"{resume.meta.get('seconds', 0)}s", f"Parsed · {resume.meta.get('mode', mode)}")]
st.markdown('<div class="stats">' + "".join(f'<div class="stat"><b>{v}</b><span>{esc(l)}</span></div>' for v, l in stats) + "</div>",
            unsafe_allow_html=True)

if resume.review_flags:
    with st.expander(f"⚠️ {len(resume.review_flags)} item(s) worth a quick check"):
        for flag in resume.review_flags:
            st.warning(flag)

tab_over, tab_exp, tab_proj, tab_high, tab_match, tab_json = st.tabs(
    ["Overview", "Experience", "Projects", "Highlights", "Job match", "JSON"])

# ------------------------------------------------------------------ overview
with tab_over:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### Skills")
        thr = parser.cfg.get("confidence", {}).get("review_threshold", 0.7)
        groups: dict[str, list[str]] = {}
        for s in resume.skills:
            groups.setdefault(parser.taxonomy.category(s), []).append(s)
        html = ""
        for cat, items in sorted(groups.items()):
            html += f'<div class="cat">{esc(cat.replace("_", " "))}</div>'
            for s in items:
                c = resume.skill_confidence.get(s, 0)
                html += f'<span class="chip {"low" if c < thr else ""}" title="confidence {c:.2f}">{esc(s)}</span>'
        st.markdown(html or "_No skills found._", unsafe_allow_html=True)
        if groups:
            st.caption("Amber chips = lower confidence. Hover a skill to see its score.")
    with right:
        st.markdown("##### Education")
        items = ""
        for e in resume.education:
            title = " in ".join(x for x in (e.degree, e.field_of_study) if x) or "—"
            sub = esc(e.institute or "")
            score = f" · {esc(e.score)}" if e.score else ""
            items += (f'<div class="item"><h4>{esc(title)}</h4><div class="muted">{sub}</div>'
                      f'<div class="when">{esc(" – ".join(x for x in (e.start_year, e.year) if x))}{score}</div></div>')
        st.markdown(f'<div class="tl">{items}</div>' if items else "_None found._", unsafe_allow_html=True)
        if resume.certifications:
            st.markdown("##### Certifications")
            st.markdown("".join(f'<div class="muted" style="margin:.35rem 0">🎓 {esc(c)}</div>' for c in resume.certifications),
                        unsafe_allow_html=True)

# ------------------------------------------------------------------ experience
with tab_exp:
    items = ""
    for e in resume.experience:
        bullets = "".join(f"<li>{esc(h)}</li>" for h in e.highlights)
        items += (f'<div class="item"><h4>{esc(e.role or "—")}</h4><div class="muted">{esc(e.company or "")}</div>'
                  f'<div class="when">{when(e.start, e.end, e.duration_months)}</div>'
                  f'{f"<ul>{bullets}</ul>" if bullets else ""}</div>')
    st.markdown(f'<div class="tl">{items}</div>' if items else "_No experience found._", unsafe_allow_html=True)

# ------------------------------------------------------------------ projects
with tab_proj:
    cards = ""
    for p in resume.projects:
        ctx = f'<div class="when">{esc(p.context)}</div>' if p.context else ""
        desc = f"<p>{esc(p.description)}</p>" if p.description else ""
        cards += f'<div class="card"><h4>{esc(p.name or "Untitled")}</h4>{ctx}{desc}{chips(p.tech)}</div>'
    st.markdown(f'<div class="grid">{cards}</div>' if cards else "_No projects found._", unsafe_allow_html=True)

# ------------------------------------------------------------------ highlights
with tab_high:
    if resume.achievements:
        st.markdown("##### Achievements & awards")
        st.markdown('<div class="grid">' + "".join(f'<div class="card">🏆 {esc(a)}</div>' for a in resume.achievements) + "</div>",
                    unsafe_allow_html=True)
    if resume.patents:
        st.markdown("##### Patents")
        for p in resume.patents:
            meta = " · ".join(x for x in (f"Application {p.application_number}" if p.application_number else None,
                                         f"Filed {p.filed}" if p.filed else None,
                                         f"Published {p.published}" if p.published else None) if x)
            st.markdown(f'<div class="card"><h4 style="margin:0">📜 {esc(p.title or "Untitled")}</h4>'
                        f'<div class="muted">{esc(meta)}</div>{chips([p.status], "") if p.status else ""}</div>',
                        unsafe_allow_html=True)
    if resume.languages:
        st.markdown("##### Languages")
        st.markdown(chips(resume.languages, ""), unsafe_allow_html=True)
    if not (resume.achievements or resume.patents or resume.languages):
        st.caption("No achievements, patents or languages found.")

# ------------------------------------------------------------------ job match
with tab_match:
    jd = st.text_area("Paste the job description", height=200, placeholder="Paste the job description here…")
    explain = st.toggle("Add LLM explanation", value=mode != "rules")
    if st.button("Match", type="primary") and jd.strip():
        matcher = JobMatcher(parser.cfg, parser.prompts, parser.taxonomy, parser.llm if explain else None)
        with st.spinner("Scoring…"):
            res = matcher.match(resume, jd, explain=explain)
        a, b = st.columns([1, 2], vertical_alignment="center")
        a.metric("Match score", f"{res['total_score']} / 100")
        b.progress(min(res["total_score"] / 100, 1.0))
        st.markdown('<div class="cat">Matched</div>' + (chips(res["matched_skills"], "") or "—"), unsafe_allow_html=True)
        st.markdown('<div class="cat">Missing</div>' + (chips(res["missing_skills"], "low") or "—"), unsafe_allow_html=True)
        with st.expander("Score breakdown"):
            st.dataframe(pd.DataFrame({"score": res["components"], "weight": res["weights"]}), use_container_width=True)
        if res.get("note"):
            st.caption(res["note"])
        if ex := res.get("explanation"):
            st.markdown(f"##### Assessment: {esc(str(ex.get('verdict', '?')))}")
            st.write(ex.get("summary", ""))
            s1, s2 = st.columns(2)
            s1.markdown("**Strengths**\n" + "\n".join(f"- {x}" for x in ex.get("strengths", [])))
            s2.markdown("**Gaps**\n" + "\n".join(f"- {x}" for x in ex.get("gaps", [])))
            st.markdown("**Suggested interview questions**\n" + "\n".join(f"1. {q}" for q in ex.get("interview_questions", [])))
        st.caption("Advisory only — a human recruiter makes the decision.")

# ------------------------------------------------------------------ json
with tab_json:
    payload = json.dumps(resume.model_dump(), indent=2, ensure_ascii=False)
    st.download_button("⬇ Download JSON", payload, file_name="parsed_resume.json", mime="application/json")
    with st.expander("Field confidence & sources"):
        st.dataframe(pd.DataFrame({"confidence": resume.confidence, "source": resume.sources}), use_container_width=True)
    st.code(payload, language="json")