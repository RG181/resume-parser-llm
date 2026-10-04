"""Explainable resume <-> job-description matching.
Numbers are computed deterministically; the LLM only explains them (it cannot change the score)."""
from __future__ import annotations

import json
import logging
import re

from .config import Prompts
from .llm_client import LLMClient, LLMError
from .patterns import DEGREE_CI, DEGREE_CS
from .privacy import anonymize, redact_contact
from .schema import ParsedResume
from .taxonomy import SkillTaxonomy

log = logging.getLogger(__name__)
_LEVELS = [(4, re.compile(r"ph\.?\s?d|doctor", re.I)),
           (3, re.compile(r"master|m\.?\s?tech|m\.?\s?sc|mba|mca|m\.?\s?com", re.I)),
           (2, re.compile(r"bachelor|b\.?\s?tech|b\.?\s?sc|bca|bba|b\.?\s?com", re.I)),
           (1, re.compile(r"diploma", re.I))]


def degree_level(text: str) -> int:
    """0 = none, 1 diploma, 2 bachelor, 3 master, 4 doctorate (highest mentioned)."""
    best = 0
    for lvl, rx in _LEVELS:
        if rx.search(text):
            best = max(best, lvl)
    if not best and DEGREE_CS.search(text):  # B.E / M.E style (case-sensitive)
        best = 3 if re.search(r"\bM\.?(E|A)\b", text) else 2
    return best


def required_years(jd: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", jd, re.I)
    return float(m.group(1)) if m else None


def _tfidf_cosine(a: str, b: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not a.strip() or not b.strip():
        return 0.0
    try:
        m = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([a, b])
    except ValueError:
        return 0.0
    return float(cosine_similarity(m[0], m[1])[0, 0])


def _profile_text(r: ParsedResume) -> str:
    parts = [r.summary or "", " ".join(r.skills)]
    for e in r.experience:
        parts += [e.role or "", " ".join(e.highlights)]
    for p in r.projects:
        parts += [p.name or "", p.description or "", " ".join(p.tech)]
    return " ".join(parts)


class JobMatcher:
    def __init__(self, cfg: dict, prompts: Prompts, taxonomy: SkillTaxonomy, llm: LLMClient | None = None):
        self.cfg, self.prompts, self.taxonomy, self._llm = cfg, prompts, taxonomy, llm

    def score(self, resume: ParsedResume, jd_text: str) -> dict:
        weights = self.cfg["matching"]["weights"]
        jd_skills = self.taxonomy.find(redact_contact(jd_text))
        have = set(resume.skills)
        matched = [s for s in jd_skills if s in have]
        missing = [s for s in jd_skills if s not in have]

        skill_overlap = len(matched) / len(jd_skills) if jd_skills else 0.0
        text_similarity = _tfidf_cosine(_profile_text(resume), jd_text)
        req = required_years(jd_text)
        years = resume.total_experience_months / 12
        experience_fit = 1.0 if req is None else min(1.0, years / req) if req > 0 else 1.0
        jd_level = degree_level(jd_text)
        res_level = max([degree_level(f"{e.degree or ''} {e.field_of_study or ''}") for e in resume.education] or [0])
        education_fit = 1.0 if jd_level == 0 else min(1.0, res_level / jd_level)

        parts = {"skill_overlap": skill_overlap, "text_similarity": text_similarity,
                 "experience_fit": experience_fit, "education_fit": education_fit}
        wsum = sum(max(weights.get(k, 0), 0) for k in parts) or 1.0
        total = sum(max(weights.get(k, 0), 0) * v for k, v in parts.items()) / wsum
        return {
            "total_score": round(total * 100, 1),
            "components": {k: round(v, 3) for k, v in parts.items()},
            "weights": {k: round(max(weights.get(k, 0), 0) / wsum, 3) for k in parts},
            "matched_skills": matched,
            "missing_skills": missing,
            "jd_required_years": req,
            "candidate_years": round(years, 1),
            "note": None if jd_skills else "No recognisable skills found in the job description.",
        }

    def match(self, resume: ParsedResume, jd_text: str, explain: bool | None = None) -> dict:
        result = self.score(resume, jd_text)
        explain = self.cfg["matching"].get("explain_with_llm", True) if explain is None else explain
        if not explain:
            return result
        profile = anonymize(resume).model_dump(
            include={"summary", "education", "experience", "skills", "projects", "certifications",
                     "total_experience_months"})
        prompt = self.prompts.render("matching", job_description=jd_text[:6000],
                                     candidate_profile=json.dumps(profile, ensure_ascii=False),
                                     scores=json.dumps(result, ensure_ascii=False))
        try:
            llm = self._llm
            if llm is None:
                raise LLMError("no LLM client configured")
            result["explanation"] = llm.complete_json(prompt, system=self.prompts.render("system"))
        except LLMError as e:
            log.warning("LLM explanation unavailable: %s", e)
            result["explanation"] = None
            result["note"] = (result["note"] or "") + f" LLM explanation unavailable: {e}"
        return result
