"""Hybrid resume-parsing pipeline: layout-aware extraction -> segmentation -> rules + LLM -> merge -> validate."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date
from pathlib import Path
from typing import BinaryIO

from pydantic import ValidationError

from .cleaning import clean_text
from .config import Prompts, load_config, load_prompts, resolve_path
from .dates import fill_durations
from .llm_client import LLMClient, LLMError
from .privacy import anonymize, redact_contact
from .rule_extractors import rule_extract
from .schema import ParsedResume
from .segmentation import is_segmented, segment
from .taxonomy import SkillTaxonomy
from .text_extraction import bytes_to_text, extract_text

log = logging.getLogger(__name__)
MODES = ("rules", "llm", "hybrid")


def _cmp(field: str, v: str | None) -> str | None:
    if not v:
        return None
    return re.sub(r"\D", "", v)[-10:] if field == "phone" else re.sub(r"[^a-z0-9]", "", v.lower())


def _pick(field: str, rule_val, llm_val, prefer: str):
    """Choose between rule and LLM scalars. Returns (value, confidence, source, flag)."""
    if field == "summary" and llm_val:
        # Rules return the whole Profile paragraph, the LLM a one-sentence summary: they are different by design,
        # so a mismatch is not a disagreement worth flagging. Prefer the concise LLM version.
        return llm_val, 0.8, "llm", None
    if rule_val and llm_val:
        if _cmp(field, rule_val) == _cmp(field, llm_val):
            return rule_val, 0.98, "rules+llm", None
        value = rule_val if prefer == "rules" else llm_val
        return value, 0.6, prefer, f"{field}: rules and LLM disagree ({rule_val!r} vs {llm_val!r}); kept {prefer}"
    if rule_val:
        return rule_val, (0.9 if field in {"email", "phone"} else 0.65), "rules", None
    if llm_val:
        return llm_val, 0.8, "llm", None
    return None, 0.0, None, None


def merge(rule: ParsedResume, llm: ParsedResume | None, taxonomy: SkillTaxonomy,
          review_threshold: float = 0.7, today: date | None = None) -> ParsedResume:
    """Confidence-based merge. Regex wins for deterministic fields, LLM for semantic ones; skills are unioned."""
    llm = llm or ParsedResume()
    out = ParsedResume()
    conf: dict[str, float] = {}
    src: dict[str, str | None] = {}
    flags: list[str] = []

    for field, prefer in (("name", "llm"), ("email", "rules"), ("phone", "rules"),
                          ("location", "llm"), ("summary", "llm")):
        val, c, s, flag = _pick(field, getattr(rule, field), getattr(llm, field), prefer)
        setattr(out, field, val)
        conf[field], src[field] = c, s
        if flag:
            flags.append(flag)
    out.links = {**llm.links, **rule.links}

    for field in ("education", "experience", "projects", "certifications", "achievements", "patents", "languages"):
        r_items, l_items = getattr(rule, field), getattr(llm, field)
        if l_items:
            items, s = l_items, "llm"
            c = 0.95 if r_items and len(r_items) == len(l_items) else 0.85 if not r_items else 0.75
        elif r_items:
            items, s, c = r_items, "rules", 0.6
        else:
            items, s, c = [], None, 0.0
        setattr(out, field, [i.model_copy(deep=True) if hasattr(i, "model_copy") else i for i in items])
        conf[field], src[field] = c, s

    # skills: union with agreement-based confidence
    rule_sk = [taxonomy.normalize(s) for s in rule.skills]
    llm_sk = [taxonomy.normalize(s) for s in llm.skills]
    skills, sk_conf = [], {}
    for s in dict.fromkeys(rule_sk + llm_sk):
        in_r, in_l = s in rule_sk, s in llm_sk
        skills.append(s)
        sk_conf[s] = 0.95 if in_r and in_l else 0.8 if in_r else 0.75
    out.skills, out.skill_confidence = skills, sk_conf
    conf["skills"] = round(sum(sk_conf.values()) / len(sk_conf), 2) if sk_conf else 0.0
    src["skills"] = "rules+llm" if rule_sk and llm_sk else "rules" if rule_sk else "llm" if llm_sk else None

    out.total_experience_months = fill_durations(out.experience, today)
    conf["total_experience_months"] = conf["experience"]
    for field, c in conf.items():
        if getattr(out, field) and c < review_threshold:
            flags.append(f"review: low confidence for '{field}' ({c:.2f})")
    out.confidence, out.sources, out.review_flags = conf, src, flags
    return out


class ResumeParser:
    def __init__(self, config: dict | None = None, llm: LLMClient | None = None):
        self.cfg = config or load_config()
        self.prompts: Prompts = load_prompts(self.cfg)
        self.taxonomy = SkillTaxonomy.from_file(resolve_path(self.cfg, self.cfg["pipeline"]["skills_file"]))
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(self.cfg, self.prompts)
        return self._llm

    # ------------------------------------------------------------ public API
    def parse_file(self, path: str | Path, **kw) -> ParsedResume:
        return self.parse_text(extract_text(path), **kw)

    def parse_bytes(self, data: bytes, filename: str, **kw) -> ParsedResume:
        return self.parse_text(bytes_to_text(data, filename), **kw)

    def parse_text(self, raw_text: str, mode: str | None = None, anonymize_output: bool | None = None,
                   today: date | None = None) -> ParsedResume:
        mode = mode or self.cfg["pipeline"]["mode"]
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        pcfg, privacy = self.cfg["pipeline"], self.cfg["privacy"]
        t0 = time.perf_counter()
        usage_before = dict(self.llm.usage) if self._llm else {}

        text = clean_text(raw_text)[: pcfg["max_resume_chars"]]
        sections = segment(text) if pcfg.get("use_segmentation", True) else {"full": text}
        rule = rule_extract(text, sections, self.taxonomy, today)

        llm_result, notes = None, []
        if mode in {"llm", "hybrid"}:
            try:
                llm_result = self._llm_extract(text, sections, redact=(mode == "hybrid" and privacy["redact_contact_before_llm"]))
            except (LLMError, ValidationError) as e:
                if mode == "llm":
                    raise LLMError(f"LLM extraction failed: {e}") from e
                log.warning("LLM failed, falling back to rules: %s", e)
                notes.append(f"LLM extraction failed ({type(e).__name__}: {str(e)[:300]}); rules-only fallback used")

        base = ParsedResume() if mode == "llm" else rule
        result = merge(base, llm_result, self.taxonomy, self.cfg["confidence"]["review_threshold"], today)
        result.review_flags = notes + result.review_flags

        if anonymize_output if anonymize_output is not None else privacy["anonymize_output"]:
            result = anonymize(result, privacy["mask_institute"], privacy["mask_graduation_year"])

        usage_after = dict(self.llm.usage) if self._llm else {}
        result.meta.update({
            "mode": mode,
            "model": (getattr(self._llm, "last_model", None) or self._llm.model) if self._llm and mode != "rules" else None,
            "prompt_version": self.prompts.version,
            "segmented": is_segmented(sections),
            "sections_found": [k for k in sections if k != "full"],
            "seconds": round(time.perf_counter() - t0, 3),
            "input_tokens": usage_after.get("input_tokens", 0) - usage_before.get("input_tokens", 0),
            "output_tokens": usage_after.get("output_tokens", 0) - usage_before.get("output_tokens", 0),
        })
        return result

    # ------------------------------------------------------------ internals
    def _llm_extract(self, text: str, sections: dict[str, str], redact: bool) -> ParsedResume:
        body = text
        if is_segmented(sections):  # give the model the detected structure
            body = "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in sections.items())
        if redact:
            body = redact_contact(body)
        prompt = self.prompts.render("extraction", resume_text=body)
        data = self.llm.complete_json(prompt, system=self.prompts.render("system"))
        for k in ("confidence", "skill_confidence", "sources", "review_flags", "meta", "total_experience_months"):
            data.pop(k, None)  # provenance/derived fields are computed in code, never trusted from the model
        return ParsedResume.model_validate(data)


def to_json(resume: ParsedResume, indent: int = 2) -> str:
    return json.dumps(resume.model_dump(), indent=indent, ensure_ascii=False)