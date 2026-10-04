from datetime import date

import pytest

from resume_parser import JobMatcher, ResumeParser
from resume_parser.llm_client import LLMError, parse_json_object
from resume_parser.privacy import anonymize
from tests.helpers import SAMPLES, FakeLLM

TODAY = date(2026, 10, 3)
LLM_PAYLOAD = {
    "name": "Riya Sharma", "email": None, "phone": None, "location": "Jaipur, Rajasthan",
    "education": [{"degree": "B.Tech", "field_of_study": "Computer Science", "institute": "JIET", "year": 2026}],
    "experience": [
        {"company": "Nimbus Analytics", "role": "Data Science Intern", "start": "2024-06", "end": "2024-08"},
        {"company": "Orbit Labs", "role": "ML Intern", "start": "2025-01", "end": "2025-03", "highlights": None},
    ],
    "skills": ["python", "sklearn", "Prompt Engineering", "Kubernetes"],
    "projects": None, "certifications": None,
}


@pytest.fixture()
def parser():
    return ResumeParser()


def sample(n=1):
    return (SAMPLES / f"resume_0{n}.txt").read_text(encoding="utf-8")


def test_rules_mode_needs_no_llm(parser):
    r = parser.parse_text(sample(1), mode="rules", today=TODAY)
    assert parser._llm is None  # no client was even created
    assert r.email == "riya.sharma@example.com" and r.total_experience_months == 6
    assert r.meta["mode"] == "rules" and r.meta["model"] is None


def test_hybrid_merge_and_confidence(parser):
    parser._llm = FakeLLM(LLM_PAYLOAD)
    r = parser.parse_text(sample(1), mode="hybrid", today=TODAY)
    assert r.name == "Riya Sharma" and r.confidence["name"] == 0.98          # both agree
    assert r.email == "riya.sharma@example.com" and r.sources["email"] == "rules"
    assert r.location == "Jaipur, Rajasthan"                                  # LLM-only field
    assert r.education[0].year == "2026"                                      # int coerced to str
    assert r.total_experience_months == 6                                     # computed in code
    assert "Python" in r.skills and "scikit-learn" in r.skills                # aliases normalised
    assert r.skill_confidence["Python"] == 0.95                               # rules + LLM agree
    assert r.skill_confidence["Kubernetes"] == 0.75                           # LLM-only
    assert r.skill_confidence["Java"] == 0.8                                  # rules-only


def test_hybrid_never_sends_contact_details_to_llm(parser):
    fake = parser._llm = FakeLLM(LLM_PAYLOAD)
    parser.parse_text(sample(1), mode="hybrid", today=TODAY)
    sent = fake.prompts_seen[0]
    for secret in ("riya.sharma@example.com", "98765", "linkedin.com/in/riya-sharma", "github.com/riyasharma"):
        assert secret not in sent
    assert "[EMAIL]" in sent and "[EDUCATION]" in sent  # redacted + structure passed along


def test_hybrid_falls_back_when_llm_fails(parser):
    parser._llm = FakeLLM(fail=LLMError("boom"))
    r = parser.parse_text(sample(2), mode="hybrid", today=TODAY)
    assert r.email == "arjun.mehta@example.org" and r.total_experience_months == 17
    assert any("rules-only fallback" in f for f in r.review_flags)


def test_llm_mode_raises_when_llm_fails(parser):
    parser._llm = FakeLLM(fail=LLMError("boom"))
    with pytest.raises(LLMError):
        parser.parse_text(sample(1), mode="llm")


def test_disagreement_is_flagged(parser):
    parser._llm = FakeLLM({**LLM_PAYLOAD, "name": "Someone Else"})
    r = parser.parse_text(sample(1), mode="hybrid", today=TODAY)
    assert r.name == "Someone Else" and r.confidence["name"] == 0.6
    assert any(f.startswith("name: rules and LLM disagree") for f in r.review_flags)


def test_ablation_without_segmentation_hurts_rules():
    from resume_parser.config import load_config
    cfg = load_config()
    cfg["pipeline"]["use_segmentation"] = False
    r = ResumeParser(cfg).parse_text(sample(1), mode="rules", today=TODAY)
    assert r.meta["segmented"] is False
    # Education date ranges now leak into "experience" -> RQ1 effect is measurable
    assert len(r.experience) > 2 or r.total_experience_months != 6


def test_anonymize(parser):
    r = anonymize(parser.parse_text(sample(1), mode="rules", today=TODAY))
    assert r.name == "Candidate" and r.email is None and r.links == {}
    assert all(e.institute == "[REDACTED]" and e.year is None for e in r.education)


def test_invalid_mode(parser):
    with pytest.raises(ValueError):
        parser.parse_text("x", mode="magic")


# ---------------------------------------------------------------- JSON handling
def test_json_parsing():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": {"b": 2}} Hope it helps') == {"a": {"b": 2}}
    for bad in ("no json", "[1,2]", '{"a": '):
        with pytest.raises(ValueError):
            parse_json_object(bad)


# ---------------------------------------------------------------- matching
def test_matching_scores(parser):
    parser._llm = None
    r = parser.parse_text(sample(1), mode="rules", today=TODAY)
    jd = (SAMPLES / "jd_data_scientist.txt").read_text(encoding="utf-8")
    m = JobMatcher(parser.cfg, parser.prompts, parser.taxonomy).match(r, jd, explain=False)
    assert "Python" in m["matched_skills"] and "AWS" in m["missing_skills"]
    assert m["jd_required_years"] == 1.0 and m["components"]["experience_fit"] == 0.5
    assert m["components"]["education_fit"] == 1.0
    assert 0 < m["total_score"] < 100 and abs(sum(m["weights"].values()) - 1) < 0.01


def test_matching_explanation_uses_anonymised_profile(parser):
    r = parser.parse_text(sample(1), mode="rules", today=TODAY)
    fake = FakeLLM({"verdict": "moderate", "summary": "ok", "strengths": [], "gaps": [], "interview_questions": []})
    m = JobMatcher(parser.cfg, parser.prompts, parser.taxonomy, fake).match(r, "Python developer, 2 years")
    assert m["explanation"]["verdict"] == "moderate"
    assert "Riya" not in fake.prompts_seen[0] and "example.com" not in fake.prompts_seen[0]


def test_matching_survives_llm_failure(parser):
    r = parser.parse_text(sample(1), mode="rules", today=TODAY)
    m = JobMatcher(parser.cfg, parser.prompts, parser.taxonomy, FakeLLM(fail=LLMError("down"))).match(r, "Python")
    assert m["explanation"] is None and "unavailable" in m["note"]


# ---------------------------------------------------------------- full stack with a simulated Gemini backend
def test_end_to_end_through_real_llmclient_with_fake_gemini(tmp_path):
    import json
    from types import SimpleNamespace

    from resume_parser.config import load_config
    from resume_parser.llm_client import LLMClient

    cfg = load_config()
    cfg["llm"].update(provider="gemini", cache_enabled=False, requests_per_minute=None)
    calls = []

    def generate_content(**kw):
        calls.append(kw)
        return SimpleNamespace(text="```json\n" + json.dumps(LLM_PAYLOAD) + "\n```", candidates=[],
                               usage_metadata=SimpleNamespace(prompt_token_count=800, candidates_token_count=200,
                                                              thoughts_token_count=0))

    parser = ResumeParser(cfg)
    parser._llm = LLMClient(cfg, parser.prompts)
    parser._llm._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    r = parser.parse_text(sample(1), mode="hybrid", today=TODAY)
    assert r.name == "Riya Sharma" and r.total_experience_months == 6 and r.meta["input_tokens"] == 800
    assert "riya.sharma@example.com" not in calls[0]["contents"]            # privacy redaction holds end-to-end
    assert "precise information-extraction engine" in calls[0]["config"].system_instruction  # prompt file is used


def test_cli_check_reports_missing_key(monkeypatch, capsys):
    import main

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    assert main.main(["check"]) == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_summary_difference_is_not_flagged_as_disagreement(parser):
    from resume_parser.pipeline import _pick
    long_rule = "Motivated student with a proven record.\nExperienced in many things across domains."
    value, conf, src, flag = _pick("summary", long_rule, "Motivated student with a proven record.", "llm")
    assert value.startswith("Motivated") and src == "llm" and conf >= 0.7 and flag is None