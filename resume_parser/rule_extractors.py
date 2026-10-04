"""Deterministic (regex + heuristic) extractors. Fast, free, and used as baseline + hybrid component."""
from __future__ import annotations

import re
from datetime import date

from .dates import RANGE_RE
from .patterns import (DEGREE_CI, EMAIL_RE, GITHUB_RE, INSTITUTE_RE, LINKEDIN_RE, PHONE_RE, ROLE_RE,
                       SPLIT_AT_RE, SPLIT_RE, URL_RE, has_degree, is_phone_like)
from .privacy import redact_contact
from .schema import Education, Experience, ParsedResume, Patent, Project
from .segmentation import heading_of, is_segmented
from .taxonomy import SkillTaxonomy

_NAME_STOP = {"resume", "cv", "curriculum", "vitae", "profile", "summary", "page"}
_SCORE_RE = re.compile(r"(?i)(?:cgpa|gpa)\s*[:\-]?\s*(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)|percentage\s*[:\-]?\s*(\d+(?:\.\d+)?\s*%?)"
                       r"|(\d{2,3}(?:\.\d+)?)\s*%")
_SCORE_STRIP = re.compile(r"(?i)\b(?:cgpa|gpa|percentage)\b\s*[:\-]?\s*[\d.]+(?:\s*/\s*\d+)?\s*%?|[\d.]+\s*%")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def _lines(text: str) -> list[str]:
    return [l.strip() for l in text.split("\n") if l.strip()]


_TERMINAL = (".", "!", "?", ";", ":", ")", '"', "'", "”", "’")


def _join_wrapped(lines: list[str]) -> list[str]:
    """PDFs wrap long bullets onto bare lines. Glue those continuations back onto their '- ' bullet so a wrapped
    sentence is never mistaken for a new project / role / certificate."""
    out: list[str] = []
    for line in lines:
        if (out and out[-1].startswith("- ") and not line.startswith("- ") and " | " not in line
                and not RANGE_RE.search(line) and (line[:1].islower() or not out[-1].endswith(_TERMINAL))):
            out[-1] += " " + line
        else:
            out.append(line)
    return out


def _split_commas(s: str) -> list[str]:
    """Split on commas that are not inside parentheses: 'HuggingFace (DistilBART, BART MNLI), Flask' -> 2 items."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth <= 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


# ------------------------------------------------------------------ contact
def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    return m.group(0).lower() if m else None


def extract_phone(text: str) -> str | None:
    for m in PHONE_RE.finditer(text):
        s = m.group(0).strip(" .-")
        if is_phone_like(s):
            return re.sub(r"\s+", " ", s)
    return None


def extract_links(text: str) -> dict[str, str]:
    links = {}
    for key, rx in (("linkedin", LINKEDIN_RE), ("github", GITHUB_RE)):
        if m := rx.search(text):
            links[key] = m.group(0).rstrip(".,;")
    return links


def extract_name(lines: list[str]) -> str | None:
    for line in lines[:8]:
        line = line.strip()
        tokens = line.split()
        if not 2 <= len(tokens) <= 4 or heading_of(line):
            continue
        if any(t.lower().strip(".,") in _NAME_STOP for t in tokens):
            continue
        if all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", t) for t in tokens):
            return line.title() if line.isupper() or line.islower() else line
    return None


_PLACE_RE = re.compile(r"[A-Z][A-Za-z.'’ -]+(?:,\s*[A-Z][A-Za-z.'’ -]+){1,2}")


def extract_location(header: list[str], name: str | None = None) -> str | None:
    """'City, State[, Country]' from the contact block, after removing email / phone / URLs."""
    for line in header[:8]:
        s = PHONE_RE.sub(" ", URL_RE.sub(" ", EMAIL_RE.sub(" ", line)))
        for seg in re.split(r"\s*[|•·]\s*|\s{2,}", s):
            seg = seg.strip(" ,;:-+#§?")
            if (_PLACE_RE.fullmatch(seg) and seg != name and not ROLE_RE.search(seg) and not has_degree(seg)
                    and not INSTITUTE_RE.search(seg)):
                return seg
    return None


# ------------------------------------------------------------------ education
def _strip_noise(line: str) -> str:
    line = RANGE_RE.sub("", line)
    line = re.sub(r"\(?\b(?:19|20)\d{2}\b\)?", "", line)
    line = _SCORE_STRIP.sub("", line)
    line = re.sub(r"\(\s*\)", "", line)
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip(" |,–—-")


def _institute_in(line: str) -> str | None:
    for part in [p for p in SPLIT_RE.split(_strip_noise(line)) if p.strip()]:
        if INSTITUTE_RE.search(part):
            if has_degree(part) and "," in part:
                for q in part.split(","):
                    if INSTITUTE_RE.search(q):
                        return q.strip()
            return part.strip()
    return None


def extract_education(text: str) -> list[Education]:
    lines = _lines(text)
    out, used = [], set()  # `used`: line indices already claimed as some entry's institute
    for i, line in enumerate(lines):
        if line.startswith("- ") or not has_degree(line):
            continue
        # "Some Higher Secondary School | City" sitting above the next entry's "Class X | 94% | 2021":
        # an institute heading (no year, no score), not a degree line of its own.
        if (INSTITUTE_RE.search(line) and not _YEAR_RE.search(line) and not _SCORE_RE.search(line)
                and i + 1 < len(lines) and has_degree(lines[i + 1])):
            continue
        window = []  # this line + up to 2 following lines, stopping at the next degree line
        for j in range(i, min(i + 3, len(lines))):
            if j > i and has_degree(lines[j]):
                break
            window.append(lines[j])
        parts = [p for p in SPLIT_RE.split(_strip_noise(line)) if p.strip()]
        degree_part = next((p for p in parts if has_degree(p)), parts[0] if parts else line)
        degree = degree_part
        if "," in degree_part:  # "Degree, Institute, City" -> keep up to the institute
            kept = []
            for q in degree_part.split(","):
                if INSTITUTE_RE.search(q):
                    break
                kept.append(q.strip())
            degree = ", ".join(kept) or degree_part
        # Institute: same line -> the line ABOVE (if no other entry already claimed it) -> following lines.
        institute, claimed = _institute_in(line), i
        if institute is None and i > 0 and (i - 1) not in used and not lines[i - 1].startswith("- "):
            institute, claimed = _institute_in(lines[i - 1]), i - 1
        if institute is None:
            for j in range(i + 1, i + len(window)):
                if not lines[j].startswith("- ") and (institute := _institute_in(lines[j])):
                    claimed = j
                    break
        if institute is not None:
            used.add(claimed)
        years = next((sorted(set(_YEAR_RE.findall(w))) for w in window if _YEAR_RE.search(w)), [])
        year, start_year = (years[-1] if years else None), (years[0] if len(years) > 1 else None)
        score = None
        for w in window:
            if m := _SCORE_RE.search(w):
                score = next(g for g in m.groups() if g).strip()
                if m.group(3) and "%" not in score:
                    score += "%"
                break
        field = None
        if m := re.match(r"(?i)^(.+?)\s+(?:in|of)\s+([A-Za-z&,/ ]+)$", degree.strip()):
            if not re.match(r"(?i)^bachelor[s']?$|^master[s']?$", m.group(1)):  # keep 'Bachelor of Technology' intact
                degree, field = m.group(1), m.group(2).strip()
        if field is None and (m := re.match(r"(?i)^(bachelor[s']?\s+of\s+\w+|master[s']?\s+of\s+\w+)\s+in\s+(.+)$", degree.strip())):
            degree, field = m.group(1), m.group(2).strip()
        out.append(Education(degree=degree.strip(), field_of_study=field, institute=institute, start_year=start_year, year=year, score=score))
    return out


# ------------------------------------------------------------------ experience
def _split_role_company(head: str) -> tuple[str | None, str | None]:
    role = company = None
    for p in [p.strip() for p in SPLIT_AT_RE.split(head) if p.strip()]:
        if role is None and ROLE_RE.search(p):
            role = p
        elif company is None:
            company = p.split(",")[0].strip()
    return role, company


def _looks_like_org(s: str) -> bool:
    """'J.P. Morgan Chase & Co.' / 'Manipal University Jaipur' yes; 'Cleaned datasets with Django.' no (a bullet sentence)."""
    words = [w for w in s.split() if w[:1].isalpha()]
    if not words or len(s.split()) > 8 or len(s) > 70 or s.endswith(":") or s.isupper():
        return False
    return sum(w[0].isupper() for w in words) / len(words) >= 0.6


def extract_experience(text: str) -> list[Experience]:
    lines = _join_wrapped(_lines(text))
    entries: list[Experience] = []
    for i, line in enumerate(lines):
        m = RANGE_RE.search(line)
        if m and not line.startswith("- "):
            head = re.sub(r"\(\s*\)", "", RANGE_RE.sub("", line)).strip(" |,–—-")
            if not head and i > 0 and not lines[i - 1].startswith("- "):
                head = lines[i - 1]
            role, company = _split_role_company(head)
            org = None  # layout: "Role | dates" with the organisation on the NEXT line
            if i + 1 < len(lines) and not lines[i + 1].startswith("- ") and not RANGE_RE.search(lines[i + 1]):
                nxt = [p.strip() for p in SPLIT_RE.split(lines[i + 1]) if p.strip()]
                org = nxt[0] if nxt and _looks_like_org(nxt[0]) else None
            if role and "," in role:  # "HR Lead, Autonomous Initiative Club"
                first, rest = role.split(",", 1)
                if ROLE_RE.search(first) and not ROLE_RE.search(rest):
                    role = first.strip()
                    company = f"{rest.strip()}, {org}" if org else rest.strip()
            if company is None:
                company = org
            entries.append(Experience(company=company, role=role, start=m.group(1), end=m.group(2)))
        elif entries and line.startswith("- ") and len(entries[-1].highlights) < 3:
            entries[-1].highlights.append(line[2:].strip())
    return entries


# ------------------------------------------------------------------ projects / certs
_AWARD_RE = re.compile(r"(?i)\b(?:place|prize|winner|award|finalist|shortlisted|selectee|hackathon|sparkathon|festival"
                       r"|sih|incubation|innovate\w*|challenge|competition|academic|course|capstone|internship)\b")
_TECH_LINE_RE = re.compile(r"(?i)^(?:tech(?:nology)?\s*stack|technologies|tools|built with|stack)\s*[:\-]\s*(.+)")


def extract_projects(text: str) -> list[Project]:
    """A project starts at every non-bullet line ('Name | tech | event'); bullets (and wrapped lines) describe it."""
    out: list[Project] = []
    for line in _join_wrapped(_lines(text)):
        if line.startswith("- "):
            body = line[2:].strip()
            if out:
                if m := _TECH_LINE_RE.match(body):
                    if not out[-1].tech:
                        out[-1].tech = _split_commas(m.group(1))
                elif not out[-1].description:
                    out[-1].description = body
            continue
        parts = [p.strip() for p in SPLIT_RE.split(line) if p.strip()]
        if not parts:
            continue
        tech, ctx = [], []
        for k, part in enumerate(parts[1:]):
            if k == 0 and not _AWARD_RE.search(part):
                tech = _split_commas(part)
            else:
                ctx.append(part)
        out.append(Project(name=parts[0], tech=tech, context=" | ".join(ctx) or None))
    return out


_CERT_DETAIL_RE = re.compile(r"(?i)^(?:score|covers?|id|credential(?: id)?|grade|duration|skills?)\s*[:\-]")


def extract_certifications(text: str) -> list[str]:
    out: list[str] = []
    for line in _lines(text):
        line = line[2:].strip() if line.startswith("- ") else line
        if out and _CERT_DETAIL_RE.match(line):
            out[-1] += " — " + line
        else:
            out.append(line)
    return out


_ICON_RE = re.compile(r"^(?:[^\w(]+\s*|[a-z?]\s+(?=[A-Z]))")  # glyph a PDF icon font turned into junk
_TRAIL_DATE_RE = re.compile(r"\s+(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Spring|Summer|Autumn|Fall|Winter)[a-z]*\.?\s+)?(?:19|20)\d{2}\s*$")


def extract_achievements(text: str) -> list[str]:
    """Title lines only. When the section has '- ' bullets, wrapped description lines also lack the bullet, so a
    title must then contain a ' – ' separator; a plain list (no bullets) keeps every line."""
    lines = _lines(text)
    has_bullets = any(l.startswith("- ") for l in lines)
    out = []
    for line in lines:
        if line.startswith("- ") or (has_bullets and not re.search(r"\s[–—-]\s", line)):
            continue
        title = _TRAIL_DATE_RE.sub("", _ICON_RE.sub("", line)).strip()
        if title:
            out.append(title)
    return out


def extract_patents(text: str) -> list[Patent]:
    """One patent per section (title lines + 'Application No.' line); multiple patents are left to the LLM."""
    lines = [l for l in _lines(text) if not l.startswith("- ")]
    flat = " ".join(_lines(text))
    number = (m.group(1) if (m := re.search(r"(?i)application\s*no\.?\s*:?\s*([\w/-]+)", flat)) else None)
    stop = next((i for i, l in enumerate(lines) if re.match(r"(?i)application\s*no", l)), None)
    title_lines = lines[:stop] if stop else lines[:1]  # title block sits above the "Application No." line
    title = " ".join(re.sub(r"(?i)\b(?:filed|published)\s*:\s*\w+\s+\d{4}", "", l).strip() for l in title_lines).strip() or None
    if not title and not number:
        return []
    date = lambda key: (m.group(1) if (m := re.search(rf"(?i)\b{key}\s*:\s*([A-Za-z]{{3,9}}\s+\d{{4}})", flat)) else None)
    status = (m.group(1).strip() if (m := re.search(r"(?i)status\s*:\s*([^.\n]+)", flat)) else None)
    return [Patent(title=title, application_number=number, filed=date("filed"), published=date("published"), status=status)]


def extract_languages(text: str) -> list[str]:
    paren = re.findall(r"([A-Z][A-Za-z]+)\s*\(([^)]+)\)", text)
    if paren:
        return [f"{lang} ({level.strip()})" for lang, level in paren]
    return [t.strip() for t in re.split(r"[,|;•\n]", text) if re.fullmatch(r"[A-Z][a-z]+", t.strip())]


# ------------------------------------------------------------------ orchestration
def rule_extract(text: str, sections: dict[str, str], taxonomy: SkillTaxonomy,
                 today: date | None = None) -> ParsedResume:
    """Run all rule-based extractors. Without segmentation every extractor sees the full text (RQ1 baseline)."""
    segmented = is_segmented(sections)
    header = _lines(sections.get("header", "")) if segmented and "header" in sections else _lines(text)
    get = (lambda k: sections.get(k, "")) if segmented else (lambda k: text)

    r = ParsedResume()
    r.name = extract_name(header)
    r.location = extract_location(header, r.name)
    r.email, r.phone, r.links = extract_email(text), extract_phone(text), extract_links(text)
    r.summary = sections.get("summary") if segmented else None
    r.education = extract_education(get("education"))
    r.experience = extract_experience(get("experience"))
    r.projects = extract_projects(get("projects")) if segmented else []
    r.certifications = extract_certifications(get("certifications")) if segmented else []
    r.achievements = extract_achievements(get("achievements")) if segmented else []
    r.patents = extract_patents(get("patents")) if segmented else []
    r.languages = extract_languages(get("languages")) if segmented else []

    skill_scope = "\n".join(v for k, v in sections.items() if k not in {"education", "header", "certifications"}) if segmented else text
    r.skills = taxonomy.find(redact_contact(skill_scope))
    return r