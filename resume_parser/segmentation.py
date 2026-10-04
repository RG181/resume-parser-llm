"""Section segmentation: heading detection with exact + fuzzy matching."""
from __future__ import annotations

import difflib
import re

SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "objective", "career objective", "about me", "about"],
    "education": ["education", "academic background", "academics", "educational qualifications",
                  "academic qualifications", "qualifications", "education and training"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "internships", "internship experience", "work history", "career history", "employment"],
    "skills": ["skills", "technical skills", "skills summary", "core competencies", "technical proficiency",
               "key skills", "tools and technologies", "technical expertise"],
    "projects": ["projects", "academic projects", "personal projects", "key projects", "project work"],
    "certifications": ["certifications", "certificates", "licenses and certifications", "courses",
                       "training and certifications"],
    "achievements": ["achievements", "awards", "honors", "accomplishments", "awards and achievements",
                     "achievements and awards", "honors and awards", "awards and honors"],
    "patents": ["patents", "patent", "patent publication", "patent publications", "patents and publications",
                "publications", "research publications", "research papers"],
    "languages": ["languages", "language skills", "spoken languages"],
    "other": ["interests", "hobbies", "references", "extracurricular activities", "volunteering",
              "extracurricular", "extracurricular and community", "leadership"],
}
_PHRASES = {p: sec for sec, phrases in SECTION_ALIASES.items() for p in phrases}


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z& ]", "", line.lower().replace("/", " "))).strip()


def heading_of(line: str) -> str | None:
    s = line.strip().rstrip(":").strip()
    if not s or len(s) > 40 or s.startswith("- "):
        return None
    n = _norm(s).replace("&", "and")
    if not n or len(n.split()) > 4:
        return None
    if n in _PHRASES:
        return _PHRASES[n]
    close = difflib.get_close_matches(n, _PHRASES.keys(), n=1, cutoff=0.88)
    return _PHRASES[close[0]] if close else None


_UPPER_PHRASES = sorted(((ph.upper(), sec) for ph, sec in _PHRASES.items() if len(ph) >= 6), key=lambda x: -len(x[0]))


def glued_heading(line: str) -> tuple[str, str] | None:
    """PDFs sometimes glue an ALL-CAPS heading to the next text: 'WORK EXPERIENCESoftware Engineer'."""
    s = line.strip()
    for phrase, sec in _UPPER_PHRASES:
        n = len(phrase)
        if s.startswith(phrase) and len(s) > n + 1 and s[n].isupper() and s[n + 1].islower():
            return sec, s[n:].strip()
    return None


def segment(text: str) -> dict[str, str]:
    """Split text into {'header', 'education', 'experience', ...}. Falls back to {'full': text}."""
    sections: dict[str, list[str]] = {}
    current = "header"
    for line in text.split("\n"):
        heading = heading_of(line)
        glued = None if heading else glued_heading(line)
        if glued:
            current = glued[0]
            sections.setdefault(current, []).append(glued[1])
        elif heading:
            current = heading
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    out = {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}
    if not [k for k in out if k != "header"]:
        return {"full": text}
    return out


def is_segmented(sections: dict[str, str]) -> bool:
    return "full" not in sections