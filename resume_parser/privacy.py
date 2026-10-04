"""PII handling: redact contact data before it reaches the LLM; anonymise parsed output."""
from __future__ import annotations

import re

from .patterns import EMAIL_RE, PHONE_RE, URL_RE, is_phone_like
from .schema import ParsedResume


def redact_contact(text: str) -> str:
    """Replace email / URL / phone with placeholders (these are extracted locally by regex)."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    return PHONE_RE.sub(lambda m: "[PHONE]" if is_phone_like(m.group(0)) else m.group(0), text)


def anonymize(resume: ParsedResume, mask_institute: bool = True, mask_year: bool = True) -> ParsedResume:
    """Return a copy with identity & bias-prone attributes removed (name, contact, location, college, years)."""
    r = resume.model_copy(deep=True)
    r.name = "Candidate"
    r.email = r.phone = r.location = None
    r.links = {}
    for ed in r.education:
        if mask_institute:
            ed.institute = "[REDACTED]"
        if mask_year:
            ed.year = ed.start_year = None
    r.meta["anonymized"] = True
    return r