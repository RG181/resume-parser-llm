"""Shared regular expressions."""
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().-]{8,17}\d(?!\w)")
_YEAR_RANGE = re.compile(r"(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)\d{2}")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s,;|)]+", re.I)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[^\s,;|)]+", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)[^\s,;|)]+|(?:linkedin|github)\.com/[^\s,;|)]+", re.I)

DEGREE_CI = re.compile(
    r"\b(?:b\.?\s?tech|m\.?\s?tech|b\.?\s?sc|m\.?\s?sc|b\.?\s?com|m\.?\s?com|bca|mca|bba|mba|ph\.?\s?d"
    r"|bachelor[s']?(?:\s+of\s+[a-z]+)?|master[s']?(?:\s+of\s+[a-z]+)?|diploma|high\s+school"
    r"|(?:senior\s+|higher\s+)?secondary|class\s*(?:x|xii|10|12)|12th|10th|hsc|ssc)(?!\w)", re.I)
DEGREE_CS = re.compile(r"\b(?:B\.?E|M\.?E|B\.?A|M\.?A)(?![\w])\.?")
INSTITUTE_RE = re.compile(r"\b(?:university|institute|college|school|iit|nit|academy|polytechnic)\b", re.I)
ROLE_RE = re.compile(
    r"\b(?:intern|engineer|developer|analyst|scientist|manager|lead|consultant|associate|trainee|researcher"
    r"|architect|administrator|specialist|assistant|executive|director|officer|designer|programmer|tester"
    r"|founder|freelanc\w*)\b", re.I)
SPLIT_RE = re.compile(r"\s*[|–—]\s*|\s-\s")
SPLIT_AT_RE = re.compile(r"\s*[|–—]\s*|\s-\s|\s+(?:at|@)\s+", re.I)


def has_degree(line: str) -> bool:
    return bool(DEGREE_CI.search(line) or DEGREE_CS.search(line))


def is_phone_like(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    return 10 <= len(digits) <= 13 and not _YEAR_RANGE.search(s)
