"""Date-range parsing and robust experience arithmetic (overlap-safe)."""
from __future__ import annotations

import re
from datetime import date

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_PAT = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
             r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_YEAR = r"(?:19|20)\d{2}"
DATE_PAT = (rf"(?:(?<![A-Za-z]){MONTH_PAT}\s*[’']?\s*(?:{_YEAR}|\d{{2}})(?!\d)"
            rf"|\d{{1,2}}/{_YEAR}(?!\d)"
            rf"|{_YEAR}-(?:0[1-9]|1[0-2])(?!\d)"
            rf"|{_YEAR}(?!\d))")
PRESENT_PAT = r"(?:present|current|now|ongoing|till date|to date|today)"
RANGE_RE = re.compile(rf"({DATE_PAT})\s*(?:-|–|—|to|until)\s*({DATE_PAT}|{PRESENT_PAT})", re.I)


def _idx(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def parse_date(s: str | None, end: bool = False, today: date | None = None) -> int | None:
    """Return an absolute month index (year*12 + month-1) or None.
    Year-only values map to Jan (start) / Dec (end). Future end dates are capped at today."""
    if not s:
        return None
    today = today or date.today()
    now = _idx(today.year, today.month)
    s = s.strip().lower().rstrip(".")
    if re.fullmatch(PRESENT_PAT, s):
        return now
    if m := re.fullmatch(rf"({_YEAR})-(\d{{2}})", s):
        return _idx(int(m[1]), int(m[2]))
    if m := re.fullmatch(rf"(\d{{1,2}})/({_YEAR})", s):
        return _idx(int(m[2]), int(m[1]))
    if m := re.fullmatch(rf"({MONTH_PAT})\s*[’']?\s*(\d{{2,4}})", s):
        year = int(m[2])
        year += 2000 if year < 100 else 0
        return _idx(year, _MONTHS.index(m[1][:3]) + 1)
    if re.fullmatch(_YEAR, s):
        v = _idx(int(s), 12 if end else 1)
        return min(v, now) if end else v
    return None


def month_range(start: str | None, end: str | None, today: date | None = None):
    a, b = parse_date(start, False, today), parse_date(end, True, today)
    if a is None or b is None or b < a:
        return None
    return a, b


def total_months(ranges: list[tuple[int, int]]) -> int:
    """Union of inclusive month ranges - concurrent roles are not double counted."""
    total, cur = 0, None
    for a, b in sorted(ranges):
        if cur and a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            if cur:
                total += cur[1] - cur[0] + 1
            cur = [a, b]
    if cur:
        total += cur[1] - cur[0] + 1
    return total


def fill_durations(experiences, today: date | None = None) -> int:
    """Set duration_months on each experience and return the overlap-safe total."""
    ranges = []
    for ex in experiences:
        r = month_range(ex.start, ex.end, today)
        if r:
            ex.duration_months = r[1] - r[0] + 1
            ranges.append(r)
    return total_months(ranges)
