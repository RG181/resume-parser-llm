"""Skill taxonomy: alias -> canonical name, category lookup, and text scanning."""
from __future__ import annotations

import json
import re
from pathlib import Path


class SkillTaxonomy:
    def __init__(self, data: dict):
        self.categories: dict[str, str] = {}
        self._alias: dict[str, str] = {}
        self._patterns: list[tuple[str, re.Pattern]] = []
        for name, meta in data.items():
            self.categories[name] = meta.get("category", "other")
            case_sensitive = bool(meta.get("case_sensitive"))
            for form in {name, *meta.get("aliases", [])}:
                self._alias[form.lower()] = name
                flags = 0 if (case_sensitive or len(form) <= 2) else re.IGNORECASE
                # boundaries exclude [A-Za-z0-9+#] so "Java" != "JavaScript" and "C" != "C++"
                rx = rf"(?<![A-Za-z0-9+#]){re.escape(form)}(?![A-Za-z0-9+#])"
                self._patterns.append((name, re.compile(rx, flags)))

    @classmethod
    def from_file(cls, path: str | Path) -> "SkillTaxonomy":
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def find(self, text: str) -> list[str]:
        """Canonical skills found in text, ordered by first appearance."""
        first: dict[str, int] = {}
        for name, rx in self._patterns:
            m = rx.search(text)
            if m and (name not in first or m.start() < first[name]):
                first[name] = m.start()
        return [n for n, _ in sorted(first.items(), key=lambda kv: kv[1])]

    def normalize(self, skill: str) -> str:
        """Map an alias to its canonical name; unknown skills are kept as written."""
        skill = skill.strip()
        return self._alias.get(skill.lower(), skill)

    def is_known(self, skill: str) -> bool:
        return skill.strip().lower() in self._alias

    def category(self, skill: str) -> str:
        return self.categories.get(self.normalize(skill), "other")
