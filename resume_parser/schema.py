"""Pydantic output schema. Lenient on input (LLM output), strict on output."""
from __future__ import annotations

from typing import Annotated, Any, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _str(v):
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _list(v):
    return [] if v is None else v


def _strlist(v):
    return [str(x).strip() for x in (v or []) if x is not None and str(x).strip()]


def _strdict(v):
    return {str(k): str(x) for k, x in (v or {}).items() if x}


OptStr = Annotated[Optional[str], BeforeValidator(_str)]
StrList = Annotated[list[str], BeforeValidator(_strlist)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Education(_Base):
    degree: OptStr = None
    field_of_study: OptStr = None
    institute: OptStr = None
    start_year: OptStr = None
    year: OptStr = None  # graduation / end year
    score: OptStr = None


class Experience(_Base):
    company: OptStr = None
    role: OptStr = None
    start: OptStr = None
    end: OptStr = None
    duration_months: Optional[int] = None  # computed in code, never trusted from the LLM
    highlights: StrList = Field(default_factory=list)


class Project(_Base):
    name: OptStr = None
    tech: StrList = Field(default_factory=list)
    description: OptStr = None
    context: OptStr = None  # event / award / course the project belongs to


class Patent(_Base):
    title: OptStr = None
    application_number: OptStr = None
    filed: OptStr = None
    published: OptStr = None
    status: OptStr = None


class ParsedResume(_Base):
    name: OptStr = None
    email: OptStr = None
    phone: OptStr = None
    location: OptStr = None
    summary: OptStr = None
    links: Annotated[dict[str, str], BeforeValidator(_strdict)] = Field(default_factory=dict)
    education: Annotated[list[Education], BeforeValidator(_list)] = Field(default_factory=list)
    experience: Annotated[list[Experience], BeforeValidator(_list)] = Field(default_factory=list)
    skills: StrList = Field(default_factory=list)
    projects: Annotated[list[Project], BeforeValidator(_list)] = Field(default_factory=list)
    certifications: StrList = Field(default_factory=list)
    achievements: StrList = Field(default_factory=list)
    patents: Annotated[list[Patent], BeforeValidator(_list)] = Field(default_factory=list)
    languages: StrList = Field(default_factory=list)
    total_experience_months: int = 0
    # --- provenance & quality ---
    confidence: dict[str, float] = Field(default_factory=dict)
    skill_confidence: dict[str, float] = Field(default_factory=dict)
    sources: dict[str, Optional[str]] = Field(default_factory=dict)
    review_flags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)