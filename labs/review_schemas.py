"""Shared schemas for the multi-agent code review pipeline (Session 45)."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    LOW = "LOW"


SEVERITY_RANK = {Severity.CRITICAL: 4, Severity.MAJOR: 3, Severity.MINOR: 2, Severity.LOW: 1}


class Finding(BaseModel):
    file: str
    line_start: int
    line_end: int
    severity: Severity
    category: Literal["security", "logic", "style", "docs"]
    title: str
    description: str
    suggestion: str
    confidence: float = Field(ge=0.0, le=1.0)


class FindingList(BaseModel):
    findings: list[Finding]
