from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SentenceRecord:
    event_id: str
    report_id: int | str
    sent_idx: int
    text: str
    link: str | None = None
    domain: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SampleRecord:
    event_id: str
    claim: str
    label: str
    explain: str
    reports: list[dict[str, Any]]


@dataclass(slots=True)
class CandidateSentence:
    event_id: str
    report_id: int | str
    sent_idx: int
    text: str
    dense_score: float
    lexical_score: float
    hybrid_score: float
    link: str | None = None
    domain: str | None = None
