from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SelectorCandidateGroup:
    claim: str
    candidates: list[dict[str, Any]]
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SelectorPrediction:
    ordered_indices: list[int]
    scores: list[float] = field(default_factory=list)
    step_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceSelector(Protocol):
    selector_type: str
    metadata: dict[str, Any]

    def select(
        self,
        claim: str,
        candidates: list[dict[str, Any]],
        candidate_scores: list[dict[str, Any]] | None = None,
        *,
        top_k: int,
    ) -> SelectorPrediction:
        ...


SelectorFactory = type[EvidenceSelector]
_SELECTOR_REGISTRY: dict[str, SelectorFactory] = {}


def register_selector_type(name: str, selector_cls: SelectorFactory) -> None:
    key = normalize_selector_type(name)
    if not key:
        raise ValueError("selector type name must be non-empty.")
    _SELECTOR_REGISTRY[key] = selector_cls


def get_selector_type(name: str) -> SelectorFactory:
    key = normalize_selector_type(name)
    try:
        return _SELECTOR_REGISTRY[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_SELECTOR_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown selector type {name!r}; registered choices: {choices}") from exc


def normalize_selector_type(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")
