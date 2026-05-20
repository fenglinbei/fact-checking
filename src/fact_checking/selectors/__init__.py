"""Evidence selector training and evaluation utilities."""

from fact_checking.selectors.base import (
    EvidenceSelector,
    SelectorCandidateGroup,
    SelectorPrediction,
    get_selector_type,
    register_selector_type,
)

__all__ = [
    "EvidenceSelector",
    "SelectorCandidateGroup",
    "SelectorPrediction",
    "get_selector_type",
    "register_selector_type",
]
