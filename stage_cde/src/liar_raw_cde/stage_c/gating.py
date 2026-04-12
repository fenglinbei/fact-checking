from __future__ import annotations

from dataclasses import dataclass

from liar_raw_cde.utils.text import count_complexity_signals


@dataclass
class GateDecision:
    should_split: bool
    reason: str
    features: dict[str, int]


class ComplexityGate:
    def __init__(
        self,
        min_claim_tokens_to_split: int = 14,
    ) -> None:
        self.min_claim_tokens_to_split = min_claim_tokens_to_split

    def decide(self, claim: str) -> GateDecision:
        feats = count_complexity_signals(claim)

        if feats["num_tokens"] < self.min_claim_tokens_to_split:
            return GateDecision(False, "too_short", feats)

        clause_like = feats["connector_count"] + (1 if feats["punctuation_count"] >= 2 else 0)
        logic_signal = feats["comparison_count"] + feats["time_count"] + (1 if feats["digit_count"] > 0 else 0)

        if clause_like <= 0 and logic_signal <= 0:
            return GateDecision(False, "too_simple", feats)

        return GateDecision(True, "complex_enough", feats)
