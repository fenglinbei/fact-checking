from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreparedSample:
    prompt: str
    target: str
    prompt_add_special_tokens: bool
    preserve_prompt_prefix: bool
    gold_id: int
    gold_label: str
    gold_explain: str
    prompt_token_count: int = 0
    target_token_count: int = 0
    evidence_count: int = 0
    was_truncated: bool = False
    claim: str = ""
    no_evidence: bool = False
    long_claim: bool = False
    label_schema: str = "liar6"
    prompt_input_ids: list[int] | None = None
