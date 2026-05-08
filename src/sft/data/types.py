from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptPreparationRecord:
    prompt_length_before_trunc: int
    prompt_length_after_trunc: int
    target_length: int
    prompt_token_budget: int
    sequence_length_before_trunc: int
    sequence_length_after_trunc: int
    evidence_count_before_trunc: int
    evidence_count_after_trunc: int
    was_truncated: bool
    overflow_before_trunc: bool
    overflow_after_trunc: bool
    claim_token_count: int
    max_report_char_count: int
    no_evidence: bool
    long_claim: bool
    duplicate_evidence: bool
    long_report: bool


@dataclass
class PreparedSample:
    prompt: str
    target: str
    prompt_add_special_tokens: bool
    preserve_prompt_prefix: bool
    gold_id: int
    gold_label: str
    gold_explain: str
    prompt_length_before_trunc: int
    prompt_length_after_trunc: int
    target_length: int
    prompt_token_budget: int
    sequence_length_before_trunc: int
    sequence_length_after_trunc: int
    evidence_count_before_trunc: int
    evidence_count_after_trunc: int
    was_truncated: bool
    overflow_before_trunc: bool
    overflow_after_trunc: bool
    claim_token_count: int
    max_report_char_count: int
    no_evidence: bool
    long_claim: bool
    duplicate_evidence: bool
    long_report: bool
