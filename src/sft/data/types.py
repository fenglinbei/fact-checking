from dataclasses import dataclass

@dataclass
class PreparedSample:
    prompt: str
    target: str
    gold_id: int
    prompt_length_before_trunc: int
    prompt_length_after_trunc: int
    evidence_count_before_trunc: int
    evidence_count_after_trunc: int
    was_truncated: bool
    overflow_before_trunc: bool
    overflow_after_trunc: bool