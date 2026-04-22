from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from transformers import AutoTokenizer


@dataclass
class PromptTruncationResult:
    prompt: str
    prompt_length_before_trunc: int
    prompt_length_after_trunc: int
    evidence_count_before_trunc: int
    evidence_count_after_trunc: int
    was_truncated: bool
    overflow_before_trunc: bool
    overflow_after_trunc: bool


class PromptTruncationStrategy:
    name = "none"

    def apply(self, claim: str, evidence_block: str, tokenizer: AutoTokenizer, max_length: int, prompt_builder: Callable[[str, str], str]) -> PromptTruncationResult:
        prompt = prompt_builder(claim, evidence_block)
        plen = len(tokenizer(prompt, truncation=False, add_special_tokens=True)["input_ids"])
        ecnt = len(_split_evidence_block(evidence_block))
        overflow = plen > int(max_length)
        return PromptTruncationResult(prompt, plen, plen, ecnt, ecnt, False, overflow, overflow)


class TailEvidenceTruncationStrategy(PromptTruncationStrategy):
    name = "tail_evidence"


def _split_evidence_block(evidence_block: str) -> list[str]:
    normalized = evidence_block.strip()
    if not normalized:
        return []
    pattern = re.compile(r"(?m)(?=^\s*(?:\[\d+\]|\(\d+\)|\d+[.)]|evidence\s+\d+\s*:|doc\s+\d+\s*:|report\s+\d+\s*:))", flags=re.IGNORECASE)
    chunks = [chunk.strip() for chunk in pattern.split(normalized) if chunk.strip()]
    return chunks if chunks else [normalized]
