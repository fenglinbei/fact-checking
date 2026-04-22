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

    def __init__(self, min_evidence_to_keep: int = 1) -> None:
        self.min_evidence_to_keep = max(0, int(min_evidence_to_keep))

    def apply(
        self,
        claim: str,
        evidence_block: str,
        tokenizer: AutoTokenizer,
        max_length: int,
        prompt_builder: Callable[[str, str], str],
    ) -> PromptTruncationResult:
        evidences = _split_evidence_block(evidence_block)
        original_count = len(evidences)

        prompt_before = prompt_builder(claim, evidence_block)
        prompt_length_before = _count_text_tokens(prompt_before, tokenizer)
        overflow_before = prompt_length_before > int(max_length)

        if not overflow_before or not evidences:
            return PromptTruncationResult(
                prompt=prompt_before,
                prompt_length_before_trunc=prompt_length_before,
                prompt_length_after_trunc=prompt_length_before,
                evidence_count_before_trunc=original_count,
                evidence_count_after_trunc=original_count,
                was_truncated=False,
                overflow_before_trunc=overflow_before,
                overflow_after_trunc=overflow_before,
            )

        kept = list(evidences)
        min_keep = min(self.min_evidence_to_keep, len(kept))
        while len(kept) > min_keep:
            kept.pop()
            candidate_block = _join_evidence_block(kept, evidence_block)
            candidate_prompt = prompt_builder(claim, candidate_block)
            candidate_length = _count_text_tokens(candidate_prompt, tokenizer)
            if candidate_length <= int(max_length):
                return PromptTruncationResult(
                    prompt=candidate_prompt,
                    prompt_length_before_trunc=prompt_length_before,
                    prompt_length_after_trunc=candidate_length,
                    evidence_count_before_trunc=original_count,
                    evidence_count_after_trunc=len(kept),
                    was_truncated=True,
                    overflow_before_trunc=overflow_before,
                    overflow_after_trunc=False,
                )

        final_block = _join_evidence_block(kept, evidence_block)
        final_prompt = prompt_builder(claim, final_block)
        final_length = _count_text_tokens(final_prompt, tokenizer)
        return PromptTruncationResult(
            prompt=final_prompt,
            prompt_length_before_trunc=prompt_length_before,
            prompt_length_after_trunc=final_length,
            evidence_count_before_trunc=original_count,
            evidence_count_after_trunc=len(kept),
            was_truncated=(len(kept) != original_count),
            overflow_before_trunc=overflow_before,
            overflow_after_trunc=final_length > int(max_length),
        )


def _count_text_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])


def _split_evidence_block(evidence_block: str) -> list[str]:
    normalized = evidence_block.strip()
    if not normalized:
        return []

    pattern = re.compile(
        r"(?m)(?=^\s*(?:\[\d+\]|\(\d+\)|\d+[.)]|evidence\s+\d+\s*:|doc\s+\d+\s*:|report\s+\d+\s*:))",
        flags=re.IGNORECASE,
    )
    chunks = [chunk.strip() for chunk in pattern.split(normalized) if chunk.strip()]
    if len(chunks) >= 2:
        return chunks

    blank_line_chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", normalized) if chunk.strip()]
    if len(blank_line_chunks) >= 2:
        return blank_line_chunks

    bullet_pattern = re.compile(r"(?m)(?=^\s*-\s+)")
    bullet_chunks = [chunk.strip() for chunk in bullet_pattern.split(normalized) if chunk.strip()]
    if len(bullet_chunks) >= 2:
        return bullet_chunks

    return [normalized]


def _join_evidence_block(evidences: list[str], reference_block: str) -> str:
    if not evidences:
        return ""

    separator = "\n\n" if "\n\n" in reference_block else "\n"
    return separator.join(evidences)
