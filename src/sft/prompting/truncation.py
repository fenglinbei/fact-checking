from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from transformers import AutoTokenizer

PromptBuilder = Callable[[str, str], str]


@dataclass
class PromptTruncationResult:
    prompt: str
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


class PromptTruncationStrategy:
    name = "none"

    def apply(
        self,
        claim: str,
        evidence_block: str,
        tokenizer: AutoTokenizer,
        max_length: int,
        prompt_builder: PromptBuilder,
        target_text: str = "",
        prompt_add_special_tokens: bool = True,
    ) -> PromptTruncationResult:
        prompt = prompt_builder(claim, evidence_block)
        plen = _count_text_tokens(prompt, tokenizer, add_special_tokens=prompt_add_special_tokens)
        target_len = _count_target_tokens(target_text, tokenizer)
        prompt_budget = max(0, int(max_length) - target_len)
        ecnt = len(_split_evidence_block(evidence_block))
        overflow = plen > prompt_budget
        return PromptTruncationResult(
            prompt=prompt,
            prompt_length_before_trunc=plen,
            prompt_length_after_trunc=plen,
            target_length=target_len,
            prompt_token_budget=prompt_budget,
            sequence_length_before_trunc=plen + target_len,
            sequence_length_after_trunc=plen + target_len,
            evidence_count_before_trunc=ecnt,
            evidence_count_after_trunc=ecnt,
            was_truncated=False,
            overflow_before_trunc=overflow,
            overflow_after_trunc=overflow,
        )


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
        prompt_builder: PromptBuilder,
        target_text: str = "",
        prompt_add_special_tokens: bool = True,
    ) -> PromptTruncationResult:
        evidences = _split_evidence_block(evidence_block)
        original_count = len(evidences)
        target_length = _count_target_tokens(target_text, tokenizer)
        prompt_budget = max(0, int(max_length) - target_length)

        prompt_before = prompt_builder(claim, evidence_block)
        prompt_length_before = _count_text_tokens(
            prompt_before,
            tokenizer,
            add_special_tokens=prompt_add_special_tokens,
        )
        overflow_before = prompt_length_before > prompt_budget

        if not overflow_before or not evidences:
            return PromptTruncationResult(
                prompt=prompt_before,
                prompt_length_before_trunc=prompt_length_before,
                prompt_length_after_trunc=prompt_length_before,
                target_length=target_length,
                prompt_token_budget=prompt_budget,
                sequence_length_before_trunc=prompt_length_before + target_length,
                sequence_length_after_trunc=prompt_length_before + target_length,
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
            candidate_length = _count_text_tokens(
                candidate_prompt,
                tokenizer,
                add_special_tokens=prompt_add_special_tokens,
            )
            if candidate_length <= prompt_budget:
                return PromptTruncationResult(
                    prompt=candidate_prompt,
                    prompt_length_before_trunc=prompt_length_before,
                    prompt_length_after_trunc=candidate_length,
                    target_length=target_length,
                    prompt_token_budget=prompt_budget,
                    sequence_length_before_trunc=prompt_length_before + target_length,
                    sequence_length_after_trunc=candidate_length + target_length,
                    evidence_count_before_trunc=original_count,
                    evidence_count_after_trunc=len(kept),
                    was_truncated=True,
                    overflow_before_trunc=overflow_before,
                    overflow_after_trunc=False,
                )

        final_block = _join_evidence_block(kept, evidence_block)
        final_block = _trim_evidence_text_to_budget(
            claim=claim,
            evidence_block=final_block,
            prompt_builder=prompt_builder,
            tokenizer=tokenizer,
            prompt_budget=prompt_budget,
            add_special_tokens=prompt_add_special_tokens,
        )
        final_prompt = prompt_builder(claim, final_block)
        final_length = _count_text_tokens(
            final_prompt,
            tokenizer,
            add_special_tokens=prompt_add_special_tokens,
        )
        return PromptTruncationResult(
            prompt=final_prompt,
            prompt_length_before_trunc=prompt_length_before,
            prompt_length_after_trunc=final_length,
            target_length=target_length,
            prompt_token_budget=prompt_budget,
            sequence_length_before_trunc=prompt_length_before + target_length,
            sequence_length_after_trunc=final_length + target_length,
            evidence_count_before_trunc=original_count,
            evidence_count_after_trunc=len(_split_evidence_block(final_block)),
            was_truncated=(final_block != evidence_block),
            overflow_before_trunc=overflow_before,
            overflow_after_trunc=final_length > prompt_budget,
        )


def _count_text_tokens(text: str, tokenizer: AutoTokenizer, add_special_tokens: bool = True) -> int:
    return len(
        tokenizer(
            text,
            truncation=False,
            add_special_tokens=add_special_tokens,
        )["input_ids"]
    )


def _count_target_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    target_ids = tokenizer(
        text.strip(),
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    return len(target_ids)


def _trim_evidence_text_to_budget(
    *,
    claim: str,
    evidence_block: str,
    prompt_builder: PromptBuilder,
    tokenizer: AutoTokenizer,
    prompt_budget: int,
    add_special_tokens: bool,
) -> str:
    if not evidence_block:
        return evidence_block

    empty_prompt = prompt_builder(claim, "")
    empty_length = _count_text_tokens(
        empty_prompt,
        tokenizer,
        add_special_tokens=add_special_tokens,
    )
    if empty_length > prompt_budget:
        return ""

    low = 0
    high = len(evidence_block)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = evidence_block[:mid].rstrip()
        candidate_prompt = prompt_builder(claim, candidate)
        candidate_length = _count_text_tokens(
            candidate_prompt,
            tokenizer,
            add_special_tokens=add_special_tokens,
        )
        if candidate_length <= prompt_budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1

    return best


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


def build_prompt_truncation_strategy(baseline_cfg: dict) -> PromptTruncationStrategy:
    trunc_cfg = baseline_cfg.get("prompt_truncation", {}) or {}
    if not bool(trunc_cfg.get("enabled", False)):
        return PromptTruncationStrategy()

    strategy_name = str(trunc_cfg.get("strategy", "tail_evidence")).strip().lower()
    if strategy_name == "tail_evidence":
        return TailEvidenceTruncationStrategy(
            min_evidence_to_keep=int(trunc_cfg.get("min_evidence_to_keep", 1))
        )

    raise ValueError(
        f"Unsupported baseline.prompt_truncation.strategy={strategy_name}. "
        "Use 'tail_evidence'."
    )
