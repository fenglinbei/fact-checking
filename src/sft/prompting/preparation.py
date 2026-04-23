from __future__ import annotations

from transformers import AutoTokenizer

from fact_checking.data.constants import LABEL2ID
from sft.data.labels import normalize_gold_label
from sft.data.types import PreparedSample, PromptPreparationRecord
from sft.prompting.output import OutputStrategy
from sft.prompting.truncation import PromptTruncationStrategy
from sft.prompting.utils import build_evidence_block, build_evidence_items

LONG_CLAIM_TOKEN_THRESHOLD = 64
LONG_REPORT_CHAR_THRESHOLD = 8000


def build_prepared_samples(
    rows: list[dict],
    top_k: int,
    use_context: bool,
    context_k: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    output_strategy: OutputStrategy,
    truncation_strategy: PromptTruncationStrategy,
) -> tuple[list[PreparedSample], list[PromptPreparationRecord]]:
    samples: list[PreparedSample] = []
    records: list[PromptPreparationRecord] = []

    for row in rows:
        gold_label = normalize_gold_label(row)
        if not gold_label:
            continue

        claim = str(row.get("claim", ""))
        target = output_strategy.build_target(row, gold_label)
        evidence_items = build_evidence_items(
            row,
            top_k=top_k,
            use_context=use_context,
            context_k=context_k,
        )
        evidence_block = build_evidence_block(
            row,
            top_k=top_k,
            use_context=use_context,
            context_k=context_k,
        )
        claim_token_count = _count_tokens(claim, tokenizer)
        max_report_char_count = _max_source_report_chars(row, top_k=top_k)
        no_evidence = len(evidence_items) == 0 or not evidence_block.strip()
        long_claim = claim_token_count >= LONG_CLAIM_TOKEN_THRESHOLD
        duplicate_evidence = _has_duplicate_evidence(evidence_items)
        context_mode = bool(use_context)
        long_report = max_report_char_count >= LONG_REPORT_CHAR_THRESHOLD
        prompt_builder = lambda candidate_claim, candidate_evidence: output_strategy.build_prompt(
            candidate_claim,
            candidate_evidence,
            tokenizer=tokenizer,
        )
        truncation_result = truncation_strategy.apply(
            claim=claim,
            evidence_block=evidence_block,
            tokenizer=tokenizer,
            max_length=max_length,
            prompt_builder=prompt_builder,
            target_text=target,
            prompt_add_special_tokens=output_strategy.prompt_add_special_tokens,
        )

        samples.append(
            PreparedSample(
                prompt=truncation_result.prompt,
                target=target,
                prompt_add_special_tokens=output_strategy.prompt_add_special_tokens,
                preserve_prompt_prefix=output_strategy.preserve_prompt_prefix,
                gold_id=LABEL2ID[gold_label],
                gold_label=gold_label,
                gold_explain=str(row.get("explain", "")).strip(),
                prompt_length_before_trunc=truncation_result.prompt_length_before_trunc,
                prompt_length_after_trunc=truncation_result.prompt_length_after_trunc,
                target_length=truncation_result.target_length,
                prompt_token_budget=truncation_result.prompt_token_budget,
                sequence_length_before_trunc=truncation_result.sequence_length_before_trunc,
                sequence_length_after_trunc=truncation_result.sequence_length_after_trunc,
                evidence_count_before_trunc=truncation_result.evidence_count_before_trunc,
                evidence_count_after_trunc=truncation_result.evidence_count_after_trunc,
                was_truncated=truncation_result.was_truncated,
                overflow_before_trunc=truncation_result.overflow_before_trunc,
                overflow_after_trunc=truncation_result.overflow_after_trunc,
                claim_token_count=claim_token_count,
                max_report_char_count=max_report_char_count,
                no_evidence=no_evidence,
                long_claim=long_claim,
                duplicate_evidence=duplicate_evidence,
                context_mode=context_mode,
                long_report=long_report,
            )
        )
        records.append(
            PromptPreparationRecord(
                prompt_length_before_trunc=truncation_result.prompt_length_before_trunc,
                prompt_length_after_trunc=truncation_result.prompt_length_after_trunc,
                target_length=truncation_result.target_length,
                prompt_token_budget=truncation_result.prompt_token_budget,
                sequence_length_before_trunc=truncation_result.sequence_length_before_trunc,
                sequence_length_after_trunc=truncation_result.sequence_length_after_trunc,
                evidence_count_before_trunc=truncation_result.evidence_count_before_trunc,
                evidence_count_after_trunc=truncation_result.evidence_count_after_trunc,
                was_truncated=truncation_result.was_truncated,
                overflow_before_trunc=truncation_result.overflow_before_trunc,
                overflow_after_trunc=truncation_result.overflow_after_trunc,
                claim_token_count=claim_token_count,
                max_report_char_count=max_report_char_count,
                no_evidence=no_evidence,
                long_claim=long_claim,
                duplicate_evidence=duplicate_evidence,
                context_mode=context_mode,
                long_report=long_report,
            )
        )

    return samples, records


def _count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    return len(
        tokenizer(
            text,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
    )


def _has_duplicate_evidence(evidence_items: list[str]) -> bool:
    normalized = [" ".join(item.lower().split()) for item in evidence_items if item.strip()]
    return len(normalized) != len(set(normalized))


def _max_source_report_chars(row: dict, top_k: int) -> int:
    candidates = sorted(
        row.get("candidates", []),
        key=lambda x: float(x.get("hybrid_score", 0.0)),
        reverse=True,
    )[:top_k]
    max_chars = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        report = candidate.get("source_report", {})
        if not isinstance(report, dict):
            continue
        content = str(report.get("content", ""))
        if not content:
            continue
        max_chars = max(max_chars, len(content))
    return max_chars
