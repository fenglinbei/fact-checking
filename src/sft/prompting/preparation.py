from __future__ import annotations

from transformers import AutoTokenizer

from fact_checking.data.constants import LABEL2ID
from sft.data.labels import normalize_gold_label
from sft.data.types import PreparedSample, PromptPreparationRecord
from sft.prompting.output import OutputStrategy
from sft.prompting.truncation import PromptTruncationStrategy
from sft.prompting.utils import build_evidence_block


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
        evidence_block = build_evidence_block(
            row,
            top_k=top_k,
            use_context=use_context,
            context_k=context_k,
        )
        truncation_result = truncation_strategy.apply(
            claim=claim,
            evidence_block=evidence_block,
            tokenizer=tokenizer,
            max_length=max_length,
            prompt_builder=output_strategy.build_prompt,
        )

        samples.append(
            PreparedSample(
                prompt=truncation_result.prompt,
                target=output_strategy.build_target(row, gold_label),
                gold_id=LABEL2ID[gold_label],
                gold_label=gold_label,
                gold_explain=str(row.get("explain", "")).strip(),
                prompt_length_before_trunc=truncation_result.prompt_length_before_trunc,
                prompt_length_after_trunc=truncation_result.prompt_length_after_trunc,
                evidence_count_before_trunc=truncation_result.evidence_count_before_trunc,
                evidence_count_after_trunc=truncation_result.evidence_count_after_trunc,
                was_truncated=truncation_result.was_truncated,
                overflow_before_trunc=truncation_result.overflow_before_trunc,
                overflow_after_trunc=truncation_result.overflow_after_trunc,
            )
        )
        records.append(
            PromptPreparationRecord(
                prompt_length_before_trunc=truncation_result.prompt_length_before_trunc,
                prompt_length_after_trunc=truncation_result.prompt_length_after_trunc,
                evidence_count_before_trunc=truncation_result.evidence_count_before_trunc,
                evidence_count_after_trunc=truncation_result.evidence_count_after_trunc,
                was_truncated=truncation_result.was_truncated,
                overflow_before_trunc=truncation_result.overflow_before_trunc,
                overflow_after_trunc=truncation_result.overflow_after_trunc,
            )
        )

    return samples, records
