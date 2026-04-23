from __future__ import annotations

import torch

from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer
from sft.dataset.datasets import EvalPromptDataset
from sft.prompting.output import OutputStrategy
from sft.prompting.truncation import PromptTruncationStrategy
from sft.data.types import PreparedSample, PromptPreparationRecord
from sft.dataset.collators import CausalLMSFTCollator
from sft.utils import _normalize_gold_label
from sft.prompting.utils import build_evidence_block
from fact_checking.data.constants import LABEL2ID
from transformers.trainer_pt_utils import LengthGroupedSampler

def _tokenize_instances(
    instances: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    max_length: int,
    padding: str,
) -> list[dict[str, list[int]]]:
    tokenized: list[dict[str, list[int]]] = []
    eos_id = tokenizer.eos_token_id

    for row in instances:
        prompt_text = row["prompt"].rstrip() + " "
        target_text = row["target"].strip()

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if eos_id is not None:
            target_ids = target_ids + [eos_id]

        # 关键：为 target 预留空间，不能让 truncation 把 label 截掉
        max_prompt_len = max_length - len(target_ids)
        if max_prompt_len <= 0:
            input_ids = target_ids[:max_length]
            labels = input_ids[:]
        else:
            if len(prompt_ids) > max_prompt_len:
                # 保留 prompt 尾部，至少保住 "Label:"
                prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

        attention_mask = [1] * len(input_ids)

        if padding == "max_length":
            pad_len = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return tokenized

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
        gold_label = _normalize_gold_label(row)
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

def build_dataloader(
    dataset: Dataset,
    collator: CausalLMSFTCollator,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    use_length_bucket: bool,
) -> DataLoader:
    sampler: Sampler | None = None
    if use_length_bucket:
        lengths = getattr(dataset, "lengths", None)
        if lengths is not None:
            sampler = LengthGroupedSampler(
                batch_size=batch_size,
                dataset=dataset,
                lengths=lengths,
            )
    return DataLoader(
        dataset,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=shuffle,
    )

def build_eval_dataloader(
    dataset: EvalPromptDataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_workers: int,
    max_length: int,
    padding: str,
) -> DataLoader:
    def _collate_fn(items):
        prompts = [str(x["prompt"]) for x in items]
        gold_ids = torch.tensor([int(x["gold_id"]) for x in items], dtype=torch.long)

        old_padding_side = tokenizer.padding_side
        old_truncation_side = getattr(tokenizer, "truncation_side", "right")

        # decoder-only generation 必须 left padding
        tokenizer.padding_side = "left"

        # 如果 prompt 过长，至少保住末尾的 "Label:"
        tokenizer.truncation_side = "left"

        try:
            enc = tokenizer(
                prompts,
                padding="max_length" if padding == "max_length" else True,
                truncation=True,
                max_length=max_length,
                # 关键：batch_size=1 时建议先去掉这个；
                # 如果你坚持保留，也必须配合 left padding。
                # pad_to_multiple_of=8,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = old_padding_side
            tokenizer.truncation_side = old_truncation_side

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "gold_ids": gold_ids,
        }

    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )