from __future__ import annotations

import torch

from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler

from sft.dataset.collators import CausalLMSFTCollator
from sft.dataset.datasets import EvalPromptDataset
from sft.runtime.model_loading import is_mistral_common_tokenizer


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
        prompt_input_ids = [x.get("prompt_input_ids") for x in items]
        prompt_add_special_tokens = bool(items[0].get("prompt_add_special_tokens", True))
        preserve_prompt_prefix = bool(items[0].get("preserve_prompt_prefix", False))
        if any(bool(x.get("prompt_add_special_tokens", True)) != prompt_add_special_tokens for x in items):
            raise ValueError("Mixed prompt_add_special_tokens values are not supported in one eval batch.")
        if any(bool(x.get("preserve_prompt_prefix", False)) != preserve_prompt_prefix for x in items):
            raise ValueError("Mixed preserve_prompt_prefix values are not supported in one eval batch.")

        gold_ids = torch.tensor([int(x["gold_id"]) for x in items], dtype=torch.long)
        sample_indices = torch.tensor([int(x["sample_idx"]) for x in items], dtype=torch.long)

        if any(ids is not None for ids in prompt_input_ids):
            if not all(isinstance(ids, list) for ids in prompt_input_ids):
                raise ValueError("Mixed pre-tokenized and string prompts are not supported in one eval batch.")
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                raise ValueError("tokenizer.pad_token_id must be set before building eval batches.")

            normalized_ids = [[int(token_id) for token_id in ids] for ids in prompt_input_ids]
            if preserve_prompt_prefix and any(len(ids) > max_length for ids in normalized_ids):
                raise ValueError(
                    "Protected eval prompt is longer than max_length after evidence truncation. "
                    "Increase sft_train.max_length or reduce evidence/context length."
                )
            if not preserve_prompt_prefix:
                normalized_ids = [ids[-max_length:] for ids in normalized_ids]

            target_len = max_length if padding == "max_length" else max(len(ids) for ids in normalized_ids)
            input_ids: list[list[int]] = []
            attention_mask: list[list[int]] = []
            for ids in normalized_ids:
                pad_len = max(0, target_len - len(ids))
                input_ids.append([pad_id] * pad_len + ids)
                attention_mask.append([0] * pad_len + [1] * len(ids))

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "gold_ids": gold_ids,
                "sample_indices": sample_indices,
            }

        if is_mistral_common_tokenizer(tokenizer):
            raise ValueError(
                "MistralCommon tokenizers require build rows with prompt_input_ids. "
                "Rebuild this run with FORCE_BUILD=true so chat prompts are stored from "
                "apply_chat_template(tokenize=True)."
            )

        old_padding_side = tokenizer.padding_side
        old_truncation_side = getattr(tokenizer, "truncation_side", "right")

        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        try:
            enc = tokenizer(
                prompts,
                padding="max_length" if padding == "max_length" else True,
                truncation=not preserve_prompt_prefix,
                max_length=max_length,
                add_special_tokens=prompt_add_special_tokens,
                return_tensors="pt",
            )
            if preserve_prompt_prefix and enc["input_ids"].shape[1] > max_length:
                raise ValueError(
                    "Protected eval prompt is longer than max_length after evidence truncation. "
                    "Increase sft_train.max_length or reduce evidence/context length."
                )
        finally:
            tokenizer.padding_side = old_padding_side
            tokenizer.truncation_side = old_truncation_side

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "gold_ids": gold_ids,
            "sample_indices": sample_indices,
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
