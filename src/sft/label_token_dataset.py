from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from fact_checking.data.constants import LABEL2ID
from sft.data.types import PreparedSample


class LabelTokenDataset(Dataset):
    def __init__(
        self,
        samples: list[PreparedSample],
        tokenizer: AutoTokenizer,
        *,
        max_length: int,
        label_prefix: str,
    ) -> None:
        self.samples = samples
        self.tokenized: list[dict[str, Any]] = []
        self.lengths: list[int] = []

        prefix_ids = tokenizer(label_prefix, add_special_tokens=False, truncation=False)["input_ids"]
        if not prefix_ids:
            raise ValueError(f"label_prefix={label_prefix!r} produced no tokens.")

        for sample_idx, sample in enumerate(samples):
            prompt_text = sample.prompt.rstrip()
            if sample.prompt_add_special_tokens:
                prompt_text += " "
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=sample.prompt_add_special_tokens,
                truncation=False,
            )["input_ids"]
            input_ids = list(prompt_ids) + list(prefix_ids)

            if len(input_ids) > max_length:
                if sample.preserve_prompt_prefix:
                    raise ValueError(
                        "Protected label-token CE prompt is longer than max_length after appending label_prefix. "
                        "Increase sft_train.max_length or reduce evidence/context length."
                    )
                input_ids = input_ids[-max_length:]

            attention_mask = [1] * len(input_ids)
            gold_id = int(sample.gold_id)
            if gold_id < 0:
                gold_id = LABEL2ID.get(sample.gold_label, -1)
            if gold_id < 0:
                raise ValueError(f"Invalid gold label for sample {sample_idx}: {sample.gold_label!r}")

            row = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "gold_id": gold_id,
                "sample_idx": sample_idx,
                "prompt": sample.prompt,
                "target": sample.target,
                "gold_label": sample.gold_label,
                "gold_explain": sample.gold_explain,
            }
            self.tokenized.append(row)
            self.lengths.append(len(input_ids))

    def __len__(self) -> int:
        return len(self.tokenized)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.tokenized[idx]


@dataclass
class LabelTokenCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[dict[str, Any]]]:
        max_len = max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be set before building label-token batches.")

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        gold_ids: list[int] = []
        sample_indices: list[int] = []
        metadata: list[dict[str, Any]] = []

        for row in features:
            pad_len = max_len - len(row["input_ids"])
            input_ids.append(list(row["input_ids"]) + [pad_id] * pad_len)
            attention_mask.append(list(row["attention_mask"]) + [0] * pad_len)
            gold_ids.append(int(row["gold_id"]))
            sample_indices.append(int(row["sample_idx"]))
            metadata.append(
                {
                    "sample_idx": int(row["sample_idx"]),
                    "prompt": str(row["prompt"]),
                    "target": str(row["target"]),
                    "gold_label": str(row["gold_label"]),
                    "gold_explain": str(row["gold_explain"]),
                }
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "gold_ids": torch.tensor(gold_ids, dtype=torch.long),
            "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
            "metadata": metadata,
        }
