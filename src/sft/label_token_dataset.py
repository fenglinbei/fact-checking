from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from fact_checking.data.constants import COVERAGE_LABEL2ID, label2id_for_schema
from sft.data.types import PreparedSample
from sft.runtime.model_loading import is_mistral_common_tokenizer


class LabelTokenDataset(Dataset):
    def __init__(
        self,
        samples: list[PreparedSample],
        tokenizer: AutoTokenizer,
        *,
        max_length: int,
        label_prefix: str,
        label_schema: str | None = None,
        coverage_enabled: bool = False,
        coverage_label_prefix: str = "Coverage:",
        allow_unlabeled: bool = False,
    ) -> None:
        self.samples = samples
        self.tokenized: list[dict[str, Any]] = []
        self.lengths: list[int] = []

        prefix_ids = tokenizer(label_prefix, add_special_tokens=False, truncation=False)["input_ids"]
        if not prefix_ids:
            raise ValueError(f"label_prefix={label_prefix!r} produced no tokens.")
        coverage_prefix_ids = tokenizer(coverage_label_prefix, add_special_tokens=False, truncation=False)["input_ids"]
        if coverage_enabled and not coverage_prefix_ids:
            raise ValueError(f"coverage_label_prefix={coverage_label_prefix!r} produced no tokens.")
        requires_prompt_input_ids = is_mistral_common_tokenizer(tokenizer)

        for sample_idx, sample in enumerate(samples):
            if sample.prompt_input_ids is not None:
                prompt_ids = list(sample.prompt_input_ids)
            elif requires_prompt_input_ids:
                raise ValueError(
                    "MistralCommon tokenizers require build rows with prompt_input_ids. "
                    "Rebuild this run with FORCE_BUILD=true so chat prompts are stored from "
                    "apply_chat_template(tokenize=True)."
                )
            else:
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
                sample_schema = str(getattr(sample, "label_schema", "") or label_schema or "liar6")
                gold_id = label2id_for_schema(sample_schema).get(sample.gold_label, -1)
            if gold_id < 0 and not allow_unlabeled:
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
            if coverage_enabled:
                coverage_gold_id = COVERAGE_LABEL2ID.get(str(sample.coverage_label), -1)
                if coverage_gold_id < 0:
                    raise ValueError(
                        f"Invalid coverage_label for sample {sample_idx}: {sample.coverage_label!r}"
                    )
                coverage_input_ids = list(prompt_ids) + list(coverage_prefix_ids)
                if len(coverage_input_ids) > max_length:
                    if sample.preserve_prompt_prefix:
                        raise ValueError(
                            "Protected coverage label-token CE prompt is longer than max_length after appending "
                            "coverage_label_prefix. Increase sft_train.max_length or reduce evidence/context length."
                        )
                    coverage_input_ids = coverage_input_ids[-max_length:]
                row.update(
                    {
                        "coverage_input_ids": coverage_input_ids,
                        "coverage_attention_mask": [1] * len(coverage_input_ids),
                        "coverage_gold_id": coverage_gold_id,
                        "coverage_label": str(sample.coverage_label),
                    }
                )
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
        coverage_enabled = "coverage_input_ids" in features[0]
        coverage_input_ids: list[list[int]] = []
        coverage_attention_mask: list[list[int]] = []
        coverage_gold_ids: list[int] = []
        sample_indices: list[int] = []
        metadata: list[dict[str, Any]] = []
        max_coverage_len = 0
        if coverage_enabled:
            max_coverage_len = max(len(x["coverage_input_ids"]) for x in features)
            if self.pad_to_multiple_of is not None:
                multiple = self.pad_to_multiple_of
                max_coverage_len = ((max_coverage_len + multiple - 1) // multiple) * multiple

        for row in features:
            pad_len = max_len - len(row["input_ids"])
            input_ids.append(list(row["input_ids"]) + [pad_id] * pad_len)
            attention_mask.append(list(row["attention_mask"]) + [0] * pad_len)
            gold_ids.append(int(row["gold_id"]))
            if coverage_enabled:
                coverage_pad_len = max_coverage_len - len(row["coverage_input_ids"])
                coverage_input_ids.append(list(row["coverage_input_ids"]) + [pad_id] * coverage_pad_len)
                coverage_attention_mask.append(list(row["coverage_attention_mask"]) + [0] * coverage_pad_len)
                coverage_gold_ids.append(int(row["coverage_gold_id"]))
            sample_indices.append(int(row["sample_idx"]))
            item = {
                "sample_idx": int(row["sample_idx"]),
                "prompt": str(row["prompt"]),
                "target": str(row["target"]),
                "gold_label": str(row["gold_label"]),
                "gold_explain": str(row["gold_explain"]),
            }
            if coverage_enabled:
                item["coverage_label"] = str(row["coverage_label"])
            metadata.append(item)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "gold_ids": torch.tensor(gold_ids, dtype=torch.long),
            "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
            "metadata": metadata,
        }
        if coverage_enabled:
            batch.update(
                {
                    "coverage_input_ids": torch.tensor(coverage_input_ids, dtype=torch.long),
                    "coverage_attention_mask": torch.tensor(coverage_attention_mask, dtype=torch.long),
                    "coverage_gold_ids": torch.tensor(coverage_gold_ids, dtype=torch.long),
                }
            )
        return batch
