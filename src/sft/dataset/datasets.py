from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from sft.data.types import PreparedSample
from sft.dataset.cache import build_cache_name
from sft.dataset.tokenization import tokenize_instances


class TokenizedDataset(Dataset):
    def __init__(self, tokenized: list[dict[str, list[int]]]) -> None:
        self.tokenized = tokenized
        self.lengths = [int(sum(x["attention_mask"])) for x in tokenized]

    def __len__(self) -> int:
        return len(self.tokenized)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.tokenized[idx]


@dataclass
class SFTDatasetBuilder:
    tokenizer: AutoTokenizer
    max_length: int
    padding: str = "max_length"
    cache_dir: Path | None = None

    def build(self, instances: list[dict[str, object]], split: str, accelerator: Accelerator) -> Dataset:
        if self.cache_dir is None:
            tokenized = tokenize_instances(
                instances=instances,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                padding=self.padding,
            )
            return TokenizedDataset(tokenized=tokenized)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / build_cache_name(
            split=split,
            max_length=self.max_length,
            tokenizer_name=self.tokenizer.name_or_path,
            instances=instances,
        )
        with accelerator.main_process_first():
            if not cache_path.exists():
                tokenized = tokenize_instances(
                    instances=instances,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    padding=self.padding,
                )
                torch.save(tokenized, cache_path)

        tokenized = torch.load(cache_path, map_location="cpu", weights_only=False)
        return TokenizedDataset(tokenized=tokenized)


class EvalPromptDataset(Dataset):
    def __init__(self, samples: list[PreparedSample]) -> None:
        self.samples: list[dict[str, str | int | bool]] = [
            {
                "sample_idx": idx,
                "prompt": sample.prompt,
                "target": sample.target,
                "prompt_add_special_tokens": sample.prompt_add_special_tokens,
                "preserve_prompt_prefix": sample.preserve_prompt_prefix,
                "gold_id": sample.gold_id,
                "gold_label": sample.gold_label,
                "gold_explain": sample.gold_explain,
                "label_schema": sample.label_schema,
            }
            for idx, sample in enumerate(samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str | int | bool]:
        return self.samples[idx]
