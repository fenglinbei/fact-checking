import torch

from dataclasses import dataclass
from pathlib import Path
from accelerate import Accelerator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, Sampler

from sft.data.types import PreparedSample
from sft.prompting.output import OutputStrategy
from sft.prompting.stats import PromptPreparationRecord
from sft.prompting.truncation import PromptTruncationStrategy
from sft.utils import _normalize_gold_label
from fact_checking.data.constants import LABEL2ID
from sft.prompting.utils import build_evidence_block
from sft.data.io import _build_cache_name
from sft.dataset.utils import _tokenize_instances


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

    def build(self, instances: list[dict[str, str]], split: str, accelerator: Accelerator) -> Dataset:
        if self.cache_dir is None:
            tokenized = _tokenize_instances(
                instances=instances,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                padding=self.padding,
            )
            return TokenizedDataset(tokenized=tokenized)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / _build_cache_name(
            split=split,
            max_length=self.max_length,
            tokenizer_name=self.tokenizer.name_or_path,
            instances=instances,
        )
        with accelerator.main_process_first():
            if not cache_path.exists():
                tokenized = _tokenize_instances(
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
        self.samples: list[dict[str, str | int]] = [
            {"prompt": sample.prompt, "gold_id": sample.gold_id}
            for sample in samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str | int]:
        return self.samples[idx]