from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from liar_raw.utils.io import read_jsonl


class GraphDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.items: list[dict[str, Any]] = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]

    def label_ids(self) -> list[int]:
        return [int(x["label_id"]) for x in self.items]
