from __future__ import annotations

import torch

from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class CausalLMSFTCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of is not None:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for x in features:
            pad_len = max_len - len(x["input_ids"])

            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }