from __future__ import annotations

import os

from sft.data.labels import normalize_gold_label as _normalize_gold_label
from sft.data.sampling import select_mini_val_rows as _select_mini_val_rows
from sft.prompting.stats import log_prompt_summary as _print_prompt_summary
from sft.runtime.config import (
    apply_runtime_output_layout as _apply_runtime_output_layout,
)
from sft.runtime.device import maybe_empty_cache

__all__ = [
    "_normalize_gold_label",
    "_print_prompt_summary",
    "_select_mini_val_rows",
    "_apply_runtime_output_layout",
    "maybe_empty_cache",
]


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
