from __future__ import annotations

import numpy as np
from accelerate import Accelerator

from fact_checking.utils.logging import init_logger

logger = init_logger(__name__)


def select_mini_val_rows(
    rows: list[dict],
    mini_val_size: int,
    mini_val_seed: int,
    accelerator: Accelerator,
) -> list[dict]:
    if mini_val_size <= 0 or mini_val_size >= len(rows):
        return rows

    rng = np.random.default_rng(mini_val_seed)
    indices = rng.choice(len(rows), size=mini_val_size, replace=False)
    mini_rows = [rows[int(i)] for i in indices.tolist()]
    if accelerator.is_main_process:
        logger.info(
            "[INFO] mini-val enabled: sampled %d / %d validation rows (seed=%d).",
            len(mini_rows),
            len(rows),
            mini_val_seed,
        )
    return mini_rows
