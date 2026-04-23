from __future__ import annotations

import torch
from accelerate import Accelerator


def enable_tf32_if_available() -> None:
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def maybe_empty_cache(accelerator: Accelerator) -> None:
    if torch.cuda.is_available():
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
