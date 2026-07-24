from __future__ import annotations

import os
import runpy

import torch


DEFAULT_TARGET_MODULE = "sft.label_token_trainer"
TARGET_MODULE_ENV = "SFT_HAMI_BOOTSTRAP_TARGET_MODULE"


def bootstrap_cuda_context() -> int:
    """Create each worker's CUDA context before importing the trainer stack.

    HAMi-core v2.8.0 can read an uninitialised device id when several workers
    enter its memory-limit hook before a CUDA context exists.  This explicit
    allocation keeps the workaround isolated to launchers that opt into this
    bootstrap module.
    """

    raw_local_rank = os.environ.get("LOCAL_RANK")
    if raw_local_rank is None:
        raise RuntimeError("LOCAL_RANK is required by the HAMi CUDA bootstrap.")
    try:
        local_rank = int(raw_local_rank)
    except ValueError as exc:
        raise RuntimeError(f"Invalid LOCAL_RANK={raw_local_rank!r}.") from exc
    if local_rank < 0:
        raise RuntimeError(f"LOCAL_RANK must be non-negative, got {local_rank}.")

    torch.cuda.set_device(local_rank)
    torch.cuda.init()
    probe = torch.empty(1, device=torch.device("cuda", local_rank))
    del probe
    print(
        f"[hami-cuda-bootstrap] initialized CUDA context for LOCAL_RANK={local_rank}",
        flush=True,
    )
    return local_rank


def main() -> None:
    bootstrap_cuda_context()
    target_module = os.environ.get(TARGET_MODULE_ENV, DEFAULT_TARGET_MODULE).strip()
    if not target_module:
        raise RuntimeError(f"{TARGET_MODULE_ENV} must name a Python module.")
    if target_module == __name__:
        raise RuntimeError(f"{TARGET_MODULE_ENV} cannot target {__name__!r} recursively.")
    print(
        f"[hami-cuda-bootstrap] launching target module {target_module}",
        flush=True,
    )
    runpy.run_module(target_module, run_name="__main__")


if __name__ == "__main__":
    main()
