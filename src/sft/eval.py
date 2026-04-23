from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from fact_checking import LABELS
from sft.data.types import PreparedSample
from sft.parser import _parse_label_id
from sft.metrics import _compute_classification_metrics, _build_confusion_matrix

def evaluate(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    max_length: int,
    max_new_tokens: int = 24,
) -> dict[str, float]:
    model.eval()

    unwrapped = accelerator.unwrap_model(model)
    old_use_cache = getattr(unwrapped.config, "use_cache", None)
    if old_use_cache is not None:
        unwrapped.config.use_cache = True

    all_pred_ids: list[torch.Tensor] = []
    all_gold_ids: list[torch.Tensor] = []
    pad_id = -100
    eval_progress = tqdm(
        total=len(dataloader),
        desc="eval",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )

    try:

        with torch.no_grad():
            for batch in dataloader:
                gold_ids = batch["gold_ids"]

                generated = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    synced_gpus=accelerator.num_processes > 1,
                )
                prompt_length = batch["input_ids"].shape[1]
                pred_ids: list[int] = []
                for i in range(generated.shape[0]):
                    gen_ids = generated[i, prompt_length:]
                    raw_pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    pred_ids.append(_parse_label_id(raw_pred))

                pred_tensor = torch.tensor(pred_ids, dtype=torch.long, device=gold_ids.device)
                pred_tensor = accelerator.pad_across_processes(pred_tensor, dim=0, pad_index=pad_id)
                gold_ids = accelerator.pad_across_processes(gold_ids, dim=0, pad_index=pad_id)
                gathered_pred = accelerator.gather(pred_tensor)
                gathered_gold = accelerator.gather(gold_ids)
                valid_mask = gathered_gold != pad_id
                if valid_mask.any():
                    all_pred_ids.append(gathered_pred[valid_mask].cpu())
                    all_gold_ids.append(gathered_gold[valid_mask].cpu())
                eval_progress.update(1)

    finally:
        if old_use_cache is not None:
            unwrapped.config.use_cache = old_use_cache

    eval_progress.close()

    if not all_gold_ids:
        model.train()
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "parse_error_rate": 0.0,
            "per_class": {},
            "confusion_matrix": np.zeros((len(LABELS), len(LABELS) + 1), dtype=np.int64),
            "confusion_labels": LABELS + ["parse_error"],
        }

    pred_np = torch.cat(all_pred_ids).numpy()
    gold_np = torch.cat(all_gold_ids).numpy()
    metrics = _compute_classification_metrics(pred_np, gold_np)
    confusion_matrix, confusion_labels = _build_confusion_matrix(pred_np, gold_np)
    metrics["confusion_matrix"] = confusion_matrix
    metrics["confusion_labels"] = confusion_labels
    model.train()
    return metrics
