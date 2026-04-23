from __future__ import annotations

from logging import Logger

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fact_checking.data.constants import LABELS
from fact_checking.utils.logging import init_logger
from sft.metrics import _build_confusion_matrix, _compute_classification_metrics
from sft.parser import _parse_label_id

module_logger = init_logger(__name__)


def _label_name_from_id(label_id: int) -> str:
    if 0 <= int(label_id) < len(LABELS):
        return LABELS[int(label_id)]
    return "parse_error"


def _truncate_for_log(text: str, max_chars: int = 240) -> str:
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _log_prediction_examples(
    prediction_records: list[dict[str, object]],
    target_logger: Logger | None,
    limit: int,
) -> None:
    if limit <= 0:
        return

    active_logger = target_logger or module_logger
    for record in prediction_records[:limit]:
        active_logger.info(
            "[EVAL_SAMPLE] idx=%s pred=%s gold=%s output=%s explain=%s prompt=%s",
            record["sample_idx"],
            record["pred_label"],
            record["gold_label"],
            _truncate_for_log(str(record["raw_output"])),
            _truncate_for_log(str(record["gold_explain"])),
            _truncate_for_log(str(record["prompt"])),
        )


def build_eval_metrics(
    pred_ids: np.ndarray,
    gold_ids: np.ndarray,
    *,
    prediction_records: list[dict[str, object]] | None = None,
    eval_logger: Logger | None = None,
    log_predictions_limit: int = 5,
    log_prediction_examples: bool = True,
) -> dict[str, object]:
    metrics = _compute_classification_metrics(pred_ids, gold_ids)
    confusion_matrix, confusion_labels = _build_confusion_matrix(pred_ids, gold_ids)
    metrics["confusion_matrix"] = confusion_matrix
    metrics["confusion_labels"] = confusion_labels

    ordered_records = list(prediction_records or [])
    ordered_records.sort(key=lambda record: int(record["sample_idx"]))
    if log_prediction_examples:
        _log_prediction_examples(
            ordered_records,
            target_logger=eval_logger,
            limit=log_predictions_limit,
        )
    metrics["prediction_records"] = ordered_records
    return metrics


def summarize_prediction_records(
    prediction_records: list[dict[str, object]],
    *,
    eval_logger: Logger | None = None,
    log_predictions_limit: int = 5,
) -> dict[str, object]:
    if not prediction_records:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "parse_error_rate": 0.0,
            "per_class": {},
            "confusion_matrix": np.zeros((len(LABELS), len(LABELS) + 1), dtype=np.int64),
            "confusion_labels": LABELS + ["parse_error"],
            "prediction_records": [],
        }

    pred_np = np.asarray([int(record["pred_id"]) for record in prediction_records], dtype=np.int64)
    gold_np = np.asarray([int(record["gold_id"]) for record in prediction_records], dtype=np.int64)
    return build_eval_metrics(
        pred_np,
        gold_np,
        prediction_records=prediction_records,
        eval_logger=eval_logger,
        log_predictions_limit=log_predictions_limit,
        log_prediction_examples=True,
    )


def evaluate(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    max_length: int,
    max_new_tokens: int = 24,
    eval_logger: Logger | None = None,
    log_predictions_limit: int = 5,
) -> dict[str, object]:
    del max_length
    model.eval()

    unwrapped = accelerator.unwrap_model(model)
    old_use_cache = getattr(unwrapped.config, "use_cache", None)
    if old_use_cache is not None:
        unwrapped.config.use_cache = True

    all_pred_ids: list[torch.Tensor] = []
    all_gold_ids: list[torch.Tensor] = []
    all_prediction_records: list[dict[str, object]] = []
    pad_id = -100
    sample_pad_id = -1
    generation_pad_id = tokenizer.pad_token_id
    if generation_pad_id is None:
        generation_pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    dataset_samples = getattr(getattr(dataloader, "dataset", None), "samples", None)
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
                sample_indices = batch["sample_indices"]

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
                continuation_ids = generated[:, prompt_length:]
                pred_ids: list[int] = []
                for i in range(continuation_ids.shape[0]):
                    raw_pred = tokenizer.decode(continuation_ids[i], skip_special_tokens=True)
                    pred_ids.append(_parse_label_id(raw_pred))

                pred_tensor = torch.tensor(pred_ids, dtype=torch.long, device=gold_ids.device)
                continuation_ids = accelerator.pad_across_processes(
                    continuation_ids,
                    dim=1,
                    pad_index=generation_pad_id,
                )
                pred_tensor = accelerator.pad_across_processes(pred_tensor, dim=0, pad_index=pad_id)
                gold_ids = accelerator.pad_across_processes(gold_ids, dim=0, pad_index=pad_id)
                sample_indices = accelerator.pad_across_processes(sample_indices, dim=0, pad_index=sample_pad_id)
                gathered_pred = accelerator.gather(pred_tensor)
                gathered_gold = accelerator.gather(gold_ids)
                gathered_sample_indices = accelerator.gather(sample_indices)
                gathered_continuation_ids = accelerator.gather(continuation_ids)
                valid_mask = (gathered_gold != pad_id) & (gathered_sample_indices != sample_pad_id)
                if valid_mask.any():
                    all_pred_ids.append(gathered_pred[valid_mask].cpu())
                    all_gold_ids.append(gathered_gold[valid_mask].cpu())
                    if accelerator.is_main_process and dataset_samples is not None:
                        valid_pred = gathered_pred[valid_mask].cpu().tolist()
                        valid_gold = gathered_gold[valid_mask].cpu().tolist()
                        valid_indices = gathered_sample_indices[valid_mask].cpu().tolist()
                        valid_continuation_ids = gathered_continuation_ids[valid_mask].cpu()
                        for sample_idx, pred_id, gold_id, token_ids in zip(
                            valid_indices,
                            valid_pred,
                            valid_gold,
                            valid_continuation_ids,
                        ):
                            sample = dataset_samples[int(sample_idx)]
                            raw_output = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
                            all_prediction_records.append(
                                {
                                    "sample_idx": int(sample_idx),
                                    "prompt": str(sample["prompt"]),
                                    "target": str(sample["target"]),
                                    "raw_output": raw_output,
                                    "pred_id": int(pred_id),
                                    "pred_label": _label_name_from_id(int(pred_id)),
                                    "gold_id": int(gold_id),
                                    "gold_label": str(sample["gold_label"]),
                                    "gold_explain": str(sample["gold_explain"]),
                                }
                            )
                eval_progress.update(1)

    finally:
        if old_use_cache is not None:
            unwrapped.config.use_cache = old_use_cache

    eval_progress.close()

    if not all_gold_ids:
        model.train()
        return summarize_prediction_records(
            [],
            eval_logger=eval_logger,
            log_predictions_limit=log_predictions_limit,
        )

    pred_np = torch.cat(all_pred_ids).numpy()
    gold_np = torch.cat(all_gold_ids).numpy()
    metrics = build_eval_metrics(
        pred_np,
        gold_np,
        prediction_records=all_prediction_records if accelerator.is_main_process else [],
        eval_logger=eval_logger,
        log_predictions_limit=log_predictions_limit,
        log_prediction_examples=accelerator.is_main_process,
    )
    model.train()
    return metrics
