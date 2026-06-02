from __future__ import annotations

from logging import Logger

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fact_checking.data.constants import LABELS, labels_for_schema
from fact_checking.utils.logging import init_logger
from sft.metrics import _build_confusion_matrix, _compute_classification_metrics
from sft.parser import _parse_label_id

from sft.infer_common import label_name_from_id

module_logger = init_logger(__name__)


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
    labels: list[str] | None = None,
    prediction_records: list[dict[str, object]] | None = None,
    eval_logger: Logger | None = None,
    log_predictions_limit: int = 5,
    log_prediction_examples: bool = True,
) -> dict[str, object]:
    metrics = _compute_classification_metrics(pred_ids, gold_ids, labels=labels)
    confusion_matrix, confusion_labels = _build_confusion_matrix(pred_ids, gold_ids, labels=labels)
    metrics["confusion_matrix"] = confusion_matrix
    metrics["confusion_labels"] = confusion_labels

    ordered_records = deduplicate_prediction_records(prediction_records or [])
    ordered_records.sort(key=lambda record: int(record["sample_idx"]))
    if log_prediction_examples:
        _log_prediction_examples(
            ordered_records,
            target_logger=eval_logger,
            limit=log_predictions_limit,
        )
    metrics["prediction_records"] = ordered_records
    return metrics


def deduplicate_by_sample_idx(
    pred_ids: np.ndarray,
    gold_ids: np.ndarray,
    sample_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep one gathered prediction per original sample index.

    Accelerate may repeat examples during distributed evaluation to make each
    process see equally sized batches. The repeated rows should not contribute
    to validation metrics or prediction artifacts.
    """
    pred_arr = np.asarray(pred_ids, dtype=np.int64)
    gold_arr = np.asarray(gold_ids, dtype=np.int64)
    sample_arr = np.asarray(sample_indices, dtype=np.int64)
    if len(pred_arr) != len(gold_arr) or len(pred_arr) != len(sample_arr):
        raise ValueError(
            "pred_ids, gold_ids, and sample_indices must have the same length "
            f"({len(pred_arr)}, {len(gold_arr)}, {len(sample_arr)})."
        )
    if len(sample_arr) == 0:
        return pred_arr, gold_arr, sample_arr

    seen: set[int] = set()
    keep: list[int] = []
    for array_idx in np.argsort(sample_arr, kind="stable").tolist():
        sample_idx = int(sample_arr[array_idx])
        if sample_idx < 0 or sample_idx in seen:
            continue
        seen.add(sample_idx)
        keep.append(int(array_idx))
    keep_arr = np.asarray(keep, dtype=np.int64)
    return pred_arr[keep_arr], gold_arr[keep_arr], sample_arr[keep_arr]


def deduplicate_prediction_records(
    prediction_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    seen: set[int] = set()
    deduped: list[dict[str, object]] = []
    for record in prediction_records:
        sample_idx = int(record["sample_idx"])
        if sample_idx < 0 or sample_idx in seen:
            continue
        seen.add(sample_idx)
        deduped.append(record)
    return deduped


def summarize_prediction_records(
    prediction_records: list[dict[str, object]],
    *,
    labels: list[str] | None = None,
    eval_logger: Logger | None = None,
    log_predictions_limit: int = 5,
) -> dict[str, object]:
    active_labels = labels or LABELS
    if not prediction_records:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "parse_error_rate": 0.0,
            "per_class": {},
            "confusion_matrix": np.zeros((len(active_labels), len(active_labels) + 1), dtype=np.int64),
            "confusion_labels": active_labels + ["parse_error"],
            "prediction_records": [],
        }

    pred_np = np.asarray([int(record["pred_id"]) for record in prediction_records], dtype=np.int64)
    gold_np = np.asarray([int(record["gold_id"]) for record in prediction_records], dtype=np.int64)
    return build_eval_metrics(
        pred_np,
        gold_np,
        labels=active_labels,
        prediction_records=prediction_records,
        eval_logger=eval_logger,
        log_predictions_limit=log_predictions_limit,
        log_prediction_examples=True,
    )


def _predict_with_logit_adjust(
    *,
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_token_ids: torch.Tensor,
    letter_token_ids: torch.Tensor,
    log_priors: torch.Tensor,
    tau: float,
    eos_id: int,
) -> tuple[list[int], torch.Tensor]:
    """One forward pass with 'Label:' appended; restricted argmax over letter tokens.

    Returns (pred_label_ids, continuation_ids). pred_label_ids[i] follows the
    active label schema order.
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    extension = prefix_token_ids.to(device).unsqueeze(0).expand(batch_size, -1)
    extension_attn = torch.ones_like(extension)
    new_input_ids = torch.cat([input_ids, extension], dim=1)
    new_attn = torch.cat([attention_mask, extension_attn], dim=1)
    outputs = model(input_ids=new_input_ids, attention_mask=new_attn, use_cache=False)
    last_logits = outputs.logits[:, -1, :]  # [B, V]
    letter_token_ids_dev = letter_token_ids.to(device)
    letter_logits = last_logits.index_select(1, letter_token_ids_dev)  # [B, K]
    log_priors_dev = log_priors.to(letter_logits.dtype).to(device)
    adjusted = letter_logits - tau * log_priors_dev.unsqueeze(0)
    pred_letter_idx = adjusted.argmax(dim=-1)  # [B]
    chosen_letter_tok = letter_token_ids_dev.index_select(0, pred_letter_idx)
    eos_col = torch.full(
        (batch_size, 1),
        int(eos_id),
        dtype=torch.long,
        device=device,
    )
    continuation_ids = torch.cat([extension, chosen_letter_tok.unsqueeze(1), eos_col], dim=1)
    return pred_letter_idx.tolist(), continuation_ids


def evaluate(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    max_length: int,
    max_new_tokens: int = 24,
    eval_logger: Logger | None = None,
    log_predictions_limit: int = 5,
    logit_adjust_cfg: dict | None = None,
) -> dict[str, object]:
    del max_length
    model.eval()

    unwrapped = accelerator.unwrap_model(model)
    old_use_cache = getattr(unwrapped.config, "use_cache", None)
    if old_use_cache is not None:
        unwrapped.config.use_cache = True

    use_logit_adjust = bool(logit_adjust_cfg and logit_adjust_cfg.get("enabled"))
    if use_logit_adjust:
        prefix_token_ids = torch.as_tensor(logit_adjust_cfg["prefix_token_ids"], dtype=torch.long)
        letter_token_ids = torch.as_tensor(logit_adjust_cfg["letter_token_ids"], dtype=torch.long)
        log_priors = torch.as_tensor(logit_adjust_cfg["log_priors"], dtype=torch.float32)
        tau = float(logit_adjust_cfg.get("tau", 1.0))
    else:
        prefix_token_ids = letter_token_ids = log_priors = None
        tau = 0.0

    all_pred_ids: list[torch.Tensor] = []
    all_gold_ids: list[torch.Tensor] = []
    all_sample_indices: list[torch.Tensor] = []
    all_prediction_records: list[dict[str, object]] = []
    pad_id = -100
    sample_pad_id = -1
    generation_pad_id = tokenizer.pad_token_id
    if generation_pad_id is None:
        generation_pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    eos_id_for_pred = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else generation_pad_id
    dataset_samples = getattr(getattr(dataloader, "dataset", None), "samples", None)
    label_schema = _dataset_label_schema(dataset_samples)
    active_labels = labels_for_schema(label_schema)
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

                if use_logit_adjust:
                    pred_ids, continuation_ids = _predict_with_logit_adjust(
                        model=model,
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        prefix_token_ids=prefix_token_ids,
                        letter_token_ids=letter_token_ids,
                        log_priors=log_priors,
                        tau=tau,
                        eos_id=int(eos_id_for_pred),
                    )
                else:
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
                        pred_ids.append(_parse_label_id(raw_pred, label_schema=label_schema))

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
                    all_sample_indices.append(gathered_sample_indices[valid_mask].cpu())
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
                                    "pred_label": label_name_from_id(int(pred_id), label_schema),
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
            labels=active_labels,
            eval_logger=eval_logger,
            log_predictions_limit=log_predictions_limit,
        )

    pred_np = torch.cat(all_pred_ids).numpy()
    gold_np = torch.cat(all_gold_ids).numpy()
    sample_indices_np = torch.cat(all_sample_indices).numpy()
    pred_np, gold_np, _ = deduplicate_by_sample_idx(pred_np, gold_np, sample_indices_np)
    metrics = build_eval_metrics(
        pred_np,
        gold_np,
        labels=active_labels,
        prediction_records=all_prediction_records if accelerator.is_main_process else [],
        eval_logger=eval_logger,
        log_predictions_limit=log_predictions_limit,
        log_prediction_examples=accelerator.is_main_process,
    )
    model.train()
    return metrics


def _dataset_label_schema(dataset_samples: object) -> str:
    if not dataset_samples:
        return "liar6"
    try:
        first = dataset_samples[0]  # type: ignore[index]
    except Exception:
        return "liar6"
    if isinstance(first, dict):
        return str(first.get("label_schema") or "liar6")
    return str(getattr(first, "label_schema", "liar6") or "liar6")
