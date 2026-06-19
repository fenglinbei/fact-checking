from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed

from fact_checking.config import load_yaml, save_yaml
from fact_checking.data.constants import (
    COVERAGE_LABELS,
    COVERAGE_LETTER_ORDER,
    labels_for_schema,
    letter_order_for_schema,
)
from fact_checking.data.io import load_jsonl
from fact_checking.utils.logging import init_logger
from sft.data.io import _save_eval_artifacts, load_prebuilt_samples, save_model
from sft.data.sampling import select_mini_val_rows
from sft.dataset.loaders import build_dataloader
from sft.eval import build_eval_metrics, deduplicate_by_sample_idx
from sft.checkpoint_selection import (
    checkpoint_selection_score as _checkpoint_selection_score,
    macro_f1_bootstrap_se_from_records,
    select_one_standard_error_checkpoint,
    true_side_macro_f1 as _true_side_macro_f1,
)
from sft.logit_adjust import build_logit_adjust_cfg_from_samples
from sft.label_token_dataset import LabelTokenCollator, LabelTokenDataset
from sft.prompting.stats import (
    build_prompt_snapshots,
    flatten_prompt_statistics,
    log_prompt_summary,
    save_prompt_statistics,
    summarize_prebuilt_prompts,
)
from sft.runtime.adapters import (
    apply_lora_if_enabled,
    checkpoint_has_hf_artifacts,
    freeze_modules_by_prefix,
    lora_enabled,
)
from sft.runtime.config import apply_runtime_output_layout, resolve_artifact_dir
from sft.runtime.deps import flash_attn2_available, fla_fast_path_available
from sft.runtime.device import maybe_empty_cache
from sft.runtime.model_loading import load_causal_lm_compatible_model, load_compatible_tokenizer
from sft.runtime.tracking import log_metrics
from sft.train_utils import setup_accelerator_and_tracker

logger = init_logger(__name__)


def _choice_text(label_prefix: str, letter: str) -> str:
    return letter if label_prefix.endswith((" ", "\n", "\t")) else " " + letter


def _build_label_token_ids(
    tokenizer: AutoTokenizer,
    *,
    label_prefix: str,
    letter_order: list[str],
) -> tuple[list[int], dict[str, Any]]:
    prefix_ids = tokenizer(label_prefix, add_special_tokens=False, truncation=False)["input_ids"]
    if not prefix_ids:
        raise ValueError(f"label_prefix={label_prefix!r} produced no tokens.")

    token_ids: list[int] = []
    token_texts: dict[str, str] = {}
    for letter in letter_order:
        token_text = _choice_text(label_prefix, letter)
        ids = tokenizer(token_text, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(
                f"label-token CE requires {token_text!r} to be one tokenizer token for letter={letter!r}; got {ids}."
            )
        token_ids.append(int(ids[0]))
        token_texts[letter] = token_text

    return token_ids, {
        "label_prefix": label_prefix,
        "prefix_token_ids": [int(x) for x in prefix_ids],
        "label_token_ids": {letter: int(token_id) for letter, token_id in zip(letter_order, token_ids)},
        "label_token_texts": token_texts,
    }


def _class_weight_tensor(train_cfg: dict[str, Any], *, labels: list[str]) -> torch.Tensor:
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    configured = label_cfg.get("class_weights", {}) or {}
    weights = [float(configured.get(label, 1.0)) for label in labels]
    return torch.tensor(weights, dtype=torch.float32)


def _coverage_label_token_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = train_cfg.get("coverage_label_token", {}) or {}
    if not isinstance(cfg, dict):
        raise TypeError("sft_train.coverage_label_token must be a mapping when configured.")
    return cfg


def _coverage_label_token_enabled(train_cfg: dict[str, Any]) -> bool:
    return bool(_coverage_label_token_cfg(train_cfg).get("enabled", False))


def _coverage_class_weight_tensor(train_cfg: dict[str, Any]) -> torch.Tensor:
    cfg = _coverage_label_token_cfg(train_cfg)
    configured = cfg.get("class_weights", {}) or {}
    weights = [float(configured.get(label, 1.0)) for label in COVERAGE_LABELS]
    return torch.tensor(weights, dtype=torch.float32)


def _ordinal_loss_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    cfg = label_cfg.get("ordinal_loss", {}) or {}
    if not isinstance(cfg, dict):
        raise TypeError("sft_train.label_token_ce.ordinal_loss must be a mapping when configured.")
    return cfg


def _ordinal_loss_meta(train_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = _ordinal_loss_cfg(train_cfg)
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "alpha": float(cfg.get("alpha", 0.0)),
        "alpha_warmup_ratio": float(cfg.get("alpha_warmup_ratio", 0.0)),
        "normalize_distance": bool(cfg.get("normalize_distance", True)),
        "distance": "absolute_rank",
    }


def _weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    denominator = sample_weights.sum().clamp_min(torch.finfo(torch.float32).eps)
    return (values * sample_weights).sum() / denominator


def _compute_label_token_losses(
    *,
    label_logits: torch.Tensor,
    gold_ids: torch.Tensor,
    class_weights: torch.Tensor,
    train_cfg: dict[str, Any],
    coverage_label_logits: torch.Tensor | None = None,
    coverage_gold_ids: torch.Tensor | None = None,
    coverage_class_weights: torch.Tensor | None = None,
    global_step: int = 0,
    max_train_steps: int = 1,
) -> dict[str, torch.Tensor]:
    logits = label_logits.float()
    gold_ids = gold_ids.to(device=logits.device, dtype=torch.long)
    weights = class_weights.to(device=logits.device, dtype=torch.float32)
    sample_weights = weights[gold_ids]

    ce_per_sample = F.cross_entropy(logits, gold_ids, reduction="none")
    ce_loss = _weighted_mean(ce_per_sample, sample_weights)

    ordinal_cfg = _ordinal_loss_cfg(train_cfg)
    ordinal_enabled = bool(ordinal_cfg.get("enabled", False))
    ordinal_alpha = float(ordinal_cfg.get("alpha", 0.0))
    ordinal_loss = logits.new_zeros(())
    if ordinal_enabled and logits.shape[-1] > 1:
        ranks = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float32)
        gold_ranks = ranks[gold_ids]
        distances = (ranks.unsqueeze(0) - gold_ranks.unsqueeze(1)).abs()
        if bool(ordinal_cfg.get("normalize_distance", True)):
            distances = distances / float(logits.shape[-1] - 1)
        probs = torch.softmax(logits, dim=-1)
        ordinal_per_sample = (probs * distances).sum(dim=-1)
        ordinal_loss = _weighted_mean(ordinal_per_sample, sample_weights)

    effective_alpha = ordinal_alpha
    if ordinal_enabled:
        alpha_warmup_ratio = float(ordinal_cfg.get("alpha_warmup_ratio", 0.0))
        if alpha_warmup_ratio > 0:
            warmup_steps = max(1, int(alpha_warmup_ratio * float(max_train_steps)))
            warmup_factor = min(1.0, float(global_step) / float(warmup_steps))
            effective_alpha = ordinal_alpha * warmup_factor

    loss = ce_loss + effective_alpha * ordinal_loss if ordinal_enabled else ce_loss
    coverage_ce_loss = logits.new_zeros(())
    if _coverage_label_token_enabled(train_cfg):
        if coverage_label_logits is None or coverage_gold_ids is None:
            raise ValueError("coverage_label_token.enabled=true requires coverage_label_logits and coverage_gold_ids.")
        coverage_logits = coverage_label_logits.float()
        coverage_gold_ids = coverage_gold_ids.to(device=coverage_logits.device, dtype=torch.long)
        if coverage_class_weights is None:
            coverage_weights = torch.ones(coverage_logits.shape[-1], device=coverage_logits.device)
        else:
            coverage_weights = coverage_class_weights.to(device=coverage_logits.device, dtype=torch.float32)
        coverage_sample_weights = coverage_weights[coverage_gold_ids]
        coverage_ce_per_sample = F.cross_entropy(coverage_logits, coverage_gold_ids, reduction="none")
        coverage_ce_loss = _weighted_mean(coverage_ce_per_sample, coverage_sample_weights)
        loss = loss + float(_coverage_label_token_cfg(train_cfg).get("loss_weight", 1.0)) * coverage_ce_loss
    return {
        "loss": loss,
        "ce_loss": ce_loss,
        "ordinal_loss": ordinal_loss,
        "coverage_ce_loss": coverage_ce_loss,
    }


def _forward_label_logits(
    model: AutoModelForCausalLM,
    batch: dict[str, Any],
    label_token_ids: torch.Tensor,
) -> torch.Tensor:
    return _forward_label_logits_for_inputs(
        model,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        label_token_ids=label_token_ids,
    )


def _forward_label_logits_for_inputs(
    model: AutoModelForCausalLM,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_token_ids: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    last_positions = attention_mask.long().sum(dim=1) - 1
    batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
    next_token_logits = outputs.logits[batch_indices, last_positions]
    return next_token_logits.index_select(1, label_token_ids.to(next_token_logits.device))


def _apply_label_logit_adjust(
    label_logits: torch.Tensor,
    logit_adjust_cfg: dict[str, Any] | None,
) -> torch.Tensor:
    if not (logit_adjust_cfg and logit_adjust_cfg.get("enabled")):
        return label_logits
    log_priors = torch.as_tensor(
        logit_adjust_cfg.get("log_priors", []),
        dtype=label_logits.dtype,
        device=label_logits.device,
    )
    if log_priors.numel() != label_logits.shape[-1]:
        raise ValueError(
            "logit_adjust log_priors length does not match label logits: "
            f"{log_priors.numel()} vs {label_logits.shape[-1]}"
        )
    tau = float(logit_adjust_cfg.get("tau", 1.0))
    return label_logits - tau * log_priors.unsqueeze(0)


def _label_name(label_id: int, *, labels: list[str]) -> str:
    if 0 <= int(label_id) < len(labels):
        return labels[int(label_id)]
    return "parse_error"


def _coverage_label_name(label_id: int) -> str:
    if 0 <= int(label_id) < len(COVERAGE_LABELS):
        return COVERAGE_LABELS[int(label_id)]
    return "parse_error"


def _best_checkpoint_policy(train_cfg: dict[str, Any]) -> str:
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    policy = label_cfg.get("best_checkpoint_policy")
    if policy is None:
        best_selection = label_cfg.get("best_selection", {}) or {}
        if isinstance(best_selection, dict):
            policy = best_selection.get("policy")
    return str(policy or "checkpoint_selection_score").strip().lower().replace("-", "_")


def _macro_f1_bootstrap_cfg(train_cfg: dict[str, Any]) -> tuple[int, int]:
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    n_bootstrap = int(label_cfg.get("bootstrap_samples", label_cfg.get("macro_f1_bootstrap_samples", 1000)))
    seed = int(label_cfg.get("bootstrap_seed", label_cfg.get("macro_f1_bootstrap_seed", 20260620)))
    return n_bootstrap, seed


def _copy_checkpoint_to_best(
    *,
    accelerator: Accelerator,
    source_dir: Path,
    best_dir: Path,
    active_logger,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not source_dir.exists():
            raise FileNotFoundError(f"one-SE selected checkpoint does not exist: {source_dir}")
        tmp_dir = best_dir.with_name(f"{best_dir.name}.tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        shutil.copytree(source_dir, tmp_dir)
        if best_dir.exists():
            shutil.rmtree(best_dir)
        tmp_dir.rename(best_dir)
        active_logger.info("[best] copied selected checkpoint %s to %s", source_dir, best_dir)
    accelerator.wait_for_everyone()


def _is_distributed_teardown_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("nccl" in text or "destroy_process_group" in text)


def _end_training_after_final_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    active_logger,
) -> None:
    try:
        accelerator.end_training()
    except Exception as exc:
        final_dir = output_dir / "final"
        if checkpoint_has_hf_artifacts(final_dir) and _is_distributed_teardown_oom(exc):
            active_logger.warning(
                "[WARN] Ignoring distributed teardown CUDA OOM after final checkpoint was saved: %s",
                exc,
            )
            return
        raise


def _bool_cfg(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_bool(name: str, *, default: bool) -> bool:
    if name not in os.environ:
        return default
    return _bool_cfg(os.environ.get(name), default=default)


def _latest_state_dir(output_dir: Path, train_cfg: dict[str, Any]) -> Path:
    configured = str(train_cfg.get("latest_state_dir", "latest_state"))
    path = Path(configured)
    return path if path.is_absolute() else output_dir / path


def _latest_state_meta_path(state_dir: Path) -> Path:
    return state_dir / "trainer_state.json"


def _read_latest_state_meta(state_dir: Path) -> dict[str, Any] | None:
    meta_path = _latest_state_meta_path(state_dir)
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise TypeError(f"Expected mapping metadata in {meta_path}")
    return meta


def _install_stop_signal_handlers(active_logger) -> dict[str, Any]:
    stop_request: dict[str, Any] = {"requested": False, "signal": None}

    def _handle_signal(signum, _frame) -> None:
        stop_request["requested"] = True
        stop_request["signal"] = int(signum)
        try:
            active_logger.warning("[signal] received %s; will save latest training state and stop.", signum)
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    return stop_request


def _json_score(score: float) -> float | None:
    return float(score) if math.isfinite(float(score)) else None


def _save_latest_training_state(
    *,
    accelerator: Accelerator,
    output_dir: Path,
    train_cfg: dict[str, Any],
    epoch: int,
    next_batch_index: int,
    global_step: int,
    best_score: float,
    no_improve_count: int,
    max_train_steps: int,
    active_logger,
    enabled: bool,
    completed: bool = False,
) -> None:
    if not enabled:
        return

    state_dir = _latest_state_dir(output_dir, train_cfg)
    tmp_state_dir = state_dir.with_name(f".{state_dir.name}.tmp")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if tmp_state_dir.exists():
            shutil.rmtree(tmp_state_dir)
        tmp_state_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.save_state(str(tmp_state_dir))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        meta = {
            "completed": bool(completed),
            "epoch": int(epoch),
            "next_batch_index": int(next_batch_index),
            "global_step": int(global_step),
            "best_score": _json_score(best_score),
            "no_improve_count": int(no_improve_count),
            "max_train_steps": int(max_train_steps),
        }
        meta_path = _latest_state_meta_path(tmp_state_dir)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if state_dir.exists():
            shutil.rmtree(state_dir)
        tmp_state_dir.rename(state_dir)
        active_logger.info("[checkpoint] latest training state saved to %s", state_dir)
    accelerator.wait_for_everyone()


def _load_latest_training_state(
    *,
    accelerator: Accelerator,
    output_dir: Path,
    train_cfg: dict[str, Any],
    active_logger,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {}

    state_dir = _latest_state_dir(output_dir, train_cfg)
    meta = _read_latest_state_meta(state_dir)
    if meta is None:
        return {}
    if bool(meta.get("completed", False)):
        return {}

    accelerator.load_state(str(state_dir))
    if accelerator.is_main_process:
        active_logger.info(
            "[checkpoint] resumed latest training state from %s at global_step=%s epoch=%s next_batch_index=%s",
            state_dir,
            meta.get("global_step"),
            meta.get("epoch"),
            meta.get("next_batch_index"),
        )
    accelerator.wait_for_everyone()
    return meta


def _write_training_complete(
    *,
    accelerator: Accelerator,
    output_dir: Path,
    train_cfg: dict[str, Any],
    global_step: int,
    best_score: float,
    active_logger,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        payload = {
            "completed": True,
            "global_step": int(global_step),
            "best_score": _json_score(best_score),
        }
        (output_dir / "training_complete.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if _bool_cfg(train_cfg.get("cleanup_latest_state_on_complete", True), default=True):
            for path in (
                _latest_state_dir(output_dir, train_cfg),
                _latest_state_dir(output_dir, train_cfg).with_name(
                    f".{_latest_state_dir(output_dir, train_cfg).name}.tmp"
                ),
            ):
                if path.exists():
                    shutil.rmtree(path)
                    active_logger.info("[checkpoint] removed completed-run latest state: %s", path)
    accelerator.wait_for_everyone()


def _evaluate_label_token(
    *,
    model: AutoModelForCausalLM,
    dataloader,
    accelerator: Accelerator,
    label_token_ids: torch.Tensor,
    class_weights: torch.Tensor,
    coverage_label_token_ids: torch.Tensor | None = None,
    coverage_class_weights: torch.Tensor | None = None,
    train_cfg: dict[str, Any],
    label_prefix: str,
    labels: list[str],
    letter_order: list[str],
    eval_logger,
    log_predictions_limit: int,
    logit_adjust_cfg: dict[str, Any] | None = None,
    global_step: int = 0,
    max_train_steps: int = 1,
) -> dict[str, Any]:
    model.eval()

    all_pred_ids: list[torch.Tensor] = []
    all_gold_ids: list[torch.Tensor] = []
    all_sample_indices: list[torch.Tensor] = []
    all_coverage_pred_ids: list[torch.Tensor] = []
    all_coverage_gold_ids: list[torch.Tensor] = []
    all_coverage_sample_indices: list[torch.Tensor] = []
    all_losses: list[torch.Tensor] = []
    all_ce_losses: list[torch.Tensor] = []
    all_ordinal_losses: list[torch.Tensor] = []
    all_coverage_ce_losses: list[torch.Tensor] = []
    pad_id = -100
    sample_pad_id = -1
    dataset_samples = getattr(getattr(dataloader, "dataset", None), "samples", None)
    progress = tqdm(
        total=len(dataloader),
        desc="label-token-eval",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )

    with torch.no_grad():
        for batch in dataloader:
            gold_ids = batch["gold_ids"]
            label_logits = _forward_label_logits(model, batch, label_token_ids)
            coverage_label_logits = None
            coverage_gold_ids = None
            if coverage_label_token_ids is not None and "coverage_input_ids" in batch:
                coverage_gold_ids = batch["coverage_gold_ids"]
                coverage_label_logits = _forward_label_logits_for_inputs(
                    model,
                    input_ids=batch["coverage_input_ids"],
                    attention_mask=batch["coverage_attention_mask"],
                    label_token_ids=coverage_label_token_ids,
                )
            losses = _compute_label_token_losses(
                label_logits=label_logits,
                gold_ids=gold_ids,
                class_weights=class_weights,
                train_cfg=train_cfg,
                coverage_label_logits=coverage_label_logits,
                coverage_gold_ids=coverage_gold_ids,
                coverage_class_weights=coverage_class_weights,
                global_step=global_step,
                max_train_steps=max_train_steps,
            )
            prediction_logits = _apply_label_logit_adjust(label_logits, logit_adjust_cfg)
            pred_ids = prediction_logits.argmax(dim=-1).to(torch.long)
            coverage_pred_ids = None
            if coverage_label_logits is not None and coverage_gold_ids is not None:
                coverage_pred_ids = coverage_label_logits.argmax(dim=-1).to(torch.long)

            pred_ids = accelerator.pad_across_processes(pred_ids, dim=0, pad_index=pad_id)
            gold_ids = accelerator.pad_across_processes(gold_ids, dim=0, pad_index=pad_id)
            sample_indices = accelerator.pad_across_processes(
                batch["sample_indices"],
                dim=0,
                pad_index=sample_pad_id,
            )
            gathered_pred = accelerator.gather(pred_ids)
            gathered_gold = accelerator.gather(gold_ids)
            gathered_sample_indices = accelerator.gather(sample_indices)
            valid_mask = (gathered_gold != pad_id) & (gathered_sample_indices != sample_pad_id)
            if valid_mask.any():
                all_pred_ids.append(gathered_pred[valid_mask].cpu())
                all_gold_ids.append(gathered_gold[valid_mask].cpu())
                all_sample_indices.append(gathered_sample_indices[valid_mask].cpu())

            if coverage_pred_ids is not None and coverage_gold_ids is not None:
                coverage_pred_ids = accelerator.pad_across_processes(coverage_pred_ids, dim=0, pad_index=pad_id)
                coverage_gold_ids = accelerator.pad_across_processes(coverage_gold_ids, dim=0, pad_index=pad_id)
                gathered_coverage_pred = accelerator.gather(coverage_pred_ids)
                gathered_coverage_gold = accelerator.gather(coverage_gold_ids)
                coverage_valid_mask = (gathered_coverage_gold != pad_id) & (
                    gathered_sample_indices != sample_pad_id
                )
                if coverage_valid_mask.any():
                    all_coverage_pred_ids.append(gathered_coverage_pred[coverage_valid_mask].cpu())
                    all_coverage_gold_ids.append(gathered_coverage_gold[coverage_valid_mask].cpu())
                    all_coverage_sample_indices.append(gathered_sample_indices[coverage_valid_mask].cpu())

            all_losses.append(accelerator.gather_for_metrics(losses["loss"].detach().float().unsqueeze(0)).cpu())
            all_ce_losses.append(accelerator.gather_for_metrics(losses["ce_loss"].detach().float().unsqueeze(0)).cpu())
            all_ordinal_losses.append(
                accelerator.gather_for_metrics(losses["ordinal_loss"].detach().float().unsqueeze(0)).cpu()
            )
            if _coverage_label_token_enabled(train_cfg):
                all_coverage_ce_losses.append(
                    accelerator.gather_for_metrics(losses["coverage_ce_loss"].detach().float().unsqueeze(0)).cpu()
                )
            progress.update(1)

    progress.close()

    if not all_gold_ids:
        model.train()
        metrics = build_eval_metrics(
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            labels=labels,
            prediction_records=[],
            eval_logger=eval_logger,
            log_predictions_limit=log_predictions_limit,
            log_prediction_examples=accelerator.is_main_process,
        )
        metrics["eval_loss"] = float("nan")
        metrics["eval_ce_loss"] = float("nan")
        metrics["eval_ordinal_loss"] = float("nan")
        return metrics

    pred_np = torch.cat(all_pred_ids).numpy()
    gold_np = torch.cat(all_gold_ids).numpy()
    sample_indices_np = torch.cat(all_sample_indices).numpy()
    pred_np, gold_np, sample_indices_np = deduplicate_by_sample_idx(pred_np, gold_np, sample_indices_np)
    coverage_by_sample_idx: dict[int, tuple[int, int]] = {}
    coverage_eval_payload: dict[str, Any] = {}
    if all_coverage_gold_ids:
        coverage_pred_np = torch.cat(all_coverage_pred_ids).numpy()
        coverage_gold_np = torch.cat(all_coverage_gold_ids).numpy()
        coverage_sample_indices_np = torch.cat(all_coverage_sample_indices).numpy()
        coverage_pred_np, coverage_gold_np, coverage_sample_indices_np = deduplicate_by_sample_idx(
            coverage_pred_np,
            coverage_gold_np,
            coverage_sample_indices_np,
        )
        coverage_by_sample_idx = {
            int(sample_idx): (int(pred_id), int(gold_id))
            for sample_idx, pred_id, gold_id in zip(
                coverage_sample_indices_np.tolist(),
                coverage_pred_np.tolist(),
                coverage_gold_np.tolist(),
            )
        }
        coverage_metrics = build_eval_metrics(
            coverage_pred_np,
            coverage_gold_np,
            labels=COVERAGE_LABELS,
            prediction_records=[],
            eval_logger=eval_logger,
            log_predictions_limit=0,
            log_prediction_examples=False,
        )
        coverage_eval_payload = {
            "coverage_accuracy": float(coverage_metrics["accuracy"]),
            "coverage_macro_precision": float(coverage_metrics["macro_precision"]),
            "coverage_macro_recall": float(coverage_metrics["macro_recall"]),
            "coverage_macro_f1": float(coverage_metrics["macro_f1"]),
            "coverage_parse_error_rate": float(coverage_metrics["parse_error_rate"]),
            "coverage_per_class": coverage_metrics.get("per_class", {}),
        }

    prediction_records: list[dict[str, object]] = []
    if accelerator.is_main_process and dataset_samples is not None:
        for sample_idx, pred_id, gold_id in zip(sample_indices_np.tolist(), pred_np.tolist(), gold_np.tolist()):
            sample = dataset_samples[int(sample_idx)]
            letter = letter_order[int(pred_id)]
            record: dict[str, object] = {
                "sample_idx": int(sample_idx),
                "prompt": str(sample.prompt),
                "target": str(sample.target),
                "raw_output": f"{label_prefix}{_choice_text(label_prefix, letter)}",
                "pred_id": int(pred_id),
                "pred_label": _label_name(int(pred_id), labels=labels),
                "gold_id": int(gold_id),
                "gold_label": str(sample.gold_label),
                "gold_explain": str(sample.gold_explain),
            }
            if int(sample_idx) in coverage_by_sample_idx:
                coverage_pred_id, coverage_gold_id = coverage_by_sample_idx[int(sample_idx)]
                coverage_letter = COVERAGE_LETTER_ORDER[coverage_pred_id]
                record.update(
                    {
                        "raw_output": (
                            f"{label_prefix}{_choice_text(label_prefix, letter)}\n"
                            f"Coverage:{_choice_text('Coverage:', coverage_letter)}"
                        ),
                        "coverage_pred_id": coverage_pred_id,
                        "coverage_pred_label": _coverage_label_name(coverage_pred_id),
                        "coverage_gold_id": coverage_gold_id,
                        "coverage_gold_label": _coverage_label_name(coverage_gold_id),
                    }
                )
            prediction_records.append(record)

    metrics = build_eval_metrics(
        pred_np,
        gold_np,
        labels=labels,
        prediction_records=prediction_records if accelerator.is_main_process else [],
        eval_logger=eval_logger,
        log_predictions_limit=log_predictions_limit,
        log_prediction_examples=accelerator.is_main_process,
    )
    if all_losses:
        metrics["eval_loss"] = float(torch.cat(all_losses).mean().item())
        metrics["eval_ce_loss"] = float(torch.cat(all_ce_losses).mean().item())
        metrics["eval_ordinal_loss"] = float(torch.cat(all_ordinal_losses).mean().item())
        if all_coverage_ce_losses:
            metrics["eval_coverage_ce_loss"] = float(torch.cat(all_coverage_ce_losses).mean().item())
    else:
        metrics["eval_loss"] = float("nan")
        metrics["eval_ce_loss"] = float("nan")
        metrics["eval_ordinal_loss"] = float("nan")
        if all_coverage_ce_losses:
            metrics["eval_coverage_ce_loss"] = float("nan")
    metrics.update(coverage_eval_payload)
    model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-token weighted CE verifier training.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mini-val-size", type=int, default=None)
    parser.add_argument("--mini-val-seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_runtime_output_layout(cfg)
    data_cfg = cfg["data"]
    baseline_cfg = cfg.get("baseline", {})
    train_cfg = cfg["sft_train"]
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    label_prefix = str(label_cfg.get("label_prefix", "Label:"))
    label_schema = str(
        train_cfg.get("label_schema")
        or cfg.get("label_schema")
        or baseline_cfg.get("label_schema")
        or "liar6"
    )
    labels = labels_for_schema(label_schema)
    letter_order = letter_order_for_schema(label_schema)

    set_seed(int(train_cfg.get("seed", 42)))
    mixed_precision = "bf16" if bool(train_cfg.get("bf16", True)) else "no"
    accelerator, tracking_setup = setup_accelerator_and_tracker(train_cfg, cfg)

    train_rows = load_jsonl(data_cfg["train_candidates"])
    val_rows = load_jsonl(data_cfg["val_candidates"])
    mini_val_size = int(
        args.mini_val_size if args.mini_val_size is not None else int(train_cfg.get("mini_val_size", 0))
    )
    mini_val_seed = int(
        args.mini_val_seed
        if args.mini_val_seed is not None
        else int(train_cfg.get("mini_val_seed", train_cfg.get("seed", 42)))
    )
    val_rows = select_mini_val_rows(
        rows=val_rows,
        mini_val_size=mini_val_size,
        mini_val_seed=mini_val_seed,
        accelerator=accelerator,
    )
    train_samples = load_prebuilt_samples(train_rows)
    val_samples = load_prebuilt_samples(val_rows)

    output_dir = Path(cfg.get("output_dir", "outputs/runs/train"))
    eval_root = resolve_artifact_dir(cfg, "eval_output_dir", output_dir, "eval")
    prompt_stats_dir = resolve_artifact_dir(cfg, "prompt_stats_output_dir", output_dir, "prompt_stats")
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        active_logger = init_logger(__name__, log_dir=output_dir / "logs", log_filename="train_loop.log")
        save_yaml(cfg, output_dir / "config.resolved.yaml")
        active_logger.info("[INFO] saved resolved config to %s", output_dir / "config.resolved.yaml")
    else:
        active_logger = init_logger(__name__)

    model_name_or_path = str(
        cfg.get("model_name_or_path")
        or baseline_cfg.get("model_name_or_path", "/data/models/Qwen3.5-9B")
    )
    tokenizer = load_compatible_tokenizer(model_name_or_path, trust_remote_code=True)

    max_length = int(train_cfg.get("max_length", 2048))
    label_token_id_list, token_meta = _build_label_token_ids(
        tokenizer,
        label_prefix=label_prefix,
        letter_order=letter_order,
    )
    label_token_ids = torch.tensor(label_token_id_list, dtype=torch.long)
    class_weights = _class_weight_tensor(train_cfg, labels=labels)
    coverage_enabled = _coverage_label_token_enabled(train_cfg)
    coverage_cfg = _coverage_label_token_cfg(train_cfg)
    coverage_label_prefix = str(coverage_cfg.get("label_prefix", "Coverage:"))
    coverage_label_token_ids = None
    coverage_token_meta: dict[str, Any] = {"enabled": False}
    coverage_class_weights = None
    if coverage_enabled:
        coverage_label_token_id_list, coverage_token_meta = _build_label_token_ids(
            tokenizer,
            label_prefix=coverage_label_prefix,
            letter_order=COVERAGE_LETTER_ORDER,
        )
        coverage_token_meta["enabled"] = True
        coverage_label_token_ids = torch.tensor(coverage_label_token_id_list, dtype=torch.long)
        coverage_class_weights = _coverage_class_weight_tensor(train_cfg)
    logit_adjust_cfg = build_logit_adjust_cfg_from_samples(
        train_cfg=train_cfg,
        tokenizer=tokenizer,
        train_samples=train_samples,
        label_schema=label_schema,
        label_prefix=label_prefix,
    )
    if logit_adjust_cfg and accelerator.is_main_process:
        logit_adjust_path = output_dir / "logit_adjust.json"
        logit_adjust_path.write_text(json.dumps(logit_adjust_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        active_logger.info(
            "[INFO] logit_adjust enabled: tau=%.3f log_priors=%s letters=%s prefix=%s",
            float(logit_adjust_cfg.get("tau", 1.0)),
            ["%.4f" % float(x) for x in logit_adjust_cfg.get("log_priors", [])],
            logit_adjust_cfg.get("letter_token_ids", []),
            logit_adjust_cfg.get("prefix_token_ids", []),
        )

    train_prompt_summary = summarize_prebuilt_prompts(train_samples, max_length=max_length, split="train")
    val_prompt_summary = summarize_prebuilt_prompts(val_samples, max_length=max_length, split="val")
    if accelerator.is_main_process:
        log_prompt_summary(train_prompt_summary, logger=active_logger)
        log_prompt_summary(val_prompt_summary, logger=active_logger)
        prompt_stats_path = save_prompt_statistics(
            output_dir=output_dir,
            train_summary=train_prompt_summary,
            val_summary=val_prompt_summary,
            train_snapshots=build_prompt_snapshots(train_samples, split="train"),
            val_snapshots=build_prompt_snapshots(val_samples, split="val"),
            stats_dir=prompt_stats_dir,
        )
        active_logger.info("[INFO] prompt statistics saved to %s", prompt_stats_path)
        meta_path = output_dir / "label_token_ce_meta.json"
        meta = {
            **token_meta,
            "label_schema": label_schema,
            "labels": labels,
            "class_weights": {label: float(weight) for label, weight in zip(labels, class_weights.tolist())},
            "ordinal_loss": _ordinal_loss_meta(train_cfg),
            "early_stopping_metric": "macro_f1",
            "configured_early_stopping_metric": str(label_cfg.get("early_stopping_metric", "macro_f1")),
            "true_side_metric_weight": float(label_cfg.get("true_side_metric_weight", 0.5)),
            "mae_metric_weight": float(label_cfg.get("mae_metric_weight", 0.3)),
            "focus_label": str(label_cfg.get("focus_label", "") or ""),
            "focus_metric_weight": float(label_cfg.get("focus_metric_weight", 0.3)),
            "logit_adjust": logit_adjust_cfg or {"enabled": False},
            "coverage_label_token": {
                **coverage_token_meta,
                "labels": COVERAGE_LABELS,
                "class_weights": (
                    {label: float(weight) for label, weight in zip(COVERAGE_LABELS, coverage_class_weights.tolist())}
                    if coverage_class_weights is not None
                    else {}
                ),
                "loss_weight": float(coverage_cfg.get("loss_weight", 1.0)),
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        active_logger.info("[INFO] label-token CE metadata saved to %s", meta_path)

    log_metrics(
        accelerator,
        flatten_prompt_statistics({"train": train_prompt_summary, "val": val_prompt_summary}),
        step=0,
        backend=tracking_setup.backend,
    )
    accelerator.wait_for_everyone()

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and mixed_precision == "bf16" else torch.float32,
    }
    if bool(train_cfg.get("use_flash_attention_2", True)):
        if flash_attn2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif accelerator.is_main_process:
            active_logger.warning(
                "[WARN] sft_train.use_flash_attention_2=true, but flash-attn is not installed. "
                "Falling back to the default attention implementation."
            )
    if accelerator.is_main_process and not fla_fast_path_available():
        active_logger.info(
            "[INFO] FLA fast path is unavailable (requires both `fla` and `causal_conv1d`). "
            "This is separate from flash-attn and does not block training."
        )

    model = load_causal_lm_compatible_model(model_name_or_path, **model_kwargs)
    model = freeze_modules_by_prefix(
        model,
        train_cfg,
        logger=active_logger if accelerator.is_main_process else None,
    )
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    model = apply_lora_if_enabled(
        model,
        train_cfg,
        gradient_checkpointing=gradient_checkpointing,
        logger=active_logger if accelerator.is_main_process else None,
    )

    padding_strategy = str(train_cfg.get("padding", "max_length"))
    if padding_strategy not in {"max_length", "longest"}:
        raise ValueError(f"Unsupported sft_train.padding={padding_strategy}. Use 'max_length' or 'longest'.")
    use_length_bucket = bool(train_cfg.get("use_length_bucket", True))
    train_ds = LabelTokenDataset(
        train_samples,
        tokenizer,
        max_length=max_length,
        label_prefix=label_prefix,
        label_schema=label_schema,
        coverage_enabled=coverage_enabled,
        coverage_label_prefix=coverage_label_prefix,
    )
    val_ds = LabelTokenDataset(
        val_samples,
        tokenizer,
        max_length=max_length,
        label_prefix=label_prefix,
        label_schema=label_schema,
        coverage_enabled=coverage_enabled,
        coverage_label_prefix=coverage_label_prefix,
    )
    collator = LabelTokenCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    num_workers = int(train_cfg.get("dataloader_num_workers", 0))
    train_dl = build_dataloader(
        train_ds,
        collator=collator,
        batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        num_workers=num_workers,
        shuffle=True,
        use_length_bucket=use_length_bucket,
    )
    val_dl = build_dataloader(
        val_ds,
        collator=collator,
        batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
        use_length_bucket=False,
    )

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found. Check full fine-tuning or LoRA configuration.")
    if accelerator.is_main_process and lora_enabled(train_cfg):
        active_logger.info("[INFO] Optimizer will update LoRA/trainable parameters only.")

    optimizer = AdamW(
        trainable_parameters,
        lr=float(train_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        fused=torch.cuda.is_available(),
    )

    num_epochs = int(math.ceil(float(train_cfg.get("num_train_epochs", 2.0))))
    effective_grad_accum_steps = int(accelerator.gradient_accumulation_steps)
    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if ds_plugin is not None:
        ds_cfg = getattr(ds_plugin, "deepspeed_config", {}) or {}
        ds_grad_accum_steps = ds_cfg.get("gradient_accumulation_steps")
        if ds_grad_accum_steps is not None:
            effective_grad_accum_steps = int(ds_grad_accum_steps)
    cfg_grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", effective_grad_accum_steps))
    if accelerator.is_main_process and effective_grad_accum_steps != cfg_grad_accum_steps:
        active_logger.warning(
            "[WARN] gradient_accumulation_steps mismatch detected: sft_train=%d, effective=%d. "
            "Using effective value for max_train_steps/progress bar.",
            cfg_grad_accum_steps,
            effective_grad_accum_steps,
        )
    pre_prepare_train_dl_len = len(train_dl)
    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)
    post_prepare_train_dl_len = len(train_dl)
    if accelerator.is_main_process:
        active_logger.info(
            "[INFO] dataloader length around accelerator.prepare: before=%d, after=%d",
            pre_prepare_train_dl_len,
            post_prepare_train_dl_len,
        )
    if accelerator.is_main_process and post_prepare_train_dl_len != pre_prepare_train_dl_len:
        active_logger.warning(
            "[WARN] len(train_dl) changed after accelerator.prepare: before=%d, after=%d. "
            "This may indicate duplicated sharding/re-partitioning across distributed samplers.",
            pre_prepare_train_dl_len,
            post_prepare_train_dl_len,
        )

    update_steps_per_epoch = max(1, math.ceil(post_prepare_train_dl_len / effective_grad_accum_steps))
    max_train_steps = num_epochs * update_steps_per_epoch
    warmup_steps = int(max_train_steps * float(train_cfg.get("warmup_ratio", 0.03)))
    lr_scheduler_kwargs = train_cfg.get("lr_scheduler_kwargs", {}) or {}
    scheduler = get_scheduler(
        name=str(train_cfg.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
        scheduler_specific_kwargs=dict(lr_scheduler_kwargs),
    )
    scheduler = accelerator.prepare(scheduler)
    if accelerator.is_main_process:
        active_logger.info(
            "[INFO] train progress setup (post-prepare): num_epochs=%d, len(train_dl)=%d, "
            "effective_grad_accum_steps=%d, update_steps_per_epoch=%d, max_train_steps=%d, "
            "warmup_steps=%d, lr_scheduler_type=%s, lr_scheduler_kwargs=%s",
            num_epochs,
            post_prepare_train_dl_len,
            effective_grad_accum_steps,
            update_steps_per_epoch,
            max_train_steps,
            warmup_steps,
            str(train_cfg.get("lr_scheduler_type", "cosine")),
            dict(lr_scheduler_kwargs),
        )

    logging_steps = int(train_cfg.get("logging_steps", 20))
    eval_steps = int(train_cfg.get("eval_steps", 500))
    save_steps = int(train_cfg.get("save_steps", 500))
    save_latest_state = _env_bool(
        "SAVE_LATEST_TRAIN_STATE",
        default=_bool_cfg(train_cfg.get("save_latest_state", False), default=False),
    )
    resume_latest_state = _env_bool(
        "RESUME_LATEST_TRAIN_STATE",
        default=_bool_cfg(train_cfg.get("resume_latest_state", save_latest_state), default=save_latest_state),
    )
    latest_state_save_steps = int(train_cfg.get("latest_state_save_steps", save_steps))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    empty_cache_steps = int(train_cfg.get("empty_cache_steps", 0))
    empty_cache_on_eval = bool(train_cfg.get("empty_cache_on_eval", False))
    empty_cache_on_save = bool(train_cfg.get("empty_cache_on_save", False))
    patience = int(train_cfg.get("early_stopping_patience", 0))
    best_checkpoint_policy = _best_checkpoint_policy(train_cfg)
    one_se_policy = best_checkpoint_policy in {"one_standard_error_macro_f1", "one_standard_error", "one_se_macro_f1"}
    bootstrap_samples, bootstrap_seed = _macro_f1_bootstrap_cfg(train_cfg)

    global_step = 0
    best_score = float("-inf")
    best_step: int | None = None
    eval_history: list[dict[str, Any]] = []
    no_improve_count = 0
    should_stop = False
    interrupted = False
    resume_epoch = 0
    resume_next_batch_index = 0

    resume_meta = _load_latest_training_state(
        accelerator=accelerator,
        output_dir=output_dir,
        train_cfg=train_cfg,
        active_logger=active_logger,
        enabled=resume_latest_state,
    )
    if resume_meta:
        global_step = int(resume_meta.get("global_step", 0))
        resume_epoch = int(resume_meta.get("epoch", 0))
        resume_next_batch_index = int(resume_meta.get("next_batch_index", 0))
        if resume_meta.get("best_score") is not None:
            best_score = float(resume_meta.get("best_score"))
        no_improve_count = int(resume_meta.get("no_improve_count", 0))
        saved_max_steps = int(resume_meta.get("max_train_steps", max_train_steps))
        if accelerator.is_main_process and saved_max_steps != max_train_steps:
            active_logger.warning(
                "[checkpoint] resumed state was created with max_train_steps=%d, current max_train_steps=%d",
                saved_max_steps,
                max_train_steps,
            )

    progress_bar = tqdm(
        total=max_train_steps,
        initial=min(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    stop_request = _install_stop_signal_handlers(active_logger)

    def run_eval_and_maybe_save_best(step: int) -> None:
        nonlocal best_score, best_step, no_improve_count, should_stop
        eval_metrics = _evaluate_label_token(
            model=model,
            dataloader=val_dl,
            accelerator=accelerator,
            label_token_ids=label_token_ids,
            class_weights=class_weights,
            coverage_label_token_ids=coverage_label_token_ids,
            coverage_class_weights=coverage_class_weights,
            train_cfg=train_cfg,
            label_prefix=label_prefix,
            labels=labels,
            letter_order=letter_order,
            eval_logger=active_logger,
            log_predictions_limit=int(train_cfg.get("eval_log_predictions", 0)),
            logit_adjust_cfg=logit_adjust_cfg,
            global_step=step,
            max_train_steps=max_train_steps,
        )
        score = _checkpoint_selection_score(eval_metrics, train_cfg)
        macro_f1 = float(eval_metrics["macro_f1"])
        true_side = _true_side_macro_f1(eval_metrics)
        per_class = eval_metrics.get("per_class", {})
        macro_f1_se = 0.0
        if one_se_policy:
            macro_f1_se = macro_f1_bootstrap_se_from_records(
                eval_metrics.get("prediction_records", []),
                labels=labels,
                n_bootstrap=bootstrap_samples,
                seed=bootstrap_seed + int(step),
            )

        if accelerator.is_main_process:
            active_logger.info(
                "[eval] step=%d loss=%.4f ce_loss=%.4f ordinal_loss=%.4f "
                "accuracy=%.4f macro_f1=%.4f macro_f1_se=%.4f true_side_macro_f1=%.4f "
                "checkpoint_selection_score=%.4f "
                "mae=%.4f mae_norm=%.4f extreme_err=%.4f",
                step,
                float(eval_metrics.get("eval_loss", float("nan"))),
                float(eval_metrics.get("eval_ce_loss", float("nan"))),
                float(eval_metrics.get("eval_ordinal_loss", float("nan"))),
                float(eval_metrics["accuracy"]),
                macro_f1,
                macro_f1_se,
                true_side,
                score,
                float(eval_metrics.get("ordinal_mae", float("nan"))),
                float(eval_metrics.get("ordinal_mae_norm", float("nan"))),
                float(eval_metrics.get("extreme_error_rate", float("nan"))),
            )
            if isinstance(per_class, dict) and per_class:
                active_logger.info("[eval] per_class:")
                for label in sorted(per_class.keys()):
                    label_metrics = per_class[label]
                    if isinstance(label_metrics, dict):
                        active_logger.info(
                            "  - %s: P=%.4f R=%.4f F1=%.4f",
                            label,
                            float(label_metrics.get("precision", 0.0)),
                            float(label_metrics.get("recall", 0.0)),
                            float(label_metrics.get("f1", 0.0)),
                        )

        log_metrics(
            accelerator,
            {
                "eval/loss": float(eval_metrics.get("eval_loss", float("nan"))),
                "eval/ce_loss": float(eval_metrics.get("eval_ce_loss", float("nan"))),
                "eval/ordinal_loss": float(eval_metrics.get("eval_ordinal_loss", float("nan"))),
                "eval/accuracy": float(eval_metrics["accuracy"]),
                "eval/macro_precision": float(eval_metrics["macro_precision"]),
                "eval/macro_recall": float(eval_metrics["macro_recall"]),
                "eval/macro_f1": macro_f1,
                "eval/macro_f1_se": macro_f1_se,
                "eval/true_side_macro_f1": true_side,
                "eval/checkpoint_selection_score": score,
                "eval/parse_error_rate": float(eval_metrics["parse_error_rate"]),
                "eval/ordinal_mae": float(eval_metrics.get("ordinal_mae", float("nan"))),
                "eval/ordinal_mae_norm": float(eval_metrics.get("ordinal_mae_norm", float("nan"))),
                "eval/extreme_error_rate": float(eval_metrics.get("extreme_error_rate", float("nan"))),
            },
            step=step,
            backend=tracking_setup.backend,
        )
        if isinstance(per_class, dict):
            for label, label_metrics in per_class.items():
                if isinstance(label_metrics, dict):
                    log_metrics(
                        accelerator,
                        {
                            f"eval/{label}/precision": float(label_metrics.get("precision", 0.0)),
                            f"eval/{label}/recall": float(label_metrics.get("recall", 0.0)),
                            f"eval/{label}/f1": float(label_metrics.get("f1", 0.0)),
                        },
                        step=step,
                        backend=tracking_setup.backend,
                    )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            step_metrics = {
                "step": step,
                "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
                "eval_ce_loss": float(eval_metrics.get("eval_ce_loss", float("nan"))),
                "eval_ordinal_loss": float(eval_metrics.get("eval_ordinal_loss", float("nan"))),
                "accuracy": float(eval_metrics["accuracy"]),
                "macro_precision": float(eval_metrics["macro_precision"]),
                "macro_recall": float(eval_metrics["macro_recall"]),
                "macro_f1": macro_f1,
                "macro_f1_se": macro_f1_se,
                "true_side_macro_f1": true_side,
                "checkpoint_selection_score": score,
                "parse_error_rate": float(eval_metrics["parse_error_rate"]),
                "ordinal_mae": float(eval_metrics.get("ordinal_mae", float("nan"))),
                "ordinal_mae_norm": float(eval_metrics.get("ordinal_mae_norm", float("nan"))),
                "extreme_error_rate": float(eval_metrics.get("extreme_error_rate", float("nan"))),
                "per_class": per_class,
            }
            for key in ("eval_coverage_ce_loss", "coverage_accuracy", "coverage_macro_f1"):
                if key in eval_metrics:
                    step_metrics[key] = float(eval_metrics[key])
            artifacts = _save_eval_artifacts(
                output_dir=output_dir,
                global_step=step,
                metrics=step_metrics,
                confusion_matrix=eval_metrics["confusion_matrix"],
                confusion_labels=eval_metrics["confusion_labels"],
                prediction_records=eval_metrics.get("prediction_records", []),
                labels=labels,
                eval_root=eval_root,
            )
            active_logger.info(
                "[eval] artifacts saved: metrics=%s confusion=%s",
                artifacts["metrics_path"],
                artifacts["confusion_png_path"],
            )

        if one_se_policy:
            eval_history.append(
                {
                    "checkpoint": f"checkpoint-{int(step)}",
                    "step": int(step),
                    "macro_f1": macro_f1,
                    "macro_f1_se": macro_f1_se,
                    "checkpoint_selection_score": score,
                }
            )
            selected = select_one_standard_error_checkpoint(eval_history)
            selected_step = int(selected["step"])
            fmax = float(selected["one_se_best_macro_f1"])
            if selected_step != best_step:
                if selected_step == int(step):
                    save_model(accelerator, model, tokenizer, output_dir / "best")
                else:
                    _copy_checkpoint_to_best(
                        accelerator=accelerator,
                        source_dir=output_dir / f"checkpoint-{selected_step}",
                        best_dir=output_dir / "best",
                        active_logger=active_logger,
                    )
                if accelerator.is_main_process:
                    (output_dir / "best_checkpoint_selection.json").write_text(
                        json.dumps(
                            {
                                "policy": best_checkpoint_policy,
                                "selected_step": selected_step,
                                "selected_checkpoint": str(selected["checkpoint"]),
                                "selected_macro_f1": float(selected["macro_f1"]),
                                "selected_macro_f1_se": float(selected.get("macro_f1_se", 0.0)),
                                "fmax_checkpoint": str(selected["one_se_best_checkpoint"]),
                                "fmax_macro_f1": fmax,
                                "fmax_macro_f1_se": float(selected["one_se_best_macro_f1_se"]),
                                "threshold": float(selected["one_se_threshold"]),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                best_step = selected_step
            if fmax > best_score:
                best_score = fmax
                no_improve_count = 0
            elif patience > 0:
                no_improve_count += 1
                if no_improve_count >= patience:
                    if accelerator.is_main_process:
                        active_logger.info(
                            "[early-stop] no val one-SE macro-F1 envelope improvement for %d evals, stopping at step=%d",
                            patience,
                            step,
                        )
                    should_stop = True
        elif score > best_score:
            best_score = score
            best_step = int(step)
            save_model(accelerator, model, tokenizer, output_dir / "best")
            no_improve_count = 0
        elif patience > 0:
            no_improve_count += 1
            if no_improve_count >= patience:
                if accelerator.is_main_process:
                    active_logger.info(
                        "[early-stop] no val checkpoint-selection-score improvement for %d evals, stopping at step=%d",
                        patience,
                        step,
                    )
                should_stop = True

        if empty_cache_on_eval:
            maybe_empty_cache(accelerator)

    for epoch in range(resume_epoch, num_epochs):
        model.train()
        epoch_train_dl = train_dl
        start_batch_index = 0
        if epoch == resume_epoch and resume_next_batch_index > 0:
            start_batch_index = min(resume_next_batch_index, len(train_dl))
            epoch_train_dl = accelerator.skip_first_batches(train_dl, start_batch_index)
            if accelerator.is_main_process:
                active_logger.info(
                    "[checkpoint] skipping %d already-processed batches in resumed epoch=%d",
                    start_batch_index,
                    epoch,
                )

        for batch_index, batch in enumerate(epoch_train_dl, start=start_batch_index):
            if stop_request["requested"]:
                interrupted = True
                should_stop = True
                break

            with accelerator.accumulate(model):
                label_logits = _forward_label_logits(model, batch, label_token_ids)
                coverage_label_logits = None
                coverage_gold_ids = None
                if coverage_label_token_ids is not None and "coverage_input_ids" in batch:
                    coverage_gold_ids = batch["coverage_gold_ids"]
                    coverage_label_logits = _forward_label_logits_for_inputs(
                        model,
                        input_ids=batch["coverage_input_ids"],
                        attention_mask=batch["coverage_attention_mask"],
                        label_token_ids=coverage_label_token_ids,
                    )
                losses = _compute_label_token_losses(
                    label_logits=label_logits,
                    gold_ids=batch["gold_ids"],
                    class_weights=class_weights,
                    train_cfg=train_cfg,
                    coverage_label_logits=coverage_label_logits,
                    coverage_gold_ids=coverage_gold_ids,
                    coverage_class_weights=coverage_class_weights,
                    global_step=global_step,
                    max_train_steps=max_train_steps,
                )
                loss = losses["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % logging_steps == 0:
                    train_loss = accelerator.gather_for_metrics(loss.detach().float().unsqueeze(0)).mean().item()
                    train_ce_loss = accelerator.gather_for_metrics(
                        losses["ce_loss"].detach().float().unsqueeze(0)
                    ).mean().item()
                    train_ordinal_loss = accelerator.gather_for_metrics(
                        losses["ordinal_loss"].detach().float().unsqueeze(0)
                    ).mean().item()
                    train_coverage_ce_loss = None
                    if coverage_enabled:
                        train_coverage_ce_loss = accelerator.gather_for_metrics(
                            losses["coverage_ce_loss"].detach().float().unsqueeze(0)
                        ).mean().item()
                    lr = scheduler.get_last_lr()[0]
                    train_log_payload = {
                        "train/loss": train_loss,
                        "train/ce_loss": train_ce_loss,
                        "train/ordinal_loss": train_ordinal_loss,
                        "train/lr": lr,
                        "train/epoch": epoch,
                    }
                    if train_coverage_ce_loss is not None:
                        train_log_payload["train/coverage_ce_loss"] = train_coverage_ce_loss
                    log_metrics(
                        accelerator,
                        train_log_payload,
                        step=global_step,
                        backend=tracking_setup.backend,
                    )

                if global_step % eval_steps == 0:
                    run_eval_and_maybe_save_best(global_step)

                if global_step % save_steps == 0:
                    save_model(accelerator, model, tokenizer, output_dir / f"checkpoint-{global_step}")
                    if empty_cache_on_save:
                        maybe_empty_cache(accelerator)

                if save_latest_state and latest_state_save_steps > 0 and global_step % latest_state_save_steps == 0:
                    _save_latest_training_state(
                        accelerator=accelerator,
                        output_dir=output_dir,
                        train_cfg=train_cfg,
                        epoch=epoch,
                        next_batch_index=batch_index + 1,
                        global_step=global_step,
                        best_score=best_score,
                        no_improve_count=no_improve_count,
                        max_train_steps=max_train_steps,
                        active_logger=active_logger,
                        enabled=True,
                    )

                if empty_cache_steps > 0 and global_step % empty_cache_steps == 0:
                    maybe_empty_cache(accelerator)

                if stop_request["requested"]:
                    interrupted = True
                    should_stop = True

                if global_step >= max_train_steps or should_stop:
                    break

        if should_stop or global_step >= max_train_steps:
            break

    if interrupted:
        _save_latest_training_state(
            accelerator=accelerator,
            output_dir=output_dir,
            train_cfg=train_cfg,
            epoch=epoch,
            next_batch_index=batch_index + 1 if "batch_index" in locals() else 0,
            global_step=global_step,
            best_score=best_score,
            no_improve_count=no_improve_count,
            max_train_steps=max_train_steps,
            active_logger=active_logger,
            enabled=save_latest_state,
        )
        if accelerator.is_main_process:
            active_logger.warning("[signal] exiting after saving latest training state at global_step=%d", global_step)
        try:
            accelerator.end_training()
        except Exception as exc:
            if accelerator.is_main_process:
                active_logger.warning("[signal] accelerator.end_training failed during interrupted exit: %s", exc)
        raise SystemExit(128 + int(stop_request.get("signal") or signal.SIGTERM))

    if best_score == float("-inf"):
        run_eval_and_maybe_save_best(global_step)

    save_model(accelerator, model, tokenizer, output_dir / "final")
    _write_training_complete(
        accelerator=accelerator,
        output_dir=output_dir,
        train_cfg=train_cfg,
        global_step=global_step,
        best_score=best_score,
        active_logger=active_logger,
    )
    _end_training_after_final_checkpoint(accelerator, output_dir, active_logger)


if __name__ == "__main__":
    main()
