"""Train Step4 sequential pointer evidence selector from Stage2 oracle rows."""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from fact_checking.selectors.metrics import (
    build_order_control_trace,
    ranked_indices_from_candidate_pool,
    ranked_indices_from_hybrid,
    random_order_controls,
    reorder_predicted_set,
    summarize_ordered_selection,
)
from fact_checking.selectors.sequential import (
    SEMANTIC_FEATURE_PROFILE_CHOICES,
    SEQUENTIAL_HEAD_FILENAME,
    SHALLOW_FEATURE_PROFILE_CHOICES,
    TARGETED_FEATURE_PROFILE_CHOICES,
    SequentialPointerSelectorModel,
    build_sequential_selection_trace,
    forward_sequential_examples,
    predict_sequential_examples,
    sequential_teacher_forcing_loss,
    summarize_sequential_step_diagnostics,
    teacher_forcing_sequential_logits,
)
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    Stage2OracleExample,
    load_stage2_oracle_examples,
    write_json,
    write_jsonl,
)


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


@dataclass(frozen=True)
class SwanLabRun:
    enabled: bool
    module: Any | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Step4 sequential pointer Stage2 evidence selector.")
    p.add_argument("--train-oracle-results", required=True)
    p.add_argument("--val-oracle-results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="microsoft/deberta-v3-base")
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=2, help="Number of claims per optimizer micro-batch.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--head-learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--seq-loss-weight", type=float, default=1.0)
    p.add_argument("--mask-loss-weight", type=float, default=0.0)
    p.add_argument("--list-hidden-size", type=int, default=256)
    p.add_argument("--list-layers", type=int, default=2)
    p.add_argument("--list-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freeze-pair-encoder", action="store_true")
    p.add_argument("--semantic-feature-profile", default="deep", choices=SEMANTIC_FEATURE_PROFILE_CHOICES)
    p.add_argument("--targeted-feature-profile", default="none", choices=TARGETED_FEATURE_PROFILE_CHOICES)
    p.add_argument("--shallow-feature-profile", default="off", choices=SHALLOW_FEATURE_PROFILE_CHOICES)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-sample-limit", type=int, default=None)
    p.add_argument("--val-sample-limit", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=250, help="Optimizer steps between validation passes.")
    p.add_argument("--early-stopping-patience", type=int, default=6)
    p.add_argument("--early-stopping-metric", default="oracle_rank_ndcg@5")
    p.add_argument("--ddp-find-unused-parameters", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-swanlab", action="store_true", help="Disable SwanLab logging for this selector run.")
    p.add_argument("--swanlab-project", default="fact-checking-stage2-sequential")
    p.add_argument("--swanlab-experiment-name", default=None)
    p.add_argument("--swanlab-workspace", default=None)
    p.add_argument("--swanlab-mode", default=None)
    p.add_argument("--swanlab-logdir", default=None)
    p.add_argument("--swanlab-tags", default="selector,sequential,step4")
    p.add_argument("--swanlab-description", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    distributed = _init_distributed(args)
    is_main = distributed.rank == 0
    _set_seed(int(args.seed) + distributed.rank)
    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    _barrier(distributed)

    train_examples = load_stage2_oracle_examples(
        args.train_oracle_results,
        expected_fingerprint=args.expected_chunk_mmr_fingerprint,
        max_candidates=args.max_candidates,
        top_k=args.top_k,
        filter_policy=args.filter_policy,
        min_margin=args.min_margin,
        sample_limit=args.train_sample_limit,
    )
    val_examples = load_stage2_oracle_examples(
        args.val_oracle_results,
        expected_fingerprint=args.expected_chunk_mmr_fingerprint,
        max_candidates=args.max_candidates,
        top_k=args.top_k,
        filter_policy=args.filter_policy,
        min_margin=args.min_margin,
        sample_limit=args.val_sample_limit,
    )
    if not train_examples:
        raise ValueError("No train examples after Stage2 audit/filtering.")
    if not val_examples:
        raise ValueError("No val examples after Stage2 audit/filtering.")

    swanlab_run = _init_swanlab(args, out_dir, is_main=is_main)

    device = distributed.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    raw_model = SequentialPointerSelectorModel(
        args.model_name,
        hidden_size=int(args.list_hidden_size),
        num_layers=int(args.list_layers),
        num_attention_heads=int(args.list_heads),
        dropout=float(args.dropout),
        max_candidates=int(args.max_candidates),
        semantic_feature_profile=str(args.semantic_feature_profile),
        targeted_feature_profile=str(args.targeted_feature_profile),
        shallow_feature_profile=str(args.shallow_feature_profile),
    )
    if bool(args.freeze_pair_encoder):
        for param in raw_model.encoder.parameters():
            param.requires_grad = False
    raw_model.to(device)

    optimizer = _build_optimizer(raw_model, args)
    model: torch.nn.Module = raw_model
    if distributed.enabled:
        model = DDP(
            raw_model,
            device_ids=[distributed.local_rank] if device.type == "cuda" else None,
            output_device=distributed.local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )

    local_train_size = _local_epoch_size(len(train_examples), distributed.world_size)
    micro_batches_per_epoch = max(math.ceil(local_train_size / max(int(args.batch_size), 1)), 1)
    total_optimizer_steps = max(
        math.ceil(micro_batches_per_epoch * int(args.epochs) / max(int(args.gradient_accumulation_steps), 1)),
        1,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_optimizer_steps * float(args.warmup_ratio)),
        num_training_steps=total_optimizer_steps,
    )

    best_metric = -1.0
    stale = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, int(args.epochs) + 1):
        epoch_examples = _distributed_epoch_examples(
            train_examples,
            epoch=epoch,
            seed=int(args.seed),
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        iterator = tqdm(
            _batches(epoch_examples, int(args.batch_size)),
            total=micro_batches_per_epoch,
            desc=f"sequential train epoch {epoch}",
            unit="batch",
            dynamic_ncols=True,
            disable=args.no_progress or not is_main,
        )
        for micro_step, batch in enumerate(iterator, start=1):
            model.train()
            output = forward_sequential_examples(
                model,
                tokenizer,
                batch,
                device=device,
                max_length=int(args.max_length),
            )
            logits = teacher_forcing_sequential_logits(
                model,
                output,
                [example.selected_indices for example in batch],
                top_k=int(args.top_k),
            )
            loss, parts = sequential_teacher_forcing_loss(
                logits,
                [example.selected_indices for example in batch],
                candidate_mask=output.candidate_mask,
                top_k=int(args.top_k),
                mask_loss_weight=float(args.mask_loss_weight),
            )
            loss = float(args.seq_loss_weight) * loss
            scaled_loss = loss / max(int(args.gradient_accumulation_steps), 1)
            scaled_loss.backward()

            should_step = (
                micro_step % max(int(args.gradient_accumulation_steps), 1) == 0
                or micro_step == micro_batches_per_epoch
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                parts["epoch"] = epoch
                parts["global_step"] = global_step
                parts["weighted_loss"] = float(loss.detach().cpu())
                if is_main:
                    history.append(parts)
                    iterator.set_postfix(loss=f"{parts['loss']:.4f}", ce=f"{parts['sequence_ce_loss']:.4f}")
                    _log_swanlab(
                        swanlab_run,
                        {
                            "train/loss": parts.get("loss"),
                            "train/sequence_ce_loss": parts.get("sequence_ce_loss"),
                            "train/mask_loss": parts.get("mask_loss"),
                            "train/n_steps": parts.get("n_steps"),
                            "train/weighted_loss": parts.get("weighted_loss"),
                            "train/epoch": parts.get("epoch"),
                            **_optimizer_lr_metrics(optimizer),
                        },
                        step=global_step,
                    )

                if int(args.eval_every) > 0 and global_step % int(args.eval_every) == 0:
                    _barrier(distributed)
                    metric_value = -1.0
                    if is_main:
                        metric_value = _validate_and_maybe_save(
                            raw_model,
                            tokenizer,
                            val_examples,
                            args,
                            out_dir,
                            history,
                            val_history,
                            swanlab_run,
                            global_step=global_step,
                            best_metric=best_metric,
                        )
                    metric_value = _broadcast_float(metric_value, distributed)
                    if metric_value > best_metric + 1e-8:
                        best_metric = metric_value
                        stale = 0
                    else:
                        stale += 1
                        if stale >= int(args.early_stopping_patience):
                            break
        if stale >= int(args.early_stopping_patience):
            break

    _barrier(distributed)
    final_metric = -1.0
    if is_main:
        final_metric = _validate_and_maybe_save(
            raw_model,
            tokenizer,
            val_examples,
            args,
            out_dir,
            history,
            val_history,
            swanlab_run,
            global_step=global_step,
            best_metric=best_metric,
            force_save=best_metric < 0,
        )
        print(f"Best/Final {args.early_stopping_metric}: {max(best_metric, final_metric):.4f}")
        print(f"Saved sequential selector under: {out_dir}")
        _finish_swanlab(swanlab_run)
    _barrier(distributed)
    _cleanup_distributed(distributed)


def _validate_and_maybe_save(
    model: SequentialPointerSelectorModel,
    tokenizer: Any,
    val_examples: list[Stage2OracleExample],
    args: argparse.Namespace,
    out_dir: Path,
    history: list[dict[str, Any]],
    val_history: list[dict[str, Any]],
    swanlab_run: SwanLabRun,
    *,
    global_step: int,
    best_metric: float,
    force_save: bool = False,
) -> float:
    traces, control_metrics = evaluate_model(
        model,
        tokenizer,
        val_examples,
        device=next(model.parameters()).device,
        max_length=int(args.max_length),
        top_k=int(args.top_k),
        no_progress=bool(args.no_progress),
    )
    metrics = summarize_ordered_selection(traces)
    step_metrics = summarize_sequential_step_diagnostics(traces)
    metric_value = float(metrics.get(args.early_stopping_metric, 0.0))
    validation_record = {
        "global_step": int(global_step),
        "metric_name": str(args.early_stopping_metric),
        "metric_value": metric_value,
        "selector": metrics,
        "controls": control_metrics,
        "step_diagnostics": step_metrics,
    }
    if val_history and int(val_history[-1].get("global_step", -1)) == int(global_step):
        val_history[-1] = validation_record
    else:
        val_history.append(validation_record)
    write_jsonl(out_dir / "val_history.jsonl", val_history)
    _log_validation_to_swanlab(swanlab_run, validation_record, step=global_step)
    if force_save or metric_value > best_metric + 1e-8:
        model.encoder.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        torch.save(model.selector_head_state_dict(), out_dir / SEQUENTIAL_HEAD_FILENAME)
        metadata = {
            "selector_type": "sequential_pointer",
            "base_model": args.model_name,
            "train_oracle_results": args.train_oracle_results,
            "val_oracle_results": args.val_oracle_results,
            "chunk_mmr_fingerprint": args.expected_chunk_mmr_fingerprint,
            "candidate_pool_policy": f"saved_stage2_candidate_pool_top{args.max_candidates}",
            "filter_policy": args.filter_policy,
            "min_margin": float(args.min_margin),
            "top_k": int(args.top_k),
            "max_candidates": int(args.max_candidates),
            "max_length": int(args.max_length),
            "model_config": model.model_config(),
            "losses": {
                "teacher_forcing_ce": float(args.seq_loss_weight),
                "selected_mask_bce": float(args.mask_loss_weight),
            },
            "semantic_feature_profile": str(args.semantic_feature_profile),
            "targeted_feature_profile": str(args.targeted_feature_profile),
            "shallow_feature_profile": str(args.shallow_feature_profile),
            "freeze_pair_encoder": bool(args.freeze_pair_encoder),
            "seed": int(args.seed),
            "best_metric": args.early_stopping_metric,
            "best_metric_value": metric_value,
            "global_step": int(global_step),
        }
        write_json(out_dir / "metadata.json", metadata)
        write_jsonl(out_dir / "val_trace.jsonl", traces)
        write_json(
            out_dir / "selection_metrics.json",
            {
                "selector": metrics,
                "controls": control_metrics,
                "step_diagnostics": step_metrics,
                "history": history,
                "val_history": val_history,
                "metadata": metadata,
            },
        )
    return metric_value


@torch.inference_mode()
def evaluate_model(
    model: SequentialPointerSelectorModel,
    tokenizer: Any,
    examples: list[Stage2OracleExample],
    *,
    device: torch.device,
    max_length: int,
    top_k: int,
    no_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    traces: list[dict[str, Any]] = []
    hybrid_traces: list[dict[str, Any]] = []
    candidate_order_traces: list[dict[str, Any]] = []
    same_set_hybrid_traces: list[dict[str, Any]] = []
    same_set_candidate_traces: list[dict[str, Any]] = []
    random_traces: list[dict[str, Any]] = []

    for batch in tqdm(
        _batches(examples, 8),
        total=max(math.ceil(len(examples) / 8), 1),
        desc="sequential val",
        unit="batch",
        dynamic_ncols=True,
        disable=no_progress,
    ):
        predictions = predict_sequential_examples(
            model,
            tokenizer,
            batch,
            device=device,
            max_length=max_length,
            top_k=top_k,
        )
        for example, prediction in zip(batch, predictions):
            trace = build_sequential_selection_trace(
                example,
                prediction,
                selector_name="sequential_pointer",
                top_k=top_k,
            )
            traces.append(trace)

            hybrid_traces.append(
                build_order_control_trace(
                    trace,
                    ranked_indices_from_hybrid(example, top_k=top_k),
                    selector_name="hybrid_score_top5",
                    top_k=top_k,
                )
            )
            candidate_order_traces.append(
                build_order_control_trace(
                    trace,
                    ranked_indices_from_candidate_pool(example, top_k=top_k),
                    selector_name="candidate_pool_order_top5",
                    top_k=top_k,
                )
            )
            predicted = [int(idx) for idx in trace["selector_ordered_indices"]]
            same_set_hybrid_traces.append(
                build_order_control_trace(
                    trace,
                    reorder_predicted_set(predicted, example=example, mode="hybrid_order"),
                    selector_name="same_set_hybrid_order",
                    top_k=top_k,
                )
            )
            same_set_candidate_traces.append(
                build_order_control_trace(
                    trace,
                    reorder_predicted_set(predicted, example=example, mode="candidate_pool_order"),
                    selector_name="same_set_candidate_pool_order",
                    top_k=top_k,
                )
            )
            random_traces.extend(
                random_order_controls(
                    predicted,
                    example=example,
                    seeds=[0, 1, 2, 3, 4],
                    top_k=top_k,
                )
            )

    controls = {
        "hybrid_score_top5": summarize_ordered_selection(hybrid_traces),
        "candidate_pool_order_top5": summarize_ordered_selection(candidate_order_traces),
        "same_set_hybrid_order": summarize_ordered_selection(same_set_hybrid_traces),
        "same_set_candidate_pool_order": summarize_ordered_selection(same_set_candidate_traces),
        "same_set_random_order_mean": summarize_ordered_selection(random_traces),
    }
    return traces, controls


def _init_swanlab(args: argparse.Namespace, out_dir: Path, *, is_main: bool) -> SwanLabRun:
    if not is_main or bool(args.no_swanlab):
        return SwanLabRun(enabled=False)
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "SwanLab logging is enabled but `swanlab` is not installed. "
            "Install it or pass --no-swanlab."
        ) from exc

    experiment_name = str(args.swanlab_experiment_name or Path(args.output_dir).name)
    tags = [
        item.strip()
        for item in str(args.swanlab_tags or "").split(",")
        if item.strip()
    ]
    config = {
        key: _json_safe_value(value)
        for key, value in vars(args).items()
        if key not in {"no_progress"}
    }
    config["selector_type"] = "sequential_pointer"
    config["output_dir"] = str(out_dir)

    init_kwargs: dict[str, Any] = {
        "project": str(args.swanlab_project),
        "experiment_name": experiment_name,
        "config": config,
    }
    if args.swanlab_workspace:
        init_kwargs["workspace"] = str(args.swanlab_workspace)
    if args.swanlab_mode:
        init_kwargs["mode"] = str(args.swanlab_mode)
    if args.swanlab_logdir:
        init_kwargs["logdir"] = str(args.swanlab_logdir)
    if tags:
        init_kwargs["tags"] = tags
    if args.swanlab_description:
        init_kwargs["description"] = str(args.swanlab_description)

    try:
        swanlab.init(**init_kwargs)
    except TypeError:
        minimal_kwargs = {
            "project": str(args.swanlab_project),
            "experiment_name": experiment_name,
            "config": config,
        }
        try:
            swanlab.init(**minimal_kwargs)
        except TypeError:
            swanlab.init(project=str(args.swanlab_project), experiment_name=experiment_name)
    return SwanLabRun(enabled=True, module=swanlab)


def _finish_swanlab(run: SwanLabRun) -> None:
    if not run.enabled or run.module is None:
        return
    finish = getattr(run.module, "finish", None)
    if callable(finish):
        finish()


def _log_swanlab(run: SwanLabRun, values: dict[str, Any], *, step: int) -> None:
    if not run.enabled or run.module is None:
        return
    payload = {
        key: scalar
        for key, value in values.items()
        for scalar in [_as_loggable_scalar(value)]
        if scalar is not None
    }
    if payload:
        run.module.log(payload, step=int(step))


def _log_validation_to_swanlab(
    run: SwanLabRun,
    validation_record: dict[str, Any],
    *,
    step: int,
) -> None:
    if not run.enabled:
        return
    metric_name = str(validation_record.get("metric_name") or "selected_metric")
    payload: dict[str, Any] = {
        "val/selected_metric": validation_record.get("metric_value"),
        f"val/selected_metric/{metric_name}": validation_record.get("metric_value"),
    }
    payload.update(_flatten_numeric_metrics("val/selector", validation_record.get("selector", {})))
    payload.update(_flatten_numeric_metrics("val/controls", validation_record.get("controls", {})))
    payload.update(
        _flatten_numeric_metrics("val/step_diagnostics", validation_record.get("step_diagnostics", {}))
    )
    _log_swanlab(run, payload, step=step)


def _flatten_numeric_metrics(prefix: str, value: Any) -> dict[str, Any]:
    scalar = _as_loggable_scalar(value)
    if scalar is not None:
        return {prefix: scalar}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            child_prefix = f"{prefix}/{_metric_key(key)}"
            out.update(_flatten_numeric_metrics(child_prefix, item))
        return out
    return {}


def _metric_key(value: Any) -> str:
    return str(value).strip().replace(" ", "_")


def _as_loggable_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        scalar = float(value.detach().float().cpu().item())
        return scalar if math.isfinite(scalar) else None
    return None


def _json_safe_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return str(value)


def _optimizer_lr_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
    metrics = {f"train/lr/group_{idx}": value for idx, value in enumerate(lrs)}
    if lrs:
        metrics["train/lr/min"] = min(lrs)
        metrics["train/lr/max"] = max(lrs)
    return metrics


def _build_optimizer(model: SequentialPointerSelectorModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
    head_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("encoder.")
    ]
    groups: list[dict[str, Any]] = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": float(args.learning_rate)})
    if head_params:
        groups.append({"params": head_params, "lr": float(args.head_learning_rate)})
    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))


def _init_distributed(args: argparse.Namespace) -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        return DistributedState(False, 0, 0, 1, device)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return DistributedState(True, rank, local_rank, world_size, device)


def _cleanup_distributed(distributed: DistributedState) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _barrier(distributed: DistributedState) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.barrier()


def _broadcast_float(value: float, distributed: DistributedState) -> float:
    if not distributed.enabled:
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float32, device=distributed.device)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


def _local_epoch_size(n_examples: int, world_size: int) -> int:
    return max(math.ceil(max(int(n_examples), 1) / max(int(world_size), 1)), 1)


def _distributed_epoch_examples(
    examples: list[Stage2OracleExample],
    *,
    epoch: int,
    seed: int,
    rank: int,
    world_size: int,
) -> list[Stage2OracleExample]:
    items = list(examples)
    rng = random.Random(int(seed) + int(epoch))
    rng.shuffle(items)
    if world_size <= 1:
        return items

    target_size = _local_epoch_size(len(items), world_size) * world_size
    if len(items) < target_size:
        repeat = target_size - len(items)
        items.extend(items[:repeat])
    return items[int(rank) : target_size : int(world_size)]


def _batches(items: list[Stage2OracleExample], batch_size: int):
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
