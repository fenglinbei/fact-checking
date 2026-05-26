#!/usr/bin/env python3
"""Train a frozen-encoder listwise utility scorer from step-0 VIG rows."""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from fact_checking.selectors.listwise import (
    FEATURE_ABLATION_CHOICES,
    LISTWISE_HEAD_FILENAME,
    SetAwareListwiseSelectorModel,
    forward_listwise_groups,
    normalize_feature_ablation,
)
from fact_checking.selectors.metrics import (
    build_order_control_trace,
    build_selection_trace,
    ranked_indices_from_candidate_pool,
    ranked_indices_from_hybrid,
    summarize_ordered_selection,
)
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    load_stage2_oracle_examples,
    write_json,
    write_jsonl,
)
from fact_checking.selectors.utility_listwise import (
    DEFAULT_UTILITY_POSITIVE_BEST_MARGIN,
    DEFAULT_UTILITY_SOFT_TAU,
    UtilityListwiseExample,
    load_utility_listwise_examples,
    permute_utility_listwise_example,
    utility_listwise_loss,
    utility_rank_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v0 utility listwise selector from saved VIG rows.")
    p.add_argument("--train-vig-cache", default="outputs/selectors/vig_utility/saved_step_train/vig_records_train.jsonl")
    p.add_argument("--val-vig-cache", default="outputs/selectors/vig_utility/saved_step_val/vig_records_val.jsonl")
    p.add_argument("--train-oracle-results", default="outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl")
    p.add_argument("--val-oracle-results", default="outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="/data/models/deberta-v3-base/")
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--pairwise-weight", type=float, default=1.0)
    p.add_argument("--soft-ce-weight", type=float, default=0.2)
    p.add_argument("--bce-weight", type=float, default=0.2)
    p.add_argument("--soft-tau", type=float, default=DEFAULT_UTILITY_SOFT_TAU)
    p.add_argument("--positive-best-margin", type=float, default=DEFAULT_UTILITY_POSITIVE_BEST_MARGIN)
    p.add_argument("--list-hidden-size", type=int, default=256)
    p.add_argument("--list-layers", type=int, default=2)
    p.add_argument("--list-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--feature-ablation",
        default="none",
        choices=FEATURE_ABLATION_CHOICES,
        help="Numeric/rank-prior ablation. Use no_rank_prior to zero rank/index features.",
    )
    p.add_argument(
        "--use-rank-embedding",
        default="auto",
        choices=["auto", "true", "false", "1", "0", "yes", "no"],
        help="Whether to add rank embeddings in the set head. auto disables them for no_rank_prior.",
    )
    p.add_argument(
        "--shuffle-probability",
        "--train-candidate-shuffle-probability",
        dest="shuffle_probability",
        type=float,
        default=0.0,
        help="Probability of permuting candidate order for each train example before scoring.",
    )
    p.add_argument("--selector-name", default="utility_listwise_v0")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--train-sample-limit", type=int, default=None)
    p.add_argument("--val-sample-limit", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--early-stopping-metric", default="jaccard@5")
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    _set_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_oracle = load_stage2_oracle_examples(
        args.train_oracle_results,
        expected_fingerprint=str(args.expected_chunk_mmr_fingerprint),
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=str(args.filter_policy),
        min_margin=float(args.min_margin),
        sample_limit=args.train_sample_limit,
    )
    val_oracle = load_stage2_oracle_examples(
        args.val_oracle_results,
        expected_fingerprint=str(args.expected_chunk_mmr_fingerprint),
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=str(args.filter_policy),
        min_margin=float(args.min_margin),
        sample_limit=args.val_sample_limit,
    )
    train_examples = load_utility_listwise_examples(
        args.train_vig_cache,
        train_oracle,
        split="train",
        max_candidates=int(args.max_candidates),
        sample_limit=args.train_sample_limit,
    )
    val_examples = load_utility_listwise_examples(
        args.val_vig_cache,
        val_oracle,
        split="val",
        max_candidates=int(args.max_candidates),
        sample_limit=args.val_sample_limit,
    )
    if not train_examples:
        raise ValueError("No utility train examples were built.")
    if not val_examples:
        raise ValueError("No utility val examples were built.")

    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    feature_ablation = normalize_feature_ablation(args.feature_ablation)
    model = SetAwareListwiseSelectorModel(
        args.model_name,
        hidden_size=int(args.list_hidden_size),
        num_layers=int(args.list_layers),
        num_attention_heads=int(args.list_heads),
        dropout=float(args.dropout),
        max_candidates=int(args.max_candidates),
        feature_ablation=feature_ablation,
        use_rank_embedding=_optional_bool(args.use_rank_embedding),
    )
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.to(device)
    model.float()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    steps_per_epoch = max(math.ceil(len(train_examples) / max(int(args.batch_size), 1)), 1)
    total_steps = max(steps_per_epoch * max(int(args.epochs), 1), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(args.warmup_ratio)),
        num_training_steps=total_steps,
    )

    metadata = _metadata(args, train_examples, val_examples, total_steps=total_steps)
    metadata["model_config"] = model.model_config()
    write_json(out_dir / "metadata.json", metadata)

    train_history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_record: dict[str, Any] | None = None
    global_step = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_examples = list(train_examples)
        random.Random(int(args.seed) + epoch).shuffle(epoch_examples)
        iterator = tqdm(
            _batches(epoch_examples, int(args.batch_size)),
            total=steps_per_epoch,
            desc=f"utility-listwise train epoch {epoch}",
            unit="batch",
            dynamic_ncols=True,
            disable=bool(args.no_progress),
        )
        for raw_batch in iterator:
            model.train()
            batch = _maybe_permute_utility_batch(
                raw_batch,
                probability=float(args.shuffle_probability),
            )
            score_groups = _forward_utility_examples(
                model,
                tokenizer,
                batch,
                device=device,
                max_length=int(args.max_length),
                max_candidates=int(args.max_candidates),
            )
            loss, parts = utility_listwise_loss(
                score_groups,
                [example.delta_margins for example in batch],
                pairwise_weight=float(args.pairwise_weight),
                soft_ce_weight=float(args.soft_ce_weight),
                bce_weight=float(args.bce_weight),
                soft_tau=float(args.soft_tau),
                positive_best_margin=float(args.positive_best_margin),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            scheduler.step()
            global_step += 1

            record = {
                "global_step": int(global_step),
                "epoch": int(epoch),
                "lr": float(scheduler.get_last_lr()[0]),
                **parts,
            }
            train_history.append(record)
            iterator.set_postfix(loss=f"{record['loss']:.4f}", pair=f"{record['pairwise_accuracy']:.3f}")

            if int(args.eval_every) > 0 and global_step % int(args.eval_every) == 0:
                best_metric, best_record = _validate_and_save_if_best(
                    model,
                    tokenizer,
                    val_examples,
                    args,
                    out_dir,
                    metadata,
                    train_history,
                    val_history,
                    global_step=global_step,
                    epoch=epoch,
                    best_metric=best_metric,
                    best_record=best_record,
                )

    best_metric, best_record = _validate_and_save_if_best(
        model,
        tokenizer,
        val_examples,
        args,
        out_dir,
        metadata,
        train_history,
        val_history,
        global_step=global_step,
        epoch=int(args.epochs),
        best_metric=best_metric,
        best_record=best_record,
        force_save=best_record is None,
    )
    if best_record is not None:
        metadata.update(
            {
                "best_metric": best_record["best_metric"],
                "best_metric_value": float(best_record["best_metric_value"]),
                "global_step": int(best_record["global_step"]),
                "epoch": int(best_record["epoch"]),
            }
        )
    metadata["best"] = best_record
    metadata["elapsed_seconds"] = round(time.time() - started_at, 3)
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "training_metrics.json", {"metadata": metadata, "best": best_record})
    write_jsonl(out_dir / "train_history.jsonl", train_history)
    write_jsonl(out_dir / "val_history.jsonl", val_history)
    print(f"Best {args.early_stopping_metric}: {best_metric:.4f}")
    print(f"Saved utility listwise selector under: {out_dir}")


def _validate_and_save_if_best(
    model: SetAwareListwiseSelectorModel,
    tokenizer: Any,
    val_examples: list[UtilityListwiseExample],
    args: argparse.Namespace,
    out_dir: Path,
    metadata: dict[str, Any],
    train_history: list[dict[str, Any]],
    val_history: list[dict[str, Any]],
    *,
    global_step: int,
    epoch: int,
    best_metric: float,
    best_record: dict[str, Any] | None,
    force_save: bool = False,
) -> tuple[float, dict[str, Any] | None]:
    traces, row_metrics, controls = evaluate_model(
        model,
        tokenizer,
        val_examples,
        device=next(model.parameters()).device,
        max_length=int(args.max_length),
        max_candidates=int(args.max_candidates),
        eval_batch_size=int(args.eval_batch_size),
        top_k=int(args.top_k),
        selector_name=str(args.selector_name),
        no_progress=bool(args.no_progress),
    )
    selection_metrics = summarize_ordered_selection(traces)
    metric_value = float(selection_metrics.get(str(args.early_stopping_metric), 0.0))
    record = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "best_metric": str(args.early_stopping_metric),
        "best_metric_value": metric_value,
        "row_metrics": row_metrics,
        "selector": selection_metrics,
        "controls": controls,
    }
    val_history.append(record)
    print(
        "eval step={step} {metric}={value:.4f} jaccard={jaccard:.4f} "
        "pair_acc={pair:.4f} top1_delta={top1:.4f}".format(
            step=global_step,
            metric=args.early_stopping_metric,
            value=metric_value,
            jaccard=float(selection_metrics.get("jaccard@5", 0.0)),
            pair=float(row_metrics.get("pairwise_accuracy", 0.0)),
            top1=float(row_metrics.get("top1_delta_match", 0.0)),
        )
    )
    should_save = force_save or metric_value > best_metric + 1.0e-8
    if not should_save:
        return best_metric, best_record

    model.encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(model.selector_head_state_dict(), out_dir / LISTWISE_HEAD_FILENAME)
    saved_metadata = dict(metadata)
    saved_metadata.update(
        {
            "best_metric": str(args.early_stopping_metric),
            "best_metric_value": metric_value,
            "global_step": int(global_step),
            "epoch": int(epoch),
        }
    )
    write_json(out_dir / "metadata.json", saved_metadata)
    write_jsonl(out_dir / "val_trace.jsonl", traces)
    write_json(
        out_dir / "selection_metrics.json",
        {
            "selector": selection_metrics,
            "row_metrics": row_metrics,
            "controls": controls,
            "history": val_history,
            "train_history": train_history,
            "metadata": saved_metadata,
        },
    )
    return metric_value, record


@torch.inference_mode()
def evaluate_model(
    model: SetAwareListwiseSelectorModel,
    tokenizer: Any,
    examples: list[UtilityListwiseExample],
    *,
    device: torch.device,
    max_length: int,
    max_candidates: int,
    eval_batch_size: int,
    top_k: int,
    selector_name: str,
    no_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    model.eval()
    traces: list[dict[str, Any]] = []
    hybrid_traces: list[dict[str, Any]] = []
    candidate_order_traces: list[dict[str, Any]] = []
    all_scores: list[torch.Tensor] = []
    all_deltas: list[list[float]] = []
    for batch in tqdm(
        _batches(examples, int(eval_batch_size)),
        total=max(math.ceil(len(examples) / max(int(eval_batch_size), 1)), 1),
        desc="utility-listwise val",
        unit="batch",
        dynamic_ncols=True,
        disable=bool(no_progress),
    ):
        score_groups = _forward_utility_examples(
            model,
            tokenizer,
            batch,
            device=device,
            max_length=int(max_length),
            max_candidates=int(max_candidates),
        )
        for example, scores_tensor in zip(batch, score_groups):
            scores = scores_tensor.detach().float().cpu()
            all_scores.append(scores)
            all_deltas.append(example.delta_margins)
            trace = build_selection_trace(
                example.oracle_example,
                scores.numpy(),
                selector_name=str(selector_name),
                top_k=int(top_k),
            )
            traces.append(trace)
            hybrid_traces.append(
                build_order_control_trace(
                    trace,
                    ranked_indices_from_hybrid(example.oracle_example, top_k=int(top_k)),
                    selector_name="hybrid_score_top5",
                    top_k=int(top_k),
                )
            )
            candidate_order_traces.append(
                build_order_control_trace(
                    trace,
                    ranked_indices_from_candidate_pool(example.oracle_example, top_k=int(top_k)),
                    selector_name="candidate_pool_order_top5",
                    top_k=int(top_k),
                )
            )
    row_metrics = utility_rank_metrics(all_scores, all_deltas)
    controls = {
        "hybrid_score_top5": summarize_ordered_selection(hybrid_traces),
        "candidate_pool_order_top5": summarize_ordered_selection(candidate_order_traces),
    }
    return traces, row_metrics, controls


def _forward_utility_examples(
    model: SetAwareListwiseSelectorModel,
    tokenizer: Any,
    examples: list[UtilityListwiseExample],
    *,
    device: torch.device,
    max_length: int,
    max_candidates: int,
) -> list[torch.Tensor]:
    return forward_listwise_groups(
        model,
        tokenizer,
        [example.as_candidate_group() for example in examples],
        device=device,
        max_length=int(max_length),
        max_candidates=int(max_candidates),
    )


def _metadata(
    args: argparse.Namespace,
    train_examples: list[UtilityListwiseExample],
    val_examples: list[UtilityListwiseExample],
    *,
    total_steps: int,
) -> dict[str, Any]:
    return {
        "selector_type": str(args.selector_name),
        "base_model": str(args.model_name),
        "train_vig_cache": str(args.train_vig_cache),
        "val_vig_cache": str(args.val_vig_cache),
        "train_oracle_results": str(args.train_oracle_results),
        "val_oracle_results": str(args.val_oracle_results),
        "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        "target_source": "vig_step0_delta_margin",
        "step_filter": 0,
        "freeze_pair_encoder": True,
        "feature_ablation": normalize_feature_ablation(args.feature_ablation),
        "use_rank_embedding": str(args.use_rank_embedding),
        "train_candidate_shuffle_probability": float(args.shuffle_probability),
        "top_k": int(args.top_k),
        "max_candidates": int(args.max_candidates),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "total_steps": int(total_steps),
        "n_train_examples": int(len(train_examples)),
        "n_val_examples": int(len(val_examples)),
        "model_config": {
            "hidden_size": int(args.list_hidden_size),
            "num_layers": int(args.list_layers),
            "num_attention_heads": int(args.list_heads),
            "dropout": float(args.dropout),
            "max_candidates": int(args.max_candidates),
            "feature_ablation": normalize_feature_ablation(args.feature_ablation),
            "use_rank_embedding": _optional_bool(args.use_rank_embedding),
        },
        "losses": {
            "pairwise_delta": float(args.pairwise_weight),
            "soft_ce_delta": float(args.soft_ce_weight),
            "positive_bce": float(args.bce_weight),
            "soft_tau": float(args.soft_tau),
            "positive_best_margin": float(args.positive_best_margin),
        },
        "filter_policy": str(args.filter_policy),
        "min_margin": float(args.min_margin),
        "train_sample_limit": int(args.train_sample_limit) if args.train_sample_limit is not None else None,
        "val_sample_limit": int(args.val_sample_limit) if args.val_sample_limit is not None else None,
        "seed": int(args.seed),
    }


def _maybe_permute_utility_batch(
    batch: list[UtilityListwiseExample],
    *,
    probability: float,
) -> list[UtilityListwiseExample]:
    if probability <= 0.0:
        return batch
    out: list[UtilityListwiseExample] = []
    for example in batch:
        if random.random() >= float(probability) or len(example.candidates) <= 1:
            out.append(example)
            continue
        perm = list(range(len(example.candidates)))
        random.shuffle(perm)
        out.append(permute_utility_listwise_example(example, perm))
    return out


def _optional_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"", "auto", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean or auto, got {value!r}.")


def _batches(items: list[Any], batch_size: int):
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
