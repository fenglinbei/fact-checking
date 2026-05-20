"""Train Step1 cross-encoder pairwise evidence selector from Stage2 oracle rows."""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from fact_checking.selectors.cross_encoder import (
    pairwise_selector_loss,
    selector_logits,
    split_flat_scores,
    tokenize_claim_candidate_pairs,
)
from fact_checking.selectors.metrics import (
    build_order_control_trace,
    build_selection_trace,
    ranked_indices_from_candidate_pool,
    ranked_indices_from_hybrid,
    random_order_controls,
    reorder_predicted_set,
    summarize_ordered_selection,
)
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    Stage2OracleExample,
    candidate_text,
    load_stage2_oracle_examples,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a cross-encoder pairwise Stage2 evidence selector.")
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
    p.add_argument("--batch-size", type=int, default=4, help="Number of claims per optimizer micro-batch.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--bce-weight", type=float, default=0.3)
    p.add_argument("--order-weight", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-sample-limit", type=int, default=None)
    p.add_argument("--val-sample-limit", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=500, help="Optimizer steps between validation passes.")
    p.add_argument("--early-stopping-patience", type=int, default=4)
    p.add_argument("--early-stopping-metric", default="jaccard@5")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        trust_remote_code=True,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    micro_batches_per_epoch = max(math.ceil(len(train_examples) / max(int(args.batch_size), 1)), 1)
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
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, int(args.epochs) + 1):
        random.shuffle(train_examples)
        iterator = tqdm(
            _batches(train_examples, int(args.batch_size)),
            total=micro_batches_per_epoch,
            desc=f"cross-encoder train epoch {epoch}",
            unit="batch",
            dynamic_ncols=True,
            disable=args.no_progress,
        )
        for micro_step, batch in enumerate(iterator, start=1):
            model.train()
            grouped_scores = _forward_grouped_scores(
                model,
                tokenizer,
                batch,
                device=device,
                max_length=int(args.max_length),
            )
            loss, parts = pairwise_selector_loss(
                grouped_scores,
                [example.selected_indices for example in batch],
                candidate_scores=[example.candidate_scores for example in batch],
                bce_weight=float(args.bce_weight),
                order_weight=float(args.order_weight),
                top_k=int(args.top_k),
            )
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
                history.append(parts)
                iterator.set_postfix(loss=f"{parts['loss']:.4f}", pair=f"{parts['pair_loss']:.4f}")

                if int(args.eval_every) > 0 and global_step % int(args.eval_every) == 0:
                    metric_value = _validate_and_maybe_save(
                        model,
                        tokenizer,
                        val_examples,
                        args,
                        out_dir,
                        history,
                        global_step=global_step,
                        best_metric=best_metric,
                    )
                    if metric_value > best_metric + 1e-8:
                        best_metric = metric_value
                        stale = 0
                    else:
                        stale += 1
                        if stale >= int(args.early_stopping_patience):
                            break
        if stale >= int(args.early_stopping_patience):
            break

    final_metric = _validate_and_maybe_save(
        model,
        tokenizer,
        val_examples,
        args,
        out_dir,
        history,
        global_step=global_step,
        best_metric=best_metric,
        force_save=best_metric < 0,
    )
    print(f"Best/Final {args.early_stopping_metric}: {max(best_metric, final_metric):.4f}")
    print(f"Saved cross-encoder selector under: {out_dir}")


def _validate_and_maybe_save(
    model: torch.nn.Module,
    tokenizer: Any,
    val_examples: list[Stage2OracleExample],
    args: argparse.Namespace,
    out_dir: Path,
    history: list[dict[str, Any]],
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
    metric_value = float(metrics.get(args.early_stopping_metric, 0.0))
    if force_save or metric_value > best_metric + 1e-8:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        metadata = {
            "selector_type": "cross_encoder_pairwise",
            "base_model": args.model_name,
            "train_oracle_results": args.train_oracle_results,
            "val_oracle_results": args.val_oracle_results,
            "chunk_mmr_fingerprint": args.expected_chunk_mmr_fingerprint,
            "candidate_pool_policy": f"saved_stage2_candidate_pool_top{args.max_candidates}",
            "filter_policy": args.filter_policy,
            "min_margin": float(args.min_margin),
            "top_k": int(args.top_k),
            "max_length": int(args.max_length),
            "losses": {
                "pairwise_logistic": 1.0,
                "selected_mask_bce": float(args.bce_weight),
                "selected_order_pair": float(args.order_weight),
            },
            "negative_sampling": "all positive-negative pairs with hybrid-top and high-hybrid weights",
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
                "history": history,
                "metadata": metadata,
            },
        )
    return metric_value


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
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
        desc="cross-encoder val",
        unit="batch",
        dynamic_ncols=True,
        disable=no_progress,
    ):
        grouped_scores = _forward_grouped_scores(
            model,
            tokenizer,
            batch,
            device=device,
            max_length=max_length,
        )
        for example, scores_tensor in zip(batch, grouped_scores):
            scores = scores_tensor.detach().float().cpu().numpy()
            trace = build_selection_trace(
                example,
                scores,
                selector_name="cross_encoder_pairwise",
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


def _forward_grouped_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: list[Stage2OracleExample],
    *,
    device: torch.device,
    max_length: int,
) -> list[torch.Tensor]:
    claims: list[str] = []
    texts: list[str] = []
    group_sizes: list[int] = []
    for example in batch:
        group_sizes.append(len(example.candidates))
        claims.extend([example.claim] * len(example.candidates))
        texts.extend(candidate_text(candidate) for candidate in example.candidates)
    enc = tokenize_claim_candidate_pairs(
        tokenizer,
        claims,
        texts,
        max_length=max_length,
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    logits = selector_logits(model(**enc).logits)
    return split_flat_scores(logits, group_sizes)


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

