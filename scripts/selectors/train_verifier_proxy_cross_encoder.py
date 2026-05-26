#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from fact_checking.selectors.cross_encoder import selector_logits, split_flat_scores, tokenize_claim_candidate_pairs
from fact_checking.selectors.question_decomp_retrieval import _compute_prediction_metrics, oracle_selected_texts_by_event
from fact_checking.selectors.question_decomp_reranker import evaluate_selected_rows, source_composition
from fact_checking.selectors.stage2_oracle import read_jsonl, write_json, write_jsonl
from fact_checking.selectors.verifier_proxy import (
    json_safe,
    pairwise_utility_accuracy,
    pearson_corr,
    spearman_corr,
    verifier_proxy_cross_encoder_loss,
)


DEFAULT_OUTPUT_DIR = "outputs/selectors/question_decomp_retrieval/verifier_proxy_cross_encoder/b3_oracle_direct_v0/cross_encoder"
DEFAULT_LABEL_DIR = "outputs/selectors/question_decomp_retrieval/verifier_proxy_cross_encoder/b3_oracle_direct_v0"
DEFAULT_TRAIN_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
DEFAULT_VAL_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate a candidate-level cross-encoder on verifier-proxy labels.")
    p.add_argument("--train-labels-jsonl", default=f"{DEFAULT_LABEL_DIR}/candidate_utility_train.jsonl")
    p.add_argument("--val-labels-jsonl", default=f"{DEFAULT_LABEL_DIR}/candidate_utility_val.jsonl")
    p.add_argument("--train-oracle-results", default=DEFAULT_TRAIN_ORACLE_RESULTS)
    p.add_argument("--val-oracle-results", default=DEFAULT_VAL_ORACLE_RESULTS)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-name", default="/data/models/bge-reranker-large")
    p.add_argument("--model-dir", default=None, help="Existing model directory for --eval-only.")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=4, help="Number of claims per train micro-batch.")
    p.add_argument("--eval-batch-size", type=int, default=8, help="Number of claims per eval batch.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--soft-tau", type=float, default=0.3)
    p.add_argument("--soft-ce-weight", type=float, default=0.2)
    p.add_argument("--regression-weight", type=float, default=0.2)
    p.add_argument("--bce-weight", type=float, default=0.1)
    p.add_argument("--utility-epsilon", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-sample-limit", type=int, default=None)
    p.add_argument("--val-sample-limit", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--early-stopping-patience", type=int, default=4)
    p.add_argument("--early-stopping-metric", default="candidate_pairwise_accuracy")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    _set_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_groups = load_label_groups(args.train_labels_jsonl, sample_limit=args.train_sample_limit)
    val_groups = load_label_groups(args.val_labels_jsonl, sample_limit=args.val_sample_limit)
    if not train_groups and not args.eval_only:
        raise ValueError("No train verifier-proxy label groups loaded.")
    if not val_groups:
        raise ValueError("No val verifier-proxy label groups loaded.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.eval_only:
        model_dir = Path(args.model_dir or args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
        model.to(device)
        history: list[dict[str, Any]] = []
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=1,
            trust_remote_code=True,
        )
        model.to(device)
        history = train_model(model, tokenizer, train_groups, val_groups, args, out_dir)

    metrics, val_trace = evaluate_and_write(
        model,
        tokenizer,
        train_groups,
        val_groups,
        args,
        out_dir,
        history=history,
        elapsed_seconds=round(time.time() - started_at, 3),
    )
    print(f"Wrote verifier-proxy cross-encoder outputs: {out_dir}")
    val = metrics["val"]
    print(
        "Val learned recall={:.4f} jaccard={:.4f}; baseline_top2+learned recall={:.4f} jaccard={:.4f}; "
        "baseline recall={:.4f}".format(
            val["learned_top5"]["oracle_selected_recall@5"],
            val["learned_top5"]["jaccard@5"],
            val["baseline_top2_plus_learned"]["oracle_selected_recall@5"],
            val["baseline_top2_plus_learned"]["jaccard@5"],
            val["baseline_top5"]["oracle_selected_recall@5"],
        )
    )
    print(
        "Val utility pairwise_acc={:.4f} positive_hit@1={:.4f} trace_rows={}".format(
            metrics["val_candidate_metrics"]["pairwise_accuracy"],
            metrics["val_candidate_metrics"]["positive_hit@1"],
            len(val_trace),
        )
    )


def train_model(
    model: torch.nn.Module,
    tokenizer: Any,
    train_groups: list[dict[str, Any]],
    val_groups: list[dict[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    micro_batches_per_epoch = max(math.ceil(len(train_groups) / max(int(args.batch_size), 1)), 1)
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
        random.shuffle(train_groups)
        iterator = tqdm(
            _batches(train_groups, int(args.batch_size)),
            total=micro_batches_per_epoch,
            desc=f"verifier-proxy xenc train epoch {epoch}",
            unit="batch",
            dynamic_ncols=True,
            disable=bool(args.no_progress),
        )
        for micro_step, batch in enumerate(iterator, start=1):
            model.train()
            grouped_scores = _forward_grouped_scores(
                model,
                tokenizer,
                batch,
                device=next(model.parameters()).device,
                max_length=int(args.max_length),
            )
            loss, parts = verifier_proxy_cross_encoder_loss(
                grouped_scores,
                [group_utilities(group) for group in batch],
                [group_positive_mask(group) for group in batch],
                utility_epsilon=float(args.utility_epsilon),
                soft_tau=float(args.soft_tau),
                soft_ce_weight=float(args.soft_ce_weight),
                regression_weight=float(args.regression_weight),
                bce_weight=float(args.bce_weight),
            )
            (loss / max(int(args.gradient_accumulation_steps), 1)).backward()
            should_step = (
                micro_step % max(int(args.gradient_accumulation_steps), 1) == 0
                or micro_step == micro_batches_per_epoch
            )
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            parts.update({"epoch": epoch, "global_step": global_step})
            history.append(parts)
            iterator.set_postfix(loss=f"{parts['loss']:.4f}", pair=f"{parts['pair_loss']:.4f}")
            if int(args.eval_every) > 0 and global_step % int(args.eval_every) == 0:
                metric_value = _eval_early_metric(model, tokenizer, val_groups, args)
                history.append({"epoch": epoch, "global_step": global_step, "early_metric": metric_value})
                if metric_value > best_metric + 1e-8:
                    best_metric = metric_value
                    stale = 0
                    _save_model(model, tokenizer, out_dir, args, best_metric=best_metric, global_step=global_step)
                else:
                    stale += 1
                    if stale >= int(args.early_stopping_patience):
                        break
        if stale >= int(args.early_stopping_patience):
            break
    if best_metric < 0:
        metric_value = _eval_early_metric(model, tokenizer, val_groups, args)
        _save_model(model, tokenizer, out_dir, args, best_metric=metric_value, global_step=global_step)
    return history


def evaluate_and_write(
    model: torch.nn.Module,
    tokenizer: Any,
    train_groups: list[dict[str, Any]],
    val_groups: list[dict[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
    *,
    history: list[dict[str, Any]],
    elapsed_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_scores = score_groups(model, tokenizer, train_groups, args) if train_groups else []
    val_scores = score_groups(model, tokenizer, val_groups, args)
    train_oracle = oracle_selected_texts_by_event(read_jsonl(args.train_oracle_results))
    val_oracle = oracle_selected_texts_by_event(read_jsonl(args.val_oracle_results))
    train_selections = selection_bundle(train_groups, train_scores, top_k=int(args.top_k)) if train_groups else {}
    val_selections = selection_bundle(val_groups, val_scores, top_k=int(args.top_k))
    train_controls = control_bundle(train_groups, top_k=int(args.top_k)) if train_groups else {}
    val_controls = control_bundle(val_groups, top_k=int(args.top_k))
    metrics = {
        "model_type": "verifier_proxy_candidate_cross_encoder",
        "train_labels_jsonl": str(args.train_labels_jsonl),
        "val_labels_jsonl": str(args.val_labels_jsonl),
        "model_name": str(args.model_name),
        "top_k": int(args.top_k),
        "max_length": int(args.max_length),
        "loss_weights": {
            "pairwise": 1.0,
            "soft_ce": float(args.soft_ce_weight),
            "regression": float(args.regression_weight),
            "bce": float(args.bce_weight),
            "soft_tau": float(args.soft_tau),
        },
        "n_train_events": len(train_groups),
        "n_val_events": len(val_groups),
        "train_candidate_metrics": candidate_metrics(train_groups, train_scores) if train_groups else {},
        "val_candidate_metrics": candidate_metrics(val_groups, val_scores),
        "train": metrics_bundle(train_selections, train_controls, train_oracle) if train_groups else {},
        "val": metrics_bundle(val_selections, val_controls, val_oracle),
        "history": history,
    }
    train_trace = trace_rows(train_groups, train_scores) if train_groups else []
    val_trace = trace_rows(val_groups, val_scores)
    write_jsonl(out_dir / "train_history.jsonl", [json_safe(row) for row in history])
    if train_groups:
        write_jsonl(out_dir / "train_trace.jsonl", [json_safe(row) for row in train_trace])
        for name, rows in train_selections.items():
            write_jsonl(out_dir / f"{name}_train.jsonl", [json_safe(row) for row in rows])
    write_jsonl(out_dir / "val_trace.jsonl", [json_safe(row) for row in val_trace])
    for name, rows in val_selections.items():
        write_jsonl(out_dir / f"{name}_val.jsonl", [json_safe(row) for row in rows])
    write_json(out_dir / "reranker_metrics.json", json_safe(metrics))
    write_json(
        out_dir / "metadata.json",
        json_safe(
            {
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
                "output_dir": str(out_dir),
                "elapsed_seconds": elapsed_seconds,
                "metrics_path": str(out_dir / "reranker_metrics.json"),
            }
        ),
    )
    return metrics, val_trace


def load_label_groups(path: str, *, sample_limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        event_id = str(row.get("event_id") or "")
        if event_id not in by_event:
            order.append(event_id)
        by_event[event_id].append(row)
    if sample_limit is not None:
        order = order[: int(sample_limit)]
    groups: list[dict[str, Any]] = []
    for event_id in order:
        items = by_event[event_id]
        items.sort(key=lambda row: int(row.get("candidate_idx") or 0))
        groups.append(
            {
                "event_id": event_id,
                "claim": str(items[0].get("claim") or ""),
                "gold_label": str(items[0].get("gold_label") or ""),
                "candidates": items,
            }
        )
    return groups


def group_utilities(group: dict[str, Any]) -> list[float]:
    return [float(row.get("target_utility") or 0.0) for row in group.get("candidates") or []]


def group_positive_mask(group: dict[str, Any]) -> list[bool]:
    return [bool(row.get("target_positive")) for row in group.get("candidates") or []]


@torch.inference_mode()
def score_groups(
    model: torch.nn.Module,
    tokenizer: Any,
    groups: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    for batch in tqdm(
        _batches(groups, int(args.eval_batch_size)),
        total=max(math.ceil(len(groups) / max(int(args.eval_batch_size), 1)), 1),
        desc="verifier-proxy xenc eval",
        unit="batch",
        dynamic_ncols=True,
        disable=bool(args.no_progress),
    ):
        grouped = _forward_grouped_scores(
            model,
            tokenizer,
            batch,
            device=next(model.parameters()).device,
            max_length=int(args.max_length),
        )
        scores.extend([item.detach().float().cpu().numpy().astype(np.float32) for item in grouped])
    return scores


def _forward_grouped_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    groups: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
) -> list[torch.Tensor]:
    claims: list[str] = []
    texts: list[str] = []
    sizes: list[int] = []
    for group in groups:
        candidates = list(group.get("candidates") or [])
        sizes.append(len(candidates))
        claims.extend([str(group.get("claim") or "")] * len(candidates))
        texts.extend([str(row.get("candidate_text") or "") for row in candidates])
    if not texts:
        return [torch.zeros((0,), device=device) for _ in sizes]
    enc = tokenize_claim_candidate_pairs(tokenizer, claims, texts, max_length=max_length)
    enc = {key: value.to(device) for key, value in enc.items()}
    flat = selector_logits(model(**enc).logits)
    return split_flat_scores(flat, sizes)


def selection_bundle(groups: list[dict[str, Any]], scores: list[np.ndarray], *, top_k: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "learned_top5": build_selected_rows(groups, scores, top_k=top_k, mode="learned_top5"),
        "baseline_top1_plus_learned": build_selected_rows(groups, scores, top_k=top_k, mode="baseline_top1_plus_learned", baseline_anchor_k=1),
        "baseline_top2_plus_learned": build_selected_rows(groups, scores, top_k=top_k, mode="baseline_top2_plus_learned", baseline_anchor_k=2),
        "baseline_top3_plus_learned": build_selected_rows(groups, scores, top_k=top_k, mode="baseline_top3_plus_learned", baseline_anchor_k=3),
    }


def build_selected_rows(
    groups: list[dict[str, Any]],
    scores: list[np.ndarray],
    *,
    top_k: int,
    mode: str,
    baseline_anchor_k: int = 0,
) -> list[dict[str, Any]]:
    selected_rows: list[dict[str, Any]] = []
    for group, group_scores in zip(groups, scores):
        candidates = list(group.get("candidates") or [])
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        if baseline_anchor_k > 0:
            anchors = [(idx, row) for idx, row in enumerate(candidates) if row.get("from_baseline")]
            anchors.sort(key=lambda item: int(item[1].get("baseline_rank") or 10**9))
            for idx, row in anchors[:baseline_anchor_k]:
                _append_selected(selected, seen, row, float(group_scores[idx]), mode)
        order = sorted(
            range(len(candidates)),
            key=lambda idx: (
                -float(group_scores[idx]),
                int(candidates[idx].get("baseline_rank") or 10**9),
                int(candidates[idx].get("qd_pool_rank") or 10**9),
                int(candidates[idx].get("candidate_idx") or idx),
            ),
        )
        for idx in order:
            if len(selected) >= int(top_k):
                break
            _append_selected(selected, seen, candidates[idx], float(group_scores[idx]), mode)
        selected_rows.append({"event_id": group.get("event_id"), "claim": group.get("claim"), "candidates": selected})
    return selected_rows


def _append_selected(selected: list[dict[str, Any]], seen: set[str], row: dict[str, Any], score: float, mode: str) -> None:
    key = str(row.get("candidate_key") or row.get("canonical_text") or "")
    if not key or key in seen:
        return
    selected.append(
        {
            "text": row.get("candidate_text", ""),
            "canonical_text": key,
            "selection_rank": len(selected) + 1,
            "model_score": float(score),
            "target_utility": float(row.get("target_utility") or 0.0),
            "target_positive": bool(row.get("target_positive")),
            "selection_mode": mode,
            "union_source": row.get("union_source"),
            "from_baseline": bool(row.get("from_baseline")),
            "from_qd": bool(row.get("from_qd")),
            "baseline_rank": row.get("baseline_rank"),
            "qd_pool_rank": row.get("qd_pool_rank"),
        }
    )
    seen.add(key)


def control_bundle(groups: list[dict[str, Any]], *, top_k: int) -> dict[str, list[dict[str, Any]]]:
    baseline_rows: list[dict[str, Any]] = []
    qd_rows: list[dict[str, Any]] = []
    union_score_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    for group in groups:
        candidates = list(group.get("candidates") or [])
        baseline = [row for row in candidates if row.get("from_baseline")]
        baseline.sort(key=lambda row: int(row.get("baseline_rank") or 10**9))
        qd = [row for row in candidates if row.get("from_qd")]
        qd.sort(key=lambda row: int(row.get("qd_pool_rank") or 10**9))
        scored = sorted(candidates, key=_union_source_score_key)
        baseline_rows.append(_selected_control_row(group, baseline[:top_k]))
        qd_rows.append(_selected_control_row(group, qd[:top_k]))
        union_score_rows.append(_selected_control_row(group, scored[:top_k]))
        pool_rows.append(_selected_control_row(group, candidates))
    return {
        "baseline_top5": baseline_rows,
        "qd_rrf_top5": qd_rows,
        "union_source_score_top5": union_score_rows,
        "union_pool": pool_rows,
    }


def _selected_control_row(group: dict[str, Any], candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected = []
    for rank, row in enumerate(candidates, start=1):
        item = {
            "text": row.get("candidate_text", ""),
            "canonical_text": row.get("candidate_key") or row.get("canonical_text") or "",
            "selection_rank": rank,
            "union_source": row.get("union_source"),
            "from_baseline": bool(row.get("from_baseline")),
            "from_qd": bool(row.get("from_qd")),
            "baseline_rank": row.get("baseline_rank"),
            "qd_pool_rank": row.get("qd_pool_rank"),
        }
        selected.append(item)
    return {"event_id": group.get("event_id"), "claim": group.get("claim"), "candidates": selected}


def _union_source_score_key(row: dict[str, Any]) -> tuple[float, int, int, int]:
    baseline_component = 0.0
    if row.get("from_baseline"):
        baseline_component += 0.04
        rank = row.get("baseline_rank")
        if rank is not None:
            baseline_component += 0.01 / max(float(rank), 1.0)
    qd_component = float(row.get("qd_rrf_score") or 0.0)
    qd_component += 0.004 * float(row.get("qd_question_hit_count") or 0.0)
    qd_component += 0.01 * float(row.get("qd_max_question_hybrid") or 0.0)
    score = baseline_component + qd_component
    return (-score, int(row.get("baseline_rank") or 10**9), int(row.get("qd_pool_rank") or 10**9), int(row.get("candidate_idx") or 0))


def metrics_bundle(
    selections: dict[str, list[dict[str, Any]]],
    controls: dict[str, list[dict[str, Any]]],
    oracle_texts: dict[str, set[str]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, rows in {**controls, **selections}.items():
        if name == "union_pool":
            metrics[name] = _compute_prediction_metrics(
                pool_rows=[
                    {"event_id": row.get("event_id"), "pool": row.get("candidates") or [], "selected": row.get("candidates") or []}
                    for row in rows
                ],
                oracle_texts=oracle_texts,
                include_pool=True,
            )
        else:
            metrics[name] = evaluate_selected_rows(rows, oracle_texts=oracle_texts)
            metrics[name]["source_composition"] = source_composition(rows)
    return metrics


def candidate_metrics(groups: list[dict[str, Any]], scores: list[np.ndarray]) -> dict[str, Any]:
    total_pairs = 0
    correct_pairs = 0
    top1_match = 0
    positive_hit1 = 0
    positive_hit5 = 0
    spearman_values: list[float] = []
    pearson_values: list[float] = []
    selected_utility_sum = 0.0
    n_groups = 0
    for group, group_scores in zip(groups, scores):
        utilities = group_utilities(group)
        positives = group_positive_mask(group)
        if not utilities:
            continue
        n_groups += 1
        correct, total = pairwise_utility_accuracy(group_scores.tolist(), utilities)
        correct_pairs += correct
        total_pairs += total
        best_score = int(np.argmax(group_scores))
        best_utility = int(np.argmax(np.asarray(utilities, dtype=np.float32)))
        top1_match += int(best_score == best_utility)
        order = np.argsort(-group_scores)
        positive_hit1 += int(bool(positives[int(order[0])]))
        positive_hit5 += int(any(bool(positives[int(idx)]) for idx in order[:5]))
        selected_utility_sum += float(sum(float(utilities[int(idx)]) for idx in order[:5]))
        sp = spearman_corr(group_scores.tolist(), utilities)
        pe = pearson_corr(group_scores.tolist(), utilities)
        if sp is not None:
            spearman_values.append(sp)
        if pe is not None:
            pearson_values.append(pe)
    return {
        "n_groups": int(n_groups),
        "n_pairs": int(total_pairs),
        "pairwise_accuracy": float(correct_pairs / total_pairs) if total_pairs else 0.0,
        "top1_utility_match": float(top1_match / n_groups) if n_groups else 0.0,
        "positive_hit@1": float(positive_hit1 / n_groups) if n_groups else 0.0,
        "positive_hit@5": float(positive_hit5 / n_groups) if n_groups else 0.0,
        "mean_selected_utility@5": float(selected_utility_sum / n_groups) if n_groups else 0.0,
        "mean_spearman": float(np.mean(spearman_values)) if spearman_values else None,
        "mean_pearson": float(np.mean(pearson_values)) if pearson_values else None,
    }


def trace_rows(groups: list[dict[str, Any]], scores: list[np.ndarray]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for group, group_scores in zip(groups, scores):
        rows = []
        candidates = list(group.get("candidates") or [])
        order = np.argsort(-group_scores)
        rank_by_idx = {int(idx): rank for rank, idx in enumerate(order.tolist(), start=1)}
        for idx, row in enumerate(candidates):
            rows.append(
                {
                    "candidate_idx": int(row.get("candidate_idx") or idx),
                    "candidate_key": row.get("candidate_key"),
                    "model_score": float(group_scores[idx]),
                    "model_rank": int(rank_by_idx[idx]),
                    "target_utility": float(row.get("target_utility") or 0.0),
                    "target_positive": bool(row.get("target_positive")),
                    "union_source": row.get("union_source"),
                    "baseline_rank": row.get("baseline_rank"),
                    "qd_pool_rank": row.get("qd_pool_rank"),
                    "text": row.get("candidate_text", ""),
                }
            )
        traces.append({"event_id": group.get("event_id"), "claim": group.get("claim"), "candidates": rows})
    return traces


def _eval_early_metric(
    model: torch.nn.Module,
    tokenizer: Any,
    val_groups: list[dict[str, Any]],
    args: argparse.Namespace,
) -> float:
    val_scores = score_groups(model, tokenizer, val_groups, args)
    metrics = candidate_metrics(val_groups, val_scores)
    return float(metrics.get("pairwise_accuracy") or 0.0)


def _save_model(
    model: torch.nn.Module,
    tokenizer: Any,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    best_metric: float,
    global_step: int,
) -> None:
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    write_json(
        out_dir / "model_metadata.json",
        {
            "selector_type": "verifier_proxy_candidate_cross_encoder",
            "base_model": str(args.model_name),
            "best_metric": str(args.early_stopping_metric),
            "best_metric_value": float(best_metric),
            "global_step": int(global_step),
            "max_length": int(args.max_length),
        },
    )


def _batches(items: Sequence[Any], batch_size: int) -> list[list[Any]]:
    return [list(items[start : start + int(batch_size)]) for start in range(0, len(items), int(batch_size))]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
