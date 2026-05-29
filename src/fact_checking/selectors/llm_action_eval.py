from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from fact_checking.selectors.llm_action import (
    ACTION_LABELS,
    ACTION_LABEL_MODE_GLOBAL_INDEX,
    ACTION_LABEL_MODE_LOCAL_CHOICE,
    CANDIDATE_ORDER_CANDIDATE_POOL,
    action_completion,
    build_action_prompt,
    build_action_prompt_from_fields,
    choice_action_labels,
    order_candidate_indices,
    score_action_choices,
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
    Stage2OracleExample,
    write_json,
    write_jsonl,
)

DECODE_STRATEGY_RAW = "raw"
DECODE_STRATEGY_CALIBRATED = "calibrated"
DECODE_STRATEGY_PERMUTATION = "permutation"
DECODE_STRATEGY_PERMUTATION_CALIBRATED = "permutation_calibrated"
DECODE_STRATEGIES = {
    DECODE_STRATEGY_RAW,
    DECODE_STRATEGY_CALIBRATED,
    DECODE_STRATEGY_PERMUTATION,
    DECODE_STRATEGY_PERMUTATION_CALIBRATED,
}
AGGREGATION_MEAN_ZSCORE = "mean_zscore"
AGGREGATION_MEAN_SCORE = "mean_score"
AGGREGATION_BORDA = "borda"
AGGREGATION_MODES = {AGGREGATION_MEAN_ZSCORE, AGGREGATION_MEAN_SCORE, AGGREGATION_BORDA}
CALIBRATION_MODE_NONE = "none"
CALIBRATION_MODE_CONTENT_FREE_WIDTH = "content_free_width"
CALIBRATION_MODES = {CALIBRATION_MODE_NONE, CALIBRATION_MODE_CONTENT_FREE_WIDTH}


def evaluate_llm_action_selection(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Stage2OracleExample],
    *,
    device: torch.device,
    split: str,
    top_k: int,
    max_length: int,
    score_mode: str,
    choice_batch_size: int,
    max_candidate_chars: int,
    include_retrieval_scores: bool,
    action_label_mode: str = ACTION_LABEL_MODE_GLOBAL_INDEX,
    candidate_order_mode: str = CANDIDATE_ORDER_CANDIDATE_POOL,
    candidate_order_seed: int = 20260524,
    decode_strategy: str = DECODE_STRATEGY_RAW,
    num_permutations: int = 1,
    permutation_seed: int = 20260524,
    permutation_include_base_order: bool = True,
    aggregation: str = AGGREGATION_MEAN_ZSCORE,
    calibration_mode: str = CALIBRATION_MODE_NONE,
    calibration_alpha: float = 0.0,
    disable_progress: bool = False,
    selector_name: str = "llm_action_selector",
) -> dict[str, Any]:
    started_at = time.time()
    decode_strategy = _normalize_decode_strategy(decode_strategy)
    aggregation = _normalize_aggregation(aggregation)
    calibration_mode = _normalize_calibration_mode(calibration_mode)
    if decode_strategy in {DECODE_STRATEGY_CALIBRATED, DECODE_STRATEGY_PERMUTATION_CALIBRATED}:
        if calibration_mode == CALIBRATION_MODE_NONE:
            calibration_mode = CALIBRATION_MODE_CONTENT_FREE_WIDTH
    effective_num_permutations = (
        max(int(num_permutations), 1)
        if decode_strategy in {DECODE_STRATEGY_PERMUTATION, DECODE_STRATEGY_PERMUTATION_CALIBRATED}
        else 1
    )
    traces: list[dict[str, Any]] = []
    hybrid_traces: list[dict[str, Any]] = []
    candidate_order_traces: list[dict[str, Any]] = []
    same_set_hybrid_traces: list[dict[str, Any]] = []
    same_set_candidate_traces: list[dict[str, Any]] = []
    random_traces: list[dict[str, Any]] = []
    estimated_forward_steps = 0

    model.eval()
    for example in tqdm(
        examples,
        desc=f"llm action eval [{split}]",
        disable=bool(disable_progress),
        dynamic_ncols=True,
    ):
        prediction = rollout_llm_action_example(
            model,
            tokenizer,
            example,
            device=device,
            top_k=int(top_k),
            max_length=int(max_length),
            score_mode=str(score_mode),
            choice_batch_size=int(choice_batch_size),
            max_candidate_chars=int(max_candidate_chars),
            include_retrieval_scores=bool(include_retrieval_scores),
            action_label_mode=str(action_label_mode),
            candidate_order_mode=str(candidate_order_mode),
            candidate_order_seed=int(candidate_order_seed),
            decode_strategy=decode_strategy,
            num_permutations=int(effective_num_permutations),
            permutation_seed=int(permutation_seed),
            permutation_include_base_order=bool(permutation_include_base_order),
            aggregation=aggregation,
            calibration_mode=calibration_mode,
            calibration_alpha=float(calibration_alpha),
        )
        estimated_forward_steps += int(prediction.get("estimated_forward_steps", len(prediction["per_step_action_scores"])))
        trace = build_selection_trace(
            example,
            _rank_vector_from_order(len(example.candidates), prediction["ordered_indices"]),
            selector_name=selector_name,
            top_k=int(top_k),
        )
        trace["selector_ordered_indices"] = [int(idx) for idx in prediction["ordered_indices"]]
        trace["per_step_action_scores"] = prediction["per_step_action_scores"]
        traces.append(trace)

        hybrid_traces.append(
            build_order_control_trace(
                trace,
                ranked_indices_from_hybrid(example, top_k=int(top_k)),
                selector_name="hybrid_score_top5",
                top_k=int(top_k),
            )
        )
        candidate_order_traces.append(
            build_order_control_trace(
                trace,
                ranked_indices_from_candidate_pool(example, top_k=int(top_k)),
                selector_name="candidate_pool_order_top5",
                top_k=int(top_k),
            )
        )
        predicted = [int(idx) for idx in trace["selector_ordered_indices"]]
        same_set_hybrid_traces.append(
            build_order_control_trace(
                trace,
                reorder_predicted_set(predicted, example=example, mode="hybrid_order"),
                selector_name="same_set_hybrid_order",
                top_k=int(top_k),
            )
        )
        same_set_candidate_traces.append(
            build_order_control_trace(
                trace,
                reorder_predicted_set(predicted, example=example, mode="candidate_pool_order"),
                selector_name="same_set_candidate_pool_order",
                top_k=int(top_k),
            )
        )
        random_traces.extend(
            random_order_controls(
                predicted,
                example=example,
                seeds=[0, 1, 2, 3, 4],
                top_k=int(top_k),
            )
        )

    elapsed = max(time.time() - started_at, 0.0)
    selector_metrics = summarize_ordered_selection(traces)
    controls = {
        "hybrid_score_top5": summarize_ordered_selection(hybrid_traces),
        "candidate_pool_order_top5": summarize_ordered_selection(candidate_order_traces),
        "same_set_hybrid_order": summarize_ordered_selection(same_set_hybrid_traces),
        "same_set_candidate_pool_order": summarize_ordered_selection(same_set_candidate_traces),
        "same_set_random_order_mean": summarize_ordered_selection(random_traces),
    }
    metrics = {
        "split": str(split),
        "n_claims": int(len(examples)),
        "top_k": int(top_k),
        "max_length": int(max_length),
        "score_mode": str(score_mode),
        "choice_batch_size": int(choice_batch_size),
        "max_candidate_chars": int(max_candidate_chars),
        "include_retrieval_scores": bool(include_retrieval_scores),
        "action_label_mode": str(action_label_mode),
        "candidate_order_mode": str(candidate_order_mode),
        "candidate_order_seed": int(candidate_order_seed),
        "decode_strategy": decode_strategy,
        "num_permutations": int(effective_num_permutations),
        "permutation_seed": int(permutation_seed),
        "permutation_include_base_order": bool(permutation_include_base_order),
        "aggregation": aggregation,
        "calibration_mode": calibration_mode,
        "calibration_alpha": float(calibration_alpha),
        "elapsed_seconds": round(float(elapsed), 3),
        "claims_per_second": float(len(examples) / elapsed) if elapsed > 0 else 0.0,
        "estimated_forward_steps": int(estimated_forward_steps),
        "decode_diagnostics": _decode_diagnostics(traces),
        "selector": selector_metrics,
        "controls": controls,
    }
    return {
        "metrics": metrics,
        "traces": traces,
        "hybrid_traces": hybrid_traces,
        "candidate_order_traces": candidate_order_traces,
    }


def rollout_llm_action_example(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Stage2OracleExample,
    *,
    device: torch.device,
    top_k: int,
    max_length: int,
    score_mode: str,
    choice_batch_size: int,
    max_candidate_chars: int,
    include_retrieval_scores: bool,
    action_label_mode: str = ACTION_LABEL_MODE_GLOBAL_INDEX,
    candidate_order_mode: str = CANDIDATE_ORDER_CANDIDATE_POOL,
    candidate_order_seed: int = 20260524,
    decode_strategy: str = DECODE_STRATEGY_RAW,
    num_permutations: int = 1,
    permutation_seed: int = 20260524,
    permutation_include_base_order: bool = True,
    aggregation: str = AGGREGATION_MEAN_ZSCORE,
    calibration_mode: str = CALIBRATION_MODE_NONE,
    calibration_alpha: float = 0.0,
) -> dict[str, Any]:
    decode_strategy = _normalize_decode_strategy(decode_strategy)
    if decode_strategy == DECODE_STRATEGY_RAW:
        return _rollout_llm_action_example_raw(
            model,
            tokenizer,
            example,
            device=device,
            top_k=int(top_k),
            max_length=int(max_length),
            score_mode=str(score_mode),
            choice_batch_size=int(choice_batch_size),
            max_candidate_chars=int(max_candidate_chars),
            include_retrieval_scores=bool(include_retrieval_scores),
            action_label_mode=str(action_label_mode),
            candidate_order_mode=str(candidate_order_mode),
            candidate_order_seed=int(candidate_order_seed),
        )
    return _rollout_llm_action_example_decode(
        model,
        tokenizer,
        example,
        device=device,
        top_k=int(top_k),
        max_length=int(max_length),
        score_mode=str(score_mode),
        choice_batch_size=int(choice_batch_size),
        max_candidate_chars=int(max_candidate_chars),
        include_retrieval_scores=bool(include_retrieval_scores),
        action_label_mode=str(action_label_mode),
        candidate_order_mode=str(candidate_order_mode),
        candidate_order_seed=int(candidate_order_seed),
        decode_strategy=decode_strategy,
        num_permutations=int(num_permutations),
        permutation_seed=int(permutation_seed),
        permutation_include_base_order=bool(permutation_include_base_order),
        aggregation=aggregation,
        calibration_mode=calibration_mode,
        calibration_alpha=float(calibration_alpha),
    )


def _rollout_llm_action_example_raw(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Stage2OracleExample,
    *,
    device: torch.device,
    top_k: int,
    max_length: int,
    score_mode: str,
    choice_batch_size: int,
    max_candidate_chars: int,
    include_retrieval_scores: bool,
    action_label_mode: str,
    candidate_order_mode: str,
    candidate_order_seed: int,
) -> dict[str, Any]:
    selected: list[int] = []
    per_step: list[dict[str, Any]] = []
    estimated_forward_steps = 0
    for step in range(int(top_k)):
        remaining = [idx for idx in range(len(example.candidates)) if idx not in selected]
        if not remaining:
            break
        ordered_remaining = order_candidate_indices(
            remaining,
            mode=str(candidate_order_mode),
            seed=int(candidate_order_seed),
            event_id=example.event_id,
            step=step,
        )
        action_labels = choice_action_labels(ordered_remaining, action_label_mode=str(action_label_mode))
        prompt = build_action_prompt(
            example,
            prefix_indices=selected,
            remaining_indices=ordered_remaining,
            max_candidate_chars=int(max_candidate_chars),
            include_retrieval_scores=bool(include_retrieval_scores),
            action_labels=action_labels,
            action_label_mode=str(action_label_mode),
        )
        sample = {
            "prompt": prompt,
            "choices": [
                {
                    "candidate_idx": int(idx),
                    "action": action_completion(action_labels[int(idx)]),
                    "action_label": action_labels[int(idx)],
                    "choice_position": int(position),
                }
                for position, idx in enumerate(ordered_remaining)
            ],
        }
        with torch.inference_mode():
            scored = score_action_choices(
                model,
                tokenizer,
                [sample],
                device=device,
                max_length=int(max_length),
                score_mode=str(score_mode),
                choice_batch_size=int(choice_batch_size),
            )
        estimated_forward_steps += 1
        scores = scored.scores[0]
        best_pos = int(torch.argmax(scores).detach().cpu().item())
        best_idx = int(scored.candidate_indices[0][best_pos])
        selected.append(best_idx)
        per_step.append(
            {
                "step": int(step),
                "selected_idx": int(best_idx),
                "selected_action": str(sample["choices"][best_pos]["action_label"]),
                "selected_score": float(scores[best_pos].detach().cpu().item()),
                "oracle_idx": int(example.selected_indices[step]) if step < len(example.selected_indices) else None,
                "action_label_mode": str(action_label_mode),
                "candidate_order_mode": str(candidate_order_mode),
                "choice_scores": [
                    {
                        "candidate_idx": int(idx),
                        "action": str(sample["choices"][pos]["action_label"]),
                        "score": float(score.detach().cpu().item()),
                    }
                    for pos, (idx, action, score) in enumerate(zip(scored.candidate_indices[0], scored.actions[0], scores))
                ],
            }
        )
    return {
        "ordered_indices": selected[: int(top_k)],
        "per_step_action_scores": per_step,
        "estimated_forward_steps": int(estimated_forward_steps),
    }


def _rollout_llm_action_example_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Stage2OracleExample,
    *,
    device: torch.device,
    top_k: int,
    max_length: int,
    score_mode: str,
    choice_batch_size: int,
    max_candidate_chars: int,
    include_retrieval_scores: bool,
    action_label_mode: str,
    candidate_order_mode: str,
    candidate_order_seed: int,
    decode_strategy: str,
    num_permutations: int,
    permutation_seed: int,
    permutation_include_base_order: bool,
    aggregation: str,
    calibration_mode: str,
    calibration_alpha: float,
) -> dict[str, Any]:
    aggregation = _normalize_aggregation(aggregation)
    calibration_mode = _normalize_calibration_mode(calibration_mode)
    use_permutation = decode_strategy in {DECODE_STRATEGY_PERMUTATION, DECODE_STRATEGY_PERMUTATION_CALIBRATED}
    use_calibration = decode_strategy in {DECODE_STRATEGY_CALIBRATED, DECODE_STRATEGY_PERMUTATION_CALIBRATED}
    if use_calibration and calibration_mode == CALIBRATION_MODE_NONE:
        calibration_mode = CALIBRATION_MODE_CONTENT_FREE_WIDTH

    selected: list[int] = []
    per_step: list[dict[str, Any]] = []
    estimated_forward_steps = 0
    for step in range(int(top_k)):
        remaining = [idx for idx in range(len(example.candidates)) if idx not in selected]
        if not remaining:
            break
        base_order = order_candidate_indices(
            remaining,
            mode=str(candidate_order_mode),
            seed=int(candidate_order_seed),
            event_id=example.event_id,
            step=step,
        )
        orders = _decode_candidate_orders(
            base_order,
            use_permutation=use_permutation,
            num_permutations=int(num_permutations),
            include_base_order=bool(permutation_include_base_order),
            seed=int(permutation_seed),
            event_id=example.event_id,
            step=step,
        )
        base_labels = choice_action_labels(base_order, action_label_mode=str(action_label_mode))
        choice_records: list[dict[str, Any]] = []
        for permutation_index, ordered_remaining in enumerate(orders):
            action_labels = choice_action_labels(ordered_remaining, action_label_mode=str(action_label_mode))
            sample = _build_choice_sample(
                example,
                prefix_indices=selected,
                ordered_remaining=ordered_remaining,
                action_labels=action_labels,
                action_label_mode=str(action_label_mode),
                max_candidate_chars=int(max_candidate_chars),
                include_retrieval_scores=bool(include_retrieval_scores),
            )
            with torch.inference_mode():
                scored = score_action_choices(
                    model,
                    tokenizer,
                    [sample],
                    device=device,
                    max_length=int(max_length),
                    score_mode=str(score_mode),
                    choice_batch_size=int(choice_batch_size),
                )
            estimated_forward_steps += 1
            raw_scores = [float(score.detach().cpu().item()) for score in scored.scores[0]]
            bias_by_label: dict[str, float] = {}
            if use_calibration:
                bias_by_label = _content_free_label_bias(
                    model,
                    tokenizer,
                    ordered_remaining=ordered_remaining,
                    action_labels=action_labels,
                    action_label_mode=str(action_label_mode),
                    step=step,
                    device=device,
                    max_length=int(max_length),
                    score_mode=str(score_mode),
                    choice_batch_size=int(choice_batch_size),
                    include_retrieval_scores=bool(include_retrieval_scores),
                )
                estimated_forward_steps += 1
            for pos, idx in enumerate(scored.candidate_indices[0]):
                action_label = str(sample["choices"][pos]["action_label"])
                raw = float(raw_scores[pos])
                calibrated = raw - float(calibration_alpha) * float(bias_by_label.get(action_label, 0.0))
                choice_records.append(
                    {
                        "permutation_index": int(permutation_index),
                        "candidate_idx": int(idx),
                        "action": action_label,
                        "raw_score": raw,
                        "calibrated_score": calibrated,
                        "selection_score": calibrated if use_calibration else raw,
                    }
                )
        aggregated = aggregate_candidate_choice_scores(choice_records, aggregation=aggregation)
        if not aggregated:
            break
        best = max(aggregated, key=lambda item: (float(item["aggregate_score"]), -int(item["candidate_idx"])))
        best_idx = int(best["candidate_idx"])
        selected.append(best_idx)
        per_step.append(
            {
                "step": int(step),
                "selected_idx": int(best_idx),
                "selected_action": str(base_labels.get(best_idx, best.get("labels_seen", [""])[0])),
                "selected_score": float(best["aggregate_score"]),
                "oracle_idx": int(example.selected_indices[step]) if step < len(example.selected_indices) else None,
                "action_label_mode": str(action_label_mode),
                "candidate_order_mode": str(candidate_order_mode),
                "decode_strategy": str(decode_strategy),
                "num_permutations": int(len(orders)),
                "aggregation": str(aggregation),
                "calibration_mode": str(calibration_mode),
                "calibration_alpha": float(calibration_alpha),
                "choice_scores": [
                    {
                        **item,
                        "action": str(base_labels.get(int(item["candidate_idx"]), "")),
                    }
                    for item in sorted(aggregated, key=lambda row: int(row["candidate_idx"]))
                ],
            }
        )
    return {
        "ordered_indices": selected[: int(top_k)],
        "per_step_action_scores": per_step,
        "estimated_forward_steps": int(estimated_forward_steps),
    }


def _build_choice_sample(
    example: Stage2OracleExample,
    *,
    prefix_indices: list[int],
    ordered_remaining: list[int],
    action_labels: dict[int, str],
    action_label_mode: str,
    max_candidate_chars: int,
    include_retrieval_scores: bool,
) -> dict[str, Any]:
    prompt = build_action_prompt(
        example,
        prefix_indices=prefix_indices,
        remaining_indices=ordered_remaining,
        max_candidate_chars=int(max_candidate_chars),
        include_retrieval_scores=bool(include_retrieval_scores),
        action_labels=action_labels,
        action_label_mode=str(action_label_mode),
    )
    return {
        "prompt": prompt,
        "choices": [
            {
                "candidate_idx": int(idx),
                "action": action_completion(action_labels[int(idx)]),
                "action_label": action_labels[int(idx)],
                "choice_position": int(position),
            }
            for position, idx in enumerate(ordered_remaining)
        ],
    }


def _content_free_label_bias(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    ordered_remaining: list[int],
    action_labels: dict[int, str],
    action_label_mode: str,
    step: int,
    device: torch.device,
    max_length: int,
    score_mode: str,
    choice_batch_size: int,
    include_retrieval_scores: bool,
) -> dict[str, float]:
    text_by_idx: dict[int, str] = {int(idx): "[candidate omitted]" for idx in ordered_remaining}
    if str(action_label_mode) == ACTION_LABEL_MODE_LOCAL_CHOICE:
        prefix_indices = [-(idx + 1) for idx in range(int(step))]
        for idx in prefix_indices:
            text_by_idx[int(idx)] = "[selected evidence omitted]"
    else:
        prefix_indices = [idx for idx in range(len(ACTION_LABELS)) if idx not in set(ordered_remaining)][: int(step)]
        for idx in prefix_indices:
            text_by_idx[int(idx)] = "[selected evidence omitted]"
    score_by_idx = {
        int(idx): {"candidate_idx": int(idx), "hybrid_rank": int(pos)}
        for pos, idx in enumerate(ordered_remaining)
    }
    prompt = build_action_prompt_from_fields(
        claim="[claim omitted]",
        prefix_indices=prefix_indices,
        remaining_indices=ordered_remaining,
        candidate_text_by_idx=text_by_idx,
        candidate_score_by_idx=score_by_idx,
        max_candidate_chars=80,
        include_retrieval_scores=bool(include_retrieval_scores),
        action_labels=action_labels,
        action_label_mode=str(action_label_mode),
    )
    sample = {
        "prompt": prompt,
        "choices": [
            {
                "candidate_idx": int(idx),
                "action": action_completion(action_labels[int(idx)]),
                "action_label": action_labels[int(idx)],
                "choice_position": int(position),
            }
            for position, idx in enumerate(ordered_remaining)
        ],
    }
    with torch.inference_mode():
        scored = score_action_choices(
            model,
            tokenizer,
            [sample],
            device=device,
            max_length=int(max_length),
            score_mode=str(score_mode),
            choice_batch_size=int(choice_batch_size),
        )
    return {
        str(sample["choices"][pos]["action_label"]): float(score.detach().cpu().item())
        for pos, score in enumerate(scored.scores[0])
    }


def _decode_candidate_orders(
    base_order: list[int],
    *,
    use_permutation: bool,
    num_permutations: int,
    include_base_order: bool,
    seed: int,
    event_id: str,
    step: int,
) -> list[list[int]]:
    if not use_permutation:
        return [list(base_order)]
    n_perms = max(int(num_permutations), 1)
    orders: list[list[int]] = []
    if include_base_order:
        orders.append(list(base_order))
    rep = 0
    while len(orders) < n_perms:
        material = "\n".join(
            [
                str(int(seed)),
                str(event_id),
                str(int(step)),
                str(int(rep)),
                ",".join(str(idx) for idx in base_order),
            ]
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        rng = np.random.default_rng(int(digest[:16], 16))
        permuted = list(base_order)
        rng.shuffle(permuted)
        orders.append([int(idx) for idx in permuted])
        rep += 1
    return orders


def aggregate_candidate_choice_scores(
    choice_records: list[dict[str, Any]],
    *,
    aggregation: str = AGGREGATION_MEAN_ZSCORE,
) -> list[dict[str, Any]]:
    aggregation = _normalize_aggregation(aggregation)
    by_perm: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in choice_records:
        by_perm[int(row["permutation_index"])].append(dict(row))

    transformed_by_candidate: dict[int, list[float]] = defaultdict(list)
    raw_by_candidate: dict[int, list[float]] = defaultdict(list)
    calibrated_by_candidate: dict[int, list[float]] = defaultdict(list)
    labels_by_candidate: dict[int, set[str]] = defaultdict(set)
    vote_counts: dict[int, int] = defaultdict(int)

    for _perm_idx, rows in sorted(by_perm.items()):
        values = np.asarray([float(row["selection_score"]) for row in rows], dtype=np.float64)
        if aggregation == AGGREGATION_MEAN_ZSCORE:
            std = float(values.std())
            transformed = (values - float(values.mean())) / std if std > 1.0e-8 else np.zeros_like(values)
        elif aggregation == AGGREGATION_BORDA:
            order = np.argsort(-values)
            transformed = np.zeros_like(values)
            for rank, pos in enumerate(order):
                transformed[int(pos)] = float(len(values) - rank)
        else:
            transformed = values

        max_value = float(values.max()) if values.size else float("-inf")
        for row, score in zip(rows, transformed):
            idx = int(row["candidate_idx"])
            transformed_by_candidate[idx].append(float(score))
            raw_by_candidate[idx].append(float(row["raw_score"]))
            calibrated_by_candidate[idx].append(float(row["calibrated_score"]))
            labels_by_candidate[idx].add(str(row["action"]))
            if abs(float(row["selection_score"]) - max_value) <= 1.0e-8:
                vote_counts[idx] += 1

    out: list[dict[str, Any]] = []
    for idx in sorted(transformed_by_candidate):
        raw_values = np.asarray(raw_by_candidate[idx], dtype=np.float64)
        calibrated_values = np.asarray(calibrated_by_candidate[idx], dtype=np.float64)
        out.append(
            {
                "candidate_idx": int(idx),
                "aggregate_score": float(np.mean(transformed_by_candidate[idx])),
                "n_appearances": int(len(transformed_by_candidate[idx])),
                "mean_raw_score": float(raw_values.mean()) if raw_values.size else 0.0,
                "std_raw_score": float(raw_values.std()) if raw_values.size else 0.0,
                "mean_calibrated_score": float(calibrated_values.mean()) if calibrated_values.size else 0.0,
                "labels_seen": sorted(labels_by_candidate[idx]),
                "selected_by_permutation_vote_count": int(vote_counts.get(idx, 0)),
            }
        )
    return out


def write_selection_eval_outputs(out_dir: Path, result: dict[str, Any]) -> None:
    metrics = dict(result["metrics"])
    write_json(out_dir / "selection_metrics.json", metrics)
    write_jsonl(out_dir / "selection_trace.jsonl", list(result.get("traces") or []))
    write_jsonl(out_dir / "control_hybrid_trace.jsonl", list(result.get("hybrid_traces") or []))
    write_jsonl(out_dir / "control_candidate_pool_trace.jsonl", list(result.get("candidate_order_traces") or []))
    write_selection_eval_markdown(out_dir / "analysis.md", metrics)


def selection_history_record(
    metrics: dict[str, Any],
    *,
    global_step: int | None = None,
    epoch: int | None = None,
    output_dir: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    selector = dict(metrics.get("selector") or {})
    record: dict[str, Any] = {
        "global_step": int(global_step) if global_step is not None else None,
        "epoch": int(epoch) if epoch is not None else None,
        "reason": reason,
        "output_dir": output_dir,
        "n_claims": int(metrics.get("n_claims", 0)),
        "elapsed_seconds": float(metrics.get("elapsed_seconds", 0.0)),
        "claims_per_second": float(metrics.get("claims_per_second", 0.0)),
        "estimated_forward_steps": int(metrics.get("estimated_forward_steps", 0)),
        "action_label_mode": metrics.get("action_label_mode"),
        "candidate_order_mode": metrics.get("candidate_order_mode"),
        "candidate_order_seed": metrics.get("candidate_order_seed"),
        "recall@5": selector.get("recall@5"),
        "jaccard@5": selector.get("jaccard@5"),
        "oracle_rank_ndcg@5": selector.get("oracle_rank_ndcg@5"),
        "top1_match": selector.get("top1_match"),
        "selector": selector,
    }
    return record


def write_selection_eval_markdown(path: Path, metrics: dict[str, Any]) -> None:
    selector = metrics.get("selector", {})
    controls = metrics.get("controls", {})
    refs = metrics.get("reference_metrics", {})
    lines = [
        "# LLM Action Selector Eval",
        "",
        f"- n_claims: {metrics.get('n_claims')}",
        f"- decode_strategy: {metrics.get('decode_strategy', DECODE_STRATEGY_RAW)}",
        f"- aggregation: {metrics.get('aggregation', AGGREGATION_MEAN_ZSCORE)}",
        f"- calibration_mode: {metrics.get('calibration_mode', CALIBRATION_MODE_NONE)}",
        f"- elapsed_seconds: {float(metrics.get('elapsed_seconds', math.nan)):.3f}",
        f"- claims_per_second: {float(metrics.get('claims_per_second', math.nan)):.4f}",
        f"- estimated_forward_steps: {metrics.get('estimated_forward_steps')}",
        f"- selector Jaccard@5: {float(selector.get('jaccard@5', math.nan)):.4f}",
        f"- selector NDCG@5: {float(selector.get('oracle_rank_ndcg@5', math.nan)):.4f}",
        f"- selector pairwise_order_acc@5: {float(selector.get('pairwise_order_acc@5', math.nan)):.4f}",
        f"- hybrid Jaccard@5: {float(controls.get('hybrid_score_top5', {}).get('jaccard@5', math.nan)):.4f}",
    ]
    single = refs.get("single_margin_step0_static", {}).get("metrics", {})
    if single:
        delta = float(selector.get("jaccard@5", math.nan)) - float(single.get("jaccard@5", math.nan))
        lines.append(f"- single_margin_step0_static Jaccard@5: {float(single.get('jaccard@5', math.nan)):.4f}")
        lines.append(f"- delta_vs_single_margin_step0_static: {delta:.4f}")
        lines.append(f"- decision: {'go' if delta > 0 else 'no_go'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decode_diagnostics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, int] = defaultdict(int)
    candidate_idx_counts: dict[str, int] = defaultdict(int)
    hybrid_ranks: list[float] = []
    aggregate_stds: list[float] = []
    mean_appearances: list[float] = []

    for trace in traces:
        score_by_idx = {
            int(row.get("candidate_idx", idx)): row
            for idx, row in enumerate(trace.get("candidate_scores") or [])
            if isinstance(row, dict)
        }
        for step in trace.get("per_step_action_scores") or []:
            action = str(step.get("selected_action") or "")
            if action:
                action_counts[action] += 1
            try:
                selected_idx = int(step.get("selected_idx"))
            except (TypeError, ValueError):
                continue
            candidate_idx_counts[str(selected_idx)] += 1
            row = score_by_idx.get(selected_idx, {})
            try:
                hybrid_ranks.append(float(row.get("hybrid_rank", selected_idx)))
            except (TypeError, ValueError):
                hybrid_ranks.append(float(selected_idx))
            choice_scores = step.get("choice_scores") or []
            aggregates = [
                float(item.get("aggregate_score", item.get("score", 0.0)))
                for item in choice_scores
                if isinstance(item, dict)
            ]
            appearances = [
                float(item.get("n_appearances", 1.0))
                for item in choice_scores
                if isinstance(item, dict)
            ]
            if aggregates:
                aggregate_stds.append(float(np.asarray(aggregates, dtype=np.float64).std()))
            if appearances:
                mean_appearances.append(float(np.asarray(appearances, dtype=np.float64).mean()))

    return {
        "selected_action_counts": dict(sorted(action_counts.items())),
        "selected_candidate_idx_counts": dict(sorted(candidate_idx_counts.items(), key=lambda item: int(item[0]))),
        "selected_action_entropy": _entropy_from_counts(action_counts),
        "selected_candidate_idx_entropy": _entropy_from_counts(candidate_idx_counts),
        "selected_hybrid_rank_mean": float(np.mean(hybrid_ranks)) if hybrid_ranks else 0.0,
        "selected_hybrid_rank_std": float(np.std(hybrid_ranks)) if hybrid_ranks else 0.0,
        "mean_choice_aggregate_score_std": float(np.mean(aggregate_stds)) if aggregate_stds else 0.0,
        "mean_choice_n_appearances": float(np.mean(mean_appearances)) if mean_appearances else 0.0,
    }


def _entropy_from_counts(counts: dict[Any, int]) -> float:
    total = float(sum(int(value) for value in counts.values()))
    if total <= 0.0:
        return 0.0
    probs = [float(value) / total for value in counts.values() if int(value) > 0]
    return float(-sum(prob * math.log(prob) for prob in probs))


def _normalize_decode_strategy(value: str) -> str:
    strategy = str(value or DECODE_STRATEGY_RAW).strip().lower()
    if strategy not in DECODE_STRATEGIES:
        choices = ", ".join(sorted(DECODE_STRATEGIES))
        raise ValueError(f"Unsupported decode_strategy={value!r}. Use one of: {choices}.")
    return strategy


def _normalize_aggregation(value: str) -> str:
    aggregation = str(value or AGGREGATION_MEAN_ZSCORE).strip().lower()
    if aggregation not in AGGREGATION_MODES:
        choices = ", ".join(sorted(AGGREGATION_MODES))
        raise ValueError(f"Unsupported aggregation={value!r}. Use one of: {choices}.")
    return aggregation


def _normalize_calibration_mode(value: str) -> str:
    mode = str(value or CALIBRATION_MODE_NONE).strip().lower()
    if mode not in CALIBRATION_MODES:
        choices = ", ".join(sorted(CALIBRATION_MODES))
        raise ValueError(f"Unsupported calibration_mode={value!r}. Use one of: {choices}.")
    return mode


def _rank_vector_from_order(n_candidates: int, ordered: list[int]) -> np.ndarray:
    scores = np.full((int(n_candidates),), -1.0e9, dtype=np.float32)
    for rank, idx in enumerate(ordered):
        if 0 <= int(idx) < scores.size:
            scores[int(idx)] = float(len(ordered) - rank)
    return scores
