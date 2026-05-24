from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from fact_checking.selectors.llm_action import (
    ACTION_LABEL_MODE_GLOBAL_INDEX,
    CANDIDATE_ORDER_CANDIDATE_POOL,
    action_completion,
    build_action_prompt,
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
    disable_progress: bool = False,
    selector_name: str = "llm_action_selector",
) -> dict[str, Any]:
    started_at = time.time()
    traces: list[dict[str, Any]] = []
    hybrid_traces: list[dict[str, Any]] = []
    candidate_order_traces: list[dict[str, Any]] = []
    same_set_hybrid_traces: list[dict[str, Any]] = []
    same_set_candidate_traces: list[dict[str, Any]] = []
    random_traces: list[dict[str, Any]] = []

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
        )
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
        "elapsed_seconds": round(float(elapsed), 3),
        "claims_per_second": float(len(examples) / elapsed) if elapsed > 0 else 0.0,
        "estimated_forward_steps": int(sum(min(int(top_k), len(example.candidates)) for example in examples)),
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
) -> dict[str, Any]:
    selected: list[int] = []
    per_step: list[dict[str, Any]] = []
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
    return {"ordered_indices": selected[: int(top_k)], "per_step_action_scores": per_step}


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


def _rank_vector_from_order(n_candidates: int, ordered: list[int]) -> np.ndarray:
    scores = np.full((int(n_candidates),), -1.0e9, dtype=np.float32)
    for rank, idx in enumerate(ordered):
        if 0 <= int(idx) < scores.size:
            scores[int(idx)] = float(len(ordered) - rank)
    return scores
