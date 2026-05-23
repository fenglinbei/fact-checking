#!/usr/bin/env python3
"""Evaluate saved-score utility rankers as selection-only oracle selectors.

This is a no-vLLM offline probe. It trains tiny ridge rankers on VIG-lite rows
whose ``delta_margin`` came from saved Stage2 oracle step scores, then converts
the row-level scores into ordered top-k evidence lists and evaluates them with
the same selector gate metrics used by the cross-encoder/listwise/sequential
selectors.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    Stage2OracleExample,
    load_stage2_oracle_examples,
    write_json,
    write_jsonl,
)
from scripts.selectors.analyze_oracle_vig_utility import (
    FEATURE_GROUPS,
    _as_float,
    _augment_rows,
    _eval_score_baseline,
    _fit_and_eval_feature_set,
    _matrix,
    _predict_ridge,
    _split_events,
)


BASELINE_SCORE_COLUMNS: dict[str, str] = {
    "single_margin": "single_margin",
    "single_gold_logprob": "single_gold_logprob",
    "hybrid_score": "hybrid_score",
    "minus_hybrid_rank": "minus_hybrid_rank",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train/evaluate lightweight saved-score utility rankers and report "
            "Stage2 ordered selection metrics."
        )
    )
    p.add_argument("--vig-cache", nargs="+", required=True, help="Eval VIG-lite row JSONL(s).")
    p.add_argument(
        "--train-vig-cache",
        nargs="+",
        default=None,
        help=(
            "Optional train VIG-lite row JSONL(s). If omitted, --vig-cache is "
            "split by event_id into train/eval subsets."
        ),
    )
    p.add_argument("--oracle-results", required=True, help="Stage2 oracle JSONL for eval traces.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260522)
    p.add_argument("--write-all-traces", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    traces_dir = out_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    examples = load_stage2_oracle_examples(
        args.oracle_results,
        expected_fingerprint=args.expected_chunk_mmr_fingerprint,
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=str(args.filter_policy),
        min_margin=float(args.min_margin),
        sample_limit=args.sample_limit,
    )
    if not examples:
        raise ValueError("No Stage2 oracle examples after audit/filtering.")
    examples_by_event = {example.event_id: example for example in examples}

    eval_rows_all = _filter_rows_for_examples(
        _augment_rows(_load_rows(args.vig_cache)),
        examples_by_event,
    )
    if not eval_rows_all:
        raise ValueError("No eval VIG rows matched the audited oracle examples.")

    if args.train_vig_cache:
        train_rows = _augment_rows(_load_rows(args.train_vig_cache))
        eval_rows = eval_rows_all
        train_event_ids = sorted({str(row.get("event_id")) for row in train_rows})
        eval_event_ids = sorted({str(row.get("event_id")) for row in eval_rows})
        split_mode = "external_train_eval"
    else:
        all_event_ids = sorted({str(row.get("event_id")) for row in eval_rows_all})
        train_events, eval_events = _split_events(
            all_event_ids,
            test_fraction=float(args.test_fraction),
            seed=int(args.seed),
        )
        train_rows = [row for row in eval_rows_all if str(row.get("event_id")) in train_events]
        eval_rows = [row for row in eval_rows_all if str(row.get("event_id")) in eval_events]
        train_event_ids = sorted(train_events)
        eval_event_ids = sorted(eval_events)
        split_mode = "event_split"

    if not train_rows or not eval_rows:
        raise ValueError("Train/eval VIG rows are empty; adjust --test-fraction or inputs.")

    eval_event_set = set(eval_event_ids)
    eval_examples = [example for example in examples if example.event_id in eval_event_set]
    if not eval_examples:
        raise ValueError("No eval oracle examples matched eval VIG event ids.")

    feature_sets = _feature_sets()
    model_results: list[dict[str, Any]] = []
    models: dict[str, dict[str, Any]] = {}
    for name, features in feature_sets.items():
        result, model = _fit_and_eval_feature_set(
            name,
            features,
            train_rows=train_rows,
            test_rows=eval_rows,
            ridge_alpha=float(args.ridge_alpha),
        )
        model_results.append(result)
        models[name] = model

    row_baselines = [
        _eval_score_baseline(name, column, eval_rows)
        for name, column in BASELINE_SCORE_COLUMNS.items()
    ]
    row_baselines.insert(0, _eval_score_baseline("true_delta_margin", "delta_margin", eval_rows))

    rows_by_event_step = _rows_by_event_step(eval_rows)
    method_traces: dict[str, list[dict[str, Any]]] = {}
    method_specs: list[dict[str, Any]] = []

    _add_teacher_forced_method(
        method_traces,
        method_specs,
        eval_examples,
        rows_by_event_step=rows_by_event_step,
        name="true_delta_teacher_forced",
        score_column="delta_margin",
        top_k=int(args.top_k),
    )
    for model_name in ("all", "no_prefix_state", "retrieval+single_verifier", "group:single_verifier"):
        model = models.get(model_name)
        if model:
            _add_teacher_forced_method(
                method_traces,
                method_specs,
                eval_examples,
                rows_by_event_step=rows_by_event_step,
                name=f"ridge_{_slug(model_name)}_teacher_forced",
                model=model,
                top_k=int(args.top_k),
            )
    for model_name in ("all", "no_prefix_state", "retrieval+single_verifier", "group:single_verifier"):
        model = models.get(model_name)
        if model:
            _add_step0_static_method(
                method_traces,
                method_specs,
                eval_examples,
                rows_by_event_step=rows_by_event_step,
                name=f"ridge_{_slug(model_name)}_step0_static",
                model=model,
                top_k=int(args.top_k),
            )
    for score_name, column in (
        ("single_margin_step0_static", "single_margin"),
        ("single_gold_logprob_step0_static", "single_gold_logprob"),
        ("hybrid_score_step0_static", "hybrid_score"),
    ):
        _add_step0_static_method(
            method_traces,
            method_specs,
            eval_examples,
            rows_by_event_step=rows_by_event_step,
            name=score_name,
            score_column=column,
            top_k=int(args.top_k),
        )

    control_traces = _control_traces(eval_examples, top_k=int(args.top_k))
    selection_metrics = {
        name: summarize_ordered_selection(traces)
        for name, traces in sorted(method_traces.items())
    }
    controls = {
        name: summarize_ordered_selection(traces)
        for name, traces in sorted(control_traces.items())
    }
    decision = _decision(selection_metrics, controls)

    trace_names_to_write = set(method_traces)
    if not bool(args.write_all_traces):
        trace_names_to_write = {
            "true_delta_teacher_forced",
            "ridge_all_teacher_forced",
            "ridge_no_prefix_state_teacher_forced",
            "ridge_all_step0_static",
            "ridge_no_prefix_state_step0_static",
            "single_margin_step0_static",
            "single_gold_logprob_step0_static",
        }
    for name, traces in sorted(method_traces.items()):
        if name in trace_names_to_write:
            write_jsonl(traces_dir / f"{name}.jsonl", traces)
    for name, traces in sorted(control_traces.items()):
        write_jsonl(traces_dir / f"control_{name}.jsonl", traces)

    ranker_models = {
        name: _jsonable_model(model)
        for name, model in sorted(models.items())
    }
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "split": str(args.split),
        "split_mode": split_mode,
        "vig_cache": [str(path) for path in args.vig_cache],
        "train_vig_cache": [str(path) for path in (args.train_vig_cache or [])],
        "oracle_results": str(args.oracle_results),
        "output_dir": str(out_dir),
        "filter_policy": str(args.filter_policy),
        "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        "top_k": int(args.top_k),
        "max_candidates": int(args.max_candidates),
        "ridge_alpha": float(args.ridge_alpha),
        "seed": int(args.seed),
        "test_fraction": float(args.test_fraction),
        "n_train_rows": int(len(train_rows)),
        "n_eval_rows": int(len(eval_rows)),
        "n_train_events": int(len(train_event_ids)),
        "n_eval_events": int(len(eval_event_ids)),
        "n_eval_examples": int(len(eval_examples)),
        "feature_group_results": model_results,
        "row_score_baselines": row_baselines,
        "method_specs": method_specs,
        "selection_metrics": selection_metrics,
        "controls": controls,
        "decision": decision,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    write_json(out_dir / "selection_metrics.json", metrics)
    write_json(out_dir / "ranker_models.json", ranker_models)
    _write_markdown(out_dir / "analysis.md", metrics)

    best = _best_method(selection_metrics)
    print(f"Wrote saved-score utility ranker metrics: {out_dir / 'selection_metrics.json'}")
    print(
        "Decision={decision}; best={best} Jaccard@5={jac:.4f}, Top1={top1:.4f}; "
        "single_margin_step0 Jaccard@5={sm_jac:.4f}".format(
            decision=decision["decision"],
            best=best,
            jac=_metric(selection_metrics.get(best), "jaccard@5"),
            top1=_metric(selection_metrics.get(best), "top1_match"),
            sm_jac=_metric(selection_metrics.get("single_margin_step0_static"), "jaccard@5"),
        )
    )


def _load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        expanded = (
            [Path(p) for p in sorted(glob.glob(raw_path))]
            if any(ch in raw_path for ch in "*?[]")
            else [Path(raw_path)]
        )
        for path in expanded:
            if not path.exists():
                raise FileNotFoundError(f"Input JSONL not found: {path}")
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def _filter_rows_for_examples(
    rows: list[dict[str, Any]],
    examples_by_event: dict[str, Stage2OracleExample],
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("event_id")) in examples_by_event]


def _feature_sets() -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    for group, features in FEATURE_GROUPS.items():
        sets[f"group:{group}"] = list(features)
    sets["retrieval+single_verifier"] = (
        list(FEATURE_GROUPS["retrieval"]) + list(FEATURE_GROUPS["single_verifier"])
    )
    sets["no_prefix_state"] = (
        list(FEATURE_GROUPS["retrieval"])
        + list(FEATURE_GROUPS["text_overlap"])
        + list(FEATURE_GROUPS["single_verifier"])
    )
    all_features: list[str] = []
    for features in FEATURE_GROUPS.values():
        all_features.extend(features)
    sets["all"] = all_features
    return sets


def _rows_by_event_step(rows: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row.get("event_id"))][int(row.get("step", -1))].append(row)
    for by_step in grouped.values():
        for step, step_rows in by_step.items():
            by_step[step] = sorted(step_rows, key=lambda row: int(row.get("candidate_idx", -1)))
    return grouped


def _add_teacher_forced_method(
    method_traces: dict[str, list[dict[str, Any]]],
    method_specs: list[dict[str, Any]],
    examples: list[Stage2OracleExample],
    *,
    rows_by_event_step: dict[str, dict[int, list[dict[str, Any]]]],
    name: str,
    top_k: int,
    model: dict[str, Any] | None = None,
    score_column: str | None = None,
) -> None:
    traces = [
        _teacher_forced_trace(
            example,
            rows_by_event_step=rows_by_event_step,
            name=name,
            model=model,
            score_column=score_column,
            top_k=top_k,
        )
        for example in examples
    ]
    method_traces[name] = traces
    method_specs.append(_method_spec(name, "teacher_forced_prefix", model=model, score_column=score_column))


def _add_step0_static_method(
    method_traces: dict[str, list[dict[str, Any]]],
    method_specs: list[dict[str, Any]],
    examples: list[Stage2OracleExample],
    *,
    rows_by_event_step: dict[str, dict[int, list[dict[str, Any]]]],
    name: str,
    top_k: int,
    model: dict[str, Any] | None = None,
    score_column: str | None = None,
) -> None:
    traces = [
        _step0_static_trace(
            example,
            rows_by_event_step=rows_by_event_step,
            name=name,
            model=model,
            score_column=score_column,
            top_k=top_k,
        )
        for example in examples
    ]
    method_traces[name] = traces
    method_specs.append(_method_spec(name, "step0_static", model=model, score_column=score_column))


def _teacher_forced_trace(
    example: Stage2OracleExample,
    *,
    rows_by_event_step: dict[str, dict[int, list[dict[str, Any]]]],
    name: str,
    top_k: int,
    model: dict[str, Any] | None = None,
    score_column: str | None = None,
) -> dict[str, Any]:
    selected: list[int] = []
    selected_set: set[int] = set()
    per_step_scores: list[dict[str, Any]] = []
    by_step = rows_by_event_step.get(example.event_id, {})
    for step in range(int(top_k)):
        rows = [row for row in by_step.get(step, []) if int(row.get("candidate_idx", -1)) not in selected_set]
        if not rows:
            continue
        scores = _score_rows(rows, model=model, score_column=score_column)
        best_pos = int(np.argmax(scores))
        best_idx = int(rows[best_pos].get("candidate_idx", -1))
        if best_idx < 0:
            continue
        selected.append(best_idx)
        selected_set.add(best_idx)
        per_step_scores.append(
            {
                "step": int(step),
                "selected_idx": int(best_idx),
                "selected_score": float(scores[best_pos]),
                "oracle_idx": int(example.selected_indices[step])
                if step < len(example.selected_indices)
                else None,
            }
        )

    score_vector = _rank_vector_from_order(len(example.candidates), selected)
    trace = build_selection_trace(example, score_vector, selector_name=name, top_k=int(top_k))
    trace["selector_ordered_indices"] = selected[: int(top_k)]
    trace.update(
        _metrics_for_order(example, selected[: int(top_k)], top_k=int(top_k))
    )
    trace["teacher_forced_prefix"] = True
    trace["per_step_selected_scores"] = per_step_scores
    return trace


def _step0_static_trace(
    example: Stage2OracleExample,
    *,
    rows_by_event_step: dict[str, dict[int, list[dict[str, Any]]]],
    name: str,
    top_k: int,
    model: dict[str, Any] | None = None,
    score_column: str | None = None,
) -> dict[str, Any]:
    rows = rows_by_event_step.get(example.event_id, {}).get(0, [])
    scores = np.full((len(example.candidates),), -1.0e9, dtype=np.float32)
    if rows:
        row_scores = _score_rows(rows, model=model, score_column=score_column)
        for row, score in zip(rows, row_scores):
            idx = int(row.get("candidate_idx", -1))
            if 0 <= idx < scores.size:
                scores[idx] = float(score)
    trace = build_selection_trace(example, scores, selector_name=name, top_k=int(top_k))
    trace["step0_static"] = True
    trace["n_scored_step0_candidates"] = int(sum(1 for value in scores if value > -1.0e8))
    return trace


def _score_rows(
    rows: list[dict[str, Any]],
    *,
    model: dict[str, Any] | None = None,
    score_column: str | None = None,
) -> np.ndarray:
    if model is not None:
        features = list(model["features"])
        x = _matrix(rows, features)
        means = np.asarray(model["means"], dtype=np.float64)
        stds = np.asarray(model["stds"], dtype=np.float64)
        x_std = (x - means) / stds
        return _predict_ridge(x_std, np.asarray(model["coef"], dtype=np.float64))
    if score_column is None:
        raise ValueError("Either model or score_column must be supplied.")
    return np.asarray([_as_float(row.get(score_column)) for row in rows], dtype=np.float64)


def _rank_vector_from_order(n_candidates: int, ordered: list[int]) -> np.ndarray:
    scores = np.full((int(n_candidates),), -1.0e9, dtype=np.float32)
    for rank, idx in enumerate(ordered):
        if 0 <= int(idx) < scores.size:
            scores[int(idx)] = float(len(ordered) - rank)
    return scores


def _metrics_for_order(
    example: Stage2OracleExample,
    ordered: list[int],
    *,
    top_k: int,
) -> dict[str, Any]:
    from fact_checking.selectors.metrics import ordered_selection_metrics

    return ordered_selection_metrics(example.selected_indices, ordered, top_k=int(top_k))


def _control_traces(
    examples: list[Stage2OracleExample],
    *,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    controls: dict[str, list[dict[str, Any]]] = {
        "hybrid_score_top5": [],
        "candidate_pool_order_top5": [],
    }
    for example in examples:
        base = build_selection_trace(
            example,
            np.zeros((len(example.candidates),), dtype=np.float32),
            selector_name="control_stub",
            top_k=int(top_k),
        )
        controls["hybrid_score_top5"].append(
            build_order_control_trace(
                base,
                ranked_indices_from_hybrid(example, top_k=int(top_k)),
                selector_name="hybrid_score_top5",
                top_k=int(top_k),
            )
        )
        controls["candidate_pool_order_top5"].append(
            build_order_control_trace(
                base,
                ranked_indices_from_candidate_pool(example, top_k=int(top_k)),
                selector_name="candidate_pool_order_top5",
                top_k=int(top_k),
            )
        )
    return controls


def _method_spec(
    name: str,
    rollout_mode: str,
    *,
    model: dict[str, Any] | None,
    score_column: str | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "rollout_mode": rollout_mode,
        "score_column": score_column,
        "model_name": model.get("name") if model else None,
        "features": list(model.get("features", [])) if model else [],
    }


def _jsonable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": model.get("name"),
        "features": list(model.get("features", [])),
        "means": [float(x) for x in np.asarray(model.get("means", []), dtype=np.float64).tolist()],
        "stds": [float(x) for x in np.asarray(model.get("stds", []), dtype=np.float64).tolist()],
        "coef": [float(x) for x in np.asarray(model.get("coef", []), dtype=np.float64).tolist()],
        "test_base": model.get("test_base", {}),
    }


def _decision(selection_metrics: dict[str, dict[str, Any]], controls: dict[str, dict[str, Any]]) -> dict[str, Any]:
    true_delta = selection_metrics.get("true_delta_teacher_forced", {})
    ridge_static = selection_metrics.get("ridge_no_prefix_state_step0_static", {})
    ridge_tf = selection_metrics.get("ridge_all_teacher_forced", {})
    single_static = selection_metrics.get("single_margin_step0_static", {})
    hybrid = controls.get("hybrid_score_top5", {})

    true_ok = _metric(true_delta, "ordered_exact_match@5") >= 0.98
    static_beats_single = _metric(ridge_static, "jaccard@5") > _metric(single_static, "jaccard@5") + 0.01
    static_beats_hybrid = _metric(ridge_static, "jaccard@5") > _metric(hybrid, "jaccard@5") + 0.03
    tf_has_signal = _metric(ridge_tf, "jaccard@5") > _metric(hybrid, "jaccard@5") + 0.03

    if not true_ok:
        decision = "fix_saved_score_ranker_alignment"
    elif static_beats_single and static_beats_hybrid:
        decision = "go_static_utility_selector"
    elif tf_has_signal:
        decision = "prefix_aware_signal_needs_train_step_scores"
    else:
        decision = "no_go_or_single_margin_baseline"
    return {
        "decision": decision,
        "true_delta_teacher_forced_exact_ok": bool(true_ok),
        "ridge_no_prefix_static_beats_single_margin": bool(static_beats_single),
        "ridge_no_prefix_static_beats_hybrid": bool(static_beats_hybrid),
        "ridge_all_teacher_forced_has_signal": bool(tf_has_signal),
        "true_delta_teacher_forced_ordered_exact_match@5": _metric(true_delta, "ordered_exact_match@5"),
        "ridge_all_teacher_forced_jaccard@5": _metric(ridge_tf, "jaccard@5"),
        "ridge_no_prefix_state_step0_static_jaccard@5": _metric(ridge_static, "jaccard@5"),
        "single_margin_step0_static_jaccard@5": _metric(single_static, "jaccard@5"),
        "hybrid_score_top5_jaccard@5": _metric(hybrid, "jaccard@5"),
    }


def _write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Saved-Score Utility Ranker Eval",
        "",
        f"- decision: `{metrics['decision']['decision']}`",
        f"- split_mode: `{metrics['split_mode']}`",
        f"- train/eval events: {metrics['n_train_events']} / {metrics['n_eval_events']}",
        f"- train/eval rows: {metrics['n_train_rows']} / {metrics['n_eval_rows']}",
        "",
        "## Selection Metrics",
        "",
        "| method | recall@5 | jaccard@5 | top1 | ndcg@5 | pairwise_order_acc@5 | exact@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(metrics["selection_metrics"].items()):
        lines.append(_selection_table_row(name, row))
    lines.extend(
        [
            "",
            "## Controls",
            "",
            "| method | recall@5 | jaccard@5 | top1 | ndcg@5 | pairwise_order_acc@5 | exact@5 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in sorted(metrics["controls"].items()):
        lines.append(_selection_table_row(name, row))
    lines.extend(
        [
            "",
            "## Row-Level Fit",
            "",
            "| feature set | R2 | RMSE | target AUROC | step top1 match |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["feature_group_results"]:
        lines.append(
            f"| {row['name']} | {_fmt(row.get('r2'))} | {_fmt(row.get('rmse'))} | "
            f"{_fmt(row.get('target_auroc'))} | {_fmt(row.get('step_top1_match'))} |"
        )
    lines.extend(
        [
            "",
            "## Score Baselines",
            "",
            "| score | target AUROC | step top1 match |",
            "|---|---:|---:|",
        ]
    )
    for row in metrics["row_score_baselines"]:
        lines.append(
            f"| {row['name']} | {_fmt(row.get('target_auroc'))} | {_fmt(row.get('step_top1_match'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`true_delta_teacher_forced` is an alignment check for the saved-score rows and the oracle selected order.",
            "`step0_static` is the deployable single-pass proxy; `teacher_forced` uses oracle prefixes and is only a prefix-aware signal probe.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection_table_row(name: str, row: dict[str, Any]) -> str:
    return (
        f"| {name} | {_fmt(row.get('recall@5'))} | {_fmt(row.get('jaccard@5'))} | "
        f"{_fmt(row.get('top1_match'))} | {_fmt(row.get('oracle_rank_ndcg@5'))} | "
        f"{_fmt(row.get('pairwise_order_acc@5'))} | {_fmt(row.get('ordered_exact_match@5'))} |"
    )


def _best_method(selection_metrics: dict[str, dict[str, Any]]) -> str:
    candidates = [
        name for name in selection_metrics
        if name != "true_delta_teacher_forced"
    ]
    if not candidates:
        return ""
    return max(candidates, key=lambda name: _metric(selection_metrics.get(name), "jaccard@5"))


def _metric(row: dict[str, Any] | None, key: str) -> float:
    if not row:
        return math.nan
    try:
        return float(row.get(key, math.nan))
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(number):
        return "nan"
    return f"{number:.4f}"


def _slug(name: str) -> str:
    return (
        str(name)
        .replace("group:", "")
        .replace("+", "_plus_")
        .replace("-", "_")
        .replace("/", "_")
    )


if __name__ == "__main__":
    main()
