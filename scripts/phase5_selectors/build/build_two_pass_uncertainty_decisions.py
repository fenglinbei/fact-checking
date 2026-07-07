#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.build.candidates import _load_prompt_tokenizer  # noqa: E402
from fact_checking.data.constants import labels_for_schema  # noqa: E402
from fact_checking.data.io import load_split  # noqa: E402
from scripts.phase5_selectors.build.build_mrec_reward_cache import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_TEACHER_RUN_DIR,
    _init_scorer,
    _load_score_cache,
    _score_request_for_indices,
    _score_requests,
    _scoring_fingerprint,
    _teacher_checkpoint,
)
from scripts.phase5_selectors.build.build_trace_verifier_data import (  # noqa: E402
    _load_experiment_config,
    _mrec_step_by_selector_idx,
    _normalize_prompt_evidence_config,
    _ordered_trace_indices,
    _prompt_cfg_for_trace_style,
    _select_prompt_evidence_indices,
)


RUN_VERSION = "mrec_two_pass_uncertainty_v0_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build two-pass uncertainty prompt evidence decisions.")
    parser.add_argument("--trace", required=True, help="Fullpool MREC selection_trace_<split>.jsonl.")
    parser.add_argument("--raw", required=True, help="Raw split JSON file.")
    parser.add_argument("--config", required=True, help="MREC policy YAML.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--dataset", default="")
    parser.add_argument("--label-schema", default="")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-file", default="")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--trace-prompt-style", default="")
    parser.add_argument("--prompt-max-length", type=int, default=0)
    parser.add_argument("--prompt-model-name-or-path", default="")
    parser.add_argument("--teacher-run-dir", default="")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--label-prefix", default="Label:")
    parser.add_argument("--scoring-backend", default="", choices=["", "auto", "vllm", "transformers"])
    parser.add_argument("--vllm-tokenizer-path", default="")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=4)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-tokenizer-mode", default="")
    parser.add_argument("--vllm-config-format", default="")
    parser.add_argument("--vllm-load-format", default="")
    parser.add_argument("--vllm-max-model-len", type=int, default=1032)
    parser.add_argument("--vllm-prompt-batch-size", type=int, default=6000)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-lora-mode", default="", choices=["", "dynamic", "merged"])
    parser.add_argument("--vllm-merge-lora-cache-dir", default="")
    parser.add_argument("--vllm-merge-lora-force-rebuild", action="store_true")
    parser.add_argument("--transformers-tokenizer-path", default="")
    parser.add_argument("--transformers-device", default="auto")
    parser.add_argument(
        "--transformers-dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--transformers-prompt-batch-size", type=int, default=24)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    cfg = _load_experiment_config(str(args.config))
    _apply_config_defaults(args, cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_trace_path = out_dir / f"two_pass_uncertainty_score_trace_{args.split}.jsonl"
    raw_scores_path = out_dir / f"two_pass_uncertainty_raw_scores_{args.split}.jsonl"
    decisions_path = out_dir / f"two_pass_uncertainty_decisions_{args.split}.jsonl"
    manifest_path = out_dir / f"two_pass_uncertainty_manifest_{args.split}.json"

    prompt_cfg = _prompt_cfg_from_config(args, cfg)
    tokenizer = _load_prompt_tokenizer(str(prompt_cfg["model_name_or_path"]))
    checkpoint = _teacher_checkpoint(args)
    scorer = _init_scorer(args, checkpoint)
    checkpoint = dict(checkpoint)
    checkpoint["scoring_fingerprint"] = _scoring_fingerprint(args, scorer)
    score_cache = _load_score_cache(raw_scores_path) if bool(args.resume) else {}

    raw_by_event = {
        sample.event_id: sample
        for sample in load_split(Path(args.raw), dataset=str(args.dataset), label_schema=str(args.label_schema))
    }
    trace_rows = _read_jsonl(Path(args.trace), sample_limit=int(args.sample_limit))
    prompt_evidence = _normalize_prompt_evidence_config(
        dict(cfg.get("prompt_evidence") or {}),
        fallback_top_k=0,
        prompt_cfg=prompt_cfg,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_version": RUN_VERSION,
        "trace": str(args.trace),
        "raw": str(args.raw),
        "config": str(args.config),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "dataset": str(args.dataset),
        "label_schema": str(args.label_schema),
        "prompt_config": prompt_cfg,
        "prompt_evidence": prompt_evidence,
        "teacher": {
            **checkpoint,
            "scoring_backend_requested": str(args.scoring_backend),
            "scoring_backend": str(getattr(scorer, "backend_name", args.scoring_backend)),
            "vllm_lora_mode": str(getattr(scorer, "vllm_lora_mode", args.vllm_lora_mode)),
        },
        "score_trace_path": str(score_trace_path),
        "raw_scores_path": str(raw_scores_path),
        "decisions_path": str(decisions_path),
        "status": "running",
    }
    _write_json(manifest_path, manifest)

    score_rows, generated_score_rows = _build_score_traces(
        trace_rows,
        raw_by_event=raw_by_event,
        args=args,
        tokenizer=tokenizer,
        prompt_cfg=prompt_cfg,
        prompt_evidence=prompt_evidence,
        checkpoint=checkpoint,
        scorer=scorer,
        score_cache=score_cache,
    )
    _write_jsonl(score_trace_path, score_rows)
    if generated_score_rows:
        mode = "a" if bool(args.resume) else "w"
        with raw_scores_path.open(mode, encoding="utf-8") as handle:
            for row in generated_score_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    calibration_file = Path(args.calibration_file) if str(args.calibration_file or "") else out_dir / "two_pass_uncertainty_calibration.json"
    if bool(args.calibrate):
        calibration = _calibrate_threshold(score_rows, label_schema=str(args.label_schema))
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        _write_json(calibration_file, calibration)
        threshold = float(calibration["threshold"])
    else:
        threshold = _resolve_threshold(args, calibration_file)

    decisions = [
        _select_decision_from_score_trace(
            event_id=str(row["event_id"]),
            split=str(args.split),
            threshold=threshold,
            score_trace=list(row.get("score_trace") or []),
        )
        for row in score_rows
    ]
    _write_jsonl(decisions_path, decisions)

    diagnostics = _decision_diagnostics(decisions)
    manifest.update(
        {
            "status": "completed",
            "threshold": float(threshold),
            "calibration_file": str(calibration_file),
            "diagnostics": diagnostics,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
    )
    _write_json(manifest_path, manifest)
    print(f"Wrote two-pass score traces: {score_trace_path}")
    print(f"Wrote two-pass decisions: {decisions_path}")
    print(f"threshold={threshold:.6f} diagnostics={diagnostics}")
    return 0


def _build_score_traces(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    raw_by_event: Mapping[str, Any],
    args: argparse.Namespace,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    prompt_evidence: dict[str, Any],
    checkpoint: Mapping[str, Any],
    scorer: Any,
    score_cache: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    iterator: Sequence[Mapping[str, Any]] = trace_rows
    if not bool(args.no_progress):
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(trace_rows, desc=f"two-pass-scores [{args.split}]", unit="claim", dynamic_ncols=True)
        except Exception:
            pass

    out_rows: list[dict[str, Any]] = []
    generated_score_rows: list[dict[str, Any]] = []
    stats = Counter()
    for trace in iterator:
        event_id = str(trace.get("event_id") or "")
        if not event_id:
            stats["missing_event_id"] += 1
            continue
        sample = raw_by_event.get(event_id)
        if sample is None:
            stats["missing_raw_sample"] += 1
            continue
        score_trace, new_score_rows = _score_trace_for_event(
            trace,
            sample=sample,
            args=args,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            prompt_evidence=prompt_evidence,
            checkpoint=checkpoint,
            scorer=scorer,
            score_cache=score_cache,
        )
        generated_score_rows.extend(new_score_rows)
        out_rows.append(
            {
                "event_id": event_id,
                "split": str(args.split),
                "gold_label": str(sample.label),
                "score_trace": score_trace,
            }
        )
        stats["events"] += 1
    if not out_rows:
        raise RuntimeError(f"No two-pass score traces were built. stats={dict(stats)}")
    return out_rows, generated_score_rows


def _score_trace_for_event(
    trace: Mapping[str, Any],
    *,
    sample: Any,
    args: argparse.Namespace,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    prompt_evidence: dict[str, Any],
    checkpoint: Mapping[str, Any],
    scorer: Any,
    score_cache: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [dict(candidate) for candidate in trace.get("candidate_pool") or []]
    if not candidates:
        raise ValueError(f"{trace.get('event_id')}: trace has no candidate_pool")
    ordered_indices = _ordered_trace_indices(dict(trace))
    ordered_indices = [idx for idx in ordered_indices if 0 <= int(idx) < len(candidates)]
    if not ordered_indices:
        ordered_indices = list(range(len(candidates)))

    initial_config = dict(prompt_evidence)
    initial_config["policy"] = "state_budget"
    initial_config["max_evidence_count"] = 0
    initial_decision = _select_prompt_evidence_indices(
        dict(trace),
        ordered_indices=ordered_indices,
        config=initial_config,
        event_id=str(trace.get("event_id") or ""),
        split=str(args.split),
    )
    initial_indices = [int(idx) for idx in initial_decision["selected_indices"]]
    ladders = _expansion_ladder_indices(ordered_indices, initial_indices)
    step_by_idx = _mrec_step_by_selector_idx(dict(trace))
    claim_atoms = [dict(atom) for atom in trace.get("claim_atoms") or [] if isinstance(atom, Mapping)]
    requests = []
    effective_prompt_budget = _effective_prompt_budget(prompt_evidence)
    for ladder_idx, selected_indices in enumerate(ladders):
        role = "initial" if ladder_idx == 0 else "expanded"
        mrec_steps = [dict(step_by_idx[idx]) for idx in selected_indices if idx in step_by_idx]
        request = _score_request_for_indices(
            trace,
            sample=sample,
            candidates=candidates,
            claim_atoms=claim_atoms,
            selected_indices=selected_indices,
            mrec_steps=mrec_steps,
            candidate_idx=None,
            role=role,
            step=ladder_idx,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            checkpoint=dict(checkpoint),
        )
        prompt_token_count = int(request.prompt_row.get("prompt_token_count") or 0)
        was_truncated = bool(request.prompt_row.get("was_truncated"))
        if role != "initial" and effective_prompt_budget > 0 and prompt_token_count > effective_prompt_budget:
            break
        if role != "initial" and was_truncated:
            break
        requests.append(request)

    scored, new_score_rows = _score_requests(requests, scorer=scorer, score_cache=score_cache)
    score_trace: list[dict[str, Any]] = []
    for request in requests:
        score = dict(scored[request.cache_key])
        prompt_row = dict(request.prompt_row)
        selected_indices = [int(idx) for idx in request.metadata.get("selected_indices") or []]
        score_trace.append(
            {
                "role": str(request.metadata.get("role") or ""),
                "selected_indices": selected_indices,
                "evidence_count": int(len(selected_indices)),
                "pred_margin": float(score.get("pred_margin", 0.0)),
                "pred_label": str(score.get("pred_label") or ""),
                "pred_letter": str(score.get("pred_letter") or ""),
                "margin": _float_or_none(score.get("margin")),
                "prompt_token_count": int(prompt_row.get("prompt_token_count") or 0),
                "was_truncated": bool(prompt_row.get("was_truncated")),
                "cache_key": str(request.cache_key),
                "scoring_backend": str(score.get("scoring_backend") or getattr(scorer, "backend_name", "")),
            }
        )
    return score_trace, new_score_rows


def _select_decision_from_score_trace(
    *,
    event_id: str,
    split: str,
    threshold: float,
    score_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    if not score_trace:
        raise ValueError(f"{split}:{event_id}: empty score_trace")
    initial = dict(score_trace[0])
    initial_indices = [int(idx) for idx in initial.get("selected_indices") or []]
    if _pred_margin(initial) >= float(threshold):
        selected = initial
        stop_reason = "confident_initial"
    else:
        confident = [
            dict(row)
            for row in score_trace[1:]
            if _pred_margin(row) >= float(threshold)
        ]
        if confident:
            selected = confident[0]
            stop_reason = "expanded_confident"
        else:
            selected = max(
                (dict(row) for row in score_trace),
                key=lambda row: (_pred_margin(row), -len(row.get("selected_indices") or [])),
            )
            stop_reason = "best_available"
    selected_indices = [int(idx) for idx in selected.get("selected_indices") or []]
    return {
        "decision_version": RUN_VERSION,
        "event_id": str(event_id),
        "split": str(split),
        "initial_indices": initial_indices,
        "selected_indices": selected_indices,
        "threshold": float(threshold),
        "uncertainty_metric": "pred_margin",
        "uncertainty_margin": float(_pred_margin(selected)),
        "initial_uncertainty_margin": float(_pred_margin(initial)),
        "prompt_evidence_expanded": bool(selected_indices != initial_indices),
        "score_trace": [dict(row) for row in score_trace],
        "stop_reason": stop_reason,
        "pred_label": str(selected.get("pred_label") or ""),
        "pred_letter": str(selected.get("pred_letter") or ""),
        "prompt_token_count_before_final_build": int(selected.get("prompt_token_count") or 0),
    }


def _calibrate_threshold(
    scored_events: Sequence[Mapping[str, Any]],
    *,
    label_schema: str,
) -> dict[str, Any]:
    initial_margins = [
        _pred_margin((event.get("score_trace") or [{}])[0])
        for event in scored_events
        if event.get("score_trace")
    ]
    thresholds = _threshold_grid(initial_margins)
    labels = labels_for_schema(label_schema)
    trials: list[dict[str, Any]] = []
    for threshold in thresholds:
        decisions = [
            _select_decision_from_score_trace(
                event_id=str(event.get("event_id") or ""),
                split=str(event.get("split") or "val"),
                threshold=float(threshold),
                score_trace=list(event.get("score_trace") or []),
            )
            for event in scored_events
        ]
        gold = [str(event.get("gold_label") or "") for event in scored_events]
        pred = [str(decision.get("pred_label") or "") for decision in decisions]
        evidence_counts = [len(decision.get("selected_indices") or []) for decision in decisions]
        truncation_rate = sum(
            1
            for decision in decisions
            if _selected_score_row(decision).get("was_truncated")
        ) / max(len(decisions), 1)
        trials.append(
            {
                "threshold": float(threshold),
                "macro_f1": _macro_f1(gold, pred, labels=labels),
                "truncation_rate": float(truncation_rate),
                "mean_evidence_count": float(sum(evidence_counts) / max(len(evidence_counts), 1)),
                "expanded_rate": float(
                    sum(1 for decision in decisions if bool(decision.get("prompt_evidence_expanded")))
                    / max(len(decisions), 1)
                ),
            }
        )
    best = max(
        trials,
        key=lambda row: (
            float(row["macro_f1"]),
            -float(row["truncation_rate"]),
            -float(row["mean_evidence_count"]),
            float(row["threshold"]),
        ),
    )
    return {
        "run_version": RUN_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_schema": str(label_schema),
        "metric": "pred_margin",
        "threshold": float(best["threshold"]),
        "selection_priority": ["macro_f1", "truncation_rate", "mean_evidence_count"],
        "best": best,
        "trials": trials,
    }


def _apply_config_defaults(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    data_cfg = dict((cfg.get("build") or {}).get("data") or {})
    prompt_cfg = dict((cfg.get("build") or {}).get("prompt") or {})
    two_pass = dict((cfg.get("prompt_evidence") or {}).get("two_pass_uncertainty") or {})
    args.dataset = str(args.dataset or data_cfg.get("dataset") or cfg.get("dataset") or "liar_raw")
    args.label_schema = str(args.label_schema or data_cfg.get("label_schema") or cfg.get("label_schema") or "liar6")
    args.trace_prompt_style = str(
        args.trace_prompt_style
        or (cfg.get("prompt_evidence") or {}).get("trace_prompt_style")
        or "mrec_min"
    )
    args.prompt_max_length = int(args.prompt_max_length or prompt_cfg.get("max_length") or 1024)
    args.prompt_model_name_or_path = str(
        args.prompt_model_name_or_path
        or prompt_cfg.get("model_name_or_path")
        or two_pass.get("base_model")
        or DEFAULT_BASE_MODEL
    )
    args.teacher_run_dir = str(args.teacher_run_dir or two_pass.get("teacher_run_dir") or DEFAULT_TEACHER_RUN_DIR)
    args.teacher_checkpoint = str(args.teacher_checkpoint or two_pass.get("teacher_checkpoint") or "best")
    args.base_model = str(args.base_model or two_pass.get("base_model") or DEFAULT_BASE_MODEL)
    args.scoring_backend = str(args.scoring_backend or two_pass.get("scoring_backend") or "auto")
    args.vllm_lora_mode = str(args.vllm_lora_mode or two_pass.get("vllm_lora_mode") or "dynamic")
    args.vllm_tokenizer_mode = str(args.vllm_tokenizer_mode or two_pass.get("vllm_tokenizer_mode") or "auto")
    args.vllm_config_format = str(args.vllm_config_format or two_pass.get("vllm_config_format") or "")
    args.vllm_load_format = str(args.vllm_load_format or two_pass.get("vllm_load_format") or "")
    args.vllm_merge_lora_cache_dir = str(
        args.vllm_merge_lora_cache_dir
        or two_pass.get("vllm_merge_lora_cache_dir")
        or "outputs/cache/merged_lora"
    )
    args.vllm_merge_lora_force_rebuild = bool(
        args.vllm_merge_lora_force_rebuild
        or two_pass.get("vllm_merge_lora_force_rebuild")
        or False
    )
    if not args.calibration_file:
        args.calibration_file = str(two_pass.get("calibration_file") or "")


def _prompt_cfg_from_config(args: argparse.Namespace, cfg: Mapping[str, Any]) -> dict[str, Any]:
    prompt_cfg = dict((cfg.get("build") or {}).get("prompt") or {})
    prompt_cfg["model_name_or_path"] = str(args.prompt_model_name_or_path or prompt_cfg.get("model_name_or_path") or args.base_model)
    prompt_cfg["auto_length"] = True
    prompt_cfg["max_length"] = int(args.prompt_max_length)
    prompt_cfg["output_mode"] = str(prompt_cfg.get("output_mode") or "label_only")
    prompt_cfg["label_format"] = str(prompt_cfg.get("label_format") or "letter")
    prompt_cfg["label_schema"] = str(args.label_schema)
    prompt_cfg.setdefault("system_prompt", None)
    return _prompt_cfg_for_trace_style(
        prompt_cfg,
        trace_prompt_style=str(args.trace_prompt_style),
        label_schema=str(args.label_schema),
    )


def _expansion_ladder_indices(ordered_indices: list[int], initial_indices: list[int]) -> list[list[int]]:
    if not initial_indices:
        initial_size = 1
    else:
        positions = [ordered_indices.index(idx) for idx in initial_indices if idx in ordered_indices]
        initial_size = max(positions) + 1 if positions else len(initial_indices)
    initial_size = max(1, min(initial_size, len(ordered_indices)))
    return [list(ordered_indices[:size]) for size in range(initial_size, len(ordered_indices) + 1)]


def _effective_prompt_budget(prompt_evidence: Mapping[str, Any]) -> int:
    guard = dict(prompt_evidence.get("max_length_guard") or {})
    if not bool(guard.get("enabled", False)):
        return 0
    return int(guard.get("effective_prompt_budget") or guard.get("effective_max_length") or 0)


def _resolve_threshold(args: argparse.Namespace, calibration_file: Path) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    if not calibration_file.exists():
        raise FileNotFoundError(
            f"Missing two-pass uncertainty calibration file: {calibration_file}. "
            "Run the val split with --calibrate first or pass --threshold."
        )
    with calibration_file.open(encoding="utf-8") as handle:
        calibration = json.load(handle)
    return float(calibration["threshold"])


def _threshold_grid(initial_margins: Sequence[float]) -> list[float]:
    values = sorted(float(value) for value in initial_margins)
    if not values:
        return [0.0]
    out = set(values)
    for percentile in range(0, 101, 5):
        out.add(_quantile(values, percentile / 100.0))
    return sorted(out)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    pos = max(0.0, min(1.0, float(q))) * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _macro_f1(gold: Sequence[str], pred: Sequence[str], *, labels: Sequence[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(sum(scores) / max(len(scores), 1))


def _selected_score_row(decision: Mapping[str, Any]) -> dict[str, Any]:
    selected = [int(idx) for idx in decision.get("selected_indices") or []]
    for row in decision.get("score_trace") or []:
        row_indices = [int(idx) for idx in row.get("selected_indices") or []]
        if row_indices == selected:
            return dict(row)
    return {}


def _decision_diagnostics(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evidence_counts = [len(decision.get("selected_indices") or []) for decision in decisions]
    expanded_count = sum(1 for decision in decisions if bool(decision.get("prompt_evidence_expanded")))
    margins = [float(decision.get("uncertainty_margin") or 0.0) for decision in decisions]
    return {
        "count": int(len(decisions)),
        "expanded_rate": float(expanded_count / max(len(decisions), 1)),
        "mean_evidence_count": float(sum(evidence_counts) / max(len(evidence_counts), 1)),
        "min_evidence_count": int(min(evidence_counts)) if evidence_counts else 0,
        "max_evidence_count": int(max(evidence_counts)) if evidence_counts else 0,
        "mean_uncertainty_margin": float(sum(margins) / max(len(margins), 1)),
    }


def _pred_margin(row: Mapping[str, Any]) -> float:
    return float(row.get("pred_margin") or 0.0)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_jsonl(path: Path, *, sample_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if sample_limit > 0 and len(rows) >= int(sample_limit):
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows read from {path}")
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
