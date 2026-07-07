#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.phase5_selectors.build.build_trace_verifier_data import (  # noqa: E402
    _build_split,
    _load_experiment_config,
    load_prompt_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate prompt evidence budget to a target prompt mean.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace", required=True, help="Train split ordered trace JSONL.")
    parser.add_argument("--raw", required=True, help="Train split raw JSON.")
    parser.add_argument("--dataset", default="liar_raw")
    parser.add_argument("--label-schema", default="liar6")
    parser.add_argument("--target-prompt-mean", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-model-name-or-path", default="/data/models/Ministral-3-8B-Instruct-2512")
    parser.add_argument("--trace-prompt-style", default="plain")
    parser.add_argument("--evidence-text-mode", default="full")
    parser.add_argument("--selection-mode", default="trace")
    parser.add_argument("--expected-selector-name", default="selector_mech_s4_atom_union_source_score_ordered")
    parser.add_argument("--expected-chunk-mmr-fingerprint", default="")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-count", type=int, default=20)
    parser.add_argument("--budget-min", type=int, default=1)
    parser.add_argument("--budget-max", type=int, default=4096)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--local-window", type=int, default=32)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    if int(args.budget_min) <= 0 or int(args.budget_max) < int(args.budget_min):
        raise SystemExit("--budget-min must be positive and <= --budget-max")
    if int(args.min_count) < 0 or int(args.max_count) < max(1, int(args.min_count)):
        raise SystemExit("--max-count must be positive and >= --min-count")

    cfg = _load_experiment_config(str(args.config))
    prompt_cfg = dict((cfg.get("build", {}) or {}).get("prompt", {}) or {})
    prompt_cfg["label_schema"] = str(args.label_schema)
    prompt_cfg["model_name_or_path"] = str(args.prompt_model_name_or_path)
    tokenizer = load_prompt_tokenizer(str(prompt_cfg["model_name_or_path"]))

    cache: dict[int, dict[str, Any]] = {}

    def evaluate(budget: int) -> dict[str, Any]:
        budget = int(budget)
        if budget in cache:
            return cache[budget]
        print(f"[budget-calibration] evaluate budget={budget}", file=sys.stderr, flush=True)
        _, report = _build_split(
            split="train",
            source_type="trace",
            source_path=Path(args.trace),
            raw_path=Path(args.raw),
            dataset=str(args.dataset),
            label_schema=str(args.label_schema),
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            selection_mode=str(args.selection_mode),
            trace_prompt_style=str(args.trace_prompt_style),
            evidence_text_mode=str(args.evidence_text_mode),
            expected_selector_name=str(args.expected_selector_name or ""),
            top_k=int(args.max_count),
            random_seed=0,
            expected_chunk_mmr_fingerprint=str(args.expected_chunk_mmr_fingerprint or ""),
            sample_limit=int(args.sample_limit) if int(args.sample_limit) > 0 else None,
            show_progress=not bool(args.no_progress),
            prompt_evidence_config={
                "policy": "budget",
                "min_evidence_count": int(args.min_count),
                "max_evidence_count": int(args.max_count),
                "evidence_token_budget": budget,
                "max_length_guard": {"enabled": True, "on_violation": "warn"},
            },
            allow_empty_evidence=False,
        )
        prompt_stats = dict(report.get("prompt_token_count") or {})
        evidence_stats = dict(report.get("evidence_count") or {})
        result = {
            "budget": budget,
            "prompt_mean": float(prompt_stats.get("mean", 0.0)),
            "prompt_p95": float(prompt_stats.get("p95", 0.0)),
            "evidence_mean": float(evidence_stats.get("mean", 0.0)),
            "truncation_rate": float(report.get("prompt_truncation_rate", 0.0)),
            "stop_reasons": dict((report.get("prompt_evidence") or {}).get("stop_reasons") or {}),
            "max_length_guard": dict(report.get("max_length_guard") or {}),
        }
        cache[budget] = result
        print(
            "[budget-calibration] budget={budget} prompt_mean={prompt_mean:.6f} "
            "prompt_p95={prompt_p95:.6f} evidence_mean={evidence_mean:.6f} "
            "truncation_rate={truncation_rate:.6f}".format(**result),
            file=sys.stderr,
            flush=True,
        )
        return result

    target = float(args.target_prompt_mean)
    low = int(args.budget_min)
    high = int(args.budget_max)
    evaluate(low)
    evaluate(high)
    while low < high:
        mid = (low + high) // 2
        observed = evaluate(mid)["prompt_mean"]
        if observed >= target:
            high = mid
        else:
            low = mid + 1

    center = low
    start = max(int(args.budget_min), center - int(args.local_window))
    end = min(int(args.budget_max), center + int(args.local_window))
    for budget in range(start, end + 1):
        evaluate(budget)

    best = min(
        cache.values(),
        key=lambda item: (abs(float(item["prompt_mean"]) - target), int(item["budget"])),
    )
    payload = {
        "target_prompt_mean": target,
        "selected_budget": int(best["budget"]),
        "selected_prompt_mean": float(best["prompt_mean"]),
        "selected_prompt_p95": float(best["prompt_p95"]),
        "delta_prompt_mean": float(best["prompt_mean"]) - target,
        "selected_evidence_mean": float(best["evidence_mean"]),
        "selected_truncation_rate": float(best["truncation_rate"]),
        "split": "train",
        "min_evidence_count": int(args.min_count),
        "max_evidence_count": int(args.max_count),
        "budget_min": int(args.budget_min),
        "budget_max": int(args.budget_max),
        "sample_limit": int(args.sample_limit),
        "inputs": {
            "config": str(args.config),
            "trace": str(args.trace),
            "raw": str(args.raw),
            "dataset": str(args.dataset),
            "label_schema": str(args.label_schema),
            "expected_selector_name": str(args.expected_selector_name or ""),
            "expected_chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint or ""),
            "prompt_model_name_or_path": str(args.prompt_model_name_or_path),
            "trace_prompt_style": str(args.trace_prompt_style),
            "evidence_text_mode": str(args.evidence_text_mode),
        },
        "trials": sorted(cache.values(), key=lambda item: int(item["budget"])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "selected_budget={budget} target_prompt_mean={target:.6f} observed_prompt_mean={observed:.6f}".format(
            budget=payload["selected_budget"],
            target=payload["target_prompt_mean"],
            observed=payload["selected_prompt_mean"],
        ),
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
