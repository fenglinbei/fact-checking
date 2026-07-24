#!/usr/bin/env python3
"""Build verifier-ready JSONL from selector/control trace files.

The input trace format is the one emitted by selector eval scripts:
``candidate_pool`` plus ``selector_ordered_indices`` in candidate-pool
coordinates.  This lets selection-only controls be promoted to an inference
dataset without rerunning retrieval or changing the verifier pipeline.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from fact_checking.build.candidates import _build_training_row, _load_prompt_tokenizer
from fact_checking.build.prompts import build_system_message, truncate_single_evidence_to_budget
from sft.runtime.model_loading import is_mistral_common_tokenizer

load_prompt_tokenizer = _load_prompt_tokenizer
build_training_row = _build_training_row
from fact_checking.config import save_yaml
from fact_checking.data.io import load_split
from fact_checking.selectors.metrics import ordered_selection_metrics, summarize_ordered_selection
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    write_json,
    write_jsonl,
)


SELECTION_MODES = (
    "trace",
    "hybrid_score_topk",
    "candidate_pool_topk",
    "same_set_hybrid_order",
    "same_set_candidate_pool_order",
    "same_set_random_order",
)
TRACE_PROMPT_STYLES = ("plain", "trace_lite", "rawfc_boundaries", "qec_min", "qec_map", "mrec_min")
EVIDENCE_TEXT_MODES = ("full", "anchor_only")
TRACE_ORDER_FIELDS = (
    "display_ordered_indices",
    "selector_full_ordered_indices",
    "selector_available_ordered_indices",
)
PROMPT_EVIDENCE_POLICIES = (
    "selected_set",
    "prefix_topk",
    "resolve_stop",
    "fixed_topk",
    "minmax",
    "budget",
    "prompt_budget",
    "state_budget",
    "two_pass_uncertainty",
)

RAWFC_BOUNDARIES_SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant for RAWFC claims. "
    "Classify claims using only the claim and retrieved evidence supplied by the user. "
    "Use these label boundaries: false means the evidence contradicts or refutes the main claim; "
    "half means the evidence supports part of the claim but leaves important qualifiers, context, "
    "or mixed factual status; true means the evidence supports the main claim."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build verifier data from selector/control traces.")
    p.add_argument("--config", default="configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml")
    p.add_argument("--train-trace", default=None)
    p.add_argument("--val-trace", default=None)
    p.add_argument("--test-trace", default=None)
    p.add_argument("--train-oracle-results", default=None)
    p.add_argument("--val-oracle-results", default=None)
    p.add_argument("--test-oracle-results", default=None)
    p.add_argument("--train-raw", default="data/raw/LIAR-RAW/train.json")
    p.add_argument("--val-raw", default="data/raw/LIAR-RAW/val.json")
    p.add_argument("--test-raw", default="data/raw/LIAR-RAW/test.json")
    p.add_argument("--dataset", default=None, help="Raw split format: liar_raw or rawfc.")
    p.add_argument("--label-schema", default=None, help="Label schema: liar6 or rawfc3.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--selection-mode", default="trace", choices=SELECTION_MODES)
    p.add_argument(
        "--trace-order-field",
        default="display_ordered_indices",
        choices=TRACE_ORDER_FIELDS,
        help=(
            "Trace field used as the ordered evidence slate when selection-mode=trace. "
            "The default preserves the historical verifier-visible order; prefix-lattice "
            "analysis should use selector_full_ordered_indices."
        ),
    )
    p.add_argument("--train-selection-plan", default=None)
    p.add_argument("--val-selection-plan", default=None)
    p.add_argument("--test-selection-plan", default=None)
    p.add_argument("--trace-prompt-style", default="plain", choices=TRACE_PROMPT_STYLES)
    p.add_argument(
        "--evidence-text-mode",
        default="full",
        choices=EVIDENCE_TEXT_MODES,
        help="Prompt-visible evidence text projection: full chunk text or ABC anchor_text only.",
    )
    p.add_argument("--prompt-output-mode", default=None)
    p.add_argument("--expected-selector-name", default="")
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--prompt-evidence-policy", default=None, choices=PROMPT_EVIDENCE_POLICIES)
    p.add_argument("--prompt-evidence-min-count", type=int, default=None)
    p.add_argument("--prompt-evidence-max-count", type=int, default=None)
    p.add_argument("--prompt-evidence-token-budget", type=int, default=None)
    p.add_argument("--prompt-evidence-prompt-token-budget", type=int, default=None)
    p.add_argument("--prompt-evidence-max-length-guard", default=None, choices=("off", "warn", "error"))
    p.add_argument("--allow-empty-evidence", action="store_true")
    p.add_argument("--forbid-prompt-truncation", action="store_true")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--prompt-model-name-or-path", default=None)
    p.add_argument("--train-model-name-or-path", default=None)
    p.add_argument("--model-base-path", default=None)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--val-only", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    build_dir = output_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_experiment_config(args.config)
    prompt_cfg = dict((cfg.get("build", {}) or {}).get("prompt", {}) or {})
    label_schema = str(
        args.label_schema
        or prompt_cfg.get("label_schema")
        or ((cfg.get("build", {}) or {}).get("data", {}) or {}).get("label_schema")
        or cfg.get("label_schema")
        or "liar6"
    )
    prompt_cfg["label_schema"] = label_schema
    if args.prompt_model_name_or_path:
        prompt_cfg["model_name_or_path"] = args.prompt_model_name_or_path
    if args.prompt_output_mode:
        prompt_cfg["output_mode"] = str(args.prompt_output_mode)
    if args.model_base_path and prompt_cfg.get("model_name_or_path"):
        prompt_cfg["model_name_or_path"] = _resolve_model_path(
            str(prompt_cfg["model_name_or_path"]),
            args.model_base_path,
        )
    tokenizer = load_prompt_tokenizer(str(prompt_cfg["model_name_or_path"]))
    prompt_evidence_config = _resolve_prompt_evidence_config(cfg, args=args, prompt_cfg=prompt_cfg)

    split_specs = []
    if not args.val_only:
        train_source = _resolve_split_source("train", args.train_trace, args.train_oracle_results)
        split_specs.append(
            (
                "train",
                train_source[0],
                train_source[1],
                args.train_raw,
                args.train_selection_plan,
            )
        )
    val_source = _resolve_split_source("val", args.val_trace, args.val_oracle_results)
    split_specs.append(
        ("val", val_source[0], val_source[1], args.val_raw, args.val_selection_plan)
    )
    test_source = _resolve_optional_split_source(args.test_trace, args.test_oracle_results)
    if test_source is not None:
        split_specs.append(
            ("test", test_source[0], test_source[1], args.test_raw, args.test_selection_plan)
        )

    split_paths: dict[str, str] = {}
    reports: dict[str, Any] = {}
    for split, source_type, source_path, raw_path, selection_plan_path in split_specs:
        rows, report = _build_split(
            split=split,
            source_type=source_type,
            source_path=Path(source_path),
            raw_path=Path(raw_path),
            dataset=args.dataset,
            label_schema=label_schema,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            selection_mode=str(args.selection_mode),
            trace_order_field=str(args.trace_order_field),
            selection_plan_path=Path(selection_plan_path) if selection_plan_path else None,
            trace_prompt_style=str(args.trace_prompt_style),
            evidence_text_mode=str(args.evidence_text_mode),
            expected_selector_name=str(args.expected_selector_name or ""),
            top_k=int(args.top_k),
            random_seed=int(args.random_seed),
            expected_chunk_mmr_fingerprint=str(args.expected_chunk_mmr_fingerprint or ""),
            sample_limit=args.sample_limit,
            show_progress=not args.no_progress,
            prompt_evidence_config=prompt_evidence_config,
            allow_empty_evidence=bool(args.allow_empty_evidence),
            forbid_prompt_truncation=bool(args.forbid_prompt_truncation),
        )
        out_path = build_dir / f"build_{split}.jsonl"
        write_jsonl(out_path, rows)
        split_paths[split] = str(out_path)
        reports[split] = report

    built_split_paths = dict(split_paths)
    if "train" not in split_paths:
        split_paths["train"] = split_paths["val"]
    if "test" not in split_paths:
        split_paths["test"] = split_paths["val"]

    train_config = _build_train_config(
        cfg=cfg,
        run_dir=output_dir,
        split_paths=split_paths,
        label_schema=label_schema,
        model_base_path=args.model_base_path,
        train_model_name_or_path=args.train_model_name_or_path,
    )
    train_config_path = output_dir / "train.resolved.yaml"
    save_yaml(train_config, train_config_path)

    report = {
        "config": args.config,
        "output_dir": str(output_dir),
        "build_dir": str(build_dir),
        "selection_mode": args.selection_mode,
        "trace_order_field": args.trace_order_field,
        "selection_plans": {
            split: path
            for split, path in (
                ("train", args.train_selection_plan),
                ("val", args.val_selection_plan),
                ("test", args.test_selection_plan),
            )
            if path
        },
        "trace_prompt_style": args.trace_prompt_style,
        "evidence_text_mode": args.evidence_text_mode,
        "expected_selector_name": args.expected_selector_name,
        "top_k": int(args.top_k),
        "random_seed": int(args.random_seed),
        "expected_chunk_mmr_fingerprint": args.expected_chunk_mmr_fingerprint,
        "prompt_evidence": _public_prompt_evidence_config(prompt_evidence_config),
        "allow_empty_evidence": bool(args.allow_empty_evidence),
        "forbid_prompt_truncation": bool(args.forbid_prompt_truncation),
        "val_only": bool(args.val_only),
        "built_splits": sorted(built_split_paths),
        "built_split_paths": built_split_paths,
        "prompt_model_name_or_path": str(prompt_cfg["model_name_or_path"]),
        "label_schema": label_schema,
        "split_paths": split_paths,
        "train_config": str(train_config_path),
        "splits": reports,
        "notes": [
            "Rows are derived from selector/control trace candidate_pool coordinates.",
            "No retrieval, selector scoring, verifier training, or oracle search is run here.",
            "Use selection_mode=same_set_random_order with multiple wrapper seeds to estimate random-order means.",
        ],
    }
    write_json(build_dir / "build_report.json", report)

    print(f"Wrote selector trace verifier data to {build_dir}")
    for split, split_report in reports.items():
        print(
            "{split}: rows={rows} skipped={skipped} trunc={trunc:.4f} "
            "mean_evidence={mean_ev:.3f}".format(
                split=split,
                rows=split_report["n_rows"],
                skipped=split_report["skipped_total"],
                trunc=split_report["prompt_truncation_rate"],
                mean_ev=float(split_report["evidence_count"].get("mean", 0.0)),
            )
        )
    print(f"Train config: {train_config_path}")


def _build_split(
    *,
    split: str,
    source_type: str,
    source_path: Path,
    raw_path: Path,
    dataset: str | None,
    label_schema: str,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    selection_mode: str,
    trace_order_field: str = "display_ordered_indices",
    selection_plan_path: Path | None = None,
    trace_prompt_style: str,
    evidence_text_mode: str = "full",
    expected_selector_name: str,
    top_k: int,
    random_seed: int,
    expected_chunk_mmr_fingerprint: str,
    sample_limit: int | None,
    show_progress: bool,
    prompt_evidence_config: dict[str, Any] | None = None,
    allow_empty_evidence: bool = False,
    forbid_prompt_truncation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if evidence_text_mode not in EVIDENCE_TEXT_MODES:
        raise ValueError(f"unsupported evidence_text_mode: {evidence_text_mode}")
    if trace_order_field not in TRACE_ORDER_FIELDS:
        raise ValueError(f"unsupported trace_order_field: {trace_order_field}")

    raw_by_event = {
        sample.event_id: sample
        for sample in load_split(raw_path, dataset=dataset, label_schema=label_schema)
    }
    source_rows = _read_jsonl(source_path)
    if sample_limit is not None:
        source_rows = source_rows[: int(sample_limit)]
    prompt_cfg_for_style = _prompt_cfg_for_trace_style(
        prompt_cfg,
        trace_prompt_style=trace_prompt_style,
        label_schema=label_schema,
    )
    prompt_evidence = _normalize_prompt_evidence_config(
        prompt_evidence_config,
        fallback_top_k=int(top_k),
        prompt_cfg=prompt_cfg_for_style,
    )
    max_length_guard = dict(prompt_evidence.get("max_length_guard") or {})
    two_pass_decisions: dict[str, dict[str, Any]] = {}
    if str(prompt_evidence.get("policy") or "") == "two_pass_uncertainty":
        two_pass_decisions = _load_two_pass_uncertainty_decisions(split, prompt_evidence)
    selection_plan: dict[str, dict[str, Any]] = {}
    if selection_plan_path is not None:
        if source_type != "trace" or selection_mode != "trace":
            raise ValueError("selection plans require source_type=trace and selection_mode=trace")
        if str(prompt_evidence.get("policy") or "") != "selected_set":
            raise ValueError("selection plans require prompt_evidence.policy=selected_set")
        selection_plan = _load_selection_plan(selection_plan_path)
    used_selection_plan_events: set[str] = set()

    out_rows: list[dict[str, Any]] = []
    metric_traces: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    selector_names: Counter[str] = Counter()
    fp_counter: Counter[str] = Counter()
    selected_len_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    coverage_label_counter: Counter[str] = Counter()
    prompt_tokens: list[int] = []
    evidence_counts: list[int] = []
    evidence_counts_before: list[int] = []
    prompt_policy_counter: Counter[str] = Counter()
    prompt_policy_stop_counter: Counter[str] = Counter()
    prompt_policy_selected_counts: list[int] = []
    two_pass_initial_counts: list[int] = []
    two_pass_final_counts: list[int] = []
    two_pass_prompt_tokens_before: list[int] = []
    two_pass_uncertainty_margins: list[float] = []
    two_pass_expanded_count = 0
    max_length_guard_violations: Counter[str] = Counter()

    iterator = tqdm(
        source_rows,
        desc=f"trace-verifier [{split}]",
        unit="claim",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for source_row in iterator:
        trace = _normalize_source_row(source_row, source_type=source_type)
        event_id = str(trace.get("event_id") or "")
        if not event_id:
            skipped["missing_event_id"] += 1
            continue

        selector_name = str(trace.get("selector_name") or "")
        selector_names[selector_name] += 1
        if expected_selector_name and selector_name != expected_selector_name:
            raise ValueError(
                f"{split}:{event_id} selector_name mismatch: "
                f"expected {expected_selector_name!r}, got {selector_name!r}."
            )

        sample = raw_by_event.get(event_id)
        if sample is None:
            skipped["missing_raw_sample"] += 1
            continue

        fingerprint = _trace_fingerprint(trace)
        fp_counter[fingerprint] += 1
        if expected_chunk_mmr_fingerprint and fingerprint != expected_chunk_mmr_fingerprint:
            raise ValueError(
                f"{split}:{event_id} chunk_mmr_fingerprint mismatch: "
                f"expected {expected_chunk_mmr_fingerprint}, got {fingerprint}."
            )

        try:
            pool = trace.get("candidate_pool") or []
            if allow_empty_evidence and not pool:
                selected_indices = []
                prompt_evidence_decision = _prompt_evidence_decision(
                    [],
                    policy=str(prompt_evidence["policy"]),
                    min_count=int(prompt_evidence.get("min_evidence_count") or 0),
                    max_count=int(prompt_evidence.get("max_evidence_count") or 0),
                    token_budget=prompt_evidence.get("evidence_token_budget"),
                    total_token_cost=0,
                    stop_reason="no_evidence",
                )
                candidates = []
            else:
                selector_top_k = int(top_k) if str(prompt_evidence["policy"]) == "prefix_topk" else 0
                selected_indices = _select_indices(
                    trace,
                    mode=selection_mode,
                    top_k=selector_top_k,
                    random_seed=random_seed,
                    trace_order_field=trace_order_field,
                )
                selection_plan_row = selection_plan.get(event_id)
                if selection_plan:
                    if selection_plan_row is None:
                        raise ValueError(
                            f"selection plan has no row for event_id={event_id!r}"
                        )
                    selected_indices = _selected_indices_from_plan(
                        trace,
                        ordered_indices=selected_indices,
                        plan=selection_plan_row,
                        event_id=event_id,
                    )
                    used_selection_plan_events.add(event_id)
                prompt_evidence_decision = _select_prompt_evidence_indices(
                    trace,
                    ordered_indices=selected_indices,
                    config=prompt_evidence,
                    event_id=event_id,
                    split=split,
                    two_pass_decision=two_pass_decisions.get(event_id),
                )
                selected_indices = list(prompt_evidence_decision["selected_indices"])
                candidates = _selected_candidates(trace, selected_indices, selection_mode=selection_mode)
        except ValueError as exc:
            raise ValueError(f"{split}:{event_id}: {exc}") from exc
        if not candidates and not allow_empty_evidence:
            skipped["no_selected_evidence"] += 1
            continue
        candidates = _apply_evidence_text_mode(candidates, evidence_text_mode)
        if str(prompt_evidence["policy"]) == "prompt_budget":
            (
                selected_indices,
                candidates,
                claim,
                qec_payload,
                mrec_payload,
                training_row,
                prompt_evidence_decision,
            ) = _fit_prompt_budget_prefix(
                sample=sample,
                trace=trace,
                candidates=candidates,
                selected_indices=selected_indices,
                tokenizer=tokenizer,
                prompt_cfg=prompt_cfg_for_style,
                prompt_evidence=prompt_evidence,
                prompt_evidence_decision=prompt_evidence_decision,
                trace_prompt_style=trace_prompt_style,
                label_schema=label_schema,
                allow_unlabeled=split == "test" and not bool(sample.label),
            )
        else:
            claim, candidates, qec_payload, mrec_payload = _apply_trace_prompt_style(
                claim=sample.claim,
                candidates=candidates,
                trace=trace,
                trace_prompt_style=trace_prompt_style,
            )
            retrieval_row = _retrieval_row_from_sample(
                sample,
                claim=claim,
                candidates=candidates,
                label_schema=label_schema,
            )
            training_row = build_training_row(
                retrieval_row,
                tokenizer,
                prompt_cfg_for_style,
                allow_unlabeled=split == "test" and not bool(sample.label),
            )
        if forbid_prompt_truncation and (
            bool(training_row.get("was_truncated"))
            or bool(training_row.get("evidence_text_truncated"))
        ):
            raise ValueError(
                f"{split}:{event_id}: prompt truncation is forbidden: "
                f"was_truncated={training_row.get('was_truncated')!r}, "
                f"evidence_text_truncated={training_row.get('evidence_text_truncated')!r}"
            )
        training_row["trace_prompt_style"] = trace_prompt_style
        training_row["evidence_text_mode"] = evidence_text_mode
        training_row.update(_prompt_evidence_row_fields(prompt_evidence, prompt_evidence_decision))
        guard_result = _max_length_guard_result(
            training_row,
            prompt_evidence=prompt_evidence,
            max_length_guard=max_length_guard,
        )
        if guard_result["enabled"]:
            training_row["prompt_evidence_max_length_guard"] = guard_result
            for reason in guard_result["violation_reasons"]:
                max_length_guard_violations[str(reason)] += 1
            if guard_result["violation_reasons"] and guard_result["on_violation"] == "error":
                raise ValueError(
                    f"{split}:{event_id}: prompt evidence max length guard failed: "
                    f"{guard_result['violation_reasons']}"
                )
        if qec_payload is not None:
            training_row["qec_steps"] = qec_payload["steps"]
            training_row["qec_diagnostics"] = qec_payload["diagnostics"]
        if mrec_payload is not None:
            training_row["mrec_prompt_steps"] = mrec_payload["steps"]
            training_row["mrec_prompt_diagnostics"] = mrec_payload["diagnostics"]
            for key in (
                "mrec_trace_version",
                "mrec_selector_name",
                "mrec_steps",
                "mrec_diagnostics",
                "atom_states_initial",
                "atom_states_final",
            ):
                if key in trace:
                    training_row[key] = trace[key]
        training_row["selector_trace"] = {
            "source_type": source_type,
            "source_path": str(source_path),
            "selector_name": selector_name,
            "selection_mode": selection_mode,
            "trace_order_field": trace_order_field,
            "selection_plan_path": str(selection_plan_path) if selection_plan_path else None,
            "selection_plan": selection_plan.get(event_id),
            "top_k": int(top_k),
            "random_seed": int(random_seed),
            "chunk_mmr_fingerprint": fingerprint,
            "oracle_ordered_indices": [int(x) for x in (trace.get("oracle_ordered_indices") or [])],
            "selected_indices": [int(x) for x in selected_indices],
        }
        out_rows.append(training_row)

        metrics = ordered_selection_metrics(
            [int(x) for x in (trace.get("oracle_ordered_indices") or [])],
            selected_indices,
            top_k=top_k,
        )
        metric_trace = {
            "event_id": event_id,
            "gold_label": training_row.get("gold_label", ""),
            "selector_name": selector_name,
        }
        metric_trace.update(metrics)
        metric_traces.append(metric_trace)

        selected_len_counter[str(len(selected_indices))] += 1
        label_counter[str(training_row.get("gold_label", ""))] += 1
        coverage_label = str(training_row.get("coverage_label") or "")
        if coverage_label:
            coverage_label_counter[coverage_label] += 1
        prompt_policy_counter[str(prompt_evidence["policy"])] += 1
        prompt_policy_stop_counter[str(prompt_evidence_decision["stop_reason"])] += 1
        prompt_policy_selected_counts.append(
            int(prompt_evidence_decision["selected_count_before_prompt_truncation"])
        )
        two_pass_payload = prompt_evidence_decision.get("two_pass_uncertainty")
        if isinstance(two_pass_payload, dict):
            two_pass_initial_counts.append(len(two_pass_payload.get("initial_indices") or []))
            two_pass_final_counts.append(len(two_pass_payload.get("selected_indices") or []))
            token_count = _int_or_none(two_pass_payload.get("prompt_token_count_before_final_build"))
            if token_count is not None:
                two_pass_prompt_tokens_before.append(int(token_count))
            margin = _float_or_none(two_pass_payload.get("uncertainty_margin"))
            if margin is not None:
                two_pass_uncertainty_margins.append(float(margin))
            if bool(two_pass_payload.get("prompt_evidence_expanded")):
                two_pass_expanded_count += 1
        prompt_tokens.append(int(training_row.get("prompt_token_count", 0)))
        evidence_counts.append(int(training_row.get("evidence_count", 0)))
        evidence_counts_before.append(
            int(training_row.get("evidence_count_before", training_row.get("evidence_count", 0)))
        )

    if selection_plan:
        unused_plan_events = sorted(set(selection_plan) - used_selection_plan_events)
        if unused_plan_events:
            raise ValueError(
                "selection plan contains events absent from the built source rows: "
                f"count={len(unused_plan_events)}, first={unused_plan_events[:10]}"
            )

    report = {
        "split": split,
        "source_type": source_type,
        "source_path": str(source_path),
        "raw_path": str(raw_path),
        "selection_mode": selection_mode,
        "trace_order_field": trace_order_field,
        "selection_plan": {
            "enabled": bool(selection_plan),
            "path": str(selection_plan_path) if selection_plan_path else None,
            "row_count": len(selection_plan),
        },
        "trace_prompt_style": trace_prompt_style,
        "evidence_text_mode": evidence_text_mode,
        "top_k": int(top_k),
        "allow_empty_evidence": bool(allow_empty_evidence),
        "forbid_prompt_truncation": bool(forbid_prompt_truncation),
        "prompt_evidence": {
            **_public_prompt_evidence_config(prompt_evidence),
            "policy_counts": dict(prompt_policy_counter),
            "stop_reasons": dict(prompt_policy_stop_counter),
            "selected_count_before_prompt_truncation": _summary(prompt_policy_selected_counts),
            "two_pass_uncertainty": _two_pass_uncertainty_report(
                two_pass_decisions=two_pass_decisions,
                row_count=len(out_rows),
                expanded_count=two_pass_expanded_count,
                initial_counts=two_pass_initial_counts,
                final_counts=two_pass_final_counts,
                prompt_tokens_before=two_pass_prompt_tokens_before,
                uncertainty_margins=two_pass_uncertainty_margins,
            ),
        },
        "max_length_guard": _max_length_guard_report(
            max_length_guard,
            violations=max_length_guard_violations,
            row_count=len(out_rows),
        ),
        "random_seed": int(random_seed),
        "n_source_rows": len(source_rows),
        "n_rows": len(out_rows),
        "skipped": dict(skipped),
        "skipped_total": int(sum(skipped.values())),
        "selector_names": dict(selector_names),
        "labels": dict(label_counter),
        "coverage_labels": dict(coverage_label_counter),
        "chunk_mmr_fingerprints": dict(fp_counter),
        "selected_index_lengths": dict(selected_len_counter),
        "selection_metrics": summarize_ordered_selection(metric_traces),
        "prompt_truncation_rate": float(
            sum(1 for row in out_rows if bool(row.get("was_truncated"))) / max(len(out_rows), 1)
        ),
        "prompt_token_count": _summary(prompt_tokens),
        "evidence_count": _summary(evidence_counts),
        "evidence_count_before": _summary(evidence_counts_before),
    }
    return out_rows, report


def _apply_evidence_text_mode(
    candidates: list[dict[str, Any]],
    evidence_text_mode: str,
) -> list[dict[str, Any]]:
    if evidence_text_mode == "full":
        return candidates
    if evidence_text_mode != "anchor_only":
        raise ValueError(f"unsupported evidence_text_mode: {evidence_text_mode}")

    rendered: list[dict[str, Any]] = []
    for candidate in candidates:
        copied = dict(candidate)
        full_text = str(copied.get("text") or "").strip()
        anchor_text = str(copied.get("anchor_text") or "").strip()
        copied["full_chunk_text"] = full_text
        copied["evidence_text_mode"] = "anchor_only"
        if anchor_text:
            copied["text"] = anchor_text
            copied["evidence_text_source"] = "anchor_text"
        else:
            copied["text"] = full_text
            copied["evidence_text_source"] = "text_fallback"
        rendered.append(copied)
    return rendered


def _apply_trace_prompt_style(
    *,
    claim: str,
    candidates: list[dict[str, Any]],
    trace: dict[str, Any],
    trace_prompt_style: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    qec_payload: dict[str, Any] | None = None
    mrec_payload: dict[str, Any] | None = None
    if trace_prompt_style == "trace_lite":
        claim, candidates = _apply_trace_lite_prompt_fields(
            claim=claim,
            candidates=candidates,
            claim_atoms=trace.get("claim_atoms") or [],
        )
    elif trace_prompt_style in {"qec_min", "qec_map"}:
        claim, candidates, qec_payload = _apply_qec_prompt_fields(
            claim=claim,
            candidates=candidates,
            claim_atoms=trace.get("claim_atoms") or [],
            chain_steps=trace.get("chain_steps") or [],
            style=trace_prompt_style,
        )
    elif trace_prompt_style == "mrec_min":
        claim, candidates, mrec_payload = _apply_mrec_prompt_fields(
            claim=claim,
            candidates=candidates,
            trace=trace,
        )
    return str(claim), candidates, qec_payload, mrec_payload


def _retrieval_row_from_sample(
    sample: Any,
    *,
    claim: str,
    candidates: list[dict[str, Any]],
    label_schema: str,
) -> dict[str, Any]:
    retrieval_row = {
        "event_id": sample.event_id,
        "claim": claim,
        "label": sample.label,
        "label_schema": label_schema,
        "explain": sample.explain,
        "candidates": candidates,
    }
    sample_metadata = getattr(sample, "metadata", {}) or {}
    for key in ("coverage_label", "coverage_score", "coverage_version", "coverage"):
        if key in sample_metadata:
            retrieval_row[key] = sample_metadata[key]
    return retrieval_row


def _fit_prompt_budget_prefix(
    *,
    sample: Any,
    trace: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected_indices: list[int],
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    prompt_evidence: dict[str, Any],
    prompt_evidence_decision: dict[str, Any],
    trace_prompt_style: str,
    label_schema: str,
    allow_unlabeled: bool,
) -> tuple[
    list[int],
    list[dict[str, Any]],
    str,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any],
]:
    requested_budget = int(prompt_evidence.get("prompt_token_budget") or 0)
    if requested_budget <= 0:
        raise ValueError("prompt_budget requires a positive prompt_token_budget")
    if not candidates or not selected_indices:
        raise ValueError("prompt_budget requires at least one selected evidence item")
    if len(candidates) != len(selected_indices):
        raise ValueError(
            "prompt_budget candidate/index alignment failed before fitting: "
            f"candidates={len(candidates)}, selected_indices={len(selected_indices)}"
        )

    considered_count = len(candidates)
    minimum_count = max(1, int(prompt_evidence.get("min_evidence_count") or 1))
    minimum_count = min(minimum_count, considered_count)
    exact_prompt_cfg = {**dict(prompt_cfg), "auto_length": False}
    rendered_cache: dict[
        int,
        tuple[
            str,
            list[dict[str, Any]],
            dict[str, Any] | None,
            dict[str, Any] | None,
            dict[str, Any],
        ],
    ] = {}

    def render_prefix(
        count: int,
    ) -> tuple[
        str,
        list[dict[str, Any]],
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any],
    ]:
        count = int(count)
        if count in rendered_cache:
            return rendered_cache[count]
        claim, rendered_candidates, qec_payload, mrec_payload = _apply_trace_prompt_style(
            claim=sample.claim,
            candidates=candidates[:count],
            trace=trace,
            trace_prompt_style=trace_prompt_style,
        )
        retrieval_row = _retrieval_row_from_sample(
            sample,
            claim=claim,
            candidates=rendered_candidates,
            label_schema=label_schema,
        )
        training_row = build_training_row(
            retrieval_row,
            tokenizer,
            exact_prompt_cfg,
            allow_unlabeled=allow_unlabeled,
        )
        rendered_cache[count] = (
            claim,
            rendered_candidates,
            qec_payload,
            mrec_payload,
            training_row,
        )
        return rendered_cache[count]

    full_render = render_prefix(considered_count)
    target_token_reserve = _prompt_target_token_reserve(full_render[-1])
    effective_budget = _effective_prompt_token_budget(
        requested_budget=requested_budget,
        target_token_reserve=target_token_reserve,
        max_length_guard=dict(prompt_evidence.get("max_length_guard") or {}),
    )

    partial_evidence = False
    if int(full_render[-1].get("prompt_token_count") or 0) <= effective_budget:
        final_count = considered_count
        final_render = full_render
        stop_reason = str(prompt_evidence_decision.get("stop_reason") or "end_of_trace")
    else:
        minimum_render = render_prefix(minimum_count)
        if int(minimum_render[-1].get("prompt_token_count") or 0) <= effective_budget:
            best_count = minimum_count
            left = minimum_count + 1
            right = considered_count - 1
            while left <= right:
                middle = (left + right) // 2
                middle_render = render_prefix(middle)
                if int(middle_render[-1].get("prompt_token_count") or 0) <= effective_budget:
                    best_count = middle
                    left = middle + 1
                else:
                    right = middle - 1
            final_count = best_count
            final_render = render_prefix(final_count)
            next_render = render_prefix(final_count + 1)
            if int(next_render[-1].get("prompt_token_count") or 0) <= effective_budget:
                raise ValueError(
                    "prompt_budget prefix search did not return the largest fitting prefix: "
                    f"selected_count={final_count}, next_prompt_token_count="
                    f"{next_render[-1].get('prompt_token_count')}, budget={effective_budget}"
                )
            stop_reason = "prompt_token_budget_exhausted"
        elif minimum_count > 1:
            raise ValueError(
                "prompt_token_budget cannot fit min_evidence_count whole evidence items: "
                f"min_evidence_count={minimum_count}, budget={effective_budget}, "
                f"prompt_token_count={minimum_render[-1].get('prompt_token_count')}"
            )
        else:
            claim, rendered_candidates, qec_payload, mrec_payload, _ = minimum_render
            first_text = str(rendered_candidates[0].get("text") or "")
            kept_texts, _, _, _, evidence_text_truncated = truncate_single_evidence_to_budget(
                claim=claim,
                evidence_text=first_text,
                tokenizer=tokenizer,
                system_msg=build_system_message(prompt_cfg.get("system_prompt"), label_schema),
                output_mode=str(prompt_cfg.get("output_mode", "label_only")).strip().lower(),
                label_format=str(prompt_cfg.get("label_format", "name")).strip().lower(),
                label_schema=label_schema,
                budget=effective_budget,
                chat_template=(
                    prompt_cfg.get("chat_template")
                    if isinstance(prompt_cfg.get("chat_template"), dict)
                    else None
                ),
            )
            if not kept_texts or not str(kept_texts[0]).strip():
                raise ValueError(
                    "prompt_token_budget is too small to retain a non-empty first evidence item "
                    f"after rendering the fixed verifier prompt: budget={effective_budget}"
                )
            truncated_candidate = dict(rendered_candidates[0])
            truncated_candidate["text"] = str(kept_texts[0])
            retrieval_row = _retrieval_row_from_sample(
                sample,
                claim=claim,
                candidates=[truncated_candidate],
                label_schema=label_schema,
            )
            training_row = build_training_row(
                retrieval_row,
                tokenizer,
                exact_prompt_cfg,
                allow_unlabeled=allow_unlabeled,
            )
            training_row["was_truncated"] = True
            training_row["evidence_text_truncated"] = bool(evidence_text_truncated)
            final_count = 1
            final_render = (
                claim,
                [truncated_candidate],
                qec_payload,
                mrec_payload,
                training_row,
            )
            partial_evidence = True
            stop_reason = "prompt_token_budget_single_evidence_truncated"

    claim, rendered_candidates, qec_payload, mrec_payload, training_row = final_render
    final_indices = [int(idx) for idx in selected_indices[:final_count]]
    decision = dict(prompt_evidence_decision)
    decision.update(
        {
            "selected_indices": final_indices,
            "selected_count_before_prompt_truncation": int(final_count),
            "selected_token_cost": _prompt_evidence_total_token_cost(trace, final_indices),
            "stop_reason": stop_reason,
            "prompt_token_budget": int(requested_budget),
            "effective_prompt_token_budget": int(effective_budget),
            "selected_prompt_token_count": int(training_row.get("prompt_token_count") or 0),
            "target_token_reserve": int(target_token_reserve),
            "partial_evidence": bool(partial_evidence),
            "considered_count": int(considered_count),
        }
    )
    _validate_prompt_budget_training_row(
        training_row,
        tokenizer=tokenizer,
        selected_indices=final_indices,
        rendered_candidates=rendered_candidates,
        qec_payload=qec_payload,
        mrec_payload=mrec_payload,
        requested_budget=requested_budget,
        effective_budget=effective_budget,
        target_token_reserve=target_token_reserve,
        max_length_guard=dict(prompt_evidence.get("max_length_guard") or {}),
        partial_evidence=partial_evidence,
    )
    return (
        final_indices,
        rendered_candidates,
        claim,
        qec_payload,
        mrec_payload,
        training_row,
        decision,
    )


def _prompt_target_token_reserve(row: dict[str, Any]) -> int:
    return max(
        0,
        int(
            row.get("target_token_count")
            or row.get("inference_target_token_reserve")
            or 0
        ),
    )


def _effective_prompt_token_budget(
    *,
    requested_budget: int,
    target_token_reserve: int,
    max_length_guard: dict[str, Any],
) -> int:
    effective_budget = int(requested_budget)
    effective_max_length = int(max_length_guard.get("effective_max_length") or 0)
    reserve_tokens = max(0, int(max_length_guard.get("reserve_tokens") or 0))
    if effective_max_length > 0:
        context_budget = effective_max_length - reserve_tokens - int(target_token_reserve)
        if context_budget <= 0:
            raise ValueError(
                "no verifier prompt capacity remains after target/context reserves: "
                f"effective_max_length={effective_max_length}, reserve_tokens={reserve_tokens}, "
                f"target_token_reserve={target_token_reserve}"
            )
        effective_budget = min(effective_budget, context_budget)
    if effective_budget <= 0:
        raise ValueError(f"invalid effective prompt token budget: {effective_budget}")
    return int(effective_budget)


def _validate_prompt_budget_training_row(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    selected_indices: list[int],
    rendered_candidates: list[dict[str, Any]],
    qec_payload: dict[str, Any] | None,
    mrec_payload: dict[str, Any] | None,
    requested_budget: int,
    effective_budget: int,
    target_token_reserve: int,
    max_length_guard: dict[str, Any],
    partial_evidence: bool,
) -> None:
    prompt_token_count = int(row.get("prompt_token_count") or 0)
    prompt_input_ids = row.get("prompt_input_ids")
    if is_mistral_common_tokenizer(tokenizer) and not isinstance(prompt_input_ids, list):
        raise ValueError("MistralCommon prompt_budget rows require prompt_input_ids")
    if isinstance(prompt_input_ids, list) and len(prompt_input_ids) != prompt_token_count:
        raise ValueError(
            "prompt_input_ids length differs from prompt_token_count: "
            f"ids={len(prompt_input_ids)}, count={prompt_token_count}"
        )
    if prompt_token_count > int(requested_budget):
        raise ValueError(
            f"final verifier prompt has {prompt_token_count} tokens, exceeding requested budget={requested_budget}"
        )
    if prompt_token_count > int(effective_budget):
        raise ValueError(
            f"final verifier prompt has {prompt_token_count} tokens, exceeding effective budget={effective_budget}"
        )

    expected_count = len(selected_indices)
    if len(rendered_candidates) != expected_count:
        raise ValueError(
            "prompt_budget rendered candidate/index mismatch: "
            f"candidates={len(rendered_candidates)}, selected_indices={expected_count}"
        )
    if int(row.get("evidence_count") or 0) != expected_count:
        raise ValueError(
            "prompt_budget evidence_count/index mismatch: "
            f"evidence_count={row.get('evidence_count')}, selected_indices={expected_count}"
        )
    if isinstance(qec_payload, dict) and len(qec_payload.get("steps") or []) != expected_count:
        raise ValueError("prompt_budget QEC steps do not align with final evidence prefix")
    if isinstance(mrec_payload, dict) and len(mrec_payload.get("steps") or []) != expected_count:
        raise ValueError("prompt_budget MREC steps do not align with final evidence prefix")

    if partial_evidence:
        if not bool(row.get("was_truncated")) or not bool(row.get("evidence_text_truncated")):
            raise ValueError("partial prompt_budget evidence must be explicitly marked as truncated")
    elif bool(row.get("was_truncated")) or bool(row.get("evidence_text_truncated")):
        raise ValueError("whole-prefix prompt_budget selection must not use implicit prompt truncation")

    effective_max_length = int(max_length_guard.get("effective_max_length") or 0)
    reserve_tokens = max(0, int(max_length_guard.get("reserve_tokens") or 0))
    if (
        effective_max_length > 0
        and prompt_token_count + int(target_token_reserve) + reserve_tokens > effective_max_length
    ):
        raise ValueError(
            "prompt plus target/context reserve exceeds effective max length: "
            f"prompt={prompt_token_count}, target_reserve={target_token_reserve}, "
            f"reserve_tokens={reserve_tokens}, effective_max_length={effective_max_length}"
        )


def _resolve_prompt_evidence_config(
    cfg: dict[str, Any],
    *,
    args: argparse.Namespace,
    prompt_cfg: dict[str, Any],
) -> dict[str, Any]:
    raw = dict(cfg.get("prompt_evidence") or {})
    if args.prompt_evidence_policy is not None:
        raw["policy"] = str(args.prompt_evidence_policy)
    if args.prompt_evidence_min_count is not None:
        raw["min_evidence_count"] = int(args.prompt_evidence_min_count)
    if args.prompt_evidence_max_count is not None:
        raw["max_evidence_count"] = int(args.prompt_evidence_max_count)
    if args.prompt_evidence_token_budget is not None:
        raw["evidence_token_budget"] = int(args.prompt_evidence_token_budget)
    if args.prompt_evidence_prompt_token_budget is not None:
        raw["prompt_token_budget"] = int(args.prompt_evidence_prompt_token_budget)

    guard = dict(raw.get("max_length_guard") or {})
    if args.prompt_evidence_max_length_guard is not None:
        if args.prompt_evidence_max_length_guard == "off":
            guard["enabled"] = False
        else:
            guard["enabled"] = True
            guard["on_violation"] = str(args.prompt_evidence_max_length_guard)
    build_prompt_max_length = int(prompt_cfg.get("max_length", 0) or 0)
    sft_train = dict(cfg.get("sft_train") or {})
    sft_train_max_length = int(sft_train.get("max_length", build_prompt_max_length) or build_prompt_max_length)
    guard.setdefault("build_prompt_max_length", build_prompt_max_length)
    guard.setdefault("sft_train_max_length", sft_train_max_length)
    raw["max_length_guard"] = guard
    return raw


def _normalize_prompt_evidence_config(
    config: dict[str, Any] | None,
    *,
    fallback_top_k: int,
    prompt_cfg: dict[str, Any],
) -> dict[str, Any]:
    raw = dict(config or {})
    policy = str(raw.get("policy") or "prefix_topk").strip().lower()
    if policy not in PROMPT_EVIDENCE_POLICIES:
        raise ValueError(f"unsupported prompt_evidence.policy: {policy!r}")

    fallback_top_k = max(0, int(fallback_top_k))
    default_max = fallback_top_k if policy in {"prefix_topk", "fixed_topk"} else 0
    min_count = max(0, _int_or_default(raw.get("min_evidence_count"), 0))
    max_count = max(0, _int_or_default(raw.get("max_evidence_count"), default_max))
    if policy == "fixed_topk":
        if max_count <= 0:
            max_count = fallback_top_k
        if min_count <= 0:
            min_count = max_count
    if policy == "prefix_topk":
        max_count = max_count or fallback_top_k
    if max_count > 0 and min_count > max_count:
        raise ValueError(
            f"prompt_evidence min_evidence_count={min_count} exceeds max_evidence_count={max_count}."
        )

    token_budget = _int_or_none(raw.get("evidence_token_budget"))
    if token_budget is not None and token_budget <= 0:
        token_budget = None
    prompt_token_budget = _int_or_none(raw.get("prompt_token_budget"))
    if prompt_token_budget is not None and prompt_token_budget <= 0:
        prompt_token_budget = None
    if policy == "prompt_budget":
        if prompt_token_budget is None:
            raise ValueError(
                "prompt_evidence.policy=prompt_budget requires a positive prompt_token_budget."
            )
        if token_budget is not None:
            raise ValueError(
                "prompt_evidence.policy=prompt_budget must not set evidence_token_budget; "
                "use prompt_token_budget for the final verifier-visible prompt cap."
            )
        min_count = max(1, min_count)
    elif prompt_token_budget is not None:
        raise ValueError(
            "prompt_evidence.prompt_token_budget is only valid with policy=prompt_budget."
        )

    guard = _normalize_max_length_guard(
        dict(raw.get("max_length_guard") or {}),
        prompt_cfg=prompt_cfg,
        evidence_token_budget=token_budget,
        prompt_token_budget=prompt_token_budget,
    )
    state_budget = _normalize_state_budget_config(
        dict(raw.get("state_budget") or {}),
        guard=guard,
    )
    two_pass_uncertainty = _normalize_two_pass_uncertainty_config(
        dict(raw.get("two_pass_uncertainty") or {})
    )
    return {
        "policy": policy,
        "min_evidence_count": int(min_count),
        "max_evidence_count": int(max_count),
        "evidence_token_budget": token_budget,
        "prompt_token_budget": prompt_token_budget,
        "max_length_guard": guard,
        "state_budget": state_budget,
        "two_pass_uncertainty": two_pass_uncertainty,
    }


def _normalize_max_length_guard(
    raw: dict[str, Any],
    *,
    prompt_cfg: dict[str, Any],
    evidence_token_budget: int | None,
    prompt_token_budget: int | None,
) -> dict[str, Any]:
    enabled = bool(raw.get("enabled", False))
    build_prompt_max_length = int(
        raw.get("build_prompt_max_length")
        or prompt_cfg.get("max_length")
        or 0
    )
    sft_train_max_length = int(raw.get("sft_train_max_length") or build_prompt_max_length)
    positive_lengths = [value for value in (build_prompt_max_length, sft_train_max_length) if value > 0]
    effective_max_length = min(positive_lengths) if positive_lengths else 0
    reserve_tokens = max(0, _int_or_default(raw.get("reserve_tokens"), 0))
    context_prompt_budget = (
        max(0, int(effective_max_length) - reserve_tokens)
        if effective_max_length > 0
        else 0
    )
    effective_prompt_budget = context_prompt_budget
    if prompt_token_budget is not None:
        effective_prompt_budget = (
            min(int(prompt_token_budget), context_prompt_budget)
            if context_prompt_budget > 0
            else int(prompt_token_budget)
        )
    on_violation = str(raw.get("on_violation") or "warn").strip().lower()
    if on_violation not in {"warn", "error"}:
        raise ValueError(f"unsupported prompt_evidence.max_length_guard.on_violation: {on_violation!r}")
    return {
        "enabled": enabled,
        "on_violation": on_violation,
        "build_prompt_max_length": int(build_prompt_max_length),
        "sft_train_max_length": int(sft_train_max_length),
        "effective_max_length": int(effective_max_length),
        "reserve_tokens": int(reserve_tokens),
        "context_prompt_budget": int(context_prompt_budget),
        "effective_prompt_budget": int(effective_prompt_budget),
        "config_conflict": bool(
            build_prompt_max_length > 0
            and sft_train_max_length > 0
            and build_prompt_max_length != sft_train_max_length
        ),
        "config_budget_conflict": bool(
            (prompt_token_budget if prompt_token_budget is not None else evidence_token_budget)
            is not None
            and context_prompt_budget > 0
            and int(prompt_token_budget if prompt_token_budget is not None else evidence_token_budget)
            > context_prompt_budget
        ),
    }


def _normalize_state_budget_config(
    raw: dict[str, Any],
    *,
    guard: dict[str, Any],
) -> dict[str, Any]:
    budget_ratio = _float_or_default(raw.get("budget_ratio"), 0.75)
    if budget_ratio < 0:
        budget_ratio = 0.0
    if budget_ratio > 1:
        budget_ratio = 1.0

    explicit_budget = _int_or_none(raw.get("effective_token_budget"))
    if explicit_budget is not None and explicit_budget <= 0:
        explicit_budget = None
    effective_guard_budget = int(guard.get("effective_prompt_budget") or 0)
    effective_token_budget = explicit_budget
    if effective_token_budget is None and effective_guard_budget > 0 and budget_ratio > 0:
        effective_token_budget = max(1, int(effective_guard_budget * budget_ratio))

    return {
        "budget_ratio": float(budget_ratio),
        "effective_token_budget": effective_token_budget,
        "lookahead_on_target_resolved": _bool_or_default(
            raw.get("lookahead_on_target_resolved"),
            True,
        ),
        "include_contrast_after_target": _bool_or_default(
            raw.get("include_contrast_after_target"),
            True,
        ),
        "unresolved_patience": max(0, _int_or_default(raw.get("unresolved_patience"), 2)),
    }


def _normalize_two_pass_uncertainty_config(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_dir": _compact_whitespace(raw.get("decision_dir") or ""),
        "calibration_file": _compact_whitespace(raw.get("calibration_file") or ""),
        "teacher_run_dir": _compact_whitespace(raw.get("teacher_run_dir") or ""),
        "teacher_checkpoint": _compact_whitespace(raw.get("teacher_checkpoint") or "best"),
        "scoring_backend": _compact_whitespace(raw.get("scoring_backend") or "auto"),
        "base_model": _compact_whitespace(raw.get("base_model") or ""),
        "decision_file_template": _compact_whitespace(
            raw.get("decision_file_template") or "two_pass_uncertainty_decisions_{split}.jsonl"
        ),
    }


def _public_prompt_evidence_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": str(config.get("policy") or "prefix_topk"),
        "min_evidence_count": int(config.get("min_evidence_count") or 0),
        "max_evidence_count": int(config.get("max_evidence_count") or 0),
        "evidence_token_budget": config.get("evidence_token_budget"),
        "prompt_token_budget": config.get("prompt_token_budget"),
        "max_length_guard": dict(config.get("max_length_guard") or {}),
        "state_budget": dict(config.get("state_budget") or {}),
        "two_pass_uncertainty": dict(config.get("two_pass_uncertainty") or {}),
    }


def _select_prompt_evidence_indices(
    trace: dict[str, Any],
    *,
    ordered_indices: list[int],
    config: dict[str, Any],
    event_id: str | None = None,
    split: str | None = None,
    two_pass_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = str(config.get("policy") or "prefix_topk")
    min_count = max(0, int(config.get("min_evidence_count") or 0))
    max_count = max(0, int(config.get("max_evidence_count") or 0))
    token_budget = config.get("evidence_token_budget")
    token_budget = int(token_budget) if token_budget is not None else None
    state_budget = dict(config.get("state_budget") or {})
    if policy == "state_budget" and token_budget is None:
        parsed_state_budget = _int_or_none(state_budget.get("effective_token_budget"))
        if parsed_state_budget is not None and parsed_state_budget > 0:
            token_budget = parsed_state_budget

    selected: list[int] = []
    total_token_cost = 0
    stop_reason = "end_of_trace"

    if policy == "two_pass_uncertainty":
        if not isinstance(two_pass_decision, dict):
            location = ":".join(item for item in (str(split or ""), str(event_id or "")) if item)
            suffix = f" for {location}" if location else ""
            raise ValueError(f"missing two-pass uncertainty decision{suffix}")
        selected = _two_pass_selected_indices_from_decision(
            trace,
            ordered_indices=ordered_indices,
            decision=two_pass_decision,
        )
        return _prompt_evidence_decision(
            selected,
            policy=policy,
            min_count=min_count,
            max_count=max_count,
            token_budget=token_budget,
            total_token_cost=_prompt_evidence_total_token_cost(trace, selected),
            stop_reason=str(two_pass_decision.get("stop_reason") or "two_pass_decision"),
            two_pass_uncertainty=two_pass_decision,
        )

    if policy == "selected_set":
        selected = list(ordered_indices)
        total_token_cost = _prompt_evidence_total_token_cost(trace, selected)
        if max_count > 0 and len(selected) > max_count:
            raise ValueError(
                f"selected_set contains {len(selected)} items, exceeding max_count={max_count}"
            )
        if token_budget is not None and total_token_cost > token_budget:
            raise ValueError(
                f"selected_set token cost {total_token_cost} exceeds token_budget={token_budget}"
            )
        return _prompt_evidence_decision(
            selected,
            policy=policy,
            min_count=len(selected),
            max_count=len(selected),
            token_budget=token_budget,
            total_token_cost=total_token_cost,
            stop_reason="selected_set_exhausted",
        )

    if policy in {"prefix_topk", "fixed_topk", "prompt_budget"}:
        limit = max_count if max_count > 0 else len(ordered_indices)
        selected = list(ordered_indices[:limit])
        if policy == "prefix_topk":
            stop_reason = "top_k"
        elif policy == "fixed_topk":
            stop_reason = "max_evidence_count"
        elif len(selected) < len(ordered_indices):
            stop_reason = "max_evidence_count"
        else:
            stop_reason = "end_of_trace"
        return _prompt_evidence_decision(
            selected,
            policy=policy,
            min_count=min_count,
            max_count=max_count,
            token_budget=token_budget,
            total_token_cost=_prompt_evidence_total_token_cost(trace, selected),
            stop_reason=stop_reason,
        )

    if policy == "state_budget":
        selected, total_token_cost, stop_reason = _select_state_budget_prompt_evidence_indices(
            trace,
            ordered_indices=ordered_indices,
            min_count=min_count,
            max_count=max_count,
            token_budget=token_budget,
            state_budget=state_budget,
        )
        return _prompt_evidence_decision(
            selected,
            policy=policy,
            min_count=min_count,
            max_count=max_count,
            token_budget=token_budget,
            total_token_cost=total_token_cost,
            stop_reason=stop_reason,
            state_budget=state_budget,
        )

    for idx in ordered_indices:
        token_cost = _prompt_evidence_token_cost(trace, idx)
        if max_count > 0 and len(selected) >= max_count:
            stop_reason = "max_evidence_count"
            break
        if (
            policy == "budget"
            and token_budget is not None
            and len(selected) >= min_count
            and total_token_cost + token_cost > token_budget
        ):
            stop_reason = "token_budget_exhausted"
            break
        selected.append(int(idx))
        total_token_cost += token_cost
        if (
            policy in {"resolve_stop", "minmax"}
            and len(selected) >= min_count
            and _prompt_evidence_target_resolved(trace, idx)
        ):
            stop_reason = "target_resolved"
            break

    if max_count > 0 and len(selected) >= max_count and stop_reason == "end_of_trace":
        stop_reason = "max_evidence_count"
    if not selected:
        raise ValueError("prompt evidence policy selected no indices")
    return _prompt_evidence_decision(
        selected,
        policy=policy,
        min_count=min_count,
        max_count=max_count,
        token_budget=token_budget,
        total_token_cost=total_token_cost,
        stop_reason=stop_reason,
    )


def _select_state_budget_prompt_evidence_indices(
    trace: dict[str, Any],
    *,
    ordered_indices: list[int],
    min_count: int,
    max_count: int,
    token_budget: int | None,
    state_budget: dict[str, Any],
) -> tuple[list[int], int, str]:
    selected: list[int] = []
    total_token_cost = 0
    stop_reason = "end_of_trace"
    previous_state: dict[str, Any] | None = None
    unresolved_no_progress = 0

    lookahead_on_target_resolved = _bool_or_default(
        state_budget.get("lookahead_on_target_resolved"),
        True,
    )
    unresolved_patience = max(0, _int_or_default(state_budget.get("unresolved_patience"), 2))

    for position, idx in enumerate(ordered_indices):
        token_cost = _prompt_evidence_token_cost(trace, idx)
        if max_count > 0 and len(selected) >= max_count:
            stop_reason = "max_evidence_count"
            break
        if (
            token_budget is not None
            and selected
            and len(selected) >= min_count
            and total_token_cost + token_cost > token_budget
        ):
            stop_reason = "token_budget_exhausted"
            break

        selected.append(int(idx))
        total_token_cost += token_cost
        current_state = _prompt_evidence_step_state(trace, idx)

        if len(selected) < min_count:
            previous_state = current_state
            continue

        if bool(current_state.get("target_resolved")):
            if not lookahead_on_target_resolved:
                stop_reason = "target_resolved"
                break
            next_idx = _next_ordered_index(ordered_indices, position)
            if next_idx is None:
                stop_reason = "target_resolved"
                break
            if _state_budget_should_include_next_after_target(
                trace,
                current_idx=idx,
                next_idx=next_idx,
                state_budget=state_budget,
            ):
                previous_state = current_state
                continue
            stop_reason = "target_resolved_stable"
            break

        if previous_state is not None:
            if _prompt_evidence_state_changed(previous_state, current_state):
                unresolved_no_progress = 0
            else:
                unresolved_no_progress += 1
            if unresolved_patience > 0 and unresolved_no_progress >= unresolved_patience:
                stop_reason = "unresolved_no_progress"
                break
        previous_state = current_state

    if max_count > 0 and len(selected) >= max_count and stop_reason == "end_of_trace":
        stop_reason = "max_evidence_count"
    if not selected:
        raise ValueError("prompt evidence policy selected no indices")
    return selected, total_token_cost, stop_reason


def _two_pass_selected_indices_from_decision(
    trace: dict[str, Any],
    *,
    ordered_indices: list[int],
    decision: dict[str, Any],
) -> list[int]:
    pool = trace.get("candidate_pool") or []
    n = len(pool) if isinstance(pool, list) else 0
    selected = []
    for item in decision.get("selected_indices") or []:
        parsed = _int_or_none(item)
        if parsed is not None:
            selected.append(int(parsed))
    selected = _dedupe_in_range(selected, n)
    if not selected:
        raise ValueError("two-pass uncertainty decision selected no indices")
    ordered_set = set(int(idx) for idx in ordered_indices)
    missing = [idx for idx in selected if idx not in ordered_set]
    if missing:
        raise ValueError(f"two-pass uncertainty decision has indices outside selector order: {missing}")
    return selected


def _next_ordered_index(ordered_indices: list[int], position: int) -> int | None:
    next_position = int(position) + 1
    if next_position >= len(ordered_indices):
        return None
    return int(ordered_indices[next_position])


def _state_budget_should_include_next_after_target(
    trace: dict[str, Any],
    *,
    current_idx: int,
    next_idx: int,
    state_budget: dict[str, Any],
) -> bool:
    current_state = _prompt_evidence_step_state(trace, current_idx)
    next_state = _prompt_evidence_step_state(trace, next_idx)
    if _prompt_evidence_state_changed(current_state, next_state):
        return True
    if not _bool_or_default(state_budget.get("include_contrast_after_target"), True):
        return False
    current_operation = _prompt_evidence_step_operation(trace, current_idx)
    next_operation = _prompt_evidence_step_operation(trace, next_idx)
    return next_operation == "CONTRAST" and current_operation != "CONTRAST"


def _prompt_evidence_decision(
    selected: list[int],
    *,
    policy: str,
    min_count: int,
    max_count: int,
    token_budget: int | None,
    total_token_cost: int,
    stop_reason: str,
    state_budget: dict[str, Any] | None = None,
    two_pass_uncertainty: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = {
        "selected_indices": [int(idx) for idx in selected],
        "policy": policy,
        "min_evidence_count": int(min_count),
        "max_evidence_count": int(max_count),
        "evidence_token_budget": token_budget,
        "selected_count_before_prompt_truncation": int(len(selected)),
        "selected_token_cost": int(total_token_cost),
        "stop_reason": stop_reason,
    }
    if state_budget is not None:
        decision["state_budget"] = dict(state_budget)
    if two_pass_uncertainty is not None:
        decision["two_pass_uncertainty"] = dict(two_pass_uncertainty)
    return decision


def _prompt_evidence_row_fields(config: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "prompt_evidence_policy": str(decision["policy"]),
        "prompt_evidence_min_count": int(decision["min_evidence_count"]),
        "prompt_evidence_max_count": int(decision["max_evidence_count"]),
        "prompt_evidence_token_budget": decision["evidence_token_budget"],
        "prompt_evidence_selected_count_before_prompt_truncation": int(
            decision["selected_count_before_prompt_truncation"]
        ),
        "prompt_evidence_selected_token_cost": int(decision["selected_token_cost"]),
        "prompt_evidence_stop_reason": str(decision["stop_reason"]),
    }
    if str(decision.get("policy") or "") == "prompt_budget":
        fields.update(
            {
                "prompt_evidence_prompt_token_budget": int(
                    decision.get("prompt_token_budget")
                    or config.get("prompt_token_budget")
                    or 0
                ),
                "prompt_evidence_effective_prompt_token_budget": int(
                    decision.get("effective_prompt_token_budget") or 0
                ),
                "prompt_evidence_selected_prompt_token_count": int(
                    decision.get("selected_prompt_token_count") or 0
                ),
                "prompt_evidence_target_token_reserve": int(
                    decision.get("target_token_reserve") or 0
                ),
                "prompt_evidence_partial_evidence": bool(
                    decision.get("partial_evidence", False)
                ),
                "prompt_evidence_considered_count": int(
                    decision.get("considered_count")
                    or decision.get("selected_count_before_prompt_truncation")
                    or 0
                ),
            }
        )
    if decision.get("state_budget") is not None:
        fields["prompt_evidence_state_budget"] = dict(decision.get("state_budget") or {})
    two_pass = decision.get("two_pass_uncertainty")
    if isinstance(two_pass, dict):
        fields.update(
            {
                "prompt_evidence_two_pass_initial_count": len(two_pass.get("initial_indices") or []),
                "prompt_evidence_uncertainty_margin": _float_or_none(two_pass.get("uncertainty_margin")),
                "prompt_evidence_expanded": bool(two_pass.get("prompt_evidence_expanded")),
                "prompt_evidence_score_trace": list(two_pass.get("score_trace") or []),
                "prompt_evidence_decision_source": str(two_pass.get("decision_source") or ""),
                "prompt_evidence_two_pass_threshold": _float_or_none(two_pass.get("threshold")),
                "prompt_evidence_prompt_token_count_before_final_build": _int_or_none(
                    two_pass.get("prompt_token_count_before_final_build")
                ),
            }
        )
    return fields


def _max_length_guard_result(
    row: dict[str, Any],
    *,
    prompt_evidence: dict[str, Any],
    max_length_guard: dict[str, Any],
) -> dict[str, Any]:
    del prompt_evidence
    enabled = bool(max_length_guard.get("enabled", False))
    reasons: list[str] = []
    prompt_token_count = int(row.get("prompt_token_count") or 0)
    effective_prompt_budget = int(max_length_guard.get("effective_prompt_budget") or 0)
    if enabled and effective_prompt_budget > 0 and prompt_token_count > effective_prompt_budget:
        reasons.append("prompt_token_count_exceeds_guard")
    return {
        **dict(max_length_guard),
        "enabled": enabled,
        "prompt_token_count": prompt_token_count,
        "violation_reasons": reasons,
    }


def _max_length_guard_report(
    max_length_guard: dict[str, Any],
    *,
    violations: Counter[str],
    row_count: int,
) -> dict[str, Any]:
    violation_count = int(sum(violations.values()))
    return {
        **dict(max_length_guard),
        "violation_count": violation_count,
        "violation_rate": float(violation_count / max(row_count, 1)),
        "violations_by_reason": dict(violations),
    }


def _load_two_pass_uncertainty_decisions(
    split: str,
    prompt_evidence: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    config = dict(prompt_evidence.get("two_pass_uncertainty") or {})
    decision_dir = _compact_whitespace(config.get("decision_dir") or "")
    if not decision_dir:
        raise ValueError("missing two-pass uncertainty decision cache: prompt_evidence.two_pass_uncertainty.decision_dir")
    template = _compact_whitespace(
        config.get("decision_file_template") or "two_pass_uncertainty_decisions_{split}.jsonl"
    )
    decision_path = Path(decision_dir) / template.format(split=str(split))
    if not decision_path.exists():
        raise ValueError(f"missing two-pass uncertainty decision cache: {decision_path}")
    decisions: dict[str, dict[str, Any]] = {}
    with decision_path.open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid two-pass uncertainty decision cache line {line_num}: {decision_path}") from exc
            event_id = str(row.get("event_id") or "")
            if not event_id:
                raise ValueError(f"two-pass uncertainty decision missing event_id at {decision_path}:{line_num}")
            copied = dict(row)
            copied["decision_source"] = str(decision_path)
            decisions[event_id] = copied
    if not decisions:
        raise ValueError(f"missing two-pass uncertainty decision cache rows: {decision_path}")
    return decisions


def _load_selection_plan(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"selection plan does not exist: {path}")
    plans: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid selection plan JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(f"selection plan row is not an object at {path}:{line_number}")
            event_id = _compact_whitespace(raw.get("event_id") or "")
            if not event_id:
                raise ValueError(f"selection plan row has no event_id at {path}:{line_number}")
            if event_id in plans:
                raise ValueError(f"selection plan has duplicate event_id={event_id!r}")
            indices = raw.get("selected_indices")
            if not isinstance(indices, list) or not indices:
                raise ValueError(
                    f"selection plan event={event_id!r} must contain non-empty selected_indices"
                )
            parsed: list[int] = []
            for value in indices:
                if isinstance(value, bool):
                    raise ValueError(
                        f"selection plan event={event_id!r} contains a boolean index"
                    )
                parsed_value = _int_or_none(value)
                if parsed_value is None:
                    raise ValueError(
                        f"selection plan event={event_id!r} contains a non-integer index"
                    )
                parsed.append(int(parsed_value))
            copied = dict(raw)
            copied["selected_indices"] = parsed
            copied["selection_plan_path"] = str(path)
            plans[event_id] = copied
    if not plans:
        raise ValueError(f"selection plan has no rows: {path}")
    return plans


def _selected_indices_from_plan(
    trace: dict[str, Any],
    *,
    ordered_indices: list[int],
    plan: dict[str, Any],
    event_id: str,
) -> list[int]:
    selected = [int(value) for value in plan.get("selected_indices") or []]
    pool = trace.get("candidate_pool") or []
    if not selected:
        raise ValueError("selection plan produced no indices")
    if len(selected) != len(set(selected)) or any(
        idx < 0 or idx >= len(pool) for idx in selected
    ):
        raise ValueError("selection plan contains duplicate or out-of-range indices")
    if selected != ordered_indices[: len(selected)]:
        raise ValueError(
            "selection plan must be an exact prefix of the configured trace order"
        )
    declared_uids = plan.get("selected_candidate_uids")
    if declared_uids is not None:
        if not isinstance(declared_uids, list) or not all(
            isinstance(value, str) for value in declared_uids
        ):
            raise ValueError("selection plan selected_candidate_uids must be a string array")
        actual_uids = [
            _compact_whitespace(pool[idx].get("candidate_uid") or "")
            if isinstance(pool[idx], dict)
            else ""
            for idx in selected
        ]
        if [str(value) for value in declared_uids] != actual_uids:
            raise ValueError(
                f"selection plan candidate UID mismatch for event_id={event_id!r}"
            )
    realized_k = _int_or_none(plan.get("prompt_feasible_prefix_k"))
    if realized_k is not None and realized_k != len(selected):
        raise ValueError(
            "selection plan prompt_feasible_prefix_k disagrees with selected_indices"
        )
    requested_k = _int_or_none(plan.get("requested_prefix_k"))
    if requested_k is not None and len(selected) > requested_k:
        raise ValueError("selection plan realized prefix exceeds requested_prefix_k")
    return selected


def _two_pass_uncertainty_report(
    *,
    two_pass_decisions: dict[str, dict[str, Any]],
    row_count: int,
    expanded_count: int,
    initial_counts: list[int],
    final_counts: list[int],
    prompt_tokens_before: list[int],
    uncertainty_margins: list[float],
) -> dict[str, Any]:
    if not two_pass_decisions:
        return {"enabled": False}
    return {
        "enabled": True,
        "decision_count": int(len(two_pass_decisions)),
        "expanded_count": int(expanded_count),
        "expanded_rate": float(expanded_count / max(int(row_count), 1)),
        "initial_evidence_count": _summary(initial_counts),
        "final_evidence_count": _summary(final_counts),
        "prompt_token_count_before_final_build": _summary(prompt_tokens_before),
        "uncertainty_margin": _float_summary(uncertainty_margins),
    }


def _prompt_evidence_total_token_cost(trace: dict[str, Any], selected: list[int]) -> int:
    return int(sum(_prompt_evidence_token_cost(trace, idx) for idx in selected))


def _prompt_evidence_token_cost(trace: dict[str, Any], idx: int) -> int:
    step = _mrec_step_by_selector_idx(trace).get(int(idx))
    if step is not None:
        parsed = _int_or_none(step.get("token_cost"))
        if parsed is not None:
            return max(0, parsed)
    pool = trace.get("candidate_pool") or []
    candidate = pool[idx] if 0 <= int(idx) < len(pool) and isinstance(pool[idx], dict) else {}
    for key in ("mrec_token_cost", "token_cost", "evidence_token_count", "prompt_token_count"):
        parsed = _int_or_none(candidate.get(key))
        if parsed is not None:
            return max(0, parsed)
    text = str(candidate.get("text") or candidate.get("evidence_text") or "")
    return max(1, len(text.split())) if text.strip() else 0


def _prompt_evidence_target_resolved(trace: dict[str, Any], idx: int) -> bool:
    step = _mrec_step_by_selector_idx(trace).get(int(idx))
    if not isinstance(step, dict):
        return False
    trace_state = step.get("trace_state")
    if isinstance(trace_state, dict) and "target_resolved" in trace_state:
        return bool(trace_state.get("target_resolved"))
    if "target_resolved" in step:
        return bool(step.get("target_resolved"))
    return False


def _prompt_evidence_step_state(trace: dict[str, Any], idx: int) -> dict[str, Any]:
    step = _mrec_step_by_selector_idx(trace).get(int(idx))
    if not isinstance(step, dict):
        return {
            "target_resolved": False,
            "resolved_atom_rate": 0.0,
            "unresolved_atom_ids": (),
            "conflicted_atom_ids": (),
            "atom_states_after": (),
        }

    trace_state = step.get("trace_state")
    state = dict(trace_state) if isinstance(trace_state, dict) else {}
    target_resolved = state.get("target_resolved")
    if target_resolved is None:
        target_resolved = step.get("target_resolved", False)
    resolved_atom_rate = _float_or_default(state.get("resolved_atom_rate"), 0.0)
    atom_states_after = state.get("atom_states_after")
    if not isinstance(atom_states_after, dict):
        atom_id = _compact_whitespace(step.get("atom_id") or "")
        state_after = _compact_whitespace(step.get("state_after") or "")
        atom_states_after = {atom_id: state_after} if atom_id and state_after else {}
    return {
        "target_resolved": bool(target_resolved),
        "resolved_atom_rate": float(resolved_atom_rate),
        "unresolved_atom_ids": tuple(sorted(_covered_atom_ids(state.get("unresolved_atom_ids")))),
        "conflicted_atom_ids": tuple(sorted(_covered_atom_ids(state.get("conflicted_atom_ids")))),
        "atom_states_after": tuple(
            sorted(
                (
                    _compact_whitespace(atom_id),
                    _compact_whitespace(atom_state),
                )
                for atom_id, atom_state in atom_states_after.items()
                if _compact_whitespace(atom_id)
            )
        ),
    }


def _prompt_evidence_state_changed(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_rate = float(left.get("resolved_atom_rate") or 0.0)
    right_rate = float(right.get("resolved_atom_rate") or 0.0)
    return (
        bool(left.get("target_resolved")) != bool(right.get("target_resolved"))
        or abs(left_rate - right_rate) > 1e-9
        or tuple(left.get("unresolved_atom_ids") or ()) != tuple(right.get("unresolved_atom_ids") or ())
        or tuple(left.get("conflicted_atom_ids") or ()) != tuple(right.get("conflicted_atom_ids") or ())
        or tuple(left.get("atom_states_after") or ()) != tuple(right.get("atom_states_after") or ())
    )


def _prompt_evidence_step_operation(trace: dict[str, Any], idx: int) -> str:
    step = _mrec_step_by_selector_idx(trace).get(int(idx))
    if not isinstance(step, dict):
        return ""
    return _compact_whitespace(step.get("operation") or "").upper()


def _mrec_step_by_selector_idx(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for fallback_idx, step in enumerate(trace.get("mrec_steps") or []):
        if not isinstance(step, dict):
            continue
        selector_idx = _int_or_none(step.get("selector_candidate_idx"))
        if selector_idx is None:
            selector_idx = _int_or_none(step.get("candidate_idx"))
        if selector_idx is None:
            selector_idx = fallback_idx
        out[int(selector_idx)] = step
    return out


def _apply_qec_prompt_fields(
    *,
    claim: str,
    candidates: list[dict[str, Any]],
    claim_atoms: list[dict[str, Any]],
    chain_steps: list[dict[str, Any]] | None = None,
    style: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if style not in {"qec_min", "qec_map"}:
        raise ValueError(f"unsupported QEC prompt style: {style}")

    atom_by_id, atom_order = _claim_atom_lookup(claim_atoms)

    rendered_candidates: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    cue_type_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    directness_counts: Counter[str] = Counter()
    focus_counts: Counter[str] = Counter()
    check_token_counts: list[int] = []
    aligned_chain_steps = _align_qec_chain_steps(chain_steps or [], candidates)

    for step_idx, candidate in enumerate(candidates, start=1):
        copied = dict(candidate)
        chain_step = aligned_chain_steps[step_idx - 1] if step_idx - 1 < len(aligned_chain_steps) else None
        original_text = _qec_step_evidence_text(chain_step, copied)
        cue = _select_qec_cue_from_chain_step(chain_step)
        if cue is None:
            cue = _select_qec_cue(copied, atom_by_id=atom_by_id, atom_order=atom_order)
        covers = _render_covered_atom_ids(
            chain_step.get("covered_atom_ids") if isinstance(chain_step, dict) else copied.get("covered_atom_ids")
        )
        relation = (
            _compact_whitespace(chain_step.get("relation") or "")
            if isinstance(chain_step, dict)
            else _compact_whitespace(copied.get("map_relation") or "")
        ) or "unknown"
        directness = (
            _compact_whitespace(chain_step.get("directness") or "")
            if isinstance(chain_step, dict)
            else _compact_whitespace(copied.get("map_directness") or "")
        ) or "unknown"

        if style == "qec_map":
            copied["text"] = (
                f"Check: {cue['check']} "
                f"[covers={covers}; relation={relation}; directness={directness}]\n"
                f"{original_text}"
            )
        else:
            copied["text"] = f"Check: {cue['check']}\n{original_text}"
        rendered_candidates.append(copied)

        cue_type = str(cue["cue_type"])
        cue_type_counts[cue_type] += 1
        relation_counts[relation] += 1
        directness_counts[directness] += 1
        if cue.get("question_focus"):
            focus_counts[str(cue["question_focus"])] += 1
        check_token_counts.append(len(str(cue["check"]).split()))

        steps.append(
            {
                "step": int(step_idx),
                "candidate_idx": int(copied.get("candidate_idx", copied.get("selector_candidate_idx", step_idx - 1))),
                "evidence_id": str(copied.get("evidence_id") or copied.get("selector_pool_evidence_id") or ""),
                "selector_rank": int(copied.get("selector_trace_rank", step_idx - 1)) + 1,
                "cue_type": cue_type,
                "check": str(cue["check"]),
                "question_id": str(cue.get("question_id") or ""),
                "question_focus": str(cue.get("question_focus") or ""),
                "question_route_rank": cue.get("question_route_rank"),
                "question_route_hybrid_score": cue.get("question_route_hybrid_score"),
                "covered_atom_ids": _covered_atom_ids(copied.get("covered_atom_ids")),
                "map_relation": relation,
                "map_directness": directness,
                "role": str(chain_step.get("role") or "") if isinstance(chain_step, dict) else "",
            }
        )

    total = max(len(rendered_candidates), 1)
    diagnostics = {
        "cue_type_counts": dict(cue_type_counts),
        "qd_cue_rate": float(cue_type_counts.get("qd_question", 0) / total),
        "atom_fallback_rate": float(cue_type_counts.get("claim_atom", 0) / total),
        "fallback_rate": float(cue_type_counts.get("fallback", 0) / total),
        "map_relation_counts": dict(relation_counts),
        "map_directness_counts": dict(directness_counts),
        "question_focus_counts": dict(focus_counts),
        "mean_check_token_count": float(np.mean(check_token_counts)) if check_token_counts else 0.0,
    }
    return str(claim), rendered_candidates, {"steps": steps, "diagnostics": diagnostics}


def _apply_mrec_prompt_fields(
    *,
    claim: str,
    candidates: list[dict[str, Any]],
    trace: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    source_name, source_steps = _mrec_prompt_source_steps(trace)
    aligned_steps = _align_qec_chain_steps(source_steps, candidates)
    atom_by_id, atom_order = _claim_atom_lookup(trace.get("claim_atoms") or [])

    rendered_candidates: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    cue_type_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    state_after_counts: Counter[str] = Counter()
    check_token_counts: list[int] = []
    token_costs: list[int] = []

    for step_idx, candidate in enumerate(candidates, start=1):
        copied = dict(candidate)
        mrec_step = aligned_steps[step_idx - 1] if step_idx - 1 < len(aligned_steps) else None
        original_text = _qec_step_evidence_text(mrec_step, copied)
        cue = _select_mrec_cue(mrec_step, source_name=source_name)
        if cue is None:
            cue = _select_mrec_fallback_cue(copied, atom_by_id=atom_by_id, atom_order=atom_order)

        copied["text"] = f"Check: {cue['check']}\n{original_text}"
        rendered_candidates.append(copied)

        cue_type = str(cue["cue_type"])
        operation = str(mrec_step.get("operation") or "").upper() if isinstance(mrec_step, dict) else ""
        state_after = str(mrec_step.get("state_after") or "").upper() if isinstance(mrec_step, dict) else ""
        token_cost = _int_or_none(mrec_step.get("token_cost")) if isinstance(mrec_step, dict) else None
        cue_type_counts[cue_type] += 1
        if operation:
            operation_counts[operation] += 1
        if state_after:
            state_after_counts[state_after] += 1
        if token_cost is not None:
            token_costs.append(token_cost)
        check_token_counts.append(len(str(cue["check"]).split()))

        steps.append(
            {
                "step": int(step_idx),
                "source": source_name,
                "candidate_idx": int(copied.get("candidate_idx", copied.get("selector_candidate_idx", step_idx - 1))),
                "evidence_id": str(copied.get("evidence_id") or copied.get("selector_pool_evidence_id") or ""),
                "selector_rank": int(copied.get("selector_trace_rank", step_idx - 1)) + 1,
                "cue_type": cue_type,
                "check": str(cue["check"]),
                "operation": operation,
                "atom_id": str(mrec_step.get("atom_id") or "") if isinstance(mrec_step, dict) else "",
                "state_before": str(mrec_step.get("state_before") or "") if isinstance(mrec_step, dict) else "",
                "state_after": str(mrec_step.get("state_after") or "") if isinstance(mrec_step, dict) else "",
                "covered_atom_ids": _covered_atom_ids(
                    mrec_step.get("covered_atom_ids") if isinstance(mrec_step, dict) else copied.get("covered_atom_ids")
                ),
                "token_cost": token_cost,
            }
        )

    total = max(len(rendered_candidates), 1)
    diagnostics = {
        "source": source_name,
        "cue_type_counts": dict(cue_type_counts),
        "operation_counts": dict(operation_counts),
        "state_after_counts": dict(state_after_counts),
        "mrec_step_cue_rate": float(cue_type_counts.get("mrec_step", 0) / total),
        "compat_chain_step_cue_rate": float(cue_type_counts.get("compat_chain_step", 0) / total),
        "fallback_rate": float(cue_type_counts.get("fallback", 0) / total),
        "mean_check_token_count": float(np.mean(check_token_counts)) if check_token_counts else 0.0,
        "total_token_cost": int(sum(token_costs)) if token_costs else 0,
        "mean_token_cost": float(np.mean(token_costs)) if token_costs else 0.0,
    }
    return str(claim), rendered_candidates, {"steps": steps, "diagnostics": diagnostics}


def _claim_atom_lookup(claim_atoms: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    atom_by_id: dict[str, dict[str, Any]] = {}
    atom_order: dict[str, int] = {}
    for idx, atom in enumerate(claim_atoms):
        if not isinstance(atom, dict):
            continue
        atom_id = _compact_whitespace(atom.get("atom_id") or atom.get("node_id") or "")
        if not atom_id:
            continue
        atom_by_id[atom_id] = atom
        atom_order[atom_id] = idx
    return atom_by_id, atom_order


def _mrec_prompt_source_steps(trace: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    for source_name in ("mrec_steps", "compat_chain_steps", "chain_steps"):
        steps = [dict(step) for step in trace.get(source_name) or [] if isinstance(step, dict)]
        if steps:
            return source_name, steps
    return "none", []


def _select_mrec_cue(step: dict[str, Any] | None, *, source_name: str) -> dict[str, Any] | None:
    if not isinstance(step, dict):
        return None
    cue_text = _compact_whitespace(step.get("cue_text") or "")
    if not cue_text:
        return None
    if source_name == "mrec_steps":
        cue_type = "mrec_step"
    elif source_name in {"compat_chain_steps", "chain_steps"}:
        cue_type = "compat_chain_step"
    else:
        cue_type = "fallback"
    return {
        "cue_type": cue_type,
        "check": cue_text,
    }


def _align_qec_chain_steps(
    chain_steps: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any] | None]:
    if not chain_steps:
        return [None for _ in candidates]
    by_selector_idx: dict[int, dict[str, Any]] = {}
    by_candidate_idx: dict[int, dict[str, Any]] = {}
    by_evidence_id: dict[str, dict[str, Any]] = {}
    for step in chain_steps:
        if not isinstance(step, dict):
            continue
        selector_idx = _int_or_none(step.get("selector_candidate_idx"))
        if selector_idx is not None:
            by_selector_idx[selector_idx] = step
        candidate_idx = _int_or_none(step.get("candidate_idx"))
        if candidate_idx is not None:
            by_candidate_idx[candidate_idx] = step
        evidence_id = _compact_whitespace(step.get("evidence_id") or "")
        if evidence_id:
            by_evidence_id[evidence_id] = step

    aligned: list[dict[str, Any] | None] = []
    for fallback_idx, candidate in enumerate(candidates):
        selector_idx = _int_or_none(candidate.get("selector_candidate_idx"))
        if selector_idx is not None and selector_idx in by_selector_idx:
            aligned.append(by_selector_idx[selector_idx])
            continue
        candidate_idx = _int_or_none(candidate.get("candidate_idx"))
        if candidate_idx is not None and candidate_idx in by_candidate_idx:
            aligned.append(by_candidate_idx[candidate_idx])
            continue
        evidence_id = _compact_whitespace(candidate.get("evidence_id") or "")
        if evidence_id and evidence_id in by_evidence_id:
            aligned.append(by_evidence_id[evidence_id])
            continue
        aligned.append(chain_steps[fallback_idx] if fallback_idx < len(chain_steps) and isinstance(chain_steps[fallback_idx], dict) else None)
    return aligned


def _select_qec_cue_from_chain_step(chain_step: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(chain_step, dict):
        return None
    cue_text = _compact_whitespace(chain_step.get("cue_text") or "")
    if not cue_text:
        return None
    return {
        "cue_type": "chain_step",
        "check": cue_text,
        "question_id": _compact_whitespace(chain_step.get("qd_question_id") or ""),
        "question_focus": _compact_whitespace(chain_step.get("question_focus") or ""),
        "question_route_rank": _int_or_none(chain_step.get("qd_question_rank")),
        "question_route_hybrid_score": _float_or_none(chain_step.get("qd_question_hybrid_score")),
    }


def _qec_step_evidence_text(chain_step: dict[str, Any] | None, candidate: dict[str, Any]) -> str:
    if str(candidate.get("evidence_text_mode") or "") == "anchor_only":
        text = str(candidate.get("text") or "").strip()
        if text:
            return text
    if isinstance(chain_step, dict):
        text = str(chain_step.get("evidence_text") or "").strip()
        if text:
            return text
    return str(candidate.get("text", "")).strip()


def _select_qec_cue(
    candidate: dict[str, Any],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    atom_order: dict[str, int],
) -> dict[str, Any]:
    route = _best_qd_route(candidate.get("qd_question_routes") or candidate.get("question_routes") or [])
    if route is not None:
        return {
            "cue_type": "qd_question",
            "check": _compact_whitespace(route.get("question") or ""),
            "question_id": _compact_whitespace(route.get("question_id") or ""),
            "question_focus": _compact_whitespace(route.get("focus") or ""),
            "question_route_rank": _int_or_none(route.get("rank")),
            "question_route_hybrid_score": _float_or_none(route.get("hybrid_score")),
        }

    atom = _best_covered_atom(candidate.get("covered_atom_ids"), atom_by_id=atom_by_id, atom_order=atom_order)
    if atom is not None:
        return {
            "cue_type": "claim_atom",
            "check": _compact_whitespace(atom.get("proposition") or atom.get("text") or ""),
        }

    return {
        "cue_type": "fallback",
        "check": "Verify the main factual claim.",
    }


def _select_mrec_fallback_cue(
    candidate: dict[str, Any],
    *,
    atom_by_id: dict[str, dict[str, Any]],
    atom_order: dict[str, int],
) -> dict[str, Any]:
    atom = _best_covered_atom(candidate.get("covered_atom_ids"), atom_by_id=atom_by_id, atom_order=atom_order)
    if atom is not None:
        check = _compact_whitespace(atom.get("proposition") or atom.get("text") or "")
        if check:
            return {
                "cue_type": "claim_atom",
                "check": check,
            }
    return {
        "cue_type": "fallback",
        "check": "Verify the main factual claim.",
    }


def _best_qd_route(routes: Any) -> dict[str, Any] | None:
    if not isinstance(routes, list):
        return None
    usable = [
        route
        for route in routes
        if isinstance(route, dict) and _compact_whitespace(route.get("question") or "")
    ]
    if not usable:
        return None
    return min(
        usable,
        key=lambda route: (
            _rank_sort_value(route.get("rank")),
            -_float_or_default(route.get("hybrid_score"), 0.0),
            _focus_priority(route.get("focus")),
            _compact_whitespace(route.get("question_id") or ""),
        ),
    )


def _best_covered_atom(
    covered_atom_ids: Any,
    *,
    atom_by_id: dict[str, dict[str, Any]],
    atom_order: dict[str, int],
) -> dict[str, Any] | None:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for atom_id in _covered_atom_ids(covered_atom_ids):
        atom = atom_by_id.get(atom_id)
        if atom is None:
            continue
        if not _compact_whitespace(atom.get("text") or ""):
            continue
        candidates.append(
            (
                _float_or_default(atom.get("importance"), 0.0),
                atom_order.get(atom_id, 10**9),
                atom,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _apply_trace_lite_prompt_fields(
    *,
    claim: str,
    candidates: list[dict[str, Any]],
    claim_atoms: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    atom_lines: list[str] = []
    for atom in claim_atoms:
        if not isinstance(atom, dict):
            continue
        atom_id = _compact_whitespace(atom.get("atom_id") or atom.get("node_id") or "")
        atom_text = _compact_whitespace(atom.get("text") or "")
        if atom_id and atom_text:
            atom_lines.append(f"{atom_id}: {atom_text}")

    rendered_claim = str(claim)
    if atom_lines:
        rendered_claim = f"{rendered_claim.rstrip()}\n\nClaim atoms:\n" + "\n".join(atom_lines)

    rendered_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        copied = dict(candidate)
        covers = _render_covered_atom_ids(copied.get("covered_atom_ids"))
        relation = _compact_whitespace(copied.get("map_relation") or "") or "unknown"
        directness = _compact_whitespace(copied.get("map_directness") or "") or "unknown"
        text = str(copied.get("text", "")).strip()
        copied["text"] = f"[covers={covers}; relation={relation}; directness={directness}]\n{text}"
        rendered_candidates.append(copied)
    return rendered_claim, rendered_candidates


def _prompt_cfg_for_trace_style(
    prompt_cfg: dict[str, Any],
    *,
    trace_prompt_style: str,
    label_schema: str,
) -> dict[str, Any]:
    styled = dict(prompt_cfg)
    if trace_prompt_style != "rawfc_boundaries":
        return styled
    if str(label_schema).strip().lower() != "rawfc3":
        raise ValueError("trace_prompt_style=rawfc_boundaries is only supported with label_schema=rawfc3.")
    styled["system_prompt"] = RAWFC_BOUNDARIES_SYSTEM_PROMPT
    return styled


def _render_covered_atom_ids(value: Any) -> str:
    rendered = _covered_atom_ids(value)
    if not rendered:
        return "none"
    return ",".join(rendered)


def _covered_atom_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = list(value) if isinstance(value, tuple) else []
    rendered = [_compact_whitespace(item) for item in items]
    return [item for item in rendered if item]


def _compact_whitespace(value: Any) -> str:
    return " ".join(str(value).split())


def _rank_sort_value(value: Any) -> int:
    parsed = _int_or_none(value)
    if parsed is None:
        return 10**9
    return parsed


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    parsed = _int_or_none(value)
    return int(default) if parsed is None else int(parsed)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_default(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = _compact_whitespace(value).lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _focus_priority(value: Any) -> int:
    focus = _compact_whitespace(value or "").lower()
    if focus in {"quantity", "attribution", "entity", "comparison", "policy", "time", "causal"}:
        return 0
    if focus == "overall":
        return 2
    return 1


def _resolve_split_source(split: str, trace_path: str | None, oracle_path: str | None) -> tuple[str, str]:
    if trace_path and oracle_path:
        raise ValueError(f"Use only one of --{split}-trace or --{split}-oracle-results.")
    if trace_path:
        return "trace", trace_path
    if oracle_path:
        return "oracle_results", oracle_path
    raise ValueError(f"--{split}-trace or --{split}-oracle-results is required.")


def _resolve_optional_split_source(
    trace_path: str | None,
    oracle_path: str | None,
) -> tuple[str, str] | None:
    if trace_path and oracle_path:
        raise ValueError("Use only one of --test-trace or --test-oracle-results.")
    if trace_path:
        return "trace", trace_path
    if oracle_path:
        return "oracle_results", oracle_path
    return None


def _normalize_source_row(row: dict[str, Any], *, source_type: str) -> dict[str, Any]:
    if source_type == "trace":
        return row
    if source_type != "oracle_results":
        raise ValueError(f"unknown source_type={source_type!r}")
    metadata = dict(row.get("candidate_pool_metadata") or {})
    return {
        "event_id": row.get("event_id", ""),
        "claim": row.get("claim", ""),
        "gold_label": row.get("gold_label", ""),
        "candidate_pool": row.get("candidate_pool") or [],
        "candidate_scores": row.get("candidate_scores") or [],
        "oracle_ordered_indices": [int(x) for x in (row.get("selected_indices") or [])],
        "selector_ordered_indices": [int(x) for x in (row.get("selected_indices") or [])],
        "selector_name": "oracle_results",
        "fingerprint": str(metadata.get("chunk_mmr_fingerprint") or ""),
        "candidate_pool_metadata": metadata,
    }


def _select_indices(
    trace: dict[str, Any],
    *,
    mode: str,
    top_k: int,
    random_seed: int,
    trace_order_field: str = "display_ordered_indices",
) -> list[int]:
    pool = trace.get("candidate_pool") or []
    if not isinstance(pool, list) or not pool:
        raise ValueError("trace has no candidate_pool")
    n = len(pool)
    limit = int(top_k) if int(top_k) > 0 else n
    if mode == "trace":
        selected = _ordered_trace_indices(trace, field=trace_order_field)
    elif mode == "hybrid_score_topk":
        selected = sorted(range(n), key=lambda idx: _hybrid_score(trace, idx), reverse=True)
    elif mode == "candidate_pool_topk":
        selected = list(range(n))
    elif mode in {"same_set_hybrid_order", "same_set_candidate_pool_order", "same_set_random_order"}:
        selected = _ordered_trace_indices(trace, field=trace_order_field)
        selected = _dedupe_in_range(selected, n)[:limit]
        selected_set = set(selected)
        if mode == "same_set_hybrid_order":
            selected = [
                idx
                for idx in sorted(range(n), key=lambda item: _hybrid_score(trace, item), reverse=True)
                if idx in selected_set
            ]
        elif mode == "same_set_candidate_pool_order":
            selected = [idx for idx in range(n) if idx in selected_set]
        else:
            rng = np.random.default_rng(int(random_seed))
            selected = list(selected)
            rng.shuffle(selected)
    else:
        raise ValueError(f"unknown selection mode: {mode}")

    selected = _dedupe_in_range(selected, n)[:limit]
    if not selected:
        raise ValueError("selection produced no indices")
    return selected


def _selected_candidates(
    trace: dict[str, Any],
    selected_indices: list[int],
    *,
    selection_mode: str,
) -> list[dict[str, Any]]:
    pool = trace.get("candidate_pool") or []
    scores_by_idx = _candidate_scores_by_idx(trace)
    selected: list[dict[str, Any]] = []
    for rank, idx in enumerate(selected_indices):
        candidate = dict(pool[idx])
        score = dict(scores_by_idx.get(idx, {}))
        candidate.update(
            {
                "selector_trace_rank": int(rank),
                "selector_candidate_idx": int(idx),
                "selector_selection_mode": selection_mode,
                "candidate_idx": int(candidate.get("candidate_idx", idx)),
                "candidate_uid": str(candidate.get("candidate_uid") or score.get("candidate_uid") or ""),
                "hybrid_rank": int(score.get("hybrid_rank", idx)),
                "dense_score": _float_or_default(score.get("dense_score", candidate.get("dense_score")), 0.0),
                "lexical_score": _float_or_default(score.get("lexical_score", candidate.get("lexical_score")), 0.0),
                "bm25_score": _float_or_default(score.get("bm25_score", candidate.get("bm25_score")), 0.0),
                "hybrid_score": _float_or_default(score.get("hybrid_score", candidate.get("hybrid_score")), 0.0),
            }
        )
        if "selector_score" in score:
            candidate["selector_score"] = _float_or_default(score.get("selector_score"), 0.0)
        if "sequential_selected_step" in score:
            candidate["sequential_selected_step"] = int(score["sequential_selected_step"])
        selected.append(candidate)
    return selected


def _ordered_trace_indices(
    trace: dict[str, Any], *, field: str = "display_ordered_indices"
) -> list[int]:
    if field not in TRACE_ORDER_FIELDS:
        raise ValueError(f"unsupported trace order field: {field}")
    raw = trace.get(field)
    if raw is None and field == "display_ordered_indices":
        raw = trace.get("selector_ordered_indices") or []
    elif raw is None:
        raise ValueError(f"trace has no {field}")
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _dedupe_in_range(indices: list[int], n: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= n or idx in seen:
            continue
        seen.add(idx)
        out.append(int(idx))
    return out


def _hybrid_score(trace: dict[str, Any], idx: int) -> float:
    scores_by_idx = _candidate_scores_by_idx(trace)
    score = scores_by_idx.get(idx, {})
    pool = trace.get("candidate_pool") or []
    candidate = pool[idx] if idx < len(pool) and isinstance(pool[idx], dict) else {}
    return _float_or_default(score.get("hybrid_score", candidate.get("hybrid_score")), 0.0)


def _candidate_scores_by_idx(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for fallback_idx, item in enumerate(trace.get("candidate_scores") or []):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("candidate_idx", fallback_idx))
        except (TypeError, ValueError):
            idx = fallback_idx
        out[idx] = item
    return out


def _trace_fingerprint(trace: dict[str, Any]) -> str:
    if trace.get("fingerprint"):
        return str(trace.get("fingerprint"))
    metadata = trace.get("candidate_pool_metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("chunk_mmr_fingerprint") or "")
    return ""


def _load_experiment_config(config_path: str) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    project_root = Path(__file__).resolve().parents[3]
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    experiment_dir = project_root / "configs" / "experiment"
    try:
        rel = path.resolve().relative_to(experiment_dir.resolve())
    except ValueError:
        cfg = OmegaConf.load(path)
        payload = dict(OmegaConf.to_container(cfg, resolve=True) or {})
        return _resolve_config_extends(path, payload)
    if len(rel.parts) != 1:
        cfg = OmegaConf.load(path)
        payload = dict(OmegaConf.to_container(cfg, resolve=True) or {})
        return _resolve_config_extends(path, payload)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(project_root / "configs")):
        cfg = compose(config_name="pipeline/default", overrides=[f"experiment={rel.stem}"])
    return dict(OmegaConf.to_container(cfg, resolve=True) or {})


def _resolve_config_extends(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    parent = payload.pop("extends", None)
    if not parent:
        return payload
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    base = _load_experiment_config(str(parent_path))
    return _deep_merge_dicts(base, payload)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _build_train_config(
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    split_paths: dict[str, str],
    label_schema: str | None,
    model_base_path: str | None,
    train_model_name_or_path: str | None,
) -> dict[str, Any]:
    train_model = train_model_name_or_path or str((cfg.get("train", {}) or {}).get("model_name_or_path", ""))
    if model_base_path and train_model:
        train_model = _resolve_model_path(train_model, model_base_path)
    sft_train = dict(cfg.get("sft_train", {}) or {})
    resolved_label_schema = str(
        label_schema
        or sft_train.get("label_schema")
        or cfg.get("label_schema")
        or ((cfg.get("build", {}) or {}).get("prompt", {}) or {}).get("label_schema")
        or ((cfg.get("build", {}) or {}).get("data", {}) or {}).get("label_schema")
        or "liar6"
    )
    sft_train["label_schema"] = resolved_label_schema
    sft_train["resolved_output_dir"] = True
    sft_train.setdefault("save_latest_state", True)
    sft_train.setdefault("resume_latest_state", True)
    train_cfg = {
        "label_schema": resolved_label_schema,
        "output_dir": str(run_dir / "train"),
        "eval_output_dir": str(run_dir / "eval"),
        "prompt_stats_output_dir": str(run_dir / "prompt_stats"),
        "data": {
            "train_candidates": split_paths["train"],
            "val_candidates": split_paths["val"],
            "test_candidates": split_paths["test"],
        },
        "model_name_or_path": train_model,
        "baseline": dict(cfg.get("baseline", {}) or {}),
        "sft_train": sft_train,
    }
    train_cfg["baseline"]["label_schema"] = resolved_label_schema
    for key in ("tracking", "wandb", "swanlab"):
        if key in cfg:
            train_cfg[key] = cfg[key]
    if isinstance(train_cfg.get("swanlab"), dict):
        swanlab = dict(train_cfg["swanlab"])
        swanlab["experiment_name"] = str(swanlab.get("experiment_name") or "selector_trace_verifier")
        train_cfg["swanlab"] = swanlab
    return train_cfg


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_model_path(raw: str, base_path: str) -> str:
    if raw.startswith("/data/models/"):
        return raw.replace("/data/models/", base_path.rstrip("/") + "/", 1)
    return raw


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _float_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


if __name__ == "__main__":
    main()
