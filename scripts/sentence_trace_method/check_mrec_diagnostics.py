#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from fact_checking.selectors.mrec_schema import validate_mrec_trace


DEFAULT_FORBIDDEN_PROMPT_PATTERNS = (
    "state_before",
    "state_after",
    "relation=",
    "directness=",
    "covers=",
    "operation=",
    r"\bOPEN\b",
    r"\bCONTRAST\b",
    r"\bCORROBORATE\b",
    r"\bFALLBACK\b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MREC trace and prompt diagnostics before full training.")
    parser.add_argument("--dataset", required=True, choices=["liar_raw", "rawfc"])
    parser.add_argument("--output-root", default="outputs/sentence_trace_method")
    parser.add_argument("--case-root", required=True, help="Built verifier data root, e.g. outputs/.../liar_raw__...__mrec_min")
    parser.add_argument("--source-selector-name", default="mrec_greedy_transition_v0_1")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--expected-trace-prompt-style", default="mrec_min")
    parser.add_argument("--expected-evidence-text-mode", default="")
    parser.add_argument("--max-empty-mrec-step-rate", type=float, default=0.0)
    parser.add_argument("--max-empty-prompt-step-rate", type=float, default=0.0)
    parser.add_argument("--max-prompt-leak-rate", type=float, default=0.0)
    parser.add_argument("--max-truncation-rate", type=float, default=0.02)
    parser.add_argument("--warn-min-mean-resolved-atom-rate", type=float, default=0.80)
    parser.add_argument("--warn-max-fallback-step-rate", type=float, default=0.25)
    parser.add_argument("--warn-max-unresolved-atom-rate", type=float, default=0.25)
    parser.add_argument(
        "--forbidden-prompt-pattern",
        action="append",
        default=[],
        help="Additional regex/string pattern that must not appear in prompt-visible text.",
    )
    parser.add_argument("--report-path", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    case_root = Path(args.case_root)
    splits = _csv(args.splits)
    forbidden_patterns = list(DEFAULT_FORBIDDEN_PROMPT_PATTERNS) + [str(item) for item in args.forbidden_prompt_pattern]

    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "case_root": str(case_root),
        "source_selector_name": str(args.source_selector_name),
        "thresholds": _thresholds(args, forbidden_patterns),
        "splits": {},
        "warnings": warnings,
        "failures": failures,
    }

    for split in splits:
        source_path = _source_trace_path(
            output_root=output_root,
            dataset=str(args.dataset),
            selector_name=str(args.source_selector_name),
            split=split,
        )
        build_path = case_root / "build" / f"build_{split}.jsonl"
        source_rows = _read_jsonl(source_path, failures, f"{split} source")
        build_rows = _read_jsonl(build_path, failures, f"{split} build")
        if not source_rows or not build_rows:
            continue
        split_report = _check_split(
            split=split,
            source_rows=source_rows,
            build_rows=build_rows,
            expected_trace_prompt_style=str(args.expected_trace_prompt_style),
            expected_evidence_text_mode=str(args.expected_evidence_text_mode),
            forbidden_patterns=forbidden_patterns,
            failures=failures,
            warnings=warnings,
            max_empty_mrec_step_rate=float(args.max_empty_mrec_step_rate),
            max_empty_prompt_step_rate=float(args.max_empty_prompt_step_rate),
            max_prompt_leak_rate=float(args.max_prompt_leak_rate),
            max_truncation_rate=float(args.max_truncation_rate),
            warn_min_mean_resolved_atom_rate=float(args.warn_min_mean_resolved_atom_rate),
            warn_max_fallback_step_rate=float(args.warn_max_fallback_step_rate),
            warn_max_unresolved_atom_rate=float(args.warn_max_unresolved_atom_rate),
        )
        report["splits"][split] = split_report

    report_path = Path(args.report_path) if args.report_path else case_root / "mrec_diagnostics_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(report_path, report)
    return 1 if failures else 0


def _thresholds(args: argparse.Namespace, forbidden_patterns: list[str]) -> dict[str, Any]:
    return {
        "expected_trace_prompt_style": str(args.expected_trace_prompt_style),
        "expected_evidence_text_mode": str(args.expected_evidence_text_mode),
        "max_empty_mrec_step_rate": float(args.max_empty_mrec_step_rate),
        "max_empty_prompt_step_rate": float(args.max_empty_prompt_step_rate),
        "max_prompt_leak_rate": float(args.max_prompt_leak_rate),
        "max_truncation_rate": float(args.max_truncation_rate),
        "warn_min_mean_resolved_atom_rate": float(args.warn_min_mean_resolved_atom_rate),
        "warn_max_fallback_step_rate": float(args.warn_max_fallback_step_rate),
        "warn_max_unresolved_atom_rate": float(args.warn_max_unresolved_atom_rate),
        "forbidden_prompt_patterns": forbidden_patterns,
    }


def _check_split(
    *,
    split: str,
    source_rows: list[dict[str, Any]],
    build_rows: list[dict[str, Any]],
    expected_trace_prompt_style: str,
    expected_evidence_text_mode: str,
    forbidden_patterns: list[str],
    failures: list[str],
    warnings: list[str],
    max_empty_mrec_step_rate: float,
    max_empty_prompt_step_rate: float,
    max_prompt_leak_rate: float,
    max_truncation_rate: float,
    warn_min_mean_resolved_atom_rate: float,
    warn_max_fallback_step_rate: float,
    warn_max_unresolved_atom_rate: float,
) -> dict[str, Any]:
    if len(source_rows) != len(build_rows):
        failures.append(f"{split}: build rows={len(build_rows)} != source rows={len(source_rows)}")

    source_by_event = _rows_by_event(source_rows, failures, f"{split} source")
    build_by_event = _rows_by_event(build_rows, failures, f"{split} build")
    if set(source_by_event) != set(build_by_event):
        missing_build = sorted(set(source_by_event) - set(build_by_event))[:5]
        missing_source = sorted(set(build_by_event) - set(source_by_event))[:5]
        failures.append(f"{split}: event_id mismatch missing_build={missing_build} missing_source={missing_source}")

    source_diag = _source_diagnostics(split=split, rows=source_rows, failures=failures, warnings=warnings)
    prompt_diag = _prompt_diagnostics(
        split=split,
        rows=build_rows,
        expected_trace_prompt_style=expected_trace_prompt_style,
        expected_evidence_text_mode=expected_evidence_text_mode,
        forbidden_patterns=forbidden_patterns,
        failures=failures,
    )

    empty_mrec_step_rate = float(source_diag["empty_mrec_step_rate"])
    empty_prompt_step_rate = float(prompt_diag["empty_prompt_step_rate"])
    prompt_leak_rate = float(prompt_diag["prompt_leak_rate"])
    truncation_rate = float(prompt_diag["truncation_rate"])
    fallback_step_rate = float(source_diag["fallback_step_rate"])
    unresolved_atom_rate = float(source_diag["unresolved_atom_rate_mean"])
    resolved_atom_rate = float(source_diag["resolved_atom_rate_mean"])

    if empty_mrec_step_rate > max_empty_mrec_step_rate:
        failures.append(f"{split}: empty_mrec_step_rate={empty_mrec_step_rate:.6f} > {max_empty_mrec_step_rate:.6f}")
    if empty_prompt_step_rate > max_empty_prompt_step_rate:
        failures.append(f"{split}: empty_prompt_step_rate={empty_prompt_step_rate:.6f} > {max_empty_prompt_step_rate:.6f}")
    if prompt_leak_rate > max_prompt_leak_rate:
        failures.append(f"{split}: prompt_leak_rate={prompt_leak_rate:.6f} > {max_prompt_leak_rate:.6f}")
    if truncation_rate > max_truncation_rate:
        failures.append(f"{split}: truncation_rate={truncation_rate:.6f} > {max_truncation_rate:.6f}")

    if resolved_atom_rate < warn_min_mean_resolved_atom_rate:
        warnings.append(
            f"{split}: resolved_atom_rate.mean={resolved_atom_rate:.6f} < {warn_min_mean_resolved_atom_rate:.6f}"
        )
    if fallback_step_rate > warn_max_fallback_step_rate:
        warnings.append(f"{split}: fallback_step_rate={fallback_step_rate:.6f} > {warn_max_fallback_step_rate:.6f}")
    if unresolved_atom_rate > warn_max_unresolved_atom_rate:
        warnings.append(
            f"{split}: unresolved_atom_rate.mean={unresolved_atom_rate:.6f} > {warn_max_unresolved_atom_rate:.6f}"
        )

    return {
        "rows": len(build_rows),
        "source": source_diag,
        "prompt": prompt_diag,
    }


def _source_diagnostics(
    *,
    split: str,
    rows: list[dict[str, Any]],
    failures: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    step_counts: list[float] = []
    resolved_rates: list[float] = []
    unresolved_rates: list[float] = []
    conflicted_rates: list[float] = []
    token_costs: list[float] = []
    invalid_trace_count = 0
    empty_step_count = 0
    invalid_transition_count = 0
    operation_counts: Counter[str] = Counter()
    state_after_counts: Counter[str] = Counter()
    cue_source_counts: Counter[str] = Counter()
    stop_reason_counts: Counter[str] = Counter()
    rejection_sums: Counter[str] = Counter()
    duplicate_evidence_rows = 0
    examples: dict[str, list[dict[str, Any]]] = {
        "invalid_transitions": [],
        "empty_mrec_steps": [],
        "invalid_traces": [],
    }

    for row_idx, row in enumerate(rows, start=1):
        event_id = str(row.get("event_id") or "")
        errors = validate_mrec_trace(row)
        if errors:
            invalid_trace_count += 1
            _append_example(examples["invalid_traces"], {"event_id": event_id, "errors": errors})
        steps = [step for step in row.get("mrec_steps") or [] if isinstance(step, dict)]
        if not steps:
            empty_step_count += 1
            _append_example(examples["empty_mrec_steps"], {"event_id": event_id, "row": row_idx})
        step_counts.append(float(len(steps)))
        evidence_ids = [str(step.get("evidence_id") or "") for step in steps if str(step.get("evidence_id") or "")]
        if len(evidence_ids) != len(set(evidence_ids)):
            duplicate_evidence_rows += 1

        for step in steps:
            operation = str(step.get("operation") or "").upper()
            state_after = str(step.get("state_after") or "").upper()
            cue_source = str(step.get("cue_source") or "")
            operation_counts[operation] += 1
            state_after_counts[state_after] += 1
            cue_source_counts[cue_source] += 1
            token_cost = _float_or_none(step.get("token_cost"))
            if token_cost is not None:
                token_costs.append(token_cost)
            if not _valid_transition(step):
                invalid_transition_count += 1
                _append_example(
                    examples["invalid_transitions"],
                    {
                        "event_id": event_id,
                        "step": step.get("step"),
                        "operation": operation,
                        "state_before": step.get("state_before"),
                        "state_after": step.get("state_after"),
                    },
                )

        diagnostics = row.get("mrec_diagnostics") or {}
        if isinstance(diagnostics, Mapping):
            stop_reason = str(diagnostics.get("stop_reason") or "")
            if stop_reason:
                stop_reason_counts[stop_reason] += 1
            for source_key, target_key in (
                ("duplicate_rejected_count", "duplicate_rejected"),
                ("background_rejected_count", "background_rejected"),
                ("no_transition_rejected_count", "no_transition_rejected"),
            ):
                rejection_sums[target_key] += _int_or_default(diagnostics.get(source_key), 0)
            resolved_rates.append(_float_or_default(diagnostics.get("resolved_atom_rate"), 0.0))

        final_states = {str(key): str(value).upper() for key, value in (row.get("atom_states_final") or {}).items()}
        if final_states:
            total_atoms = len(final_states)
            unresolved_rates.append(sum(1 for state in final_states.values() if state == "U") / total_atoms)
            conflicted_rates.append(sum(1 for state in final_states.values() if state == "C") / total_atoms)

    if invalid_trace_count:
        failures.append(f"{split}: invalid_mrec_trace_count={invalid_trace_count}")
    if invalid_transition_count:
        failures.append(f"{split}: invalid_transition_count={invalid_transition_count}")
    if duplicate_evidence_rows:
        warnings.append(f"{split}: duplicate evidence appears in {duplicate_evidence_rows} MREC rows")

    total_steps = sum(operation_counts.values())
    fallback_steps = int(operation_counts.get("FALLBACK", 0))
    return {
        "rows": len(rows),
        "invalid_mrec_trace_count": invalid_trace_count,
        "empty_mrec_step_count": empty_step_count,
        "empty_mrec_step_rate": _rate(empty_step_count, len(rows)),
        "step_count": _numeric_summary(step_counts),
        "operation_counts": _clean_counts(operation_counts),
        "state_after_counts": _clean_counts(state_after_counts),
        "cue_source_counts": _clean_counts(cue_source_counts),
        "stop_reason_counts": _clean_counts(stop_reason_counts),
        "fallback_step_rate": _rate(fallback_steps, total_steps),
        "duplicate_evidence_row_rate": _rate(duplicate_evidence_rows, len(rows)),
        "resolved_atom_rate_mean": _mean(resolved_rates),
        "unresolved_atom_rate_mean": _mean(unresolved_rates),
        "conflicted_atom_rate_mean": _mean(conflicted_rates),
        "token_cost": _numeric_summary(token_costs),
        "rejection_sums": dict(rejection_sums),
        "invalid_transition_count": invalid_transition_count,
        "examples": examples,
    }


def _prompt_diagnostics(
    *,
    split: str,
    rows: list[dict[str, Any]],
    expected_trace_prompt_style: str,
    expected_evidence_text_mode: str,
    forbidden_patterns: list[str],
    failures: list[str],
) -> dict[str, Any]:
    prompt_token_counts: list[float] = []
    evidence_counts: list[float] = []
    check_counts: list[float] = []
    empty_prompt_step_count = 0
    truncated_count = 0
    leak_rows = 0
    missing_trace_fields = 0
    prompt_source_counts: Counter[str] = Counter()
    cue_type_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    evidence_mode_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        "prompt_leaks": [],
        "empty_prompt_steps": [],
        "missing_mrec_fields": [],
        "bad_style_or_mode": [],
    }

    for row_idx, row in enumerate(rows, start=1):
        event_id = str(row.get("event_id") or "")
        style = str(row.get("trace_prompt_style") or "")
        mode = str(row.get("evidence_text_mode") or "")
        style_counts[style] += 1
        if mode:
            evidence_mode_counts[mode] += 1
        if expected_trace_prompt_style and style != expected_trace_prompt_style:
            _append_example(examples["bad_style_or_mode"], {"event_id": event_id, "trace_prompt_style": style})
        if expected_evidence_text_mode and mode != expected_evidence_text_mode:
            _append_example(examples["bad_style_or_mode"], {"event_id": event_id, "evidence_text_mode": mode})

        prompt_steps = [step for step in row.get("mrec_prompt_steps") or [] if isinstance(step, dict)]
        if not prompt_steps:
            empty_prompt_step_count += 1
            _append_example(examples["empty_prompt_steps"], {"event_id": event_id, "row": row_idx})
        for step in prompt_steps:
            prompt_source_counts[str(step.get("source") or "")] += 1
            cue_type_counts[str(step.get("cue_type") or "")] += 1
            operation = str(step.get("operation") or "").upper()
            if operation:
                operation_counts[operation] += 1

        diagnostics = row.get("mrec_prompt_diagnostics") or {}
        if isinstance(diagnostics, Mapping):
            check_counts.append(_float_or_default(diagnostics.get("mean_check_token_count"), 0.0))

        if any(key not in row for key in ("mrec_steps", "mrec_diagnostics", "atom_states_final")):
            missing_trace_fields += 1
            _append_example(
                examples["missing_mrec_fields"],
                {"event_id": event_id, "missing": [key for key in ("mrec_steps", "mrec_diagnostics", "atom_states_final") if key not in row]},
            )

        if _truthy(row.get("was_truncated")) or _truthy(row.get("evidence_text_truncated")):
            truncated_count += 1
        prompt_token_count = _float_or_none(row.get("prompt_token_count"))
        if prompt_token_count is not None:
            prompt_token_counts.append(prompt_token_count)
        evidence_count = _float_or_none(row.get("evidence_count"))
        if evidence_count is not None:
            evidence_counts.append(evidence_count)

        leaks = _prompt_leaks(row, forbidden_patterns=forbidden_patterns)
        if leaks:
            leak_rows += 1
            _append_example(examples["prompt_leaks"], {"event_id": event_id, "matches": leaks[:5]})

    bad_style_rows = len(examples["bad_style_or_mode"])
    if bad_style_rows:
        failures.append(f"{split}: rows with unexpected prompt style/mode={bad_style_rows}")
    if missing_trace_fields:
        failures.append(f"{split}: build rows missing MREC diagnostic fields={missing_trace_fields}")

    return {
        "rows": len(rows),
        "style_counts": _clean_counts(style_counts),
        "evidence_mode_counts": _clean_counts(evidence_mode_counts),
        "empty_prompt_step_count": empty_prompt_step_count,
        "empty_prompt_step_rate": _rate(empty_prompt_step_count, len(rows)),
        "missing_mrec_field_count": missing_trace_fields,
        "prompt_source_counts": _clean_counts(prompt_source_counts),
        "cue_type_counts": _clean_counts(cue_type_counts),
        "operation_counts": _clean_counts(operation_counts),
        "prompt_leak_rows": leak_rows,
        "prompt_leak_rate": _rate(leak_rows, len(rows)),
        "truncated_rows": truncated_count,
        "truncation_rate": _rate(truncated_count, len(rows)),
        "prompt_token_count": _numeric_summary(prompt_token_counts),
        "evidence_count": _numeric_summary(evidence_counts),
        "mean_check_token_count": _mean(check_counts),
        "examples": examples,
    }


def _valid_transition(step: Mapping[str, Any]) -> bool:
    operation = str(step.get("operation") or "").upper()
    before = str(step.get("state_before") or "").upper()
    after = str(step.get("state_after") or "").upper()
    if operation == "OPEN":
        return before == "U" and after in {"S", "R", "Q"}
    if operation == "CONTRAST":
        return before in {"S", "R", "Q", "C"} and after in {"Q", "C"}
    if operation == "CORROBORATE":
        return before == after and after in {"S", "R", "Q"}
    if operation == "BRIDGE":
        return before == after
    if operation == "FALLBACK":
        return before == after
    return False


def _prompt_leaks(row: Mapping[str, Any], *, forbidden_patterns: list[str]) -> list[str]:
    visible_parts = [str(row.get("prompt") or "")]
    for candidate in row.get("candidates") or []:
        if isinstance(candidate, Mapping):
            visible_parts.append(str(candidate.get("text") or ""))
    visible = "\n".join(visible_parts)
    matches: list[str] = []
    for pattern in forbidden_patterns:
        if re.search(pattern, visible):
            matches.append(pattern)
    return matches


def _source_trace_path(*, output_root: Path, dataset: str, selector_name: str, split: str) -> Path:
    return output_root / "_sources" / dataset / selector_name / split / f"selection_trace_{split}.jsonl"


def _read_jsonl(path: Path, failures: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        failures.append(f"{label}: missing {path}")
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(f"{label}: invalid JSON at {path}:{line_no}: {exc}")
                return []
            if not isinstance(row, dict):
                failures.append(f"{label}: non-object JSON row at {path}:{line_no}")
                return []
            rows.append(row)
    if not rows:
        failures.append(f"{label}: empty {path}")
    return rows


def _rows_by_event(rows: list[dict[str, Any]], failures: list[str], label: str) -> dict[str, dict[str, Any]]:
    by_event: dict[str, dict[str, Any]] = {}
    for row_idx, row in enumerate(rows, start=1):
        event_id = str(row.get("event_id") or "")
        if not event_id:
            failures.append(f"{label}: missing event_id at row={row_idx}")
            continue
        if event_id in by_event:
            failures.append(f"{label}: duplicate event_id={event_id}")
            continue
        by_event[event_id] = row
    return by_event


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _clean_counts(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in dict(counter).items() if str(key)}


def _numeric_summary(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0.0}
    return {
        "count": float(len(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(mean(ordered)),
    }


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(mean(values)) if values else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _append_example(items: list[dict[str, Any]], item: dict[str, Any], *, limit: int = 20) -> None:
    if len(items) < limit:
        items.append(item)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    return float(default) if parsed is None else parsed


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _print_summary(report_path: Path, report: Mapping[str, Any]) -> None:
    print(f"[mrec-diagnostics] report={report_path}")
    for split, split_report in (report.get("splits") or {}).items():
        source = split_report.get("source") or {}
        prompt = split_report.get("prompt") or {}
        print(
            "[mrec-diagnostics] "
            f"{split} rows={split_report.get('rows', 0)} "
            f"steps.mean={(source.get('step_count') or {}).get('mean', 0.0):.3f} "
            f"resolved={source.get('resolved_atom_rate_mean', 0.0):.3f} "
            f"fallback_step_rate={source.get('fallback_step_rate', 0.0):.3f} "
            f"trunc={prompt.get('truncation_rate', 0.0):.3f} "
            f"leak={prompt.get('prompt_leak_rate', 0.0):.3f}"
        )
    warnings = list(report.get("warnings") or [])
    failures = list(report.get("failures") or [])
    if warnings:
        print("[mrec-diagnostics] WARNINGS")
        for warning in warnings:
            print(f"  - {warning}")
    if failures:
        print("[mrec-diagnostics] FAILED")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("[mrec-diagnostics] PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
