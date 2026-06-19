from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


CASE_MAP = {
    "F1": {
        "selector": "aa_qec_full_atom_facts_abc_primary_fallback_no_secondary_qd_prefer_top20_min5_10",
        "suffix": "__aa_qec_f1_atom_facts_abc_primary_fallback_no_secondary",
    },
    "F2": {
        "selector": "aa_qec_full_atom_facts_abc_primary_secondary_fallback_qd_prefer_top20_min5_10",
        "suffix": "__aa_qec_f2_atom_facts_abc_primary_secondary_fallback",
    },
    "F3": {
        "selector": "aa_qec_full_atom_facts_abc_primary_secondary_dynamic_qd_prefer_top20",
        "suffix": "__aa_qec_f3_atom_facts_abc_primary_secondary_dynamic",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check AA-QEC Stage3 build quality before full training.")
    parser.add_argument("--output-root", default="outputs/sentence_trace_method")
    parser.add_argument("--graph-root", default="outputs/selectors/atom_anchored_qec/liar_raw")
    parser.add_argument("--source-selector-name", default="v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10")
    parser.add_argument("--baseline-run", default="")
    parser.add_argument("--model", default="ministral3_8b")
    parser.add_argument("--lora-suffix", default="_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw")
    parser.add_argument("--cases", default="F1,F2,F3")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--prompt-splits", default="train,val,test")
    parser.add_argument(
        "--atom-coverage-baseline-selector",
        default="aa_qec_constrained_atom_facts_abc_primary_secondary_fallback_qd_prefer_selected_min5_10",
    )
    parser.add_argument("--atom-coverage-tolerance", type=float, default=0.0005)
    parser.add_argument("--min-atom-coverage", type=float, default=0.846)
    parser.add_argument("--min-qd-cue-rate", type=float, default=0.95)
    parser.add_argument("--qd-hard-splits", default="train,val")
    parser.add_argument("--max-duplicate-rate", type=float, default=0.0)
    parser.add_argument("--max-truncation-rate", type=float, default=0.02)
    parser.add_argument("--max-truncation-delta-vs-baseline", type=float, default=0.01)
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    graph_root = Path(args.graph_root)
    source_selector = str(args.source_selector_name)
    baseline_run = args.baseline_run or (
        "liar_raw__ministral3_8b__v0_7_atom_facts_abc_bm_adaptive5_10__qec_min"
        "_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
    )
    cases = _csv(args.cases)
    splits = _csv(args.splits)
    prompt_splits = _csv(args.prompt_splits)
    qd_hard_splits = set(_csv(args.qd_hard_splits))

    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"cases": {}, "thresholds": _thresholds(args), "warnings": warnings}
    baseline_prompt_stats = _load_prompt_stats(output_root / baseline_run)
    baseline_atom_floors = _load_atom_coverage_floors(
        graph_root=graph_root,
        selector_name=str(args.atom_coverage_baseline_selector),
        splits=splits,
        warnings=warnings,
    )

    for case_id in cases:
        if case_id not in CASE_MAP:
            failures.append(f"unsupported case: {case_id}")
            continue
        case = CASE_MAP[case_id]
        selector_name = str(case["selector"])
        case_name = f"liar_raw__{args.model}{case['suffix']}"
        case_root = output_root / case_name
        lora_root = output_root / f"{case_name}{args.lora_suffix}"
        case_report: dict[str, Any] = {"selector_name": selector_name, "case_name": case_name, "splits": {}}
        report["cases"][case_id] = case_report

        for split in splits:
            source_trace = _source_trace_path(output_root, source_selector, split)
            graph_trace = graph_root / f"{selector_name}_{split}" / f"selection_trace_{split}.jsonl"
            staged_trace = _source_trace_path(output_root, selector_name, split)
            build_trace = case_root / "build" / f"build_{split}.jsonl"

            source_rows = _read_jsonl(source_trace, failures, f"{case_id}/{split} source")
            graph_rows = _read_jsonl(graph_trace, failures, f"{case_id}/{split} graph")
            staged_rows = _read_jsonl(staged_trace, failures, f"{case_id}/{split} staged")
            build_rows = _read_jsonl(build_trace, failures, f"{case_id}/{split} build")
            if not source_rows or not graph_rows or not staged_rows or not build_rows:
                continue
            atom_floor_info = baseline_atom_floors.get(split)
            if atom_floor_info:
                atom_coverage_floor = float(atom_floor_info["value"])
                atom_coverage_floor_source = str(atom_floor_info["source"])
            else:
                atom_coverage_floor = float(args.min_atom_coverage)
                atom_coverage_floor_source = "fixed_threshold"

            split_report = _check_split(
                case_id=case_id,
                split=split,
                source_rows=source_rows,
                graph_rows=graph_rows,
                staged_rows=staged_rows,
                build_rows=build_rows,
                failures=failures,
                warnings=warnings,
                atom_coverage_floor=atom_coverage_floor,
                atom_coverage_floor_source=atom_coverage_floor_source,
                atom_coverage_tolerance=float(args.atom_coverage_tolerance),
                min_qd_cue_rate=float(args.min_qd_cue_rate),
                qd_hard_splits=qd_hard_splits,
                max_duplicate_rate=float(args.max_duplicate_rate),
            )
            case_report["splits"][split] = split_report

        prompt_stats = _load_prompt_stats(case_root) or _load_prompt_stats(lora_root)
        case_report["prompt_stats_path"] = str(prompt_stats.get("_path", "")) if prompt_stats else ""
        case_report["prompt_splits"] = _check_prompt_quality(
            case_id=case_id,
            prompt_splits=prompt_splits,
            prompt_stats=prompt_stats,
            case_roots=[case_root, lora_root],
            baseline_prompt_stats=baseline_prompt_stats,
            baseline_roots=[output_root / baseline_run],
            failures=failures,
            warnings=warnings,
            max_truncation_rate=float(args.max_truncation_rate),
            max_delta=float(args.max_truncation_delta_vs_baseline),
        )

    report_path = Path(args.report_path) if args.report_path else output_root / "aa_qec_stage3_build_gate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[aa-qec-stage3-build-gate] report={report_path}")
    for case_id, case_report in report["cases"].items():
        for split, split_report in case_report.get("splits", {}).items():
            print(
                "[aa-qec-stage3-build-gate] "
                f"{case_id}/{split} rows={split_report['rows']} "
                f"outside_rows={split_report['outside_source_selected_rows']} "
                f"atom_coverage={split_report['atom_coverage_rate_mean']:.6f} "
                f"qd_cue={split_report['qd_cue_rate_mean']:.6f} "
                f"dup={split_report['duplicate_evidence_rate_mean']:.6f}"
            )

    if warnings:
        print("[aa-qec-stage3-build-gate] WARNINGS")
        for warning in warnings:
            print(f"  - {warning}")
    if failures:
        print("[aa-qec-stage3-build-gate] FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("[aa-qec-stage3-build-gate] PASSED")
    return 0


def _thresholds(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "atom_coverage_baseline_selector": str(args.atom_coverage_baseline_selector),
        "atom_coverage_tolerance": float(args.atom_coverage_tolerance),
        "min_atom_coverage": float(args.min_atom_coverage),
        "min_qd_cue_rate": float(args.min_qd_cue_rate),
        "qd_hard_splits": _csv(args.qd_hard_splits),
        "max_duplicate_rate": float(args.max_duplicate_rate),
        "max_truncation_rate": float(args.max_truncation_rate),
        "max_truncation_delta_vs_baseline": float(args.max_truncation_delta_vs_baseline),
    }


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _source_trace_path(output_root: Path, selector_name: str, split: str) -> Path:
    return output_root / "_sources" / "liar_raw" / selector_name / split / f"selection_trace_{split}.jsonl"


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
                failures.append(f"{label}: invalid json at {path}:{line_no}: {exc}")
                return []
            if isinstance(row, dict):
                rows.append(row)
            else:
                failures.append(f"{label}: non-object json row at {path}:{line_no}")
                return []
    if not rows:
        failures.append(f"{label}: empty {path}")
    return rows


def _read_jsonl_optional(path: Path, warnings: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
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
                warnings.append(f"{label}: invalid json at {path}:{line_no}: {exc}")
                return []
            if isinstance(row, dict):
                rows.append(row)
            else:
                warnings.append(f"{label}: non-object json row at {path}:{line_no}")
                return []
    return rows


def _load_atom_coverage_floors(
    *,
    graph_root: Path,
    selector_name: str,
    splits: list[str],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    floors: dict[str, dict[str, Any]] = {}
    for split in splits:
        path = graph_root / f"{selector_name}_{split}" / f"selection_trace_{split}.jsonl"
        rows = _read_jsonl_optional(path, warnings, f"atom coverage baseline {split}")
        if not rows:
            continue
        diagnostics = [row.get("chain_diagnostics") for row in rows]
        usable_diag = [item for item in diagnostics if isinstance(item, dict) and item]
        if not usable_diag:
            warnings.append(f"{split}: atom coverage baseline has no usable diagnostics at {path}")
            continue
        floors[split] = {
            "value": _metric_mean(usable_diag, "atom_coverage_rate"),
            "source": selector_name,
            "path": str(path),
        }
    return floors


def _check_split(
    *,
    case_id: str,
    split: str,
    source_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    staged_rows: list[dict[str, Any]],
    build_rows: list[dict[str, Any]],
    failures: list[str],
    warnings: list[str],
    atom_coverage_floor: float,
    atom_coverage_floor_source: str,
    atom_coverage_tolerance: float,
    min_qd_cue_rate: float,
    qd_hard_splits: set[str],
    max_duplicate_rate: float,
) -> dict[str, Any]:
    source_count = len(source_rows)
    if len(graph_rows) != source_count:
        failures.append(f"{case_id}/{split}: graph rows={len(graph_rows)} != source rows={source_count}")
    if len(staged_rows) != source_count:
        failures.append(f"{case_id}/{split}: staged rows={len(staged_rows)} != source rows={source_count}")
    if len(build_rows) != len(staged_rows):
        failures.append(f"{case_id}/{split}: build rows={len(build_rows)} != staged rows={len(staged_rows)}")

    diagnostics = [row.get("chain_diagnostics") for row in graph_rows]
    missing_diag = sum(1 for item in diagnostics if not isinstance(item, dict) or not item)
    if missing_diag:
        failures.append(f"{case_id}/{split}: chain_diagnostics missing/non-empty failures={missing_diag}")
    usable_diag = [item for item in diagnostics if isinstance(item, dict) and item]

    outside_rows = 0
    for source_row, graph_row in zip(source_rows, graph_rows):
        source_selected = set(_int_list(source_row.get("selector_ordered_indices") or source_row.get("selected_indices")))
        selected = set(_int_list(graph_row.get("selector_ordered_indices") or graph_row.get("selected_indices")))
        if selected and not selected.issubset(source_selected):
            outside_rows += 1
    if outside_rows <= 0:
        failures.append(f"{case_id}/{split}: top20 selector never selected outside source selected set")

    duplicate_mean = _metric_mean(usable_diag, "duplicate_evidence_rate")
    atom_mean = _metric_mean(usable_diag, "atom_coverage_rate")
    qd_mean = _metric_mean(usable_diag, "qd_cue_rate")
    if duplicate_mean > max_duplicate_rate:
        failures.append(f"{case_id}/{split}: duplicate_evidence_rate.mean={duplicate_mean:.6f} > {max_duplicate_rate:.6f}")
    if atom_mean + atom_coverage_tolerance < atom_coverage_floor:
        failures.append(
            f"{case_id}/{split}: atom_coverage_rate.mean={atom_mean:.6f} < "
            f"{atom_coverage_floor:.6f} ({atom_coverage_floor_source})"
        )
    if qd_mean < min_qd_cue_rate:
        message = f"{case_id}/{split}: qd_cue_rate.mean={qd_mean:.6f} < {min_qd_cue_rate:.6f}"
        if split in qd_hard_splits:
            failures.append(message)
        else:
            warnings.append(message)

    return {
        "rows": source_count,
        "graph_rows": len(graph_rows),
        "staged_rows": len(staged_rows),
        "build_rows": len(build_rows),
        "missing_chain_diagnostics": missing_diag,
        "outside_source_selected_rows": outside_rows,
        "duplicate_evidence_rate_mean": duplicate_mean,
        "atom_coverage_rate_mean": atom_mean,
        "atom_coverage_floor": atom_coverage_floor,
        "atom_coverage_floor_source": atom_coverage_floor_source,
        "qd_cue_rate_mean": qd_mean,
    }


def _int_list(value: Any) -> list[int]:
    out: list[int] = []
    for item in value if isinstance(value, list) else []:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key, 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    return float(mean(values)) if values else 0.0


def _load_prompt_stats(run_root: Path) -> dict[str, Any]:
    for path in (
        run_root / "prompt_stats" / "prompt_stats.json",
        run_root / "train" / "prompt_stats" / "prompt_stats.json",
    ):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data["_path"] = str(path)
            return data
    return {}


def _check_prompt_quality(
    *,
    case_id: str,
    prompt_splits: list[str],
    prompt_stats: dict[str, Any],
    case_roots: list[Path],
    baseline_prompt_stats: dict[str, Any],
    baseline_roots: list[Path],
    failures: list[str],
    warnings: list[str],
    max_truncation_rate: float,
    max_delta: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in prompt_splits:
        split_stats = _prompt_split_stats(prompt_stats, split)
        if not split_stats:
            split_stats = _build_prompt_stats_from_roots(case_roots, split, warnings, f"{case_id}/{split}")
        if not split_stats:
            failures.append(f"{case_id}/{split}: missing prompt_stats or build prompt rows")
            continue
        rate = float(split_stats["truncation_rate"])
        baseline_split_stats = _prompt_split_stats(baseline_prompt_stats, split)
        if not baseline_split_stats:
            baseline_split_stats = _build_prompt_stats_from_roots(
                baseline_roots,
                split,
                warnings,
                f"{case_id}/{split} baseline",
            )
        baseline_rate = (
            float(baseline_split_stats["truncation_rate"]) if baseline_split_stats else None
        )
        out[split] = {**split_stats, "baseline_truncation_rate": baseline_rate}
        if baseline_split_stats:
            out[split]["baseline_source"] = baseline_split_stats.get("source", "")
            out[split]["baseline_path"] = baseline_split_stats.get("path", "")
        else:
            warnings.append(f"{case_id}/{split}: missing baseline prompt stats/build rows for truncation delta")
        if rate > max_truncation_rate:
            failures.append(f"{case_id}/{split}: truncation_rate={rate:.6f} > {max_truncation_rate:.6f}")
        if baseline_rate is not None and rate - baseline_rate > max_delta:
            failures.append(
                f"{case_id}/{split}: truncation_rate delta vs baseline={rate - baseline_rate:.6f} > {max_delta:.6f}"
            )
    return out


def _prompt_split_stats(prompt_stats: dict[str, Any], split: str) -> dict[str, Any]:
    rate = _prompt_truncation_rate(prompt_stats, split)
    if rate is None:
        return {}
    return {
        "source": "prompt_stats",
        "path": str(prompt_stats.get("_path", "")),
        "truncation_rate": rate,
    }


def _prompt_truncation_rate(prompt_stats: dict[str, Any], split: str) -> float | None:
    split_stats = prompt_stats.get(split)
    if not isinstance(split_stats, dict):
        return None
    trunc = split_stats.get("evidence_truncation")
    if not isinstance(trunc, dict):
        return None
    try:
        return float(trunc.get("truncation_rate"))
    except (TypeError, ValueError):
        return None


def _build_prompt_stats_from_roots(
    run_roots: list[Path],
    split: str,
    warnings: list[str],
    label: str,
) -> dict[str, Any]:
    for run_root in run_roots:
        stats = _build_prompt_stats(run_root / "build" / f"build_{split}.jsonl", warnings, label)
        if stats:
            return stats
    return {}


def _build_prompt_stats(path: Path, warnings: list[str], label: str) -> dict[str, Any]:
    rows = _read_jsonl_optional(path, warnings, f"{label} build prompt stats")
    if not rows:
        return {}
    truncated_rows = 0
    prompt_token_counts: list[float] = []
    evidence_counts: list[float] = []
    for row in rows:
        if _truthy(row.get("was_truncated")) or _truthy(row.get("evidence_text_truncated")):
            truncated_rows += 1
        prompt_token_count = _float_or_none(row.get("prompt_token_count"))
        if prompt_token_count is not None:
            prompt_token_counts.append(prompt_token_count)
        evidence_count = _float_or_none(row.get("evidence_count"))
        if evidence_count is not None:
            evidence_counts.append(evidence_count)
    total_rows = len(rows)
    return {
        "source": "build_rows",
        "path": str(path),
        "rows": total_rows,
        "truncated_rows": truncated_rows,
        "truncation_rate": float(truncated_rows / total_rows) if total_rows else 0.0,
        "mean_prompt_token_count": _mean_or_none(prompt_token_counts),
        "max_prompt_token_count": max(prompt_token_counts) if prompt_token_counts else None,
        "mean_evidence_count": _mean_or_none(evidence_counts),
        "max_evidence_count": max(evidence_counts) if evidence_counts else None,
    }


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


def _mean_or_none(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
