#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_MODES = ["baseline_only", "atom_only", "union_no_mmr", "union_full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the LIAR-RAW Atom-Union pool ablation.")
    parser.add_argument(
        "--ablation-root",
        default="outputs/selectors/atom_union_pool_ablation/liar_raw_abc_n20",
    )
    parser.add_argument("--case-output-root", default="outputs/sentence_trace_method")
    parser.add_argument("--case-suffix", default="")
    parser.add_argument("--pool-modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation_root = Path(args.ablation_root)
    case_output_root = Path(args.case_output_root)
    output_dir = Path(args.output_dir) if args.output_dir else ablation_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        collect_row(
            pool_mode=pool_mode,
            split=split,
            ablation_root=ablation_root,
            case_output_root=case_output_root,
            case_suffix=str(args.case_suffix),
        )
        for pool_mode in args.pool_modes
        for split in ("val", "test")
    ]
    (output_dir / "summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.md").write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote Atom-Union pool ablation summary: {output_dir / 'summary.md'}")


def collect_row(
    *,
    pool_mode: str,
    split: str,
    ablation_root: Path,
    case_output_root: Path,
    case_suffix: str,
) -> dict[str, Any]:
    variant_root = ablation_root / pool_mode
    case_root = case_output_root / (
        f"liar_raw__ministral3_8b__atom_union_pool_ablation_{pool_mode}_reuse_main_ckpt{case_suffix}"
    )
    pool_path = variant_root / "03_atom_union" / f"atom_union_candidate_pool_{split}.jsonl"
    map_manifest = variant_root / "04_evidence_map" / f"postprocess_manifest_{split}.json"
    diagnostics_path = (
        variant_root
        / "05_mrec_v0_2_learned_marginal_proxy_fullpool_minmax5_10"
        / f"mrec_diagnostics_{split}.json"
    )
    metrics_path = case_root / "eval" / split / "best" / "label_token" / "metrics.json"
    tau_metrics_path = (
        case_root
        / "eval"
        / split
        / "best"
        / "label_token_logit_adjust_tau0p75"
        / "metrics.json"
    )

    pool_rows = read_jsonl(pool_path)
    metrics = read_json(metrics_path)
    tau_metrics = read_json(tau_metrics_path)
    diagnostics = read_json(diagnostics_path)
    map_payload = read_json(map_manifest)
    candidate_counts = [len(row.get("candidates") or []) for row in pool_rows]
    mmr_rows = sum(bool(row.get("union_mmr_applied")) for row in pool_rows)

    return {
        "pool_mode": pool_mode,
        "split": split,
        "status": "ok" if metrics else "missing_metrics",
        "num_samples": metrics.get("num_samples"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "tau075_accuracy": tau_metrics.get("accuracy"),
        "tau075_macro_f1": tau_metrics.get("macro_f1"),
        "mean_candidate_count": mean(candidate_counts),
        "min_candidate_count": min(candidate_counts) if candidate_counts else None,
        "max_candidate_count": max(candidate_counts) if candidate_counts else None,
        "mmr_row_rate": (mmr_rows / len(pool_rows)) if pool_rows else None,
        "map_parse_status_counts": map_payload.get("parse_status_counts") or {},
        "mean_selected_steps": nested(diagnostics, "step_count", "mean"),
        "mean_resolved_atom_rate": nested(diagnostics, "resolved_atom_rate", "mean"),
        "metrics_path": str(metrics_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "pool_mode",
        "split",
        "status",
        "num_samples",
        "accuracy",
        "macro_f1",
        "tau075_accuracy",
        "tau075_macro_f1",
        "mean_candidate_count",
        "min_candidate_count",
        "max_candidate_count",
        "mmr_row_rate",
        "mean_selected_steps",
        "mean_resolved_atom_rate",
        "metrics_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LIAR-RAW Atom-Union Candidate-Pool Ablation",
        "",
        "All rows reuse the main learned-marginal selector weights and the same verifier checkpoint.",
        "",
        "| Pool mode | Split | Acc | Macro-F1 | Tau 0.75 F1 | Mean pool | MMR rate | Mean K* | Resolved atoms | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {pool_mode} | {split} | {accuracy} | {macro_f1} | {tau_f1} | {pool} | {mmr} | {steps} | {resolved} | {status} |".format(
                pool_mode=row["pool_mode"],
                split=row["split"],
                accuracy=fmt(row.get("accuracy")),
                macro_f1=fmt(row.get("macro_f1")),
                tau_f1=fmt(row.get("tau075_macro_f1")),
                pool=fmt(row.get("mean_candidate_count"), digits=2),
                mmr=fmt(row.get("mmr_row_rate")),
                steps=fmt(row.get("mean_selected_steps"), digits=2),
                resolved=fmt(row.get("mean_resolved_atom_rate")),
                status=row.get("status") or "",
            )
        )
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nested(payload: dict[str, Any], key: str, child: str) -> Any:
    value = payload.get(key)
    return value.get(child) if isinstance(value, dict) else None


def mean(values: list[int]) -> float | None:
    return (sum(values) / len(values)) if values else None


def fmt(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    main()
