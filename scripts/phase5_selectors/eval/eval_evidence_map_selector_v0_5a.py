#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fact_checking.selectors.evidence_map_selector import (
    EVIDENCE_MAP_SELECTOR,
    build_all_evidence_map_traces,
    evidence_map_diagnostics,
    render_case_study_markdown,
    summarize_evidence_map_traces,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_OUTPUT_DIR = "outputs/selectors/evidence_map_selector/v0_5a_val"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate v0.5a claim-atom evidence-map selector.")
    p.add_argument("--candidate-features", required=True)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--case-ids", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.candidate_features)
    if args.sample_limit is not None:
        rows = rows[: int(args.sample_limit)]
    if not rows:
        raise ValueError(f"No rows loaded from {args.candidate_features}")
    traces = build_all_evidence_map_traces(rows, top_k=int(args.top_k))
    selector_metrics = summarize_evidence_map_traces(traces)
    diagnostics = evidence_map_diagnostics(rows, traces, selector_metrics)
    case_ids = [item.strip() for item in str(args.case_ids or "").split(",") if item.strip()]
    case_study = render_case_study_markdown(traces, case_ids=case_ids)

    trace_path = out_dir / f"selection_trace_{args.split}.jsonl"
    write_jsonl(traces, trace_path)
    save_json(selector_metrics, out_dir / "selector_metrics.json")
    save_json(diagnostics, out_dir / "evidence_map_diagnostics.json")
    (out_dir / "case_studies.md").write_text(case_study, encoding="utf-8")
    _write_analysis(out_dir / "analysis_summary.md", selector_metrics, diagnostics)
    manifest: dict[str, Any] = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "candidate_features": str(args.candidate_features),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "top_k": int(args.top_k),
        "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
        "case_ids": case_ids,
        "n_events": len(rows),
        "n_traces": len(traces),
        "decision": diagnostics.get("decision"),
        "outputs": {
            "selection_trace": str(trace_path),
            "selector_metrics": str(out_dir / "selector_metrics.json"),
            "diagnostics": str(out_dir / "evidence_map_diagnostics.json"),
            "analysis_summary": str(out_dir / "analysis_summary.md"),
            "case_studies": str(out_dir / "case_studies.md"),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    save_json(manifest, out_dir / "manifest.json")
    primary = selector_metrics.get(EVIDENCE_MAP_SELECTOR, {})
    print(f"Wrote evidence-map selector eval under: {out_dir}")
    print(
        "Decision={decision}; jaccard={jaccard:.4f}; recall={recall:.4f}; weighted_coverage={coverage:.4f}".format(
            decision=diagnostics.get("decision"),
            jaccard=float(primary.get("jaccard@5", 0.0)),
            recall=float(primary.get("recall@5", 0.0)),
            coverage=float(primary.get("mean_weighted_atom_coverage@5", 0.0)),
        )
    )


def _write_analysis(path: Path, selector_metrics: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    lines = [
        "# Claim-Atom Evidence Map Selector v0.5a",
        "",
        f"- decision: `{diagnostics.get('decision')}`",
        f"- n_events: `{diagnostics.get('n_events', 0)}`",
        f"- n_candidates: `{diagnostics.get('n_candidates', 0)}`",
        "",
        "## Selector Metrics",
        "",
        "| selector | recall@5 | jaccard@5 | top1 | ndcg@5 | atom_cov | weighted_cov | direct/partial | background | duplicate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selector, metrics in sorted(selector_metrics.items()):
        lines.append(
            "| {selector} | {recall:.4f} | {jaccard:.4f} | {top1:.4f} | {ndcg:.4f} | {atom:.4f} | {watom:.4f} | {direct:.4f} | {background:.4f} | {duplicate:.4f} |".format(
                selector=selector,
                recall=float(metrics.get("recall@5", 0.0)),
                jaccard=float(metrics.get("jaccard@5", 0.0)),
                top1=float(metrics.get("top1_match", 0.0)),
                ndcg=float(metrics.get("oracle_rank_ndcg@5", 0.0)),
                atom=float(metrics.get("mean_atom_coverage@5", 0.0)),
                watom=float(metrics.get("mean_weighted_atom_coverage@5", 0.0)),
                direct=float(metrics.get("direct_or_partial_map_rate@5", 0.0)),
                background=float(metrics.get("background_only_map_rate@5", 0.0)),
                duplicate=float(metrics.get("duplicate_group_collapse_rate@5", 0.0)),
            )
        )
    lines.extend(["", "## Deltas", ""])
    for name in ("primary_vs_fusion_refit", "primary_vs_base_only"):
        delta = diagnostics.get(name) or {}
        lines.append(f"### {name}")
        for key, value in sorted(delta.items()):
            lines.append(f"- {key}: `{float(value):.4f}`")
        lines.append("")
    lines.extend(["## Go Criteria", ""])
    for key, value in sorted((diagnostics.get("go_criteria") or {}).items()):
        lines.append(f"- {key}: `{bool(value)}`")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
