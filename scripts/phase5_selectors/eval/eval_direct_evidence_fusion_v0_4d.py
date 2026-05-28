#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.selectors.direct_evidence_fusion_selector import (
    DEFAULT_LAMBDAS,
    LogisticParams,
    add_fusion_candidate_features,
    build_all_fusion_traces,
    decide_fusion,
    fusion_diagnostics,
    merge_oracle_and_direct_ce_rows,
    parse_lambdas,
    run_refit_fusion,
    summarize_fusion_traces,
    validate_lambda_zero_reproduces_baseline,
)
from fact_checking.selectors.direct_evidence_cross_encoder import score_sanity_summary
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_ORACLE_SCORED = (
    "outputs/selectors/oracle_likelihood_constrained_selector/v0_3_1_val/"
    "pointwise_all_features/candidate_oracle_likelihood_scores_val.jsonl"
)
DEFAULT_DIRECT_CE_SCORED = (
    "outputs/selectors/direct_evidence_cross_encoder/v0_4a_1_val_default_query/"
    "direct_ce_scored_candidates_val.jsonl"
)
DEFAULT_OUTPUT_DIR = "outputs/selectors/direct_evidence_cross_encoder/v0_4d_val_default_query_fusion"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate v0.4d light fusion between v0.3.1 oracle-likelihood and v0.4a.1 direct CE.")
    p.add_argument("--oracle-likelihood-scored", default=DEFAULT_ORACLE_SCORED)
    p.add_argument("--direct-ce-scored", default=DEFAULT_DIRECT_CE_SCORED)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    p.add_argument("--cross-fit-folds", type=int, default=5)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--dev-fraction", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lambdas = parse_lambdas(str(args.lambdas))
    if 0.0 not in {round(value, 12) for value in lambdas}:
        raise ValueError("--lambdas must include 0.0 so lambda=0 can reproduce v0.3.1.")

    oracle_rows = _load_rows(args.oracle_likelihood_scored, sample_limit=args.sample_limit)
    direct_rows = _load_rows(args.direct_ce_scored, sample_limit=args.sample_limit)
    direct_sanity = score_sanity_summary(direct_rows)
    if not bool(direct_sanity.get("passes_score_sanity_gate")):
        raise RuntimeError(f"Direct CE score sanity failed; refusing fusion. summary={direct_sanity}")

    merged_rows = merge_oracle_and_direct_ce_rows(oracle_rows, direct_rows)
    add_fusion_candidate_features(merged_rows, lambdas=lambdas)
    params = LogisticParams(
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        patience=int(args.patience),
        eval_every=int(args.eval_every),
        seed=int(args.seed),
        dev_fraction=float(args.dev_fraction),
    )
    fused_rows, models, fold_records, feature_names, feature_importance = run_refit_fusion(
        merged_rows,
        folds=int(args.cross_fit_folds),
        params=params,
    )
    traces = build_all_fusion_traces(fused_rows, top_k=int(args.top_k), lambdas=lambdas)
    validate_lambda_zero_reproduces_baseline(traces)
    selector_metrics = summarize_fusion_traces(traces)
    diagnostics = fusion_diagnostics(
        fused_rows,
        traces,
        selector_metrics,
        feature_importance=feature_importance,
    )
    decision = decide_fusion(selector_metrics, diagnostics)
    diagnostics["decision"] = asdict(decision)
    diagnostics["folds"] = fold_records

    candidate_path = out_dir / f"candidate_fusion_scores_{args.split}.jsonl"
    trace_path = out_dir / f"selection_trace_{args.split}.jsonl"
    write_jsonl(fused_rows, candidate_path)
    write_jsonl(traces, trace_path)
    save_json(selector_metrics, out_dir / "selector_metrics.json")
    save_json(_json_safe(diagnostics), out_dir / "fusion_diagnostics.json")
    save_json({"features": _json_safe(feature_importance)}, out_dir / "feature_importance.json")
    save_json(
        _json_safe(
            {
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
                "split": str(args.split),
                "oracle_likelihood_scored": str(args.oracle_likelihood_scored),
                "direct_ce_scored": str(args.direct_ce_scored),
                "output_dir": str(out_dir),
                "top_k": int(args.top_k),
                "lambdas": lambdas,
                "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
                "cross_fit_folds": int(args.cross_fit_folds),
                "params": params.__dict__,
                "n_events": len(fused_rows),
                "n_candidates": sum(len(row.get("candidates") or []) for row in fused_rows),
                "n_features": len(feature_names),
                "feature_names": feature_names,
                "direct_ce_score_sanity": direct_sanity,
                "outputs": {
                    "candidate_fusion_scores": str(candidate_path),
                    "selection_trace": str(trace_path),
                    "selector_metrics": str(out_dir / "selector_metrics.json"),
                    "diagnostics": str(out_dir / "fusion_diagnostics.json"),
                    "analysis_summary": str(out_dir / "analysis_summary.md"),
                    "feature_importance": str(out_dir / "feature_importance.json"),
                },
                "decision": asdict(decision),
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
        ),
        out_dir / "manifest.json",
    )
    _write_analysis(out_dir / "analysis_summary.md", selector_metrics, diagnostics, decision)
    print(f"Wrote direct evidence fusion eval under: {out_dir}")
    print(
        "Decision={decision}; best={best}; delta_jaccard={dj:.4f}; delta_recall={dr:.4f}".format(
            decision=decision.decision,
            best=decision.best_selector,
            dj=decision.delta_jaccard,
            dr=decision.delta_recall,
        )
    )


def _load_rows(path: str, *, sample_limit: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if sample_limit is not None:
        rows = rows[: int(sample_limit)]
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def _write_analysis(
    path: Path,
    selector_metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    decision: Any,
) -> None:
    lines = [
        "# Direct Evidence Light Fusion v0.4d",
        "",
        f"- decision: `{decision.decision}`",
        f"- best_selector: `{decision.best_selector}`",
        f"- baseline_jaccard@5: `{decision.baseline_jaccard:.4f}`",
        f"- best_jaccard@5: `{decision.best_jaccard:.4f}`",
        f"- delta_jaccard@5: `{decision.delta_jaccard:.4f}`",
        f"- baseline_recall@5: `{decision.baseline_recall:.4f}`",
        f"- best_recall@5: `{decision.best_recall:.4f}`",
        f"- delta_recall@5: `{decision.delta_recall:.4f}`",
        f"- refit_direct_ce_mean_abs_weight: `{decision.refit_direct_ce_mean_abs_weight:.6f}`",
        "",
        "## Selector Metrics",
        "",
        "| selector | recall@5 | jaccard@5 | top1 | ndcg@5 | pairwise@5 | fusion@5 | oracle@5 | direct_ce@5 | source_entropy | stance_entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selector, metrics in sorted(selector_metrics.items()):
        lines.append(
            "| {selector} | {recall:.4f} | {jaccard:.4f} | {top1:.4f} | {ndcg:.4f} | {pairwise:.4f} | {fusion:.4f} | {oracle:.4f} | {direct:.4f} | {source:.4f} | {stance:.4f} |".format(
                selector=selector,
                recall=float(metrics.get("recall@5", 0.0)),
                jaccard=float(metrics.get("jaccard@5", 0.0)),
                top1=float(metrics.get("top1_match", 0.0)),
                ndcg=float(metrics.get("oracle_rank_ndcg@5", 0.0)),
                pairwise=float(metrics.get("pairwise_order_acc@5", 0.0)),
                fusion=float(metrics.get("mean_fusion_score@5", 0.0)),
                oracle=float(metrics.get("mean_oracle_likelihood_score@5", 0.0)),
                direct=float(metrics.get("mean_direct_ce_score@5", 0.0)),
                source=float(metrics.get("source_entropy@5", 0.0)),
                stance=float(metrics.get("stance_bucket_entropy@5", 0.0)),
            )
        )
    candidate = diagnostics.get("candidate_metrics") or {}
    lines.extend(["", "## Candidate Metrics", ""])
    for name, metrics in sorted(candidate.items()):
        lines.append(
            "- {name}: auroc=`{auroc:.4f}`, auprc=`{auprc:.4f}`, brier=`{brier:.4f}`, log_loss=`{log_loss:.4f}`, transform=`{transform}`".format(
                name=name,
                auroc=float(metrics.get("auroc", 0.0)),
                auprc=float(metrics.get("auprc", 0.0)),
                brier=float(metrics.get("brier", 0.0)),
                log_loss=float(metrics.get("log_loss", 0.0)),
                transform=str(metrics.get("probability_transform") or ""),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
