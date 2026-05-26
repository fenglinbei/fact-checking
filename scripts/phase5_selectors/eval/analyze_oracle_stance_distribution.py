"""Analyze whether NLI stance scores carry signal for Stage2 oracle selection."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.selectors.stage2_oracle import write_json


STANCE_LABELS = ("support", "refute", "neutral")
STANCE_FEATURES = (
    "support_score",
    "refute_score",
    "neutral_score",
    "stance_confidence",
    "stance_polarity",
    "support_refute_abs_margin",
    "qualify_proxy_score",
)
RETRIEVAL_FEATURES = ("hybrid_score", "dense_score", "lexical_score", "bm25_score", "hybrid_rank")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize oracle-selected vs pool stance/NLI distributions.")
    p.add_argument("--stance-scores", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-lift-pp", type=float, default=5.0)
    p.add_argument("--min-auroc", type=float, default=0.57)
    p.add_argument("--top-examples", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl_many(args.stance_scores)
    if not rows:
        raise ValueError("No stance score rows found.")
    selected_rows = [row for row in rows if bool(row.get("selected"))]
    nonselected_rows = [row for row in rows if not bool(row.get("selected"))]

    oracle_vs_pool = _oracle_vs_pool_payload(rows, selected_rows)
    by_gold = _grouped_payload(rows, group_key="gold_label")
    by_step = _oracle_step_payload(rows)
    probe = _selected_probe_payload(rows)
    set_patterns = _set_patterns_payload(rows, top_examples=int(args.top_examples))
    decision = _decision_payload(
        oracle_vs_pool,
        probe,
        min_lift_pp=float(args.min_lift_pp),
        min_auroc=float(args.min_auroc),
    )

    write_json(out_dir / "oracle_vs_pool_stance_distribution.json", oracle_vs_pool)
    write_json(out_dir / "stance_by_gold_label.json", by_gold)
    write_json(out_dir / "stance_by_oracle_step.json", by_step)
    write_json(out_dir / "selected_vs_nonselected_probe.json", probe)
    write_json(out_dir / "stance_set_patterns.json", set_patterns)
    write_json(
        out_dir / "analysis_summary.json",
        {
            "stance_scores": [str(path) for path in args.stance_scores],
            "n_candidates": len(rows),
            "n_selected": len(selected_rows),
            "n_nonselected": len(nonselected_rows),
            "n_events": len({str(row.get("event_id") or "") for row in rows}),
            "decision": decision,
        },
    )
    _write_markdown(
        out_dir / "analysis.md",
        stance_scores=[str(path) for path in args.stance_scores],
        oracle_vs_pool=oracle_vs_pool,
        probe=probe,
        set_patterns=set_patterns,
        decision=decision,
    )

    print(f"Wrote stance analysis under: {out_dir}")
    print(
        "Decision={decision}; selected stance lift={lift:.2f}pp; best stance AUROC={auc:.4f}".format(
            decision=decision["decision"],
            lift=float(decision["support_refute_selected_lift_pp"]),
            auc=float(decision["best_stance_feature_separability_auc"]),
        )
    )


def _oracle_vs_pool_payload(rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pool": _bucket_summary(rows),
        "oracle_selected": _bucket_summary(selected_rows),
        "deltas": _distribution_deltas(_distribution(rows), _distribution(selected_rows)),
    }


def _grouped_payload(rows: list[dict[str, Any]], *, group_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "")].append(row)
    payload: dict[str, Any] = {}
    for key in sorted(grouped):
        group_rows = grouped[key]
        selected_rows = [row for row in group_rows if bool(row.get("selected"))]
        payload[key] = {
            "pool": _bucket_summary(group_rows),
            "oracle_selected": _bucket_summary(selected_rows),
            "deltas": _distribution_deltas(_distribution(group_rows), _distribution(selected_rows)),
        }
    return payload


def _oracle_step_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(int(row.get("oracle_step", -1)))].append(row)
    return {key: _bucket_summary(grouped[key]) for key in sorted(grouped, key=lambda item: int(item))}


def _selected_probe_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([1 if bool(row.get("selected")) else 0 for row in rows], dtype=np.int64)
    features: dict[str, Any] = {}
    for feature in STANCE_FEATURES + RETRIEVAL_FEATURES:
        values = np.asarray([_as_float(row.get(feature)) for row in rows], dtype=np.float64)
        valid = ~np.isnan(values)
        if int(valid.sum()) == 0:
            continue
        y = labels[valid]
        x = values[valid]
        auc = _roc_auc_score(y, x)
        features[feature] = {
            "n": int(valid.sum()),
            "selected": _numeric_summary(x[y == 1]),
            "nonselected": _numeric_summary(x[y == 0]),
            "mean_delta_selected_minus_nonselected": _safe_mean(x[y == 1]) - _safe_mean(x[y == 0]),
            "auroc_selected_positive": auc,
            "separability_auc": max(auc, 1.0 - auc) if not math.isnan(auc) else math.nan,
        }

    correlations: dict[str, Any] = {}
    hybrid = np.asarray([_as_float(row.get("hybrid_score")) for row in rows], dtype=np.float64)
    for feature in STANCE_FEATURES:
        values = np.asarray([_as_float(row.get(feature)) for row in rows], dtype=np.float64)
        valid = ~np.isnan(values) & ~np.isnan(hybrid)
        if int(valid.sum()) < 2:
            continue
        correlations[feature] = {
            "pearson_with_hybrid_score": _pearson(values[valid], hybrid[valid]),
            "spearman_with_hybrid_score": _pearson(_rankdata(values[valid]), _rankdata(hybrid[valid])),
        }

    return {
        "features": features,
        "stance_feature_names": list(STANCE_FEATURES),
        "retrieval_feature_names": list(RETRIEVAL_FEATURES),
        "stance_vs_hybrid_correlations": correlations,
    }


def _set_patterns_payload(rows: list[dict[str, Any]], *, top_examples: int) -> dict[str, Any]:
    events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        events[str(row.get("event_id") or "")].append(row)

    per_event: list[dict[str, Any]] = []
    for event_id, event_rows in events.items():
        selected = sorted(
            [row for row in event_rows if bool(row.get("selected"))],
            key=lambda row: int(row.get("oracle_step", 999)),
        )
        pool_counts = Counter(str(row.get("stance_label") or "unknown") for row in event_rows)
        selected_counts = Counter(str(row.get("stance_label") or "unknown") for row in selected)
        per_event.append(
            {
                "event_id": event_id,
                "gold_label": str(event_rows[0].get("gold_label") or ""),
                "n_candidates": len(event_rows),
                "n_selected": len(selected),
                "pool_stance_counts": dict(pool_counts),
                "selected_stance_counts": dict(selected_counts),
                "selected_stance_order": [str(row.get("stance_label") or "unknown") for row in selected],
                "has_pool_support_refute_mix": pool_counts.get("support", 0) > 0 and pool_counts.get("refute", 0) > 0,
                "has_selected_support_refute_mix": selected_counts.get("support", 0) > 0
                and selected_counts.get("refute", 0) > 0,
                "pool_stance_entropy": _entropy_from_counts(pool_counts),
                "selected_stance_entropy": _entropy_from_counts(selected_counts),
                "max_support_score": _max_feature(event_rows, "support_score"),
                "max_refute_score": _max_feature(event_rows, "refute_score"),
                "max_neutral_score": _max_feature(event_rows, "neutral_score"),
                "selected_max_support_score": _max_feature(selected, "support_score"),
                "selected_max_refute_score": _max_feature(selected, "refute_score"),
                "selected_max_neutral_score": _max_feature(selected, "neutral_score"),
            }
        )

    return {
        "aggregate": _set_pattern_aggregate(per_event),
        "per_event": sorted(per_event, key=lambda row: row["event_id"]),
        "examples": {
            "top_selected_support": _top_rows(
                [row for row in rows if bool(row.get("selected"))],
                feature="support_score",
                limit=top_examples,
            ),
            "top_selected_refute": _top_rows(
                [row for row in rows if bool(row.get("selected"))],
                feature="refute_score",
                limit=top_examples,
            ),
            "top_selected_neutral": _top_rows(
                [row for row in rows if bool(row.get("selected"))],
                feature="neutral_score",
                limit=top_examples,
            ),
        },
    }


def _set_pattern_aggregate(per_event: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_event:
        return {}
    return {
        "n_events": len(per_event),
        "pool_support_refute_mix_rate": _mean_bool(row["has_pool_support_refute_mix"] for row in per_event),
        "selected_support_refute_mix_rate": _mean_bool(row["has_selected_support_refute_mix"] for row in per_event),
        "mean_pool_stance_entropy": _mean(row["pool_stance_entropy"] for row in per_event),
        "mean_selected_stance_entropy": _mean(row["selected_stance_entropy"] for row in per_event),
        "mean_max_support_score": _mean(row["max_support_score"] for row in per_event),
        "mean_max_refute_score": _mean(row["max_refute_score"] for row in per_event),
        "mean_max_neutral_score": _mean(row["max_neutral_score"] for row in per_event),
        "mean_selected_max_support_score": _mean(row["selected_max_support_score"] for row in per_event),
        "mean_selected_max_refute_score": _mean(row["selected_max_refute_score"] for row in per_event),
        "mean_selected_max_neutral_score": _mean(row["selected_max_neutral_score"] for row in per_event),
    }


def _decision_payload(
    oracle_vs_pool: dict[str, Any],
    probe: dict[str, Any],
    *,
    min_lift_pp: float,
    min_auroc: float,
) -> dict[str, Any]:
    pool_props = oracle_vs_pool["pool"]["stance_distribution"]["proportions"]
    selected_props = oracle_vs_pool["oracle_selected"]["stance_distribution"]["proportions"]
    pool_support_refute = float(pool_props.get("support", 0.0)) + float(pool_props.get("refute", 0.0))
    selected_support_refute = float(selected_props.get("support", 0.0)) + float(selected_props.get("refute", 0.0))
    lift_pp = (selected_support_refute - pool_support_refute) * 100.0

    stance_features = {
        key: value
        for key, value in probe.get("features", {}).items()
        if key in STANCE_FEATURES and not math.isnan(float(value.get("separability_auc", math.nan)))
    }
    best_name = ""
    best_auc = math.nan
    if stance_features:
        best_name, best_payload = max(
            stance_features.items(),
            key=lambda item: float(item[1].get("separability_auc", math.nan)),
        )
        best_auc = float(best_payload.get("separability_auc", math.nan))

    go_by_lift = lift_pp >= float(min_lift_pp)
    go_by_auc = not math.isnan(best_auc) and best_auc >= float(min_auroc)
    decision = "go_selector_ablation" if go_by_lift or go_by_auc else "stop_or_calibrate_nli"
    return {
        "decision": decision,
        "min_lift_pp": float(min_lift_pp),
        "min_auroc": float(min_auroc),
        "support_refute_pool_rate": pool_support_refute,
        "support_refute_selected_rate": selected_support_refute,
        "support_refute_selected_lift_pp": lift_pp,
        "best_stance_feature": best_name,
        "best_stance_feature_separability_auc": best_auc,
        "go_by_lift": bool(go_by_lift),
        "go_by_auc": bool(go_by_auc),
    }


def _bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "stance_distribution": _distribution(rows),
        "probability_means": {feature: _safe_mean_values(rows, feature) for feature in STANCE_FEATURES},
        "retrieval_means": {feature: _safe_mean_values(rows, feature) for feature in RETRIEVAL_FEATURES},
    }


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("stance_label") or "unknown") for row in rows)
    for label in STANCE_LABELS:
        counts.setdefault(label, 0)
    total = sum(counts.values())
    return {
        "counts": dict(sorted(counts.items())),
        "proportions": {
            key: (float(value) / float(total) if total else 0.0)
            for key, value in sorted(counts.items())
        },
    }


def _distribution_deltas(pool_dist: dict[str, Any], selected_dist: dict[str, Any]) -> dict[str, Any]:
    labels = sorted(set(pool_dist["proportions"]) | set(selected_dist["proportions"]))
    return {
        label: float(selected_dist["proportions"].get(label, 0.0)) - float(pool_dist["proportions"].get(label, 0.0))
        for label in labels
    }


def _read_jsonl_many(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    return rows


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_markdown(
    path: Path,
    *,
    stance_scores: list[str],
    oracle_vs_pool: dict[str, Any],
    probe: dict[str, Any],
    set_patterns: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    pool = oracle_vs_pool["pool"]
    selected = oracle_vs_pool["oracle_selected"]
    features = probe.get("features", {})
    stance_feature_rows = [
        (name, payload)
        for name, payload in features.items()
        if name in STANCE_FEATURES
    ]
    stance_feature_rows.sort(key=lambda item: float(item[1].get("separability_auc", -1.0)), reverse=True)
    aggregate = set_patterns.get("aggregate", {})

    lines = [
        "# Oracle Stance/NLI Distribution Analysis",
        "",
        f"- stance_scores: {', '.join(f'`{item}`' for item in stance_scores)}",
        f"- candidates: {pool['n']}",
        f"- oracle_selected: {selected['n']}",
        f"- decision: `{decision['decision']}`",
        "",
        "## Pool vs Oracle Selected",
        "",
        "| label | pool | selected | delta_pp |",
        "|---|---:|---:|---:|",
    ]
    labels = sorted(
        set(pool["stance_distribution"]["proportions"])
        | set(selected["stance_distribution"]["proportions"])
    )
    for label in labels:
        pool_prop = float(pool["stance_distribution"]["proportions"].get(label, 0.0))
        selected_prop = float(selected["stance_distribution"]["proportions"].get(label, 0.0))
        lines.append(f"| {label} | {pool_prop:.4f} | {selected_prop:.4f} | {(selected_prop - pool_prop) * 100.0:.2f} |")

    lines.extend(
        [
            "",
            "## Selected Probe",
            "",
            "| feature | selected_mean | nonselected_mean | delta | auroc | separability_auc |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, payload in stance_feature_rows:
        selected_mean = payload["selected"]["mean"]
        nonselected_mean = payload["nonselected"]["mean"]
        delta = payload["mean_delta_selected_minus_nonselected"]
        auc = payload["auroc_selected_positive"]
        sep = payload["separability_auc"]
        lines.append(f"| {name} | {selected_mean:.6f} | {nonselected_mean:.6f} | {delta:.6f} | {auc:.4f} | {sep:.4f} |")

    lines.extend(
        [
            "",
            "## Set-Level Patterns",
            "",
            f"- pool_support_refute_mix_rate: {float(aggregate.get('pool_support_refute_mix_rate', math.nan)):.4f}",
            f"- selected_support_refute_mix_rate: {float(aggregate.get('selected_support_refute_mix_rate', math.nan)):.4f}",
            f"- mean_pool_stance_entropy: {float(aggregate.get('mean_pool_stance_entropy', math.nan)):.4f}",
            f"- mean_selected_stance_entropy: {float(aggregate.get('mean_selected_stance_entropy', math.nan)):.4f}",
            "",
            "## Stop/Go",
            "",
            f"- support_refute_selected_lift_pp: {float(decision['support_refute_selected_lift_pp']):.2f}",
            f"- best_stance_feature: `{decision['best_stance_feature']}`",
            f"- best_stance_feature_separability_auc: {float(decision['best_stance_feature_separability_auc']):.4f}",
            f"- go_by_lift: {decision['go_by_lift']}",
            f"- go_by_auc: {decision['go_by_auc']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _top_rows(rows: list[dict[str, Any]], *, feature: str, limit: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: _sort_float(row.get(feature)), reverse=True)
    output: list[dict[str, Any]] = []
    for row in sorted_rows[: max(int(limit), 0)]:
        output.append(
            {
                "event_id": row.get("event_id"),
                "gold_label": row.get("gold_label"),
                "oracle_step": row.get("oracle_step"),
                "candidate_idx": row.get("candidate_idx"),
                "candidate_uid": row.get("candidate_uid"),
                "stance_label": row.get("stance_label"),
                "support_score": row.get("support_score"),
                "refute_score": row.get("refute_score"),
                "neutral_score": row.get("neutral_score"),
                "hybrid_score": row.get("hybrid_score"),
                "claim": _truncate(str(row.get("claim") or ""), 180),
                "text": _truncate(str(row.get("text") or ""), 220),
            }
        )
    return output


def _numeric_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "p25": math.nan, "p50": math.nan, "p75": math.nan, "max": math.nan}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def _roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = ~np.isnan(scores)
    labels = labels[valid]
    scores = scores[valid]
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _rankdata(scores)
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return math.nan
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return math.nan
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (x_std * y_std))


def _entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        if value > 0:
            p = float(value) / float(total)
            entropy -= p * math.log(p)
    return entropy


def _max_feature(rows: list[dict[str, Any]], feature: str) -> float:
    values = [_as_float(row.get(feature)) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return max(values) if values else math.nan


def _safe_mean_values(rows: list[dict[str, Any]], feature: str) -> float:
    return _mean(_as_float(row.get(feature)) for row in rows)


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return float(np.mean(values)) if values.size else math.nan


def _mean(values: Any) -> float:
    clean = [
        float(value)
        for value in values
        if value is not None and not math.isnan(float(value)) and not math.isinf(float(value))
    ]
    return float(sum(clean) / len(clean)) if clean else math.nan


def _mean_bool(values: Any) -> float:
    values = list(values)
    return float(sum(1 for value in values if value) / len(values)) if values else math.nan


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if not math.isnan(number) and not math.isinf(number) else math.nan


def _sort_float(value: Any) -> float:
    number = _as_float(value)
    return number if not math.isnan(number) else -math.inf


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= int(limit) else text[: int(limit) - 3] + "..."


if __name__ == "__main__":
    main()
