"""Offline retrospective evaluation for soft-label RL-MMR policies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.learned_lambda.cache_utils import (
    load_experiment_build_cfg,
    pick_retrieval_value,
    resolve_chunk_mmr_cache_path,
)
from fact_checking.rl_mmr.soft_label_dataset import (
    SoftLabelDataset,
    parse_lambda_grid,
    utility_vector_from_record,
)
from fact_checking.rl_mmr.soft_label_selector import (
    load_soft_label_policy,
    predict_policy_proba,
    select_lambdas_from_probs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a soft-label lambda policy by oracle utility lookup.")
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--oracle-logprobs", type=str, required=True)
    p.add_argument("--chunk-mmr-cache", type=str, default=None)
    p.add_argument("--chunk-mmr-cache-root", type=str, default="outputs/cache/chunk_mmr")
    p.add_argument("--experiment", type=str, default="b3_mmr_topk_sweep_1024")
    p.add_argument("--config-overrides", nargs="*", default=[])
    p.add_argument("--split-name", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--lambda-grid", type=str, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--weight-mode", type=str, default=None, choices=["margin", "gap", "none"])
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--alpha-dense", type=float, default=None)
    p.add_argument("--alpha-lexical", type=float, default=None)
    p.add_argument("--alpha-bm25", type=float, default=None)
    p.add_argument("--inference-modes", type=str, default="argmax,expected,sample")
    p.add_argument("--sample-temperature", type=float, default=0.5)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--fixed-lambda", type=float, default=0.7)
    p.add_argument("--continuous-lookup", type=str, default="nearest", choices=["nearest", "linear"])
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--sample-limit", type=int, default=None)
    return p.parse_args()


def _resolve_cache(
    build_cfg: dict[str, Any],
    *,
    explicit_path: str | None,
    split_name: str,
    cache_root: str,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return resolve_chunk_mmr_cache_path(build_cfg, split_name=split_name, cache_root=cache_root)


def _utility_at_lambda(
    record: dict[str, Any],
    lambda_grid: np.ndarray,
    chosen_lambda: float,
    *,
    mode: str,
) -> tuple[float, float]:
    utility = utility_vector_from_record(record, lambda_grid)
    grid = np.asarray(lambda_grid, dtype=np.float32)
    lam = float(chosen_lambda)
    if mode == "linear":
        order = np.argsort(grid)
        grid_sorted = grid[order]
        util_sorted = utility[order]
        return float(np.interp(lam, grid_sorted, util_sorted)), lam
    idx = int(np.argmin(np.abs(grid - lam)))
    return float(utility[idx]), float(grid[idx])


def _oracle_margin(record: dict[str, Any], lambda_grid: np.ndarray) -> float:
    values = np.sort(utility_vector_from_record(record, lambda_grid))[::-1]
    if len(values) < 2:
        return 0.0
    return float(values[0] - values[1])


def _target_calibration(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    pred = np.argmax(probs, axis=1)
    conf = probs[np.arange(len(probs)), pred]
    soft_acc = targets[np.arange(len(targets)), pred]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi >= 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            rows.append({"lo": float(lo), "hi": float(hi), "n": 0, "confidence": 0.0, "soft_accuracy": 0.0})
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(soft_acc[mask].mean())
        frac = float(mask.mean())
        ece += frac * abs(avg_acc - avg_conf)
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()), "confidence": avg_conf, "soft_accuracy": avg_acc})
    return {"ece": float(ece), "bins": rows}


def _write_calibration_plot(calibration: dict[str, Any], output_path: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = calibration["bins"]
    xs = [(row["lo"] + row["hi"]) / 2.0 for row in rows]
    conf = [row["confidence"] for row in rows]
    acc = [row["soft_accuracy"] for row in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1)
    ax.plot(xs, conf, marker="o", label="confidence")
    ax.plot(xs, acc, marker="o", label="soft target prob")
    ax.set_xlabel("confidence bin")
    ax.set_ylabel("probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _bucket_assignments(raw_feature_dicts: list[dict[str, Any]], margins: np.ndarray) -> dict[str, list[str]]:
    n_candidates = np.array([float(row.get("n_candidates", 0.0)) for row in raw_feature_dicts], dtype=np.float32)
    sensitivity = np.array([float(row.get("sens_0p30_0p70", 0.0)) for row in raw_feature_dicts], dtype=np.float32)
    redundancy = np.array([float(row.get("pool_redundancy", 0.0)) for row in raw_feature_dicts], dtype=np.float32)
    sens_med = float(np.median(sensitivity)) if sensitivity.size else 0.0
    red_med = float(np.median(redundancy)) if redundancy.size else 0.0
    assignments: dict[str, list[str]] = {
        "candidate_count": [],
        "sensitivity": [],
        "pool_redundancy": [],
        "oracle_margin": [],
        "sensitivity_x_redundancy": [],
    }
    for i in range(len(raw_feature_dicts)):
        if n_candidates[i] <= 5:
            assignments["candidate_count"].append("n<=5")
        elif n_candidates[i] <= 16:
            assignments["candidate_count"].append("6<=n<=16")
        else:
            assignments["candidate_count"].append("n>16")

        sens_label = "high" if sensitivity[i] >= sens_med else "low"
        red_label = "high" if redundancy[i] >= red_med else "low"
        assignments["sensitivity"].append(sens_label)
        assignments["pool_redundancy"].append(red_label)
        assignments["sensitivity_x_redundancy"].append(f"{sens_label}_sensitivity__{red_label}_redundancy")

        if margins[i] < 0.01:
            assignments["oracle_margin"].append("margin<0.01")
        elif margins[i] < 0.05:
            assignments["oracle_margin"].append("0.01<=margin<0.05")
        elif margins[i] < 0.10:
            assignments["oracle_margin"].append("0.05<=margin<0.10")
        else:
            assignments["oracle_margin"].append("margin>=0.10")
    return assignments


def _summarize_buckets(
    assignments: dict[str, list[str]],
    fixed_utilities: np.ndarray,
    mode_utilities: dict[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bucket_name, labels in assignments.items():
        bucket_rows: dict[str, Any] = {}
        for label in sorted(set(labels)):
            mask = np.array([value == label for value in labels], dtype=bool)
            if not np.any(mask):
                continue
            row: dict[str, Any] = {
                "n": int(mask.sum()),
                "fixed_mean_utility": float(fixed_utilities[mask].mean()),
            }
            for mode, utilities in mode_utilities.items():
                row[f"{mode}_mean_utility"] = float(utilities[mask].mean())
                row[f"{mode}_delta_vs_fixed"] = float(utilities[mask].mean() - fixed_utilities[mask].mean())
            bucket_rows[label] = row
        output[bucket_name] = bucket_rows
    return output


def main() -> None:
    args = parse_args()
    policy = load_soft_label_policy(args.model_dir)
    lambda_grid = parse_lambda_grid(args.lambda_grid or policy.stats.get("lambda_grid") or policy.lambda_grid)
    temperature = float(args.temperature if args.temperature is not None else policy.stats.get("temperature", 1.0))
    weight_mode = args.weight_mode or str(policy.stats.get("weight_mode", "none"))

    build_cfg = load_experiment_build_cfg(args.experiment, args.config_overrides)
    retrieval_cfg = dict(build_cfg.get("retrieval", {}) or {})
    chunk_cache = _resolve_cache(
        build_cfg,
        explicit_path=args.chunk_mmr_cache,
        split_name=args.split_name,
        cache_root=args.chunk_mmr_cache_root,
    )
    top_k = int(pick_retrieval_value(args.top_k, retrieval_cfg, "top_k", policy.stats.get("top_k", 5)))
    alpha_dense = float(pick_retrieval_value(args.alpha_dense, retrieval_cfg, "alpha_dense", 0.70))
    alpha_lexical = float(pick_retrieval_value(args.alpha_lexical, retrieval_cfg, "alpha_lexical", 0.20))
    alpha_bm25 = float(pick_retrieval_value(args.alpha_bm25, retrieval_cfg, "alpha_bm25", 0.10))

    ds = SoftLabelDataset.from_oracle_and_cache(
        oracle_jsonl=args.oracle_logprobs,
        chunk_cache_pkl=chunk_cache,
        lambda_grid=lambda_grid,
        temperature=temperature,
        weight_mode=weight_mode,
        top_k=top_k,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
        feature_mean=np.array(policy.stats["mean"], dtype=np.float32),
        feature_std=np.array(policy.stats["std"], dtype=np.float32),
        sample_limit=args.sample_limit,
    )
    probs = predict_policy_proba(policy, ds.features)
    modes = [x.strip() for x in args.inference_modes.split(",") if x.strip()]
    output_dir = Path(args.output_dir or (policy.model_dir / f"eval_{args.split_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_utilities = np.zeros(len(ds.event_ids), dtype=np.float32)
    records = ds.oracle_records or []
    for i, rec in enumerate(records):
        fixed_utilities[i] = _utility_at_lambda(
            rec,
            lambda_grid,
            args.fixed_lambda,
            mode=args.continuous_lookup,
        )[0]

    mode_utilities: dict[str, np.ndarray] = {}
    mode_chosen: dict[str, np.ndarray] = {}
    mode_lookup_lambdas: dict[str, np.ndarray] = {}
    summary_modes: dict[str, Any] = {}
    for mode in modes:
        chosen = select_lambdas_from_probs(
            probs,
            policy.lambda_grid,
            inference_mode=mode,
            sample_temperature=args.sample_temperature,
            random_seed=args.random_seed,
        )
        utilities = np.zeros(len(ds.event_ids), dtype=np.float32)
        lookup_lambdas = np.zeros(len(ds.event_ids), dtype=np.float32)
        for i, rec in enumerate(records):
            utility, lookup_lambda = _utility_at_lambda(
                rec,
                lambda_grid,
                float(chosen[i]),
                mode=args.continuous_lookup,
            )
            utilities[i] = utility
            lookup_lambdas[i] = lookup_lambda
        mode_utilities[mode] = utilities
        mode_chosen[mode] = chosen
        mode_lookup_lambdas[mode] = lookup_lambdas
        summary_modes[mode] = {
            "mean_utility": float(utilities.mean()) if utilities.size else 0.0,
            "delta_vs_fixed": float(utilities.mean() - fixed_utilities.mean()) if utilities.size else 0.0,
            "chosen_lambda_mean": float(chosen.mean()) if chosen.size else 0.0,
            "chosen_lambda_std": float(chosen.std()) if chosen.size else 0.0,
            "lookup_lambda_mean": float(lookup_lambdas.mean()) if lookup_lambdas.size else 0.0,
        }

    calibration = _target_calibration(probs, ds.soft_targets)
    margins = np.array([_oracle_margin(rec, lambda_grid) for rec in records], dtype=np.float32)
    bucket_summary = _summarize_buckets(
        _bucket_assignments(ds.raw_feature_dicts or [], margins),
        fixed_utilities,
        mode_utilities,
    )

    summary = {
        "model_dir": str(policy.model_dir),
        "model_type": policy.model_type,
        "split_name": args.split_name,
        "num_samples": len(ds.event_ids),
        "lambda_grid": [float(x) for x in lambda_grid.tolist()],
        "policy_lambda_grid": [float(x) for x in policy.lambda_grid.tolist()],
        "fixed_lambda": float(args.fixed_lambda),
        "fixed_mean_utility": float(fixed_utilities.mean()) if fixed_utilities.size else 0.0,
        "continuous_lookup": args.continuous_lookup,
        "modes": summary_modes,
        "calibration": calibration,
        "oracle_margin_mean": float(margins.mean()) if margins.size else 0.0,
    }
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2, ensure_ascii=False)
    with (output_dir / "eval_by_bucket.json").open("w", encoding="utf-8") as writer:
        json.dump(bucket_summary, writer, indent=2, ensure_ascii=False)

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as writer:
        for i, event_id in enumerate(ds.event_ids):
            row = {
                "event_id": event_id,
                "gold_label": records[i].get("gold_label", "") if i < len(records) else "",
                "fixed_utility": float(fixed_utilities[i]),
                "oracle_margin": float(margins[i]),
                "probs_by_lambda": {
                    f"{float(policy.lambda_grid[j]):.2f}": float(probs[i, j])
                    for j in range(len(policy.lambda_grid))
                },
            }
            for mode in modes:
                row[f"{mode}_chosen_lambda"] = float(mode_chosen[mode][i])
                row[f"{mode}_lookup_lambda"] = float(mode_lookup_lambdas[mode][i])
                row[f"{mode}_utility"] = float(mode_utilities[mode][i])
                row[f"{mode}_delta_vs_fixed"] = float(mode_utilities[mode][i] - fixed_utilities[i])
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_calibration_plot(calibration, output_dir / "calibration.png")

    print(f"Samples: {len(ds.event_ids)}", flush=True)
    print(f"Fixed λ={args.fixed_lambda:.2f}: mean utility={summary['fixed_mean_utility']:.5f}", flush=True)
    for mode, metrics in summary_modes.items():
        print(
            f"{mode}: mean utility={metrics['mean_utility']:.5f} "
            f"delta={metrics['delta_vs_fixed']:+.5f} "
            f"chosen_mean={metrics['chosen_lambda_mean']:.3f}",
            flush=True,
        )
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
