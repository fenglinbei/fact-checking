"""Train a tabular soft-label lambda policy for RL-MMR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from fact_checking.learned_lambda.cache_utils import (
    load_experiment_build_cfg,
    pick_retrieval_value,
    resolve_chunk_mmr_cache_path,
)
from fact_checking.rl_mmr.soft_label_dataset import (
    SoftLabelDataset,
    parse_lambda_grid,
)
from fact_checking.rl_mmr.soft_label_selector import (
    SoftLabelMLP,
    SoftLabelRegressorEnsemble,
    _normalize_prob_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train soft-label lambda policy from oracle utility curves.")
    p.add_argument("--oracle-logprobs", type=str, required=True, help="Train oracle JSONL from compute_oracle_lambda.py")
    p.add_argument("--val-oracle-logprobs", type=str, default=None, help="Optional val oracle JSONL")
    p.add_argument("--chunk-mmr-cache", type=str, default=None, help="Train Chunk-MMR cache pickle")
    p.add_argument("--val-chunk-mmr-cache", type=str, default=None, help="Optional val Chunk-MMR cache pickle")
    p.add_argument("--chunk-mmr-cache-root", type=str, default="outputs/cache/chunk_mmr")
    p.add_argument("--experiment", type=str, default="b3_mmr_topk_sweep_1024")
    p.add_argument("--config-overrides", nargs="*", default=[])
    p.add_argument("--split-name", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--val-split-name", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--lambda-grid", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--weight-mode", type=str, default="margin", choices=["margin", "gap", "none"])
    p.add_argument("--model-type", type=str, default="lightgbm", choices=["lr", "lightgbm", "mlp"])
    p.add_argument("--output-dir", type=str, default="outputs/rl_mmr/soft_label/lightgbm")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--alpha-dense", type=float, default=None)
    p.add_argument("--alpha-lexical", type=float, default=None)
    p.add_argument("--alpha-bm25", type=float, default=None)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-limit", type=int, default=None, help="Optional smoke-test cap")
    p.add_argument("--C-grid", type=str, default="0.01,0.1,1.0,10.0", help="LR C candidates")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def _log(message: str, *, show_progress: bool) -> None:
    if show_progress:
        tqdm.write(message)
    else:
        print(message, flush=True)


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


def _build_dataset(
    *,
    oracle_logprobs: str,
    chunk_cache: Path,
    lambda_grid: np.ndarray,
    temperature: float,
    weight_mode: str,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
    sample_limit: int | None = None,
) -> SoftLabelDataset:
    if not chunk_cache.exists():
        raise FileNotFoundError(f"Chunk-MMR cache not found: {chunk_cache}")
    return SoftLabelDataset.from_oracle_and_cache(
        oracle_jsonl=oracle_logprobs,
        chunk_cache_pkl=chunk_cache,
        lambda_grid=lambda_grid,
        temperature=temperature,
        weight_mode=weight_mode,
        top_k=top_k,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
        feature_mean=feature_mean,
        feature_std=feature_std,
        sample_limit=sample_limit,
    )


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least two samples to create a validation split.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = min(max(1, int(round(n * val_fraction))), n - 1)
    return indices[n_val:], indices[:n_val]


def _subset_arrays(ds: SoftLabelDataset, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return ds.features[indices], ds.soft_targets[indices], ds.sample_weights[indices]


def _ensure_positive_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if weights.size == 0:
        return weights
    if float(weights.sum()) <= 1e-8:
        return np.ones_like(weights, dtype=np.float32)
    return weights


def _kl_divergence(targets: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float32), 1e-8, 1.0)
    q = np.clip(np.asarray(targets, dtype=np.float32), 1e-8, 1.0)
    return float(np.mean(np.sum(q * (np.log(q) - np.log(p)), axis=1)))


def _ece(targets: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    p = _normalize_prob_rows(probs)
    pred = np.argmax(p, axis=1)
    conf = p[np.arange(len(p)), pred]
    soft_acc = targets[np.arange(len(targets)), pred]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi >= 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(soft_acc[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def _distribution_summary(lambda_grid: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    p = _normalize_prob_rows(probs)
    argmax = np.argmax(p, axis=1)
    counts = {
        f"{float(lambda_grid[i]):.2f}": int((argmax == i).sum())
        for i in range(len(lambda_grid))
    }
    entropy = -(p * np.log(np.clip(p, 1e-8, 1.0))).sum(axis=1)
    max_count = max(counts.values()) if counts else 0
    return {
        "argmax_counts": counts,
        "argmax_max_fraction": float(max_count / max(len(p), 1)),
        "pred_entropy_mean": float(entropy.mean()) if entropy.size else 0.0,
        "pred_entropy_std": float(entropy.std()) if entropy.size else 0.0,
        "expected_lambda_mean": float((p * lambda_grid[None, :]).sum(axis=1).mean()) if len(p) else 0.0,
        "expected_lambda_std": float((p * lambda_grid[None, :]).sum(axis=1).std()) if len(p) else 0.0,
    }


def _eval_metrics(lambda_grid: np.ndarray, targets: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    probs = _normalize_prob_rows(probs)
    metrics = {
        "kl_divergence": _kl_divergence(targets, probs),
        "ece": _ece(targets, probs),
    }
    metrics.update(_distribution_summary(lambda_grid, probs))
    return metrics


def _expanded_soft_label_training_set(
    x: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, n_classes = targets.shape
    x_rep = np.repeat(x, n_classes, axis=0)
    y_rep = np.tile(np.arange(n_classes, dtype=np.int64), n)
    w_rep = (sample_weights[:, None] * targets).reshape(-1).astype(np.float32)
    keep = w_rep > 1e-10
    return x_rep[keep], y_rep[keep], w_rep[keep]


def _train_lr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lambda_grid: np.ndarray,
    c_grid: list[float],
    max_iter: int,
    show_progress: bool,
) -> tuple[Any, dict[str, Any], np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    x_rep, y_rep, w_rep = _expanded_soft_label_training_set(x_train, y_train, w_train)
    best_model = None
    best_metrics: dict[str, Any] | None = None
    best_probs: np.ndarray | None = None
    for c_value in c_grid:
        model = LogisticRegression(C=float(c_value), max_iter=max_iter, solver="lbfgs")
        model.fit(x_rep, y_rep, sample_weight=w_rep)
        probs = _normalize_prob_rows(model.predict_proba(x_val))
        metrics = _eval_metrics(lambda_grid, y_val, probs)
        metrics["C"] = float(c_value)
        _log(f"LR C={c_value:g}: val_KL={metrics['kl_divergence']:.5f}", show_progress=show_progress)
        if best_metrics is None or metrics["kl_divergence"] < best_metrics["kl_divergence"]:
            best_model = model
            best_metrics = metrics
            best_probs = probs
    assert best_model is not None and best_metrics is not None and best_probs is not None
    return best_model, best_metrics, best_probs


def _make_gbdt_regressor(args: argparse.Namespace) -> tuple[type, dict[str, Any], str]:
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor, {
            "n_estimators": int(args.n_estimators),
            "num_leaves": int(args.num_leaves),
            "learning_rate": float(args.learning_rate),
            "random_state": int(args.seed),
            "verbose": -1,
        }, "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor, {
            "max_iter": int(args.n_estimators),
            "max_leaf_nodes": int(args.num_leaves),
            "learning_rate": float(args.learning_rate),
            "random_state": int(args.seed),
        }, "sklearn_hist_gradient_boosting"


def _train_gbdt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    lambda_grid: np.ndarray,
    args: argparse.Namespace,
    *,
    show_progress: bool,
) -> tuple[SoftLabelRegressorEnsemble, dict[str, Any], np.ndarray]:
    estimator_cls, estimator_kwargs, backend = _make_gbdt_regressor(args)
    estimators = []
    for j in tqdm(
        range(y_train.shape[1]),
        desc="train gbdt heads",
        unit="class",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        estimator = estimator_cls(**estimator_kwargs)
        estimator.fit(x_train, y_train[:, j], sample_weight=w_train)
        estimators.append(estimator)
    model = SoftLabelRegressorEnsemble(estimators, backend=backend)
    probs = model.predict_proba(x_val)
    metrics = _eval_metrics(lambda_grid, y_val, probs)
    metrics["backend"] = backend
    _log(f"GBDT backend={backend}: val_KL={metrics['kl_divergence']:.5f}", show_progress=show_progress)
    return model, metrics, probs


def _weighted_kl_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_sample = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    denom = weights.sum().clamp(min=1e-8)
    return (per_sample * weights).sum() / denom


def _predict_mlp(model: SoftLabelMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x.astype(np.float32, copy=False)))
        return torch.softmax(logits, dim=-1).numpy()


def _train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    lambda_grid: np.ndarray,
    args: argparse.Namespace,
    *,
    show_progress: bool,
) -> tuple[SoftLabelMLP, dict[str, Any], np.ndarray]:
    torch.manual_seed(int(args.seed))
    model = SoftLabelMLP(
        input_dim=x_train.shape[1],
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        n_classes=y_train.shape[1],
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train.astype(np.float32, copy=False)),
            torch.from_numpy(y_train.astype(np.float32, copy=False)),
            torch.from_numpy(w_train.astype(np.float32, copy=False)),
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = None
    best_metrics: dict[str, Any] | None = None
    best_probs: np.ndarray | None = None
    patience = 0
    epoch_iter = tqdm(
        range(1, int(args.epochs) + 1),
        desc="train mlp",
        unit="epoch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb, wb in loader:
            logits = model(xb)
            loss = _weighted_kl_loss(logits, yb, wb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        probs = _predict_mlp(model, x_val)
        metrics = _eval_metrics(lambda_grid, y_val, probs)
        improved = best_metrics is None or metrics["kl_divergence"] < best_metrics["kl_divergence"]
        if improved:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            best_probs = probs
            patience = 0
        else:
            patience += 1
        if show_progress:
            epoch_iter.set_postfix({
                "train_loss": f"{total_loss / max(n_batches, 1):.5f}",
                "val_kl": f"{metrics['kl_divergence']:.5f}",
                "best": f"{best_metrics['kl_divergence']:.5f}",
                "patience": patience,
            })
        if epoch == 1 or epoch % 10 == 0:
            _log(
                f"Epoch {epoch:3d}: train_loss={total_loss / max(n_batches, 1):.5f} "
                f"val_KL={metrics['kl_divergence']:.5f}",
                show_progress=show_progress,
            )
        if patience >= int(args.patience):
            _log(f"Early stopping at epoch {epoch}", show_progress=show_progress)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    assert best_metrics is not None and best_probs is not None
    return model, best_metrics, best_probs


def _save_artifacts(
    *,
    model: Any,
    model_type: str,
    output_dir: Path,
    train_ds: SoftLabelDataset,
    args: argparse.Namespace,
    train_cache: Path,
    val_cache: Path | None,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_type == "mlp":
        torch.save(model.state_dict(), output_dir / "model.pt")
    else:
        import joblib

        joblib.dump(model, output_dir / "model.joblib")

    stats = {
        "feature_mode": "soft_label_tabular",
        "feature_names": train_ds.feature_names,
        "mean": train_ds.feature_mean.tolist(),
        "std": train_ds.feature_std.tolist(),
        "input_dim": int(train_ds.features.shape[1]),
        "model_type": model_type,
        "lambda_grid": [float(x) for x in train_ds.lambda_grid.tolist()],
        "temperature": float(args.temperature),
        "weight_mode": args.weight_mode,
        "top_k": int(args.top_k_resolved),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "train_chunk_mmr_cache": str(train_cache),
        "val_chunk_mmr_cache": str(val_cache) if val_cache else None,
        "experiment": args.experiment,
        "config_overrides": args.config_overrides,
    }
    with (output_dir / "feature_stats.json").open("w", encoding="utf-8") as writer:
        json.dump(stats, writer, indent=2)

    meta = {
        "model_type": model_type,
        "n_train_total": int(train_ds.features.shape[0]),
        "lambda_grid": stats["lambda_grid"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "oracle_logprobs": args.oracle_logprobs,
        "val_oracle_logprobs": args.val_oracle_logprobs,
        "train_chunk_mmr_cache": str(train_cache),
        "val_chunk_mmr_cache": str(val_cache) if val_cache else None,
        "temperature": float(args.temperature),
        "weight_mode": args.weight_mode,
        "seed": int(args.seed),
    }
    with (output_dir / "training_meta.json").open("w", encoding="utf-8") as writer:
        json.dump(meta, writer, indent=2)


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    lambda_grid = parse_lambda_grid(args.lambda_grid)

    build_cfg = load_experiment_build_cfg(args.experiment, args.config_overrides)
    retrieval_cfg = dict(build_cfg.get("retrieval", {}) or {})
    train_cache = _resolve_cache(
        build_cfg,
        explicit_path=args.chunk_mmr_cache,
        split_name=args.split_name,
        cache_root=args.chunk_mmr_cache_root,
    )
    val_cache = None
    if args.val_oracle_logprobs:
        val_cache = _resolve_cache(
            build_cfg,
            explicit_path=args.val_chunk_mmr_cache,
            split_name=args.val_split_name,
            cache_root=args.chunk_mmr_cache_root,
        )

    top_k = int(pick_retrieval_value(args.top_k, retrieval_cfg, "top_k", 5))
    alpha_dense = float(pick_retrieval_value(args.alpha_dense, retrieval_cfg, "alpha_dense", 0.70))
    alpha_lexical = float(pick_retrieval_value(args.alpha_lexical, retrieval_cfg, "alpha_lexical", 0.20))
    alpha_bm25 = float(pick_retrieval_value(args.alpha_bm25, retrieval_cfg, "alpha_bm25", 0.10))
    args.top_k_resolved = top_k

    _log(f"Loaded build config from experiment={args.experiment}", show_progress=show_progress)
    _log(f"train_chunk_mmr_cache={train_cache}", show_progress=show_progress)
    _log(
        f"lambda_grid={','.join(f'{x:.2f}' for x in lambda_grid)} top_k={top_k} "
        f"alpha=({alpha_dense:.2f},{alpha_lexical:.2f},{alpha_bm25:.2f})",
        show_progress=show_progress,
    )

    train_ds = _build_dataset(
        oracle_logprobs=args.oracle_logprobs,
        chunk_cache=train_cache,
        lambda_grid=lambda_grid,
        temperature=args.temperature,
        weight_mode=args.weight_mode,
        top_k=top_k,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
        sample_limit=args.sample_limit,
    )
    _log(
        f"Train dataset: n={len(train_ds.event_ids)} dim={train_ds.features.shape[1]} "
        f"weight_mean={train_ds.sample_weights.mean():.3f}",
        show_progress=show_progress,
    )

    output_dir = Path(args.output_dir)
    train_ds.save_npz(output_dir / "train_dataset.npz")

    if args.val_oracle_logprobs:
        assert val_cache is not None
        val_ds = _build_dataset(
            oracle_logprobs=args.val_oracle_logprobs,
            chunk_cache=val_cache,
            lambda_grid=lambda_grid,
            temperature=args.temperature,
            weight_mode=args.weight_mode,
            top_k=top_k,
            alpha_dense=alpha_dense,
            alpha_lexical=alpha_lexical,
            alpha_bm25=alpha_bm25,
            feature_mean=train_ds.feature_mean,
            feature_std=train_ds.feature_std,
            sample_limit=args.sample_limit,
        )
        val_ds.save_npz(output_dir / "val_dataset.npz")
        x_train, y_train, w_train = train_ds.features, train_ds.soft_targets, train_ds.sample_weights
        x_val, y_val, _w_val = val_ds.features, val_ds.soft_targets, val_ds.sample_weights
        _log(f"Val dataset: n={len(val_ds.event_ids)}", show_progress=show_progress)
    else:
        train_idx, val_idx = _split_indices(len(train_ds.event_ids), args.val_fraction, args.seed)
        x_train, y_train, w_train = _subset_arrays(train_ds, train_idx)
        x_val, y_val, _w_val = _subset_arrays(train_ds, val_idx)
        _log(f"Train/val split from train: train={len(train_idx)} val={len(val_idx)}", show_progress=show_progress)

    w_train = _ensure_positive_weights(w_train)
    train_prior_probs = np.average(y_train, axis=0, weights=w_train)
    train_prior_probs = _normalize_prob_rows(train_prior_probs[None, :])
    train_metrics = _eval_metrics(lambda_grid, y_train, np.repeat(train_prior_probs, len(y_train), axis=0))

    if args.model_type == "lr":
        c_grid = [float(x.strip()) for x in args.C_grid.split(",") if x.strip()]
        model, val_metrics, _val_probs = _train_lr(
            x_train,
            y_train,
            w_train,
            x_val,
            y_val,
            lambda_grid=lambda_grid,
            c_grid=c_grid,
            max_iter=int(args.max_iter),
            show_progress=show_progress,
        )
    elif args.model_type == "lightgbm":
        model, val_metrics, _val_probs = _train_gbdt(
            x_train,
            y_train,
            w_train,
            x_val,
            y_val,
            lambda_grid,
            args,
            show_progress=show_progress,
        )
    else:
        model, val_metrics, _val_probs = _train_mlp(
            x_train,
            y_train,
            w_train,
            x_val,
            y_val,
            lambda_grid,
            args,
            show_progress=show_progress,
        )

    _log(
        f"Best val: KL={val_metrics['kl_divergence']:.5f} ECE={val_metrics['ece']:.5f} "
        f"entropy={val_metrics['pred_entropy_mean']:.3f}",
        show_progress=show_progress,
    )
    _save_artifacts(
        model=model,
        model_type=args.model_type,
        output_dir=output_dir,
        train_ds=train_ds,
        args=args,
        train_cache=train_cache,
        val_cache=val_cache,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    _log(f"Saved soft-label policy to {output_dir}", show_progress=show_progress)


if __name__ == "__main__":
    main()
