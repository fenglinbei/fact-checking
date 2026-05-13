"""Step 3: Train the λ predictor from oracle λ labels and chunk embeddings.

Usage:
    PYTHONPATH=src python scripts/learned_lambda/train_predictor.py \
        --oracle-lambdas outputs/learned_lambda/oracle_lambda_train.jsonl \
        --experiment b3_mmr_topk_sweep_1024 \
        --output-dir outputs/learned_lambda/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from fact_checking.build.candidates import _load_pickle
from fact_checking.learned_lambda.cache_utils import (
    load_experiment_build_cfg,
    pick_retrieval_value,
    resolve_chunk_mmr_cache_path,
)
from fact_checking.learned_lambda.embedding_features import (
    CHUNK_EMBEDDING_FEATURE_MODE,
    build_matched_chunk_embedding_arrays,
)
from fact_checking.learned_lambda.predictor import ChunkEmbeddingLambdaEncoder, save_predictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train λ predictor from chunk embeddings.")
    p.add_argument("--oracle-lambdas", type=str, required=True, help="JSONL from compute_oracle_lambda.py")
    p.add_argument("--chunk-mmr-cache", type=str, default=None, help="Chunk-MMR cache pickle")
    p.add_argument("--chunk-mmr-cache-root", type=str, default="outputs/cache/chunk_mmr")
    p.add_argument("--experiment", type=str, default="b3_mmr_topk_sweep_1024")
    p.add_argument("--config-overrides", nargs="*", default=[])
    p.add_argument("--split-name", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--candidate-top-k",
        type=int,
        default=None,
        help="Number of hybrid-ranked chunk candidates to feed the predictor. Omit to use the full chunk pool.",
    )
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument(
        "--objective",
        type=str,
        default="regression",
        choices=["regression", "classification", "soft_classification"],
        help="Training objective. Classification treats oracle λ as a discrete grid label.",
    )
    p.add_argument("--regression-loss", type=str, default="mse", choices=["mse", "huber"])
    p.add_argument("--huber-delta", type=float, default=0.1)
    p.add_argument(
        "--lambda-grid",
        type=str,
        default="auto",
        help="Comma-separated λ grid for classification objectives, or 'auto' to infer from oracle records.",
    )
    p.add_argument("--softmax-temperature", type=float, default=1.0)
    p.add_argument("--alpha-dense", type=float, default=None)
    p.add_argument("--alpha-lexical", type=float, default=None)
    p.add_argument("--alpha-bm25", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return p.parse_args()


def _log(message: str, *, show_progress: bool) -> None:
    if show_progress:
        tqdm.write(message)
    else:
        print(message, flush=True)


def _resolve_lambda_grid(grid_arg: str, oracle_records: list[dict]) -> np.ndarray:
    if grid_arg.strip().lower() != "auto":
        values = [float(x.strip()) for x in grid_arg.split(",") if x.strip()]
        if len(values) < 2:
            raise ValueError("--lambda-grid must contain at least two values for classification objectives.")
        return np.array(sorted(set(values)), dtype=np.float32)

    logprob_values: set[float] = set()
    target_values: set[float] = set()
    for rec in oracle_records:
        target_values.add(float(rec["oracle_lambda"]))
        lp_by_lambda = rec.get("logprobs_by_lambda")
        if isinstance(lp_by_lambda, dict):
            for key in lp_by_lambda:
                logprob_values.add(float(key))

    values = sorted(logprob_values or target_values)
    if len(values) < 2:
        values = [i / 10 for i in range(11)]
    return np.array(values, dtype=np.float32)


def _nearest_grid_indices(targets: np.ndarray, lambda_grid: np.ndarray) -> np.ndarray:
    distances = np.abs(targets[:, None] - lambda_grid[None, :])
    return np.argmin(distances, axis=1).astype(np.int64)


def _softmax_np(values: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    scaled = values / temp
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full_like(values, 1.0 / len(values), dtype=np.float32)
    return (exp_values / total).astype(np.float32)


def _soft_targets_from_oracle(
    oracle_records: list[dict],
    hard_class_targets: np.ndarray,
    lambda_grid: np.ndarray,
    temperature: float,
) -> np.ndarray:
    soft_targets = np.zeros((len(oracle_records), len(lambda_grid)), dtype=np.float32)
    rounded_grid = [round(float(v), 6) for v in lambda_grid]
    for i, rec in enumerate(oracle_records):
        lp_by_lambda = rec.get("logprobs_by_lambda")
        if not isinstance(lp_by_lambda, dict) or not lp_by_lambda:
            soft_targets[i, hard_class_targets[i]] = 1.0
            continue

        lp_float = {round(float(k), 6): float(v) for k, v in lp_by_lambda.items()}
        values = np.array([lp_float.get(v, -100.0) for v in rounded_grid], dtype=np.float32)
        if np.all(values <= -99.0):
            soft_targets[i, hard_class_targets[i]] = 1.0
        else:
            soft_targets[i] = _softmax_np(values, temperature)
    return soft_targets


def _compute_objective_loss(
    model: ChunkEmbeddingLambdaEncoder,
    claim_emb: torch.Tensor,
    candidate_emb: torch.Tensor,
    candidate_mask: torch.Tensor,
    y_obj: torch.Tensor,
    y_float: torch.Tensor,
    *,
    objective: str,
    regression_loss: str,
    huber_delta: float,
) -> torch.Tensor:
    if objective == "regression":
        pred = model(claim_emb, candidate_emb, candidate_mask)
        if regression_loss == "huber":
            return F.huber_loss(pred, y_float, delta=huber_delta)
        return F.mse_loss(pred, y_float)

    logits = model.forward_logits(claim_emb, candidate_emb, candidate_mask)
    if objective == "soft_classification":
        return -(y_obj * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    return F.cross_entropy(logits, y_obj.long())


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    build_cfg = load_experiment_build_cfg(args.experiment, args.config_overrides)
    retrieval_cfg = dict(build_cfg.get("retrieval", {}) or {})
    chunk_mmr_cache = (
        Path(args.chunk_mmr_cache)
        if args.chunk_mmr_cache
        else resolve_chunk_mmr_cache_path(
            build_cfg,
            split_name=args.split_name,
            cache_root=args.chunk_mmr_cache_root,
        )
    )
    candidate_top_k = int(args.candidate_top_k) if args.candidate_top_k is not None else None
    retrieval_top_k = int(pick_retrieval_value(None, retrieval_cfg, "top_k", 16))
    alpha_dense = float(pick_retrieval_value(args.alpha_dense, retrieval_cfg, "alpha_dense", 0.70))
    alpha_lexical = float(pick_retrieval_value(args.alpha_lexical, retrieval_cfg, "alpha_lexical", 0.20))
    alpha_bm25 = float(pick_retrieval_value(args.alpha_bm25, retrieval_cfg, "alpha_bm25", 0.10))
    if not chunk_mmr_cache.exists():
        raise FileNotFoundError(
            f"Chunk-MMR cache not found: {chunk_mmr_cache}. "
            "Run scripts/learned_lambda/run_generate_oracle_prompts.sh first, "
            "or pass --chunk-mmr-cache."
        )

    _log(f"Loaded build config from experiment={args.experiment}", show_progress=show_progress)
    _log(f"chunk_mmr_cache={chunk_mmr_cache}", show_progress=show_progress)
    _log(
        f"candidate_pool={'full' if candidate_top_k is None else f'top_{candidate_top_k}'}, "
        f"prompt_top_k={retrieval_top_k}, alpha_dense={alpha_dense}, "
        f"alpha_lexical={alpha_lexical}, alpha_bm25={alpha_bm25}",
        show_progress=show_progress,
    )

    # Load oracle λ
    oracle_by_eid: dict[str, dict] = {}
    with open(args.oracle_lambdas) as f:
        for line in f:
            rec = json.loads(line.strip())
            rec["oracle_lambda"] = float(rec["oracle_lambda"])
            oracle_by_eid[rec["event_id"]] = rec
    _log(f"Loaded {len(oracle_by_eid)} oracle λ values", show_progress=show_progress)

    # Load ChunkMMR cache and construct embedding tensors
    chunk_samples = _load_pickle(chunk_mmr_cache)
    _log(f"Loaded {len(chunk_samples)} ChunkMMR samples", show_progress=show_progress)

    arrays, targets, oracle_records_list, skipped = build_matched_chunk_embedding_arrays(
        tqdm(
            chunk_samples,
            desc="build embedding tensors",
            unit="sample",
            dynamic_ncols=True,
            disable=not show_progress,
        ),
        oracle_by_eid,
        candidate_top_k=candidate_top_k,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
    )
    if skipped > 0:
        _log(f"Skipped {skipped} samples without oracle λ", show_progress=show_progress)

    embedding_dim = int(arrays["claim_emb"].shape[1])
    candidate_capacity = int(arrays["candidate_emb"].shape[1])
    candidate_counts = arrays["candidate_counts"]
    n_nonempty = int((arrays["candidate_mask"].sum(axis=1) > 0).sum())
    _log(
        f"Dataset: {len(targets)} samples, embedding_dim={embedding_dim}, "
        f"candidate_capacity={candidate_capacity}, "
        f"candidate_count_mean={candidate_counts.mean():.1f}, "
        f"candidate_count_max={candidate_counts.max()}, nonempty={n_nonempty}",
        show_progress=show_progress,
    )
    _log(f"Target λ: mean={targets.mean():.3f}, std={targets.std():.3f}", show_progress=show_progress)

    lambda_grid: np.ndarray | None = None
    objective_targets: np.ndarray
    if args.objective == "regression":
        objective_targets = targets
    else:
        lambda_grid = _resolve_lambda_grid(args.lambda_grid, oracle_records_list)
        hard_class_targets = _nearest_grid_indices(targets, lambda_grid)
        if args.objective == "soft_classification":
            objective_targets = _soft_targets_from_oracle(
                oracle_records_list,
                hard_class_targets,
                lambda_grid,
                args.softmax_temperature,
            )
        else:
            objective_targets = hard_class_targets
        _log(
            f"Lambda grid: {', '.join(f'{x:.2f}' for x in lambda_grid)}",
            show_progress=show_progress,
        )

    # Train / val split
    n = len(targets)
    indices = rng.permutation(n)
    n_val = max(1, int(n * args.val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    claim_train = torch.from_numpy(arrays["claim_emb"][train_idx])
    cand_train = torch.from_numpy(arrays["candidate_emb"][train_idx])
    mask_train = torch.from_numpy(arrays["candidate_mask"][train_idx])
    y_train = torch.from_numpy(objective_targets[train_idx])
    y_train_float = torch.from_numpy(targets[train_idx])
    claim_val = torch.from_numpy(arrays["claim_emb"][val_idx])
    cand_val = torch.from_numpy(arrays["candidate_emb"][val_idx])
    mask_val = torch.from_numpy(arrays["candidate_mask"][val_idx])
    y_val = torch.from_numpy(objective_targets[val_idx])
    y_val_float = torch.from_numpy(targets[val_idx])
    _log(f"Train: {len(train_idx)}, Val: {len(val_idx)}", show_progress=show_progress)

    train_loader = DataLoader(
        TensorDataset(claim_train, cand_train, mask_train, y_train, y_train_float),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Model
    model_type = "regression"
    if args.objective != "regression":
        model_type = "classifier"
    model = ChunkEmbeddingLambdaEncoder(
        embedding_dim=embedding_dim,
        encoder_dim=args.hidden_dim,
        dropout=args.dropout,
        lambda_grid=lambda_grid if args.objective != "regression" else None,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_val_objective_loss = float("inf")
    best_state = None
    patience_counter = 0

    epoch_progress = tqdm(
        range(1, args.epochs + 1),
        desc="train predictor",
        unit="epoch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for epoch in epoch_progress:
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        batch_iter = tqdm(
            train_loader,
            desc=f"epoch {epoch}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress or len(train_loader) <= 1,
        )
        for claim_b, cand_b, mask_b, yb, yb_float in batch_iter:
            loss = _compute_objective_loss(
                model,
                claim_b,
                cand_b,
                mask_b,
                yb,
                yb_float,
                objective=args.objective,
                regression_loss=args.regression_loss,
                huber_delta=args.huber_delta,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(claim_val, cand_val, mask_val)
            val_mse = mse_loss_fn(val_pred, y_val_float).item()
            val_objective_loss = _compute_objective_loss(
                model,
                claim_val,
                cand_val,
                mask_val,
                y_val,
                y_val_float,
                objective=args.objective,
                regression_loss=args.regression_loss,
                huber_delta=args.huber_delta,
            ).item()

        train_loss = epoch_loss / max(n_batches, 1)
        improved = val_mse < best_val_mse
        if improved:
            best_val_mse = val_mse
            best_val_objective_loss = val_objective_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if show_progress:
            epoch_progress.set_postfix({
                "train_loss": f"{train_loss:.5f}",
                "val_mse": f"{val_mse:.5f}",
                "best": f"{best_val_mse:.5f}",
                "patience": patience_counter,
            })
        if epoch % 10 == 0 or epoch == 1:
            _log(
                f"Epoch {epoch:3d}: train_loss={train_loss:.5f}  "
                f"val_loss={val_objective_loss:.5f}  val_mse={val_mse:.5f}",
                show_progress=show_progress,
            )

        if not improved and patience_counter >= args.patience:
            _log(
                f"Early stopping at epoch {epoch} (best val MSE={best_val_mse:.5f})",
                show_progress=show_progress,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on val set
    model.eval()
    with torch.no_grad():
        val_pred = model(claim_val, cand_val, mask_val).numpy()
    val_targets = y_val_float.numpy()
    mae = float(np.mean(np.abs(val_pred - val_targets)))
    rmse = float(np.sqrt(np.mean((val_pred - val_targets) ** 2)))
    _log(
        f"\nFinal val metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, best_MSE={best_val_mse:.5f}",
        show_progress=show_progress,
    )

    # Save
    output_dir = Path(args.output_dir)
    save_predictor(
        model,
        None,
        None,
        output_dir,
        model_type=model_type,
        lambda_grid=lambda_grid,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        feature_mode=CHUNK_EMBEDDING_FEATURE_MODE,
        feature_names=["claim_emb", "candidate_emb", "candidate_mask"],
        input_dim=embedding_dim,
        embedding_dim=embedding_dim,
        candidate_top_k=candidate_top_k,
        retrieval_config={
            "experiment": args.experiment,
            "chunk_mmr_cache": str(chunk_mmr_cache),
            "candidate_top_k": candidate_top_k,
            "prompt_top_k": retrieval_top_k,
            "alpha_dense": alpha_dense,
            "alpha_lexical": alpha_lexical,
            "alpha_bm25": alpha_bm25,
        },
    )
    _log(f"Saved predictor to {output_dir}", show_progress=show_progress)

    # Save training metadata
    meta = {
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "best_val_mse": best_val_mse,
        "best_val_objective_loss": best_val_objective_loss,
        "val_mae": mae,
        "val_rmse": rmse,
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
        "feature_mode": CHUNK_EMBEDDING_FEATURE_MODE,
        "feature_names": ["claim_emb", "candidate_emb", "candidate_mask"],
        "embedding_dim": embedding_dim,
        "candidate_top_k": candidate_top_k,
        "candidate_capacity": candidate_capacity,
        "candidate_count_mean": float(candidate_counts.mean()),
        "candidate_count_max": int(candidate_counts.max()),
        "prompt_top_k": retrieval_top_k,
        "chunk_mmr_cache": str(chunk_mmr_cache),
        "experiment": args.experiment,
        "config_overrides": args.config_overrides,
        "objective": args.objective,
        "model_type": model_type,
        "lambda_grid": lambda_grid.tolist() if lambda_grid is not None else None,
        "regression_loss": args.regression_loss,
        "huber_delta": args.huber_delta,
        "softmax_temperature": args.softmax_temperature,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "seed": args.seed,
    }
    with (output_dir / "training_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    _log(f"Saved training metadata to {output_dir / 'training_meta.json'}", show_progress=show_progress)


if __name__ == "__main__":
    main()
