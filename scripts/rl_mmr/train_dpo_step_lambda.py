"""Train DPO step-wise λ policy from preference pairs.

Usage:
    PYTHONPATH=src python scripts/rl_mmr/train_dpo_step_lambda.py \\
        --train-pairs outputs/rl_mmr/dpo_stepwise/preference_pairs/train_pairs.npz \\
        --val-pairs outputs/rl_mmr/dpo_stepwise/preference_pairs/val_pairs.npz \\
        --output-dir outputs/rl_mmr/dpo_stepwise/checkpoints \\
        --beta 1.0 --lr 1e-3 --epochs 200 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from fact_checking.rl_mmr.dpo_policy import (
    FixedReferencePolicy,
    StepLambdaPolicy,
    argmax_distribution,
    dpo_loss,
    evaluate_policy_metrics,
    policy_entropy,
)
from fact_checking.rl_mmr.trajectory import LAMBDA_GRID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DPO step-wise λ policy.")
    p.add_argument("--train-pairs", type=str, required=True)
    p.add_argument("--val-pairs", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--lambda-grid", nargs="*", type=float, default=LAMBDA_GRID)
    p.add_argument("--ref-center", type=float, default=0.7, help="Reference policy center λ.")
    p.add_argument("--ref-temperature", type=float, default=0.3,
                   help="Reference policy temperature (lower = sharper).")
    p.add_argument("--beta", type=float, default=1.0, help="DPO β parameter.")
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[64, 32])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def _log(msg: str, show_progress: bool) -> None:
    if show_progress:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _load_pairs(npz_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(npz_path, allow_pickle=True)
    return (
        torch.from_numpy(data["win_features"].astype(np.float32, copy=False)),
        torch.from_numpy(data["win_lambdas"].astype(np.int64, copy=False)),
        torch.from_numpy(data["lose_features"].astype(np.float32, copy=False)),
        torch.from_numpy(data["lose_lambdas"].astype(np.int64, copy=False)),
    )


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    torch.manual_seed(args.seed)

    lambda_grid = [float(x) for x in args.lambda_grid]
    n_actions = len(lambda_grid)

    # Load data
    train_wf, train_wl, train_lf, train_ll = _load_pairs(args.train_pairs)
    val_wf, val_wl, val_lf, val_ll = _load_pairs(args.val_pairs)

    n_train_pairs, K, D = train_wf.shape
    n_val_pairs = val_wf.shape[0]
    _log(f"Train pairs: {n_train_pairs}  Val pairs: {n_val_pairs}  K={K}  D={D}  actions={n_actions}", show_progress)

    # Create dataloader
    train_dataset = TensorDataset(train_wf, train_wl, train_lf, train_ll)
    val_dataset = TensorDataset(val_wf, val_wl, val_lf, val_ll)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create models
    policy = StepLambdaPolicy(
        input_dim=D, hidden_dims=args.hidden_dims,
        dropout=args.dropout, n_actions=n_actions,
    )
    ref_policy = FixedReferencePolicy(
        lambda_grid=lambda_grid,
        center=args.ref_center, temperature=args.ref_temperature,
    )
    _log(f"Policy: {sum(p.numel() for p in policy.parameters())} params", show_progress)
    _log(f"Reference: center={args.ref_center} temp={args.ref_temperature}", show_progress)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    patience_counter = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_iter = tqdm(
        range(1, args.epochs + 1), desc="train dpo", unit="epoch",
        dynamic_ncols=True, disable=not show_progress,
    )

    train_losses: list[float] = []
    val_metrics_list: list[dict] = []

    for epoch in epoch_iter:
        # Training
        policy.train()
        epoch_loss = 0.0
        n_batches = 0
        for wf_b, wl_b, lf_b, ll_b in train_loader:
            loss = dpo_loss(
                policy, ref_policy, wf_b, wl_b, lf_b, ll_b, beta=args.beta,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        val_metrics = evaluate_policy_metrics(
            policy, ref_policy, val_wf, val_wl, val_lf, val_ll, beta=args.beta,
        )

        # Argmax distribution (on val winner features only)
        val_argmax = argmax_distribution(policy, val_wf.reshape(-1, D), n_actions)
        val_metrics["argmax_counts"] = {
            f"{lambda_grid[i]:.2f}": int(val_argmax[i]) for i in range(n_actions)
        }
        val_metrics["argmax_max_frac"] = float(val_argmax.max() / max(val_argmax.sum(), 1))

        val_metrics_list.append(val_metrics)
        val_loss = val_metrics["dpo_loss"]

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if show_progress:
            epoch_iter.set_postfix({
                "train": f"{avg_train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "acc": f"{val_metrics['accuracy']:.3f}",
                "H": f"{val_metrics['entropy']:.3f}",
                "best": f"{best_val_loss:.4f}",
                "pat": patience_counter,
            })

        if epoch == 1 or epoch % 10 == 0:
            _log(
                f"Epoch {epoch:3d}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
                f"acc={val_metrics['accuracy']:.3f} entropy={val_metrics['entropy']:.3f} "
                f"argmax_max={val_metrics['argmax_max_frac']:.2f}",
                show_progress,
            )

        if patience_counter >= args.patience:
            _log(f"Early stopping at epoch {epoch}", show_progress)
            break

    # Save best model
    if best_state is not None:
        policy.load_state_dict(best_state)

    torch.save(policy.state_dict(), output_dir / "model_best.pt")
    _log(f"Saved best model to {output_dir / 'model_best.pt'}", show_progress)

    # Save feature stats (placeholder - standardization computed from train data)
    all_features = train_wf.reshape(-1, D).numpy()
    feature_stats = {
        "input_dim": D,
        "n_actions": n_actions,
        "lambda_grid": lambda_grid,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "mean": all_features.mean(axis=0).tolist(),
        "std": np.where(all_features.std(axis=0) < 1e-8, 1.0, all_features.std(axis=0)).tolist(),
        "ref_center": args.ref_center,
        "ref_temperature": args.ref_temperature,
        "beta": args.beta,
    }
    with (output_dir / "feature_stats.json").open("w", encoding="utf-8") as f:
        json.dump(feature_stats, f, indent=2, ensure_ascii=False)

    # Save training metrics
    training_meta = {
        "beta": args.beta,
        "lr": args.lr,
        "n_train_pairs": n_train_pairs,
        "n_val_pairs": n_val_pairs,
        "K": K,
        "input_dim": D,
        "n_actions": n_actions,
        "lambda_grid": lambda_grid,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
        "final_val_metrics": val_metrics_list[-1] if val_metrics_list else None,
    }
    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2, ensure_ascii=False)

    _log(f"Training complete. Best val loss: {best_val_loss:.4f}", show_progress)


if __name__ == "__main__":
    main()
