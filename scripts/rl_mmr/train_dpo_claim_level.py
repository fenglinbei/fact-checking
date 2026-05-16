"""Train a claim-level DPO λ policy using only pool features (no step features).

This eliminates the endogenous state problem: the policy sees the same features
regardless of which λ schedule was used, so it can learn genuine causal signal.

Usage:
    PYTHONPATH=src python scripts/rl_mmr/train_dpo_claim_level.py \\
        --train-pairs ... --val-pairs ... \\
        --output-dir outputs/rl_mmr/dpo_stepwise/checkpoints_claim \\
        --beta 3.0 --lr 1e-3 --epochs 200
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
    dpo_loss,
    evaluate_policy_metrics,
)
from fact_checking.rl_mmr.trajectory import LAMBDA_GRID


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-pairs", type=str, required=True)
    p.add_argument("--val-pairs", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--lambda-grid", nargs="*", type=float, default=LAMBDA_GRID)
    p.add_argument("--ref-center", type=float, default=0.7)
    p.add_argument("--ref-temperature", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[64, 32])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def _log(msg, show):
    if show: tqdm.write(msg)
    else: print(msg, flush=True)


def _load_pairs(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return (
        torch.from_numpy(data["win_features"].astype(np.float32)),
        torch.from_numpy(data["win_lambdas"].astype(np.int64)),
        torch.from_numpy(data["lose_features"].astype(np.float32)),
        torch.from_numpy(data["lose_lambdas"].astype(np.int64)),
    )


def _pool_features_only(wf, wl, lf, ll):
    """Extract step-0 pool features (dims 0-7 of the OLD 20-dim format) + mean lambda.

    Since we're using the v3 features (13 dims, no pool), we need step-0 features
    which only have deterministic values. Instead, we use the STEP FEATURES at step 0
    to represent the pool, plus the mean λ of the trajectory as the "label".

    For claim-level: input = step-0 features (12 dims, same for all trajectories
    of the same claim) + step_fraction(=0) + n_selected(=0). All trajectories of
    the same claim have identical step-0 features.

    Action = majority λ of the trajectory (rounded to nearest grid point).
    """
    B, K, D = wf.shape
    # Use step-0 features only (identical for same-claim trajectories)
    # These are the step features before any selection
    wf0 = wf[:, 0, :]  # [B, D]
    lf0 = lf[:, 0, :]  # [B, D]
    # Both should be nearly identical for same claim (step 0, nothing selected)

    # Action: majority λ across all 5 steps
    def _majority_lambda(lambdas):
        # lambdas: [B, K] indices
        result = np.zeros(B, dtype=np.int64)
        for b in range(B):
            counts = np.bincount(lambdas[b].numpy(), minlength=5)
            result[b] = int(np.argmax(counts))
        return torch.from_numpy(result)

    wl_maj = _majority_lambda(wl)  # [B]
    ll_maj = _majority_lambda(ll)

    return wf0, wl_maj, lf0, ll_maj


def main():
    args = parse_args()
    show = not args.no_progress
    torch.manual_seed(args.seed)

    lambda_grid = [float(x) for x in args.lambda_grid]
    n_actions = len(lambda_grid)

    train_wf, train_wl, train_lf, train_ll = _load_pairs(args.train_pairs)
    val_wf, val_wl, val_lf, val_ll = _load_pairs(args.val_pairs)

    # Convert to claim-level: use step-0 features + majority λ
    tr_wf, tr_wl, tr_lf, tr_ll = _pool_features_only(train_wf, train_wl, train_lf, train_ll)
    va_wf, va_wl, va_lf, va_ll = _pool_features_only(val_wf, val_wl, val_lf, val_ll)

    # Reshape to [B, 1, D] for the step-wise DPO loss function
    tr_wf = tr_wf.unsqueeze(1)
    tr_lf = tr_lf.unsqueeze(1)
    tr_wl = tr_wl.unsqueeze(1)
    tr_ll = tr_ll.unsqueeze(1)
    va_wf = va_wf.unsqueeze(1)
    va_lf = va_lf.unsqueeze(1)
    va_wl = va_wl.unsqueeze(1)
    va_ll = va_ll.unsqueeze(1)

    B, K, D = tr_wf.shape
    _log(f"Train pairs: {B}  Val pairs: {va_wf.shape[0]}  K=1 (claim-level)  D={D}  actions={n_actions}", show)

    train_loader = DataLoader(
        TensorDataset(tr_wf, tr_wl, tr_lf, tr_ll),
        batch_size=args.batch_size, shuffle=True,
    )

    policy = StepLambdaPolicy(input_dim=D, hidden_dims=args.hidden_dims, dropout=args.dropout, n_actions=n_actions)
    ref = FixedReferencePolicy(lambda_grid=lambda_grid, center=args.ref_center, temperature=args.ref_temperature)
    _log(f"Policy: {sum(p.numel() for p in policy.parameters())} params", show)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    patience = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(1, args.epochs + 1), desc="train claim dpo", disable=not show):
        policy.train()
        epoch_loss = 0.0
        n_batches = 0
        for wf_b, wl_b, lf_b, ll_b in train_loader:
            loss = dpo_loss(policy, ref, wf_b, wl_b, lf_b, ll_b, beta=args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        val_metrics = evaluate_policy_metrics(policy, ref, va_wf, va_wl, va_lf, va_ll, beta=args.beta)
        val_loss = val_metrics["dpo_loss"]

        # Argmax distribution
        policy.eval()
        with torch.no_grad():
            val_logits = policy(va_wf.reshape(-1, D))
            val_argmax = torch.argmax(val_logits, dim=-1).cpu().numpy()
        val_counts = np.bincount(val_argmax, minlength=n_actions)
        argmax_max_frac = float(val_counts.max()) / max(val_counts.sum(), 1)

        if show:
            tqdm.write(
                f"Epoch {epoch:3d}: train_loss={epoch_loss/max(n_batches,1):.4f} "
                f"val_loss={val_loss:.4f} acc={val_metrics['accuracy']:.3f} "
                f"H={val_metrics['entropy']:.3f} max_frac={argmax_max_frac:.2f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= args.patience:
            _log(f"Early stopping at epoch {epoch}", show)
            break

    if best_state:
        policy.load_state_dict(best_state)
    torch.save(policy.state_dict(), output_dir / "model_best.pt")

    # Save stats
    all_feats = tr_wf.reshape(-1, D).numpy()
    feature_stats = {
        "input_dim": D, "n_actions": n_actions, "lambda_grid": lambda_grid,
        "hidden_dims": args.hidden_dims, "dropout": args.dropout,
        "mean": all_feats.mean(axis=0).tolist(),
        "std": np.where(all_feats.std(axis=0) < 1e-8, 1.0, all_feats.std(axis=0)).tolist(),
        "ref_center": args.ref_center, "ref_temperature": args.ref_temperature, "beta": args.beta,
        "policy_type": "claim_level",
    }
    with (output_dir / "feature_stats.json").open("w") as f:
        json.dump(feature_stats, f, indent=2, ensure_ascii=False)

    _log(f"Best val loss: {best_val_loss:.4f}", show)
    _log(f"Final argmax: {dict(zip([f'{g:.2f}' for g in lambda_grid], val_counts.tolist()))}", show)


if __name__ == "__main__":
    main()
