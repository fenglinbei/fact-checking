"""Step 3: Train the λ predictor MLP from oracle λ labels and PreMMR features.

Usage:
    PYTHONPATH=src python scripts/learned_lambda/train_predictor.py \
        --oracle-lambdas outputs/learned_lambda/oracle_lambda_train.jsonl \
        --premmr-cache outputs/cache/pre_mmr/53a3588e485d/train.pkl \
        --output-dir outputs/learned_lambda/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fact_checking.build.candidates import _load_pickle
from fact_checking.learned_lambda.features import FEATURE_NAMES, extract_features
from fact_checking.learned_lambda.predictor import LambdaPredictor, save_predictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train λ predictor MLP.")
    p.add_argument("--oracle-lambdas", type=str, required=True, help="JSONL from compute_oracle_lambda.py")
    p.add_argument("--premmr-cache", type=str, required=True, help="PreMMR cache pickle")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--alpha-dense", type=float, default=0.70)
    p.add_argument("--alpha-lexical", type=float, default=0.20)
    p.add_argument("--alpha-bm25", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load oracle λ
    oracle_by_eid: dict[str, float] = {}
    with open(args.oracle_lambdas) as f:
        for line in f:
            rec = json.loads(line.strip())
            oracle_by_eid[rec["event_id"]] = rec["oracle_lambda"]
    print(f"Loaded {len(oracle_by_eid)} oracle λ values", flush=True)

    # Load PreMMR cache and extract features
    pre_samples = _load_pickle(Path(args.premmr_cache))
    print(f"Loaded {len(pre_samples)} PreMMR samples", flush=True)

    features_list: list[np.ndarray] = []
    targets_list: list[float] = []
    skipped = 0
    for pre in pre_samples:
        if pre.event_id not in oracle_by_eid:
            skipped += 1
            continue
        feat = extract_features(pre, args.alpha_dense, args.alpha_lexical, args.alpha_bm25)
        features_list.append(feat)
        targets_list.append(oracle_by_eid[pre.event_id])

    if skipped > 0:
        print(f"Skipped {skipped} samples without oracle λ", flush=True)

    features = np.stack(features_list)
    targets = np.array(targets_list, dtype=np.float32)
    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features", flush=True)
    print(f"Target λ: mean={targets.mean():.3f}, std={targets.std():.3f}", flush=True)

    # z-score normalization
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    features_norm = (features - feat_mean) / feat_std

    # Train / val split
    n = len(features_norm)
    indices = rng.permutation(n)
    n_val = max(1, int(n * args.val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = torch.from_numpy(features_norm[train_idx])
    y_train = torch.from_numpy(targets[train_idx])
    X_val = torch.from_numpy(features_norm[val_idx])
    y_val = torch.from_numpy(targets[val_idx])
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}", flush=True)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Model
    model = LambdaPredictor(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        train_loss = epoch_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train_mse={train_loss:.5f}  val_mse={val_loss:.5f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val MSE={best_val_loss:.5f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on val set
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).numpy()
    val_targets = y_val.numpy()
    mae = float(np.mean(np.abs(val_pred - val_targets)))
    rmse = float(np.sqrt(np.mean((val_pred - val_targets) ** 2)))
    print(f"\nFinal val metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, best_MSE={best_val_loss:.5f}", flush=True)

    # Save
    output_dir = Path(args.output_dir)
    save_predictor(model, feat_mean, feat_std, output_dir)
    print(f"Saved predictor to {output_dir}", flush=True)

    # Save training metadata
    meta = {
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "best_val_mse": best_val_loss,
        "val_mae": mae,
        "val_rmse": rmse,
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
        "feature_names": FEATURE_NAMES,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "seed": args.seed,
    }
    with (output_dir / "training_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved training metadata to {output_dir / 'training_meta.json'}", flush=True)


if __name__ == "__main__":
    main()
