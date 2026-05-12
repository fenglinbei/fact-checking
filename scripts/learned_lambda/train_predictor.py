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
from tqdm.auto import tqdm

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
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return p.parse_args()


def _log(message: str, *, show_progress: bool) -> None:
    if show_progress:
        tqdm.write(message)
    else:
        print(message, flush=True)


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load oracle λ
    oracle_by_eid: dict[str, float] = {}
    with open(args.oracle_lambdas) as f:
        for line in f:
            rec = json.loads(line.strip())
            oracle_by_eid[rec["event_id"]] = rec["oracle_lambda"]
    _log(f"Loaded {len(oracle_by_eid)} oracle λ values", show_progress=show_progress)

    # Load PreMMR cache and extract features
    pre_samples = _load_pickle(Path(args.premmr_cache))
    _log(f"Loaded {len(pre_samples)} PreMMR samples", show_progress=show_progress)

    features_list: list[np.ndarray] = []
    targets_list: list[float] = []
    skipped = 0
    for pre in tqdm(
        pre_samples,
        desc="extract features",
        unit="sample",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        if pre.event_id not in oracle_by_eid:
            skipped += 1
            continue
        feat = extract_features(pre, args.alpha_dense, args.alpha_lexical, args.alpha_bm25)
        features_list.append(feat)
        targets_list.append(oracle_by_eid[pre.event_id])

    if skipped > 0:
        _log(f"Skipped {skipped} samples without oracle λ", show_progress=show_progress)

    features = np.stack(features_list)
    targets = np.array(targets_list, dtype=np.float32)
    _log(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features", show_progress=show_progress)
    _log(f"Target λ: mean={targets.mean():.3f}, std={targets.std():.3f}", show_progress=show_progress)

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
    _log(f"Train: {len(train_idx)}, Val: {len(val_idx)}", show_progress=show_progress)

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
        for xb, yb in batch_iter:
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
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if show_progress:
            epoch_progress.set_postfix({
                "train_mse": f"{train_loss:.5f}",
                "val_mse": f"{val_loss:.5f}",
                "best": f"{best_val_loss:.5f}",
                "patience": patience_counter,
            })
        if epoch % 10 == 0 or epoch == 1:
            _log(
                f"Epoch {epoch:3d}: train_mse={train_loss:.5f}  val_mse={val_loss:.5f}",
                show_progress=show_progress,
            )

        if not improved and patience_counter >= args.patience:
            _log(
                f"Early stopping at epoch {epoch} (best val MSE={best_val_loss:.5f})",
                show_progress=show_progress,
            )
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
    _log(
        f"\nFinal val metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, best_MSE={best_val_loss:.5f}",
        show_progress=show_progress,
    )

    # Save
    output_dir = Path(args.output_dir)
    save_predictor(model, feat_mean, feat_std, output_dir)
    _log(f"Saved predictor to {output_dir}", show_progress=show_progress)

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
    _log(f"Saved training metadata to {output_dir / 'training_meta.json'}", show_progress=show_progress)


if __name__ == "__main__":
    main()
