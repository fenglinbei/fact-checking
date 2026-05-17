"""Train a lightweight pointwise oracle evidence selector."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fact_checking.oracle_pointwise import (
    DEFAULT_FEATURE_NAMES,
    average_precision,
    bce_loss,
    claim_selection_metrics,
    compute_row_weights,
    feature_matrix,
    labels_array,
    read_jsonl,
    roc_auc,
    selected_evidence_rows,
    sigmoid,
    split_event_ids_by_label,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train logistic pointwise oracle selector.")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--feature-schema", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="logreg", choices=["logreg"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--eval-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.train_jsonl)
    if not rows:
        raise ValueError(f"No rows found in {args.train_jsonl}")

    feature_names = _load_feature_names(args.feature_schema)
    train_eids, val_eids = split_event_ids_by_label(rows, args.val_fraction, args.seed)
    train_rows = [r for r in rows if str(r["event_id"]) in train_eids]
    val_rows = [r for r in rows if str(r["event_id"]) in val_eids]
    if not train_rows or not val_rows:
        raise ValueError("Train/dev split produced an empty side.")

    x_train_raw = feature_matrix(train_rows, feature_names)
    x_val_raw = feature_matrix(val_rows, feature_names)
    y_train = labels_array(train_rows)
    y_val = labels_array(val_rows)
    w_train = compute_row_weights(train_rows)
    w_val = compute_row_weights(val_rows)

    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train_raw - mean) / std
    x_val = (x_val_raw - mean) / std

    weights, bias, history = _train_logreg(
        x_train,
        y_train,
        w_train,
        x_val,
        y_val,
        w_val,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        patience=args.patience,
        eval_every=args.eval_every,
    )

    train_scores = sigmoid(x_train @ weights + bias)
    val_scores = sigmoid(x_val @ weights + bias)
    hybrid_train = np.array([float(r["features"].get("hybrid_score", 0.0)) for r in train_rows], dtype=np.float32)
    hybrid_val = np.array([float(r["features"].get("hybrid_score", 0.0)) for r in val_rows], dtype=np.float32)

    train_metrics = _row_metrics(y_train, train_scores, w_train)
    val_metrics = _row_metrics(y_val, val_scores, w_val)
    train_claim = claim_selection_metrics(train_rows, train_scores, top_k=args.top_k, score_name="model")
    val_claim = claim_selection_metrics(val_rows, val_scores, top_k=args.top_k, score_name="model")
    hybrid_val_claim = claim_selection_metrics(val_rows, hybrid_val, top_k=args.top_k, score_name="hybrid_score")

    model_path = out_dir / "model.npz"
    np.savez(
        model_path,
        weights=weights.astype(np.float32),
        bias=np.array([bias], dtype=np.float32),
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        feature_names=np.array(feature_names, dtype=object),
    )

    metrics = {
        "model_type": "numpy_logistic_regression",
        "train_jsonl": args.train_jsonl,
        "n_rows": len(rows),
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "n_train_claims": len(train_eids),
        "n_val_claims": len(val_eids),
        "positive_rate": float(labels_array(rows).mean()),
        "supervision_summary": _supervision_summary(rows),
        "train": train_metrics,
        "val": val_metrics,
        "train_claim_selection": train_claim,
        "val_claim_selection": val_claim,
        "val_hybrid_baseline_selection": hybrid_val_claim,
        "history": history,
        "feature_names": feature_names,
        "model_path": str(model_path),
    }
    write_json(out_dir / "training_metrics.json", metrics)

    feature_importance = [
        {"feature": name, "weight": float(weight)}
        for name, weight in sorted(zip(feature_names, weights), key=lambda x: abs(float(x[1])), reverse=True)
    ]
    write_json(out_dir / "feature_importance.json", {"features": feature_importance})

    pred_rows = []
    for row, score, hybrid in zip(val_rows, val_scores, hybrid_val):
        item = dict(row)
        item["model_score"] = float(score)
        item["hybrid_score"] = float(hybrid)
        pred_rows.append(item)
    write_jsonl(out_dir / "dev_predictions.jsonl", pred_rows)
    write_jsonl(out_dir / "dev_selected_evidence.jsonl", selected_evidence_rows(val_rows, val_scores, top_k=args.top_k))

    print(f"Saved model: {model_path}")
    print(f"Val AUPRC={val_metrics['auprc']:.4f}, AUROC={val_metrics['auroc']:.4f}")
    print(
        "Val Recall@{k}={rec:.4f}, Jaccard@{k}={jac:.4f}; hybrid Jaccard@{k}={hjac:.4f}".format(
            k=args.top_k,
            rec=val_claim["recall_at_k"],
            jac=val_claim["jaccard_at_k"],
            hjac=hybrid_val_claim["jaccard_at_k"],
        )
    )


def _load_feature_names(schema_path: str | None) -> list[str]:
    if not schema_path:
        return list(DEFAULT_FEATURE_NAMES)
    with Path(schema_path).open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return list(payload.get("feature_names") or DEFAULT_FEATURE_NAMES)


def _train_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2: float,
    patience: int,
    eval_every: int,
) -> tuple[np.ndarray, float, list[dict]]:
    weights = np.zeros(x_train.shape[1], dtype=np.float32)
    bias = 0.0
    best_weights = weights.copy()
    best_bias = bias
    best_score = -1.0
    stale = 0
    history: list[dict] = []
    train_weight_sum = max(float(w_train.sum()), 1e-8)

    for epoch in range(1, epochs + 1):
        probs = sigmoid(x_train @ weights + bias)
        err = (probs - y_train) * w_train
        grad_w = (x_train.T @ err) / train_weight_sum + l2 * weights
        grad_b = float(err.sum() / train_weight_sum)
        weights -= lr * grad_w.astype(np.float32)
        bias -= lr * grad_b

        if epoch % eval_every != 0 and epoch != epochs:
            continue
        train_probs = sigmoid(x_train @ weights + bias)
        val_probs = sigmoid(x_val @ weights + bias)
        val_ap = average_precision(y_val, val_probs)
        record = {
            "epoch": epoch,
            "train_loss": bce_loss(y_train, train_probs, w_train),
            "val_loss": bce_loss(y_val, val_probs, w_val),
            "val_auprc": val_ap,
            "val_auroc": roc_auc(y_val, val_probs),
        }
        history.append(record)
        if val_ap > best_score + 1e-6:
            best_score = val_ap
            best_weights = weights.copy()
            best_bias = bias
            stale = 0
        else:
            stale += eval_every
            if stale >= patience:
                break
    return best_weights, float(best_bias), history


def _row_metrics(y: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "loss": bce_loss(y, scores, weights),
        "auprc": average_precision(y, scores),
        "auroc": roc_auc(y, scores),
        "positive_rate": float(y.mean()) if y.size else 0.0,
    }


def _supervision_summary(rows: list[dict]) -> dict[str, object]:
    event_seen: set[str] = set()
    bucket_claim_counts: Counter[str] = Counter()
    label_claim_counts: Counter[str] = Counter()
    bucket_weight_sums: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        eid = str(row.get("event_id", ""))
        if eid in event_seen:
            continue
        event_seen.add(eid)
        bucket = str(row.get("filter_bucket", "unknown"))
        label = str(row.get("gold_label", ""))
        weight = float(row.get("supervision_weight", 1.0))
        bucket_claim_counts[bucket] += 1
        label_claim_counts[label] += 1
        bucket_weight_sums[bucket] += weight
    return {
        "n_claims": len(event_seen),
        "claim_counts_by_bucket": dict(bucket_claim_counts),
        "claim_counts_by_label": dict(label_claim_counts),
        "supervision_weight_sum_by_bucket": {
            key: float(value)
            for key, value in bucket_weight_sums.items()
        },
    }


if __name__ == "__main__":
    main()
