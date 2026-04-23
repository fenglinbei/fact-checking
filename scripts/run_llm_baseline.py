from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from fact_checking.data.constants import LABEL2ID, LABELS
from fact_checking.baselines.llm_baseline import BaselineConfig, run_inference
from fact_checking.config import load_yaml


def _load_predictions(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_label_to_id(label: str) -> int:
    return LABEL2ID.get(str(label).strip().lower(), LABEL2ID["half-true"])


def _save_confusion_matrix_figure(cm: np.ndarray, labels: list[str], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Gold Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    max_value = int(cm.max()) if cm.size else 0
    threshold = max_value / 2.0 if max_value else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _report_metrics(out_path: Path) -> None:
    rows = _load_predictions(out_path)
    if not rows:
        print("No predictions found, skip metrics.")
        return

    y_true = np.array([_safe_label_to_id(str(row.get("gold_label", ""))) for row in rows], dtype=np.int64)
    y_pred = np.array([_safe_label_to_id(str(row.get("pred_label", ""))) for row in rows], dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))

    metrics = {
        "num_samples": int(len(rows)),
        "accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(per_p[idx]),
                "recall": float(per_r[idx]),
                "f1": float(per_f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(LABELS)
        },
    }
    cm_payload = {
        "labels": LABELS,
        "matrix": cm.tolist(),
    }

    metrics_path = out_path.with_suffix(".metrics.json")
    cm_json_path = out_path.with_suffix(".confusion_matrix.json")
    cm_png_path = out_path.with_suffix(".confusion_matrix.png")

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    cm_json_path.write_text(json.dumps(cm_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    has_png = _save_confusion_matrix_figure(cm, LABELS, cm_png_path)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro P/R/F1: {float(macro_p):.4f} / {float(macro_r):.4f} / {float(macro_f1):.4f}")
    print("Per-class P/R/F1:")
    for idx, label in enumerate(LABELS):
        print(
            f"  - {label:12s}  P={float(per_p[idx]):.4f}  R={float(per_r[idx]):.4f}  "
            f"F1={float(per_f1[idx]):.4f}  N={int(support[idx])}"
        )
    print(f"Saved metrics JSON: {metrics_path}")
    print(f"Saved confusion matrix JSON: {cm_json_path}")
    if has_png:
        print(f"Saved confusion matrix PNG: {cm_png_path}")
    else:
        print("matplotlib is not installed, skipped confusion matrix PNG output.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B0/B1 LLM baselines.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    baseline_cfg = cfg["baseline"]

    baseline = BaselineConfig(
        model_name_or_path=str(baseline_cfg.get("model_name_or_path", "/data/models/Qwen3.5-9B")),
        top_k=int(baseline_cfg.get("top_k", 8)),
        use_context=bool(baseline_cfg.get("use_context", False)),
        context_k=int(baseline_cfg.get("context_k", 1)),
        prompt_mode=str(baseline_cfg.get("prompt_mode", "few_shot")),
        few_shot_k=int(baseline_cfg.get("few_shot_k", 10)),
        few_shot_mmr_lambda=float(baseline_cfg.get("few_shot_mmr_lambda", 0.7)),
        retrieval_model=str(baseline_cfg.get("retrieval_model", "/home/fenglin/project/models/bge-base-en-v1.5/")),
        retrieval_batch_size=int(baseline_cfg.get("retrieval_batch_size", 64)),
        retrieval_max_length=int(baseline_cfg.get("retrieval_max_length", 256)),
        max_new_tokens=int(baseline_cfg.get("max_new_tokens", 24)),
        temperature=float(baseline_cfg.get("temperature", 0.0)),
        do_sample=bool(baseline_cfg.get("do_sample", False)),
    )

    split_map = {
        "train": str(data_cfg["train_candidates"]),
        "val": str(data_cfg["val_candidates"]),
        "test": str(data_cfg["test_candidates"]),
    }
    input_path = split_map[args.split]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    variant = str(baseline_cfg.get("variant", "")).strip() or timestamp
    run_output_dir = Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline")) / f"{variant}_{timestamp}" / "test"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_output_dir / f"{variant}_{timestamp}.jsonl"

    run_inference(
        cfg=baseline,
        input_path=input_path,
        output_path=out_path,
        train_path_for_few_shot=split_map["train"],
    )
    print(f"Wrote {out_path}")
    _report_metrics(out_path)


if __name__ == "__main__":
    main()
