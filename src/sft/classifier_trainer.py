from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import swanlab
from swanlab.integration.transformers import SwanLabCallback

from fact_checking.config import load_yaml, save_yaml
from fact_checking.data.constants import LABELS, LABELS_3CLASS, LABEL2ID, LABEL_MAP_6TO3
from fact_checking.models.ordinal import coral_decode, coral_loss
from fact_checking.utils.logging import init_logger
from sft.classifier_dataset import ClassifierDataset
from sft.data.io import _save_eval_artifacts
from sft.metrics import _build_confusion_matrix, _compute_classification_metrics
from sft.runtime.config import apply_runtime_output_layout

logger = init_logger(__name__)


@dataclass
class _MetricsCapture:
    """Stash full eval results from compute_metrics so the callback can reuse them."""

    full_metrics: dict[str, Any] | None = None
    pred_ids: np.ndarray | None = None
    gold_ids: np.ndarray | None = None


def _make_compute_metrics(capture: _MetricsCapture, *, loss_kind: str = "ce", labels: list[str] | None = None):
    def _compute_metrics(eval_pred) -> dict[str, float]:
        if loss_kind == "coral":
            preds = coral_decode(eval_pred.predictions)
        else:
            preds = np.argmax(eval_pred.predictions, axis=-1)
        gold = np.asarray(eval_pred.label_ids)
        full = _compute_classification_metrics(np.asarray(preds), gold, labels=labels)
        capture.full_metrics = full
        capture.pred_ids = np.asarray(preds)
        capture.gold_ids = gold
        return {
            "accuracy": full["accuracy"],
            "macro_f1": full["macro_f1"],
            "macro_precision": full["macro_precision"],
            "macro_recall": full["macro_recall"],
        }

    return _compute_metrics


class EvalArtifactsCallback(TrainerCallback):
    """Mirror sft.trainer eval logic: print summary + per-class lines and save artifacts."""

    def __init__(self, output_dir: Path, capture: _MetricsCapture, *, labels: list[str] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.capture = capture
        self.labels = labels

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        if not state.is_world_process_zero:
            return
        full = self.capture.full_metrics
        if full is None or self.capture.pred_ids is None or self.capture.gold_ids is None:
            return

        global_step = int(state.global_step)
        eval_loss = float(metrics.get("eval_loss", float("nan"))) if metrics else float("nan")
        per_class = full.get("per_class", {}) or {}

        summary = (
            f"[eval] step={global_step} "
            f"loss={eval_loss:.4f} "
            f"accuracy={float(full['accuracy']):.4f} "
            f"macro_precision={float(full['macro_precision']):.4f} "
            f"macro_recall={float(full['macro_recall']):.4f} "
            f"macro_f1={float(full['macro_f1']):.4f} "
            f"parse_error_rate={float(full['parse_error_rate']):.4f}"
        )
        logger.info(summary)
        if isinstance(per_class, dict) and per_class:
            logger.info("[eval] per_class:")
            for label in sorted(per_class.keys()):
                m = per_class[label]
                if isinstance(m, dict):
                    logger.info(
                        "  - %s: P=%.4f R=%.4f F1=%.4f",
                        label,
                        float(m.get("precision", 0.0)),
                        float(m.get("recall", 0.0)),
                        float(m.get("f1", 0.0)),
                    )

        cm, cm_labels = _build_confusion_matrix(self.capture.pred_ids, self.capture.gold_ids, labels=self.labels)

        # Log per-class metrics to SwanLab (if initialized)
        if swanlab.get_run() is not None:
            swanlab_log: dict[str, float] = {}
            for label, m in per_class.items():
                if isinstance(m, dict):
                    swanlab_log[f"eval/{label}/precision"] = float(m.get("precision", 0.0))
                    swanlab_log[f"eval/{label}/recall"] = float(m.get("recall", 0.0))
                    swanlab_log[f"eval/{label}/f1"] = float(m.get("f1", 0.0))
            swanlab.log(swanlab_log, step=global_step)

        artifacts = _save_eval_artifacts(
            output_dir=self.output_dir,
            global_step=global_step,
            metrics={
                "step": global_step,
                "eval_loss": eval_loss,
                "accuracy": float(full["accuracy"]),
                "macro_precision": float(full["macro_precision"]),
                "macro_recall": float(full["macro_recall"]),
                "macro_f1": float(full["macro_f1"]),
                "parse_error_rate": float(full["parse_error_rate"]),
                "per_class": per_class,
            },
            confusion_matrix=cm,
            confusion_labels=cm_labels,
            prediction_records=None,
        )
        logger.info(
            "[eval] artifacts saved: metrics=%s confusion=%s",
            artifacts["metrics_path"],
            artifacts["confusion_png_path"],
        )


class CoralTrainer(Trainer):
    """Trainer that uses CORAL ordinal regression loss instead of CE."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self._coral_num_classes = num_classes

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = coral_loss(outputs.logits.float(), labels, num_classes=self._coral_num_classes)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Discriminative classifier trainer for fact-checking (b4).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_runtime_output_layout(cfg)
    data_cfg = cfg["data"]
    train_cfg = cfg["sft_train"]
    loss_cfg = train_cfg.get("loss", {})
    loss_kind = str(loss_cfg.get("kind", "ce")).lower()
    label_map_name = train_cfg.get("label_map")
    if label_map_name == "6to3":
        effective_labels: list[str] = list(LABELS_3CLASS)
        effective_label2id: dict[str, int] = {l: i for i, l in enumerate(effective_labels)}
        label_map_dict: dict[int, int] | None = dict(LABEL_MAP_6TO3)
    else:
        effective_labels = list(LABELS)
        effective_label2id = dict(LABEL2ID)
        label_map_dict = None

    # SwanLab tracking setup
    swanlab_cfg: dict[str, Any] = dict(cfg.get("swanlab", {}) or {})
    swanlab_project = str(swanlab_cfg.get("project", "fact-checking"))
    swanlab_experiment = str(swanlab_cfg.get("experiment_name", cfg.get("experiment", {}).get("name", "classifier")))
    swanlab_tags: list[str] = [str(t) for t in swanlab_cfg.get("tags", [])]
    swanlab_tags.append("classifier")
    swanlab_callback = SwanLabCallback(
        project=swanlab_project,
        experiment_name=swanlab_experiment,
        tags=swanlab_tags,
    )

    model_path = str(cfg.get("model_name_or_path") or cfg.get("train", {}).get("model_name_or_path") or "")
    if not model_path:
        raise ValueError("classifier_trainer: model_name_or_path is empty")

    set_seed(int(train_cfg.get("seed", 42)))

    output_dir = Path(cfg.get("output_dir", "outputs/runs/train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, output_dir / "config.resolved.yaml")
    logger.info("[INFO] resolved config saved to %s", output_dir / "config.resolved.yaml")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs: dict = dict(
        num_labels=len(effective_labels) - 1 if loss_kind == "coral" else len(effective_labels),
        torch_dtype=torch.bfloat16 if bool(train_cfg.get("bf16", True)) else torch.float32,
    )
    if loss_kind != "coral":
        model_kwargs["id2label"] = {i: effective_labels[i] for i in range(len(effective_labels))}
        model_kwargs["label2id"] = effective_label2id
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    train_ds = ClassifierDataset(
        data_cfg["train_candidates"],
        tokenizer,
        top_k_evidence=int(train_cfg.get("top_k_evidence", 16)),
        max_length=int(train_cfg.get("max_length", 2048)),
        label_map=label_map_dict,
    )
    val_ds = ClassifierDataset(
        data_cfg["val_candidates"],
        tokenizer,
        top_k_evidence=int(train_cfg.get("top_k_evidence", 16)),
        max_length=int(train_cfg.get("max_length", 2048)),
        label_map=label_map_dict,
    )
    logger.info("[INFO] train=%d val=%d", len(train_ds), len(val_ds))

    collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(train_cfg.get("learning_rate", 2.0e-5)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.06)),
        bf16=bool(train_cfg.get("bf16", True)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy="steps",
        eval_steps=int(train_cfg.get("eval_steps", 50)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 50)),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 2)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        seed=int(train_cfg.get("seed", 42)),
        report_to=[],
        max_steps=int(train_cfg.get("max_steps", -1)),
    )

    capture = _MetricsCapture()
    trainer_cls = CoralTrainer if loss_kind == "coral" else Trainer
    trainer_kwargs: dict = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_make_compute_metrics(capture, loss_kind=loss_kind, labels=effective_labels),
        callbacks=[
            EvalArtifactsCallback(output_dir=output_dir, capture=capture, labels=effective_labels),
            swanlab_callback,
        ],
    )
    if loss_kind == "coral":
        trainer_kwargs["num_classes"] = len(effective_labels)
    trainer = trainer_cls(**trainer_kwargs)

    trainer.train()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info("[INFO] best checkpoint saved to %s", best_dir)

    eval_metrics = trainer.evaluate()
    preds = trainer.predict(val_ds)
    if loss_kind == "coral":
        pred_ids = coral_decode(preds.predictions)
    else:
        pred_ids = np.argmax(preds.predictions, axis=-1)
    gold_ids = np.asarray(preds.label_ids)
    full_metrics = _compute_classification_metrics(pred_ids, gold_ids, labels=effective_labels)
    cm, cm_labels = _build_confusion_matrix(pred_ids, gold_ids, labels=effective_labels)
    summary = {
        "eval": eval_metrics,
        "val_full_metrics": full_metrics,
        "val_confusion_matrix": {"labels": cm_labels, "matrix": cm.tolist()},
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("[INFO] train_summary written; macro_f1=%.4f acc=%.4f",
                full_metrics["macro_f1"], full_metrics["accuracy"])


if __name__ == "__main__":
    main()
