from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from fact_checking.config import load_yaml
from fact_checking.data.constants import LABELS, LABELS_3CLASS, LABEL_MAP_6TO3
from fact_checking.utils.logging import init_logger
from sft.classifier_dataset import ClassifierDataset
from sft.data.io import save_eval_artifacts
from sft.eval import log_eval_summary
from sft.metrics import _build_confusion_matrix, _compute_classification_metrics

logger = init_logger(__name__)


def _label_name(idx: int, *, labels: list[str] | None = None) -> str:
    _labels = labels if labels is not None else LABELS
    return _labels[idx] if 0 <= idx < len(_labels) else "parse_error"


def run_classifier_inference(
    *,
    run_dir: str | Path,
    checkpoint: str,
    split: str,
    config_path: str | Path,
    infer_cfg: dict[str, Any],
    eval_dir: str | Path,
    log_dir: str | Path,
) -> dict[str, str]:
    train_cfg = load_yaml(str(config_path))
    sft_train_cfg = train_cfg.get("sft_train", {})
    loss_cfg = sft_train_cfg.get("loss", {})
    loss_kind = str(loss_cfg.get("kind", "ce")).lower()
    label_map_name = sft_train_cfg.get("label_map")
    if label_map_name == "6to3":
        effective_labels: list[str] = list(LABELS_3CLASS)
        label_map_dict: dict[int, int] | None = dict(LABEL_MAP_6TO3)
    else:
        effective_labels = list(LABELS)
        label_map_dict = None
    data_cfg = train_cfg["data"]

    split_key = f"{split}_candidates"
    if split_key not in data_cfg:
        raise KeyError(f"infer: split '{split}' not found in train config data section (have {list(data_cfg)})")

    ckpt_dir = Path(run_dir) / str(checkpoint)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"infer: checkpoint dir not found: {ckpt_dir}")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)

    dtype_str = str(infer_cfg.get("dtype", "bfloat16")).lower()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_str, torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir), torch_dtype=dtype)
    model = model.to(device).eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    ds = ClassifierDataset(
        data_cfg[split_key],
        tokenizer,
        top_k_evidence=int(sft_train_cfg.get("top_k_evidence", 16)),
        max_length=int(sft_train_cfg.get("max_length", 2048)),
        label_map=label_map_dict,
    )

    collator = DataCollatorWithPadding(tokenizer)
    batch_size = int(infer_cfg.get("batch_size", 8))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=int(infer_cfg.get("dataloader_num_workers", 0)),
    )

    pred_ids: list[int] = []
    gold_ids: list[int] = []
    prediction_records: list[dict[str, object]] = []
    sample_idx = 0

    progress = tqdm(loader, desc=f"infer[{split}/{checkpoint}]", unit="batch", dynamic_ncols=True)
    for batch in progress:
        labels = batch.pop("labels")
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.float()
        if loss_kind == "coral":
            cum_p = torch.sigmoid(logits).cpu().numpy()  # (B, C-1)
            ids_np = (cum_p > 0.5).sum(axis=-1)
            # marginal probs: p_k = P(y>k-1) - P(y>k); boundaries P(y>-1)=1, P(y>C-1)=0
            left = np.concatenate([np.ones((cum_p.shape[0], 1)), cum_p], axis=1)
            right = np.concatenate([cum_p, np.zeros((cum_p.shape[0], 1))], axis=1)
            marginals = np.clip(left - right, 0.0, None)
            probs = marginals / marginals.sum(axis=1, keepdims=True)
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            ids_np = np.argmax(probs, axis=-1)
        for local_i, pid in enumerate(ids_np):
            row = ds.rows[sample_idx]
            gold_id = int(labels[local_i].item())
            prediction_records.append(
                {
                    "sample_idx": sample_idx,
                    "event_id": str(row.get("event_id", "")),
                    "claim": str(row.get("claim", "")),
                    "gold_label": _label_name(gold_id, labels=effective_labels),
                    "gold_id": gold_id,
                    "pred_label": _label_name(int(pid), labels=effective_labels),
                    "pred_id": int(pid),
                    "probs": probs[local_i].tolist(),
                }
            )
            pred_ids.append(int(pid))
            gold_ids.append(gold_id)
            sample_idx += 1

    pred_arr = np.asarray(pred_ids, dtype=np.int64)
    gold_arr = np.asarray(gold_ids, dtype=np.int64)
    metrics = _compute_classification_metrics(pred_arr, gold_arr, labels=effective_labels)
    log_eval_summary(metrics, eval_logger=logger, split=split, checkpoint=str(checkpoint))
    cm, cm_labels = _build_confusion_matrix(pred_arr, gold_arr, labels=effective_labels)
    artifacts = save_eval_artifacts(
        eval_dir=eval_path,
        metrics=metrics,
        confusion_matrix=cm,
        confusion_labels=cm_labels,
        prediction_records=prediction_records,
        predictions_filename="predictions.jsonl",
        title=f"b4 classifier @ {split}/{checkpoint}",
    )
    return artifacts
