import hashlib
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable

from datetime import datetime
from pathlib import Path
from typing import Iterable
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from fact_checking.data.types import SampleRecord, SentenceRecord
from sft.prompting.utils import clean_text, robust_sentence_split
from fact_checking.utils.logging import init_logger
from fact_checking.data.constants import LABEL2ID, LABELS

logger = init_logger(__name__)

def save_model(
    accelerator: Accelerator,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: Path,
) -> None:
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        output_path.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)

    try:
        # 关键：不要放在 if accelerator.is_main_process 里面
        state_dict = accelerator.get_state_dict(model)

        # 关键：所有 rank 都调用 save_pretrained，但只有 main process 真正写文件
        unwrapped.save_pretrained(
            str(output_path),
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )

        if accelerator.is_main_process:
            tokenizer.save_pretrained(str(output_path))

    except ValueError as exc:
        if "stage3_gather_16bit_weights_on_model_save" not in str(exc):
            raise
        if not hasattr(model, "save_checkpoint"):
            raise

        ds_ckpt_dir = output_path / "ds_checkpoint"

        # 关键：DeepSpeed save_checkpoint 必须所有 rank 调用
        model.save_checkpoint(str(ds_ckpt_dir))

        if accelerator.is_main_process:
            tokenizer.save_pretrained(str(output_path))
            logger.warning(
                "[WARN] DeepSpeed ZeRO-3 16-bit gather is disabled; saved a DeepSpeed checkpoint to %s. "
                "Convert to fp32 using zero_to_fp32.py or enable stage3_gather_16bit_weights_on_model_save.",
                ds_ckpt_dir,
            )

    accelerator.wait_for_everyone()


def load_split(path: str | Path) -> list[SampleRecord]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records: list[SampleRecord] = []
    for item in payload:
        label = clean_text(str(item["label"])).lower()
        if label not in LABEL2ID:
            raise ValueError(f"Unknown label: {label!r} in {path}")
        records.append(
            SampleRecord(
                event_id=str(item["event_id"]),
                claim=clean_text(str(item["claim"])),
                label=label,
                explain=clean_text(str(item.get("explain", ""))),
                reports=item.get("reports", []),
            )
        )
    return records



def iter_sentences(sample: SampleRecord, min_char_len: int = 10) -> Iterable[SentenceRecord]:
    for report in sample.reports:
        report_id = report.get("report_id", "unknown")
        link = report.get("link")
        domain = report.get("domain")
        content = clean_text(str(report.get("content", "")))
        for sent_idx, sent in enumerate(robust_sentence_split(content)):
            if len(sent) < min_char_len:
                continue
            yield SentenceRecord(
                event_id=sample.event_id,
                report_id=report_id,
                sent_idx=sent_idx,
                text=sent,
                link=link,
                domain=domain,
                raw=report,
            )

def _build_cache_name(split: str, max_length: int, tokenizer_name: str, instances: list[dict[str, str]]) -> str:
    tok_hash = hashlib.md5(tokenizer_name.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    data_hash = hashlib.md5(usedforsecurity=False)
    data_hash.update(str(len(instances)).encode("utf-8"))
    for row in instances:
        data_hash.update(row["prompt"].encode("utf-8"))
        data_hash.update(row["target"].encode("utf-8"))
    return f"{split}_ml{max_length}_{tok_hash}_{data_hash.hexdigest()[:16]}.pt"


def _save_eval_artifacts(
    output_dir: Path,
    global_step: int,
    metrics: dict[str, float | dict[str, dict[str, float]]],
    confusion_matrix: np.ndarray,
    confusion_labels: list[str],
) -> dict[str, str]:
    step_dir = output_dir / "eval" / f"step-{global_step}"
    step_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = step_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    confusion_data_path = step_dir / "confusion_matrix.json"
    with confusion_data_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "gold_labels": LABELS,
                "pred_labels": confusion_labels,
                "matrix": confusion_matrix.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(confusion_labels)))
    ax.set_xticklabels(confusion_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(f"Confusion Matrix @ step {global_step}")
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    confusion_png_path = step_dir / "confusion_matrix.png"
    fig.savefig(confusion_png_path, dpi=200)
    plt.close(fig)

    return {
        "metrics_path": str(metrics_path),
        "confusion_data_path": str(confusion_data_path),
        "confusion_png_path": str(confusion_png_path),
    }
