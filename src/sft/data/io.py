from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from fact_checking.data.constants import LABELS, label2id_for_schema
from fact_checking.data.types import SampleRecord, SentenceRecord
from fact_checking.utils.logging import init_logger
from fact_checking.utils.text import clean_text, robust_sentence_split
from sft.data.types import PreparedSample
from sft.runtime.adapters import checkpoint_has_hf_artifacts, is_peft_model

logger = init_logger(__name__)

_LORA_KEY_MARKERS = (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B.")


def _coerce_prompt_input_ids(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    ids: list[int] = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            return None
    return ids


def load_prebuilt_samples(
    rows: list[dict],
    *,
    include_unlabeled: bool = False,
) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    for row in rows:
        gold_label = str(row.get("gold_label", ""))
        if not gold_label and not include_unlabeled:
            continue
        label_schema = str(row.get("label_schema") or "liar6")
        label2id = label2id_for_schema(label_schema)
        samples.append(PreparedSample(
            prompt=str(row["prompt"]),
            target=str(row["target"]),
            prompt_add_special_tokens=bool(row.get("prompt_add_special_tokens", False)),
            preserve_prompt_prefix=bool(row.get("preserve_prompt_prefix", True)),
            gold_id=int(row.get("gold_id", label2id.get(gold_label, -1))),
            gold_label=gold_label,
            gold_explain=str(row.get("gold_explain", "")),
            prompt_token_count=int(row.get("prompt_token_count", 0)),
            target_token_count=int(row.get("target_token_count", 0)),
            evidence_count=int(row.get("evidence_count", 0)),
            was_truncated=bool(row.get("was_truncated", False)),
            claim=str(row.get("claim", "")),
            no_evidence=int(row.get("evidence_count", 0)) == 0,
            long_claim=len(str(row.get("claim", "")).split()) > 64,
            label_schema=label_schema,
            prompt_input_ids=_coerce_prompt_input_ids(row.get("prompt_input_ids")),
            coverage_label=str(row.get("coverage_label", "")),
        ))
    return samples


def _is_lora_state_key(key: str) -> bool:
    return any(marker in key for marker in _LORA_KEY_MARKERS)


def _strip_default_adapter_name(key: str) -> str:
    for marker in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        key = key.replace(f".{marker}.default.", f".{marker}.")
    return key


def _normalize_adapter_state_key(key: str) -> str:
    key = str(key)
    while key.startswith("_orig_mod."):
        key = key.removeprefix("_orig_mod.")
    while key.startswith("module."):
        key = key.removeprefix("module.")
    return _strip_default_adapter_name(key)


def _adapter_weight_path(output_path: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = output_path / name
        if path.exists():
            return path
    return None


def _count_saved_lora_tensors(path: Path) -> int:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            return sum(1 for key in f.keys() if _is_lora_state_key(str(key)))

    import torch

    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return 0
    return sum(1 for key in state.keys() if _is_lora_state_key(str(key)))


def _collect_lora_state_dict(model: AutoModelForCausalLM) -> dict[str, object]:
    import torch

    state: dict[str, object] = {}
    for key, tensor in model.state_dict().items():
        key = _normalize_adapter_state_key(key)
        if not _is_lora_state_key(key):
            continue
        if torch.is_tensor(tensor):
            state[key] = tensor.detach().cpu().contiguous()
    return state


def _repair_or_validate_peft_adapter(output_path: Path, model: AutoModelForCausalLM) -> None:
    weight_path = _adapter_weight_path(output_path)
    if weight_path is not None:
        lora_count = _count_saved_lora_tensors(weight_path)
        if lora_count > 0:
            logger.info("[INFO] saved LoRA adapter contains %d LoRA tensors: %s", lora_count, weight_path)
            return
        logger.warning("[WARN] saved adapter has no LoRA tensors; rebuilding adapter weights from model state.")

    fallback_state = _collect_lora_state_dict(model)
    if not fallback_state:
        raise RuntimeError(
            f"Saved PEFT adapter at {output_path} has no LoRA tensors, and no LoRA tensors could be collected "
            "from the in-memory model. The checkpoint would behave like the base model."
        )

    import safetensors.torch

    repaired_path = output_path / "adapter_model.safetensors"
    safetensors.torch.save_file(fallback_state, str(repaired_path), metadata={"format": "pt"})
    logger.warning(
        "[WARN] Rewrote %s with %d LoRA tensors collected from the in-memory model.",
        repaired_path,
        len(fallback_state),
    )


def _export_zero3_checkpoint_to_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ds_ckpt_dir: Path,
    output_path: Path,
) -> bool:
    try:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    except ImportError:
        logger.warning(
            "[WARN] DeepSpeed is unavailable when exporting %s; skipped automatic HF export.",
            ds_ckpt_dir,
        )
        return False

    try:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(str(ds_ckpt_dir))
        model.save_pretrained(str(output_path), state_dict=state_dict)
        tokenizer.save_pretrained(str(output_path))
    except Exception as exc:
        logger.warning(
            "[WARN] Failed to export DeepSpeed checkpoint %s to Hugging Face weights: %s",
            ds_ckpt_dir,
            exc,
        )
        return False

    logger.info("[INFO] Exported Hugging Face weights to %s from %s", output_path, ds_ckpt_dir)
    return True


def _drop_zero3_placeholder_safetensors(output_path: Path) -> None:
    single_file = output_path / "model.safetensors"
    index_file = output_path / "model.safetensors.index.json"
    if not single_file.exists() or not index_file.exists():
        return

    try:
        from safetensors import safe_open

        has_zero_tensor = False
        with safe_open(str(single_file), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for key in keys:
                if 0 in tuple(handle.get_tensor(key).shape):
                    has_zero_tensor = True
                    break
    except Exception as exc:
        logger.warning("[WARN] Could not inspect %s for ZeRO-3 placeholders: %s", single_file, exc)
        return

    if not has_zero_tensor:
        return

    single_file.unlink()
    logger.warning(
        "[WARN] Removed ZeRO-3 placeholder %s because sharded Hugging Face weights are available via %s.",
        single_file,
        index_file,
    )


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
        if is_peft_model(unwrapped):
            try:
                from peft import get_peft_model_state_dict
            except ImportError as exc:
                raise RuntimeError("Saving a LoRA checkpoint requires the `peft` package.") from exc

            state_dict = get_peft_model_state_dict(
                unwrapped,
                state_dict=accelerator.get_state_dict(model),
            )
            unwrapped.save_pretrained(
                str(output_path),
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state_dict,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(str(output_path))
                _repair_or_validate_peft_adapter(output_path, unwrapped)
            accelerator.wait_for_everyone()
            return

        state_dict = accelerator.get_state_dict(model)
        unwrapped.save_pretrained(
            str(output_path),
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )

        if accelerator.is_main_process:
            tokenizer.save_pretrained(str(output_path))
            _drop_zero3_placeholder_safetensors(output_path)

    except ValueError as exc:
        if "stage3_gather_16bit_weights_on_model_save" not in str(exc):
            raise
        if not hasattr(model, "save_checkpoint"):
            raise

        ds_ckpt_dir = output_path / "ds_checkpoint"
        model.save_checkpoint(str(ds_ckpt_dir))

        if accelerator.is_main_process:
            tokenizer.save_pretrained(str(output_path))
            logger.warning(
                "[WARN] DeepSpeed ZeRO-3 16-bit gather is disabled; saved a DeepSpeed checkpoint to %s. "
                "Convert to fp32 using zero_to_fp32.py or enable stage3_gather_16bit_weights_on_model_save.",
                ds_ckpt_dir,
            )
            _export_zero3_checkpoint_to_hf(unwrapped, tokenizer, ds_ckpt_dir, output_path)
            _drop_zero3_placeholder_safetensors(output_path)

    accelerator.wait_for_everyone()


def load_split(
    path: str | Path,
    *,
    dataset: str | None = None,
    label_schema: str | None = None,
) -> list[SampleRecord]:
    from fact_checking.data.io import load_split as _load_split

    return _load_split(path, dataset=dataset, label_schema=label_schema)


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


def save_eval_artifacts(
    eval_dir: Path,
    metrics: dict[str, float | dict[str, dict[str, float]]],
    confusion_matrix: np.ndarray,
    confusion_labels: list[str],
    prediction_records: list[dict[str, object]] | None = None,
    predictions_filename: str = "predictions.jsonl",
    title: str = "Confusion Matrix",
    labels: list[str] | None = None,
) -> dict[str, str]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    display_labels = list(labels or LABELS)

    metrics_path = eval_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    confusion_data_path = eval_dir / "confusion_matrix.json"
    with confusion_data_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "gold_labels": display_labels,
                "pred_labels": confusion_labels,
                "matrix": confusion_matrix.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    predictions_path = eval_dir / predictions_filename
    with predictions_path.open("w", encoding="utf-8") as f:
        for record in prediction_records or []:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(confusion_matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(confusion_labels)))
        ax.set_xticklabels(confusion_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(display_labels)))
        ax.set_yticklabels(display_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(title)
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        confusion_png_path = eval_dir / "confusion_matrix.png"
        fig.savefig(confusion_png_path, dpi=200)
        plt.close(fig)
    except Exception:
        confusion_png_path = ""

    return {
        "metrics_path": str(metrics_path),
        "confusion_data_path": str(confusion_data_path),
        "confusion_png_path": str(confusion_png_path),
        "predictions_path": str(predictions_path),
    }


def _save_eval_artifacts(
    output_dir: Path,
    global_step: int,
    metrics: dict[str, float | dict[str, dict[str, float]]],
    confusion_matrix: np.ndarray,
    confusion_labels: list[str],
    prediction_records: list[dict[str, object]] | None = None,
    labels: list[str] | None = None,
    eval_root: Path | None = None,
) -> dict[str, str]:
    step_dir = (Path(eval_root) if eval_root is not None else output_dir / "eval") / f"step-{global_step}"
    return save_eval_artifacts(
        eval_dir=step_dir,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        confusion_labels=confusion_labels,
        prediction_records=prediction_records,
        predictions_filename="val_predictions.jsonl",
        title=f"Confusion Matrix @ step {global_step}",
        labels=labels,
    )
