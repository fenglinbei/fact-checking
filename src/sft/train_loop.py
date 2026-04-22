from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

from fact_checking.baselines.llm_baseline import load_jsonl
from fact_checking import LABELS, LABEL2ID
from fact_checking.baselines.llm_baseline import build_evidence_block
from fact_checking.config import load_yaml
from prompting.output_strategy import OutputStrategy, build_output_strategy, _infer_output_mode
from prompting.stats import PromptPreparationRecord, save_prompt_statistics, summarize_prompt_preparation
from prompting.truncation import (
    PromptTruncationStrategy,
    TailEvidenceTruncationStrategy,
)


def _flash_attn2_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _fla_fast_path_available() -> bool:
    return importlib.util.find_spec("fla") is not None and importlib.util.find_spec("causal_conv1d") is not None


@dataclass
class CausalLMSFTCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of is not None:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for x in features:
            pad_len = max_len - len(x["input_ids"])

            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class PreparedSample:
    prompt: str
    target: str
    gold_id: int
    prompt_length_before_trunc: int
    prompt_length_after_trunc: int
    evidence_count_before_trunc: int
    evidence_count_after_trunc: int
    was_truncated: bool
    overflow_before_trunc: bool
    overflow_after_trunc: bool


def build_prompt_truncation_strategy(baseline_cfg: dict) -> PromptTruncationStrategy:
    trunc_cfg = baseline_cfg.get("prompt_truncation", {}) or {}
    if not bool(trunc_cfg.get("enabled", False)):
        return PromptTruncationStrategy()

    strategy_name = str(trunc_cfg.get("strategy", "tail_evidence")).strip().lower()
    if strategy_name == "tail_evidence":
        return TailEvidenceTruncationStrategy(
            min_evidence_to_keep=int(trunc_cfg.get("min_evidence_to_keep", 1))
        )

    raise ValueError(
        f"Unsupported baseline.prompt_truncation.strategy={strategy_name}. "
        "Use 'tail_evidence'."
    )


def _normalize_gold_label(row: dict) -> str:
    gold_label = str(row.get("label", "")).strip().lower()
    if gold_label not in LABEL2ID:
        return ""
    return gold_label


def build_prepared_samples(
    rows: list[dict],
    top_k: int,
    use_context: bool,
    context_k: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    output_strategy: OutputStrategy,
    truncation_strategy: PromptTruncationStrategy,
) -> tuple[list[PreparedSample], list[PromptPreparationRecord]]:
    samples: list[PreparedSample] = []
    records: list[PromptPreparationRecord] = []

    for row in rows:
        gold_label = _normalize_gold_label(row)
        if not gold_label:
            continue

        claim = str(row.get("claim", ""))
        evidence_block = build_evidence_block(
            row,
            top_k=top_k,
            use_context=use_context,
            context_k=context_k,
        )
        truncation_result = truncation_strategy.apply(
            claim=claim,
            evidence_block=evidence_block,
            tokenizer=tokenizer,
            max_length=max_length,
            prompt_builder=output_strategy.build_prompt,
        )

        samples.append(
            PreparedSample(
                prompt=truncation_result.prompt,
                target=output_strategy.build_target(row, gold_label),
                gold_id=LABEL2ID[gold_label],
                prompt_length_before_trunc=truncation_result.prompt_length_before_trunc,
                prompt_length_after_trunc=truncation_result.prompt_length_after_trunc,
                evidence_count_before_trunc=truncation_result.evidence_count_before_trunc,
                evidence_count_after_trunc=truncation_result.evidence_count_after_trunc,
                was_truncated=truncation_result.was_truncated,
                overflow_before_trunc=truncation_result.overflow_before_trunc,
                overflow_after_trunc=truncation_result.overflow_after_trunc,
            )
        )
        records.append(
            PromptPreparationRecord(
                prompt_length_before_trunc=truncation_result.prompt_length_before_trunc,
                prompt_length_after_trunc=truncation_result.prompt_length_after_trunc,
                evidence_count_before_trunc=truncation_result.evidence_count_before_trunc,
                evidence_count_after_trunc=truncation_result.evidence_count_after_trunc,
                was_truncated=truncation_result.was_truncated,
                overflow_before_trunc=truncation_result.overflow_before_trunc,
                overflow_after_trunc=truncation_result.overflow_after_trunc,
            )
        )

    return samples, records


def _print_prompt_summary(summary: dict[str, object]) -> None:
    before = summary["prompt_length_before_truncation"]
    after = summary["prompt_length_after_truncation"]
    trunc = summary["evidence_truncation"]
    print(
        f"[PROMPT_STATS] split={summary['split']} mode={summary['output_mode']} "
        f"strategy={summary['prompt_truncation_strategy']} "
        f"pre_mean={before['mean']:.2f} pre_p95={before['p95']:.0f} pre_overflow={int(before['overflow_count'])} "
        f"post_mean={after['mean']:.2f} post_p95={after['p95']:.0f} post_overflow={int(after['overflow_count'])} "
        f"truncated={trunc['truncated_count']} trunc_rate={trunc['truncation_rate']:.4f}"
    )
@dataclass
class SFTDatasetBuilder:
    tokenizer: AutoTokenizer
    max_length: int
    padding: str = "max_length"
    cache_dir: Path | None = None

    def build(self, instances: list[dict[str, str]], split: str, accelerator: Accelerator) -> Dataset:
        if self.cache_dir is None:
            tokenized = _tokenize_instances(
                instances=instances,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                padding=self.padding,
            )
            return TokenizedDataset(tokenized=tokenized)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / _build_cache_name(
            split=split,
            max_length=self.max_length,
            tokenizer_name=self.tokenizer.name_or_path,
            instances=instances,
        )
        with accelerator.main_process_first():
            if not cache_path.exists():
                tokenized = _tokenize_instances(
                    instances=instances,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    padding=self.padding,
                )
                torch.save(tokenized, cache_path)

        tokenized = torch.load(cache_path, map_location="cpu", weights_only=False)
        return TokenizedDataset(tokenized=tokenized)


def _build_cache_name(split: str, max_length: int, tokenizer_name: str, instances: list[dict[str, str]]) -> str:
    tok_hash = hashlib.md5(tokenizer_name.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    data_hash = hashlib.md5(usedforsecurity=False)
    data_hash.update(str(len(instances)).encode("utf-8"))
    for row in instances:
        data_hash.update(row["prompt"].encode("utf-8"))
        data_hash.update(row["target"].encode("utf-8"))
    return f"{split}_ml{max_length}_{tok_hash}_{data_hash.hexdigest()[:16]}.pt"


def _tokenize_instances(
    instances: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    max_length: int,
    padding: str,
) -> list[dict[str, list[int]]]:
    tokenized: list[dict[str, list[int]]] = []
    eos_id = tokenizer.eos_token_id

    for row in instances:
        prompt_text = row["prompt"].rstrip() + " "
        target_text = row["target"].strip()

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if eos_id is not None:
            target_ids = target_ids + [eos_id]

        # 关键：为 target 预留空间，不能让 truncation 把 label 截掉
        max_prompt_len = max_length - len(target_ids)
        if max_prompt_len <= 0:
            input_ids = target_ids[:max_length]
            labels = input_ids[:]
        else:
            if len(prompt_ids) > max_prompt_len:
                # 保留 prompt 尾部，至少保住 "Label:"
                prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

        attention_mask = [1] * len(input_ids)

        if padding == "max_length":
            pad_len = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return tokenized


class TokenizedDataset(Dataset):
    def __init__(self, tokenized: list[dict[str, list[int]]]) -> None:
        self.tokenized = tokenized
        self.lengths = [int(sum(x["attention_mask"])) for x in tokenized]

    def __len__(self) -> int:
        return len(self.tokenized)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.tokenized[idx]


class EvalPromptDataset(Dataset):
    def __init__(self, samples: list[PreparedSample]) -> None:
        self.samples: list[dict[str, str | int]] = [
            {"prompt": sample.prompt, "gold_id": sample.gold_id}
            for sample in samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str | int]:
        return self.samples[idx]


def build_dataloader(
    dataset: Dataset,
    collator: DataCollatorForLanguageModeling | CausalLMSFTCollator,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    use_length_bucket: bool,
) -> DataLoader:
    sampler: Sampler | None = None
    if use_length_bucket:
        lengths = getattr(dataset, "lengths", None)
        if lengths is not None:
            sampler = LengthGroupedSampler(
                batch_size=batch_size,
                dataset=dataset,
                lengths=lengths,
            )
    return DataLoader(
        dataset,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=shuffle,
    )


def build_eval_dataloader(
    dataset: EvalPromptDataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_workers: int,
    max_length: int,
    padding: str,
) -> DataLoader:
    def _collate_fn(items):
        prompts = [str(x["prompt"]) for x in items]
        gold_ids = torch.tensor([int(x["gold_id"]) for x in items], dtype=torch.long)

        old_padding_side = tokenizer.padding_side
        old_truncation_side = getattr(tokenizer, "truncation_side", "right")

        # decoder-only generation 必须 left padding
        tokenizer.padding_side = "left"

        # 如果 prompt 过长，至少保住末尾的 "Label:"
        tokenizer.truncation_side = "left"

        try:
            enc = tokenizer(
                prompts,
                padding="max_length" if padding == "max_length" else True,
                truncation=True,
                max_length=max_length,
                # 关键：batch_size=1 时建议先去掉这个；
                # 如果你坚持保留，也必须配合 left padding。
                # pad_to_multiple_of=8,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = old_padding_side
            tokenizer.truncation_side = old_truncation_side

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "gold_ids": gold_ids,
        }

    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )


_LABEL_PATTERNS = [
    ("pants-fire", r"\bpants\s*-?\s*fire\b"),
    ("barely-true", r"\bbarely\s*-?\s*true\b"),
    ("half-true", r"\bhalf\s*-?\s*true\b"),
    ("mostly-true", r"\bmostly\s*-?\s*true\b"),
    ("false", r"\bfalse\b"),
    ("true", r"\btrue\b"),
]

def _parse_label_id(raw_text: str) -> int:
    label_line_match = re.search(r"(?mi)^\s*label\s*:\s*([^\n\r]+)", raw_text)
    if label_line_match:
        label_candidate = label_line_match.group(1)
        label_from_line = _parse_label_id(label_candidate)
        if label_from_line >= 0:
            return label_from_line

    clean = raw_text.strip().lower()
    clean = re.sub(r"[_/]+", " ", clean)
    clean = re.sub(r"[^a-z\-\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean)

    for label, pattern in _LABEL_PATTERNS:
        if re.search(pattern, clean):
            return LABEL2ID[label]

    return -1


def _compute_classification_metrics(pred_ids: np.ndarray, gold_ids: np.ndarray) -> dict[str, float | dict[str, dict[str, float]]]:
    eps = 1e-12
    per_class: dict[str, dict[str, float]] = {}
    p_list: list[float] = []
    r_list: list[float] = []
    f1_list: list[float] = []
    for label_id, label in enumerate(LABELS):
        tp = float(np.sum((pred_ids == label_id) & (gold_ids == label_id)))
        fp = float(np.sum((pred_ids == label_id) & (gold_ids != label_id)))
        fn = float(np.sum((pred_ids != label_id) & (gold_ids == label_id)))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        p_list.append(precision)
        r_list.append(recall)
        f1_list.append(f1)
    macro_p = float(np.mean(p_list))
    macro_r = float(np.mean(r_list))
    macro_f1 = float(np.mean(f1_list))
    parse_error_rate = float(np.mean(pred_ids < 0))
    accuracy = float(np.mean(pred_ids == gold_ids))
    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "parse_error_rate": parse_error_rate,
        "per_class": per_class,
    }


def _build_confusion_matrix(pred_ids: np.ndarray, gold_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels_with_parse = LABELS + ["parse_error"]
    mat = np.zeros((len(LABELS), len(labels_with_parse)), dtype=np.int64)
    for g, p in zip(gold_ids.tolist(), pred_ids.tolist()):
        pred_idx = p if p >= 0 else len(LABELS)
        mat[g, pred_idx] += 1
    return mat, labels_with_parse


def _summarize_prompt_lengths(
    instances: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    split: str,
    max_length: int,
) -> dict[str, float]:
    if not instances:
        return {
            "count": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "overflow_count": 0.0,
            "overflow_rate": 0.0,
        }

    lengths = np.array(
        [len(tokenizer(row["prompt"], truncation=False, add_special_tokens=True)["input_ids"]) for row in instances],
        dtype=np.int64,
    )
    overflow = lengths > int(max_length)
    summary = {
        "count": float(lengths.size),
        "min": float(lengths.min()),
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": float(lengths.max()),
        "mean": float(lengths.mean()),
        "overflow_count": float(np.sum(overflow)),
        "overflow_rate": float(np.mean(overflow)),
    }
    print(
        f"[PROMPT_STATS] split={split} count={int(summary['count'])} "
        f"min={summary['min']:.0f} p50={summary['p50']:.0f} p90={summary['p90']:.0f} "
        f"p95={summary['p95']:.0f} p99={summary['p99']:.0f} max={summary['max']:.0f} "
        f"mean={summary['mean']:.2f} overflow(>{max_length})={int(summary['overflow_count'])} "
        f"rate={summary['overflow_rate']:.4f}"
    )
    return summary


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


def _select_mini_val_rows(
    rows: list[dict],
    mini_val_size: int,
    mini_val_seed: int,
    accelerator: Accelerator,
) -> list[dict]:
    if mini_val_size <= 0 or mini_val_size >= len(rows):
        return rows

    rng = np.random.default_rng(mini_val_seed)
    indices = rng.choice(len(rows), size=mini_val_size, replace=False)
    mini_rows = [rows[int(i)] for i in indices.tolist()]
    if accelerator.is_main_process:
        print(
            f"[INFO] mini-val enabled: sampled {len(mini_rows)} / {len(rows)} "
            f"validation rows (seed={mini_val_seed})."
        )
    return mini_rows


def evaluate(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    max_length: int,
    max_new_tokens: int = 24,
) -> dict[str, float]:
    model.eval()

    unwrapped = accelerator.unwrap_model(model)
    old_use_cache = getattr(unwrapped.config, "use_cache", None)
    if old_use_cache is not None:
        unwrapped.config.use_cache = True

    all_pred_ids: list[torch.Tensor] = []
    all_gold_ids: list[torch.Tensor] = []
    pad_id = -100
    eval_progress = tqdm(
        total=len(dataloader),
        desc="eval",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )

    try:

        with torch.no_grad():
            for batch in dataloader:
                gold_ids = batch["gold_ids"]

                generated = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    synced_gpus=accelerator.num_processes > 1,
                )
                prompt_length = batch["input_ids"].shape[1]
                pred_ids: list[int] = []
                for i in range(generated.shape[0]):
                    gen_ids = generated[i, prompt_length:]
                    raw_pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    pred_ids.append(_parse_label_id(raw_pred))

                pred_tensor = torch.tensor(pred_ids, dtype=torch.long, device=gold_ids.device)
                pred_tensor = accelerator.pad_across_processes(pred_tensor, dim=0, pad_index=pad_id)
                gold_ids = accelerator.pad_across_processes(gold_ids, dim=0, pad_index=pad_id)
                gathered_pred = accelerator.gather(pred_tensor)
                gathered_gold = accelerator.gather(gold_ids)
                valid_mask = gathered_gold != pad_id
                if valid_mask.any():
                    all_pred_ids.append(gathered_pred[valid_mask].cpu())
                    all_gold_ids.append(gathered_gold[valid_mask].cpu())
                eval_progress.update(1)

    finally:
        if old_use_cache is not None:
            unwrapped.config.use_cache = old_use_cache

    eval_progress.close()

    if not all_gold_ids:
        model.train()
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "parse_error_rate": 0.0,
            "per_class": {},
            "confusion_matrix": np.zeros((len(LABELS), len(LABELS) + 1), dtype=np.int64),
            "confusion_labels": LABELS + ["parse_error"],
        }

    pred_np = torch.cat(all_pred_ids).numpy()
    gold_np = torch.cat(all_gold_ids).numpy()
    metrics = _compute_classification_metrics(pred_np, gold_np)
    confusion_matrix, confusion_labels = _build_confusion_matrix(pred_np, gold_np)
    metrics["confusion_matrix"] = confusion_matrix
    metrics["confusion_labels"] = confusion_labels
    model.train()
    return metrics


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
            print(
                "[WARN] DeepSpeed ZeRO-3 16-bit gather is disabled; saved a DeepSpeed checkpoint to "
                f"{ds_ckpt_dir}. Convert to fp32 using zero_to_fp32.py or enable "
                "stage3_gather_16bit_weights_on_model_save."
            )

    accelerator.wait_for_everyone()


def maybe_empty_cache(accelerator: Accelerator) -> None:
    if torch.cuda.is_available():
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT train for LLM baselines (Accelerate).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mini-val-size",
        type=int,
        default=None,
        help="If > 0, randomly sample this many examples from val set for faster eval (e.g., 32/64/128).",
    )
    parser.add_argument(
        "--mini-val-seed",
        type=int,
        default=None,
        help="Random seed for mini val sampling. Falls back to sft_train.seed or 42.",
    )
    parser.add_argument(
        "--prompt-length-stats-only",
        action="store_true",
        help="Only build prompts and report prompt token length statistics, then exit.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    baseline_cfg = cfg["baseline"]
    train_cfg = cfg["sft_train"]
    wandb_cfg = cfg.get("wandb", {})

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_cfg.get("project", "fact-checking-stage-ab")))
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
        os.environ.setdefault("WANDB_LOG_MODEL", str(wandb_cfg.get("log_model", "false")))
        os.environ.setdefault("WANDB_WATCH", str(wandb_cfg.get("watch", "false")))

    mixed_precision = "bf16" if bool(train_cfg.get("bf16", True)) else "no"
    report_to = "wandb" if wandb_enabled else None
    run_name = str(wandb_cfg.get("run_name", "llm_baseline_sft"))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        mixed_precision=mixed_precision,
        log_with=report_to,
    )

    if wandb_enabled:
        accelerator.init_trackers(project_name=os.environ["WANDB_PROJECT"], config=cfg, init_kwargs={"wandb": {"name": run_name}})

    train_rows = load_jsonl(data_cfg["train_candidates"])
    val_rows = load_jsonl(data_cfg["val_candidates"])

    mini_val_size = int(
        args.mini_val_size if args.mini_val_size is not None else int(train_cfg.get("mini_val_size", 0))
    )
    mini_val_seed = int(
        args.mini_val_seed
        if args.mini_val_seed is not None
        else int(train_cfg.get("mini_val_seed", train_cfg.get("seed", 42)))
    )
    val_rows = _select_mini_val_rows(
        rows=val_rows,
        mini_val_size=mini_val_size,
        mini_val_seed=mini_val_seed,
        accelerator=accelerator,
    )

    use_context = bool(baseline_cfg.get("use_context", False))
    top_k = int(baseline_cfg.get("top_k", 8))
    context_k = int(baseline_cfg.get("context_k", 1))
    output_mode = _infer_output_mode(baseline_cfg)
    output_strategy = build_output_strategy(baseline_cfg)
    truncation_strategy = build_prompt_truncation_strategy(baseline_cfg)

    output_base_dir = Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline"))
    output_dir = output_base_dir
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    model_name_or_path = str(baseline_cfg.get("model_name_or_path", "/data/models/Qwen3.5-9B"))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(train_cfg.get("max_length", 2048))
    train_samples, train_prompt_records = build_prepared_samples(
        train_rows,
        top_k=top_k,
        use_context=use_context,
        context_k=context_k,
        tokenizer=tokenizer,
        max_length=max_length,
        output_strategy=output_strategy,
        truncation_strategy=truncation_strategy,
    )
    val_samples, val_prompt_records = build_prepared_samples(
        val_rows,
        top_k=top_k,
        use_context=use_context,
        context_k=context_k,
        tokenizer=tokenizer,
        max_length=max_length,
        output_strategy=output_strategy,
        truncation_strategy=truncation_strategy,
    )

    train_instances = [{"prompt": sample.prompt, "target": sample.target} for sample in train_samples]
    val_instances = [{"prompt": sample.prompt, "target": sample.target} for sample in val_samples]
    val_eval_ds = EvalPromptDataset(val_samples)

    train_prompt_summary = summarize_prompt_preparation(
        train_prompt_records,
        max_length=max_length,
        split="train",
        truncation_strategy_name=truncation_strategy.name,
        output_mode=output_strategy.name,
    )
    val_prompt_summary = summarize_prompt_preparation(
        val_prompt_records,
        max_length=max_length,
        split="val",
        truncation_strategy_name=truncation_strategy.name,
        output_mode=output_strategy.name,
    )
    if accelerator.is_main_process:
        _print_prompt_summary(train_prompt_summary)
        _print_prompt_summary(val_prompt_summary)
        prompt_stats_path = save_prompt_statistics(
            output_dir=output_dir,
            train_summary=train_prompt_summary,
            val_summary=val_prompt_summary,
        )
        print(f"[INFO] prompt statistics saved to {prompt_stats_path}")

    accelerator.wait_for_everyone()

    if args.prompt_length_stats_only:
        if accelerator.is_main_process:
            print("[INFO] prompt length stats finished. Exiting due to --prompt-length-stats-only.")
        return

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if torch.cuda.is_available() and mixed_precision == "bf16" else torch.float32,
    }
    if bool(train_cfg.get("use_flash_attention_2", True)):
        if _flash_attn2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif accelerator.is_main_process:
            print(
                "[WARN] sft_train.use_flash_attention_2=true, but flash-attn is not installed. "
                "Falling back to the default attention implementation."
            )
    if accelerator.is_main_process and not _fla_fast_path_available():
        print(
            "[INFO] FLA fast path is unavailable (requires both `fla` and `causal_conv1d`). "
            "This is separate from flash-attn and does not block training."
        )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    cache_dir_cfg = train_cfg.get("tokenized_cache_dir")
    cache_dir = Path(str(cache_dir_cfg)) if cache_dir_cfg else output_dir / "tokenized_cache"
    padding_strategy = str(train_cfg.get("padding", "max_length"))
    if padding_strategy not in {"max_length", "longest"}:
        raise ValueError(f"Unsupported sft_train.padding={padding_strategy}. Use 'max_length' or 'longest'.")
    use_length_bucket = bool(train_cfg.get("use_length_bucket", True))

    builder = SFTDatasetBuilder(
        tokenizer=tokenizer,
        max_length=max_length,
        padding=padding_strategy,
        cache_dir=cache_dir,
    )
    train_ds = builder.build(train_instances, split="train", accelerator=accelerator)
    val_ds = builder.build(val_instances, split="val", accelerator=accelerator)

    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    data_collator = CausalLMSFTCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    num_workers = int(train_cfg.get("dataloader_num_workers", 0))
    train_dl = build_dataloader(
        train_ds,
        collator=data_collator,
        batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        num_workers=num_workers,
        shuffle=True,
        use_length_bucket=use_length_bucket,
    )
    val_dl = build_dataloader(
        val_ds,
        collator=data_collator,
        batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
        use_length_bucket=False,
    )

    max_length = int(train_cfg.get("max_length", 2048))
    val_eval_dl = build_eval_dataloader(
        val_eval_ds,
        tokenizer=tokenizer,
        batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        num_workers=num_workers,
        max_length=max_length,
        padding=padding_strategy,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        fused=torch.cuda.is_available(),
    )

    num_epochs = int(math.ceil(float(train_cfg.get("num_train_epochs", 2.0))))
    effective_grad_accum_steps = int(accelerator.gradient_accumulation_steps)
    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if ds_plugin is not None:
        ds_cfg = getattr(ds_plugin, "deepspeed_config", {}) or {}
        ds_grad_accum_steps = ds_cfg.get("gradient_accumulation_steps")
        if ds_grad_accum_steps is not None:
            effective_grad_accum_steps = int(ds_grad_accum_steps)
    cfg_grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", effective_grad_accum_steps))
    if accelerator.is_main_process and effective_grad_accum_steps != cfg_grad_accum_steps:
        print(
            "[WARN] gradient_accumulation_steps mismatch detected: "
            f"sft_train={cfg_grad_accum_steps}, effective={effective_grad_accum_steps}. "
            "Using effective value for max_train_steps/progress bar."
        )
    pre_prepare_train_dl_len = len(train_dl)
    model, optimizer, train_dl, val_dl, val_eval_dl = accelerator.prepare(
        model, optimizer, train_dl, val_dl, val_eval_dl
    )
    post_prepare_train_dl_len = len(train_dl)
    if accelerator.is_main_process:
        print(
            "[INFO] dataloader length around accelerator.prepare: "
            f"before={pre_prepare_train_dl_len}, after={post_prepare_train_dl_len}"
        )
    if accelerator.is_main_process and post_prepare_train_dl_len != pre_prepare_train_dl_len:
        print(
            "[WARN] len(train_dl) changed after accelerator.prepare: "
            f"before={pre_prepare_train_dl_len}, after={post_prepare_train_dl_len}. "
            "This may indicate duplicated sharding/re-partitioning across distributed samplers."
        )
    update_steps_per_epoch = max(1, math.ceil(post_prepare_train_dl_len / effective_grad_accum_steps))
    max_train_steps = num_epochs * update_steps_per_epoch
    warmup_steps = int(max_train_steps * float(train_cfg.get("warmup_ratio", 0.03)))
    scheduler = get_scheduler(
        name=str(train_cfg.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    scheduler = accelerator.prepare(scheduler)
    if accelerator.is_main_process:
        print(
            "[INFO] train progress setup (post-prepare): "
            f"num_epochs={num_epochs}, "
            f"len(train_dl)={post_prepare_train_dl_len}, "
            f"effective_grad_accum_steps={effective_grad_accum_steps}, "
            f"update_steps_per_epoch={update_steps_per_epoch}, "
            f"max_train_steps={max_train_steps}"
        )

    logging_steps = int(train_cfg.get("logging_steps", 20))
    eval_steps = int(train_cfg.get("eval_steps", 500))
    save_steps = int(train_cfg.get("save_steps", 500))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    empty_cache_steps = int(train_cfg.get("empty_cache_steps", 0))
    empty_cache_on_eval = bool(train_cfg.get("empty_cache_on_eval", False))
    empty_cache_on_save = bool(train_cfg.get("empty_cache_on_save", False))

    progress_bar = tqdm(total=max_train_steps, disable=not accelerator.is_local_main_process)
    global_step = 0
    best_val_loss = float("-inf")

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % logging_steps == 0:
                    train_loss = accelerator.gather_for_metrics(loss.detach().float().unsqueeze(0)).mean().item()
                    lr = scheduler.get_last_lr()[0]
                    accelerator.log({"train/loss": train_loss, "train/lr": lr, "train/epoch": epoch}, step=global_step)

                if global_step % eval_steps == 0:
                    print(f"[rank {accelerator.process_index}] before evaluate step={global_step}", flush=True)
                    eval_metrics = evaluate(
                        model,
                        val_eval_dl,
                        tokenizer,
                        accelerator,
                        max_length=max_length,
                        max_new_tokens=int(baseline_cfg.get("max_new_tokens", 24)),
                    )
                    print(f"[rank {accelerator.process_index}] after evaluate step={global_step}", flush=True)
                    macro_f1 = float(eval_metrics["macro_f1"])
                    accelerator.log(
                        {
                            "eval/accuracy": float(eval_metrics["accuracy"]),
                            "eval/macro_precision": float(eval_metrics["macro_precision"]),
                            "eval/macro_recall": float(eval_metrics["macro_recall"]),
                            "eval/macro_f1": macro_f1,
                            "eval/parse_error_rate": float(eval_metrics["parse_error_rate"]),
                        },
                        step=global_step,
                    )
                    per_class = eval_metrics.get("per_class", {})
                    if isinstance(per_class, dict):
                        for label, label_metrics in per_class.items():
                            if isinstance(label_metrics, dict):
                                accelerator.log(
                                    {
                                        f"eval/{label}/precision": float(label_metrics.get("precision", 0.0)),
                                        f"eval/{label}/recall": float(label_metrics.get("recall", 0.0)),
                                        f"eval/{label}/f1": float(label_metrics.get("f1", 0.0)),
                                    },
                                    step=global_step,
                                )

                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        artifacts = _save_eval_artifacts(
                            output_dir=output_dir,
                            global_step=global_step,
                            metrics={
                                "step": global_step,
                                "accuracy": float(eval_metrics["accuracy"]),
                                "macro_precision": float(eval_metrics["macro_precision"]),
                                "macro_recall": float(eval_metrics["macro_recall"]),
                                "macro_f1": macro_f1,
                                "parse_error_rate": float(eval_metrics["parse_error_rate"]),
                                "per_class": per_class,
                            },
                            confusion_matrix=eval_metrics["confusion_matrix"],
                            confusion_labels=eval_metrics["confusion_labels"],
                        )
                        accelerator.log(
                            {
                                "eval/metrics_path": artifacts["metrics_path"],
                                "eval/confusion_data_path": artifacts["confusion_data_path"],
                                "eval/confusion_png_path": artifacts["confusion_png_path"],
                            },
                            step=global_step,
                        )

                    if macro_f1 > best_val_loss:
                        best_val_loss = macro_f1
                        print(f"[rank {accelerator.process_index}] before save best step={global_step}", flush=True)
                        save_model(accelerator, model, tokenizer, output_dir / "best")
                        print(f"[rank {accelerator.process_index}] after save best step={global_step}", flush=True)

                    if empty_cache_on_eval:
                        maybe_empty_cache(accelerator)

                if global_step % save_steps == 0:
                    print(f"[rank {accelerator.process_index}] before save best step={global_step}", flush=True)
                    save_model(accelerator, model, tokenizer, output_dir / f"checkpoint-{global_step}")
                    print(f"[rank {accelerator.process_index}] after save best step={global_step}", flush=True)

                    if empty_cache_on_save:
                        maybe_empty_cache(accelerator)

                if empty_cache_steps > 0 and global_step % empty_cache_steps == 0:
                    maybe_empty_cache(accelerator)

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    save_model(accelerator, model, tokenizer, output_dir / "final")
    accelerator.end_training()


if __name__ == "__main__":
    main()
