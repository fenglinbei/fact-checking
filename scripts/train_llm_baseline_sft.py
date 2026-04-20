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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

from liar_raw.baselines.llm_baseline import build_sft_instances, load_jsonl
from liar_raw import LABELS, LABEL2ID
from liar_raw.baselines.llm_baseline import build_evidence_block, build_zero_shot_prompt
from liar_raw.config import load_yaml


def _flash_attn2_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _fla_fast_path_available() -> bool:
    return importlib.util.find_spec("fla") is not None and importlib.util.find_spec("causal_conv1d") is not None


@dataclass
class SFTDatasetBuilder:
    tokenizer: AutoTokenizer
    max_length: int
    cache_dir: Path | None = None

    def build(self, instances: list[dict[str, str]], split: str, accelerator: Accelerator) -> Dataset:
        if self.cache_dir is None:
            tokenized = _tokenize_instances(instances=instances, tokenizer=self.tokenizer, max_length=self.max_length)
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
                tokenized = _tokenize_instances(instances=instances, tokenizer=self.tokenizer, max_length=self.max_length)
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


def _tokenize_instances(instances: list[dict[str, str]], tokenizer: AutoTokenizer, max_length: int) -> list[dict[str, list[int]]]:
    tokenized: list[dict[str, list[int]]] = []
    for row in instances:
        text = f"{row['prompt']} {row['target']}"
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        tokenized.append(
            {
                "input_ids": ids,
                "attention_mask": mask,
                "labels": ids[:],
            }
        )
    return tokenized


class TokenizedDataset(Dataset):
    def __init__(self, tokenized: list[dict[str, list[int]]]) -> None:
        self.tokenized = tokenized

    def __len__(self) -> int:
        return len(self.tokenized)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.tokenized[idx]


class EvalPromptDataset(Dataset):
    def __init__(self, rows: list[dict], top_k: int, use_context: bool, context_k: int) -> None:
        self.samples: list[dict[str, str | int]] = []
        for row in rows:
            gold_label = str(row.get("label", "")).strip().lower()
            if gold_label not in LABEL2ID:
                continue
            evidence_block = build_evidence_block(row, top_k=top_k, use_context=use_context, context_k=context_k)
            prompt = build_zero_shot_prompt(claim=str(row.get("claim", "")), evidence_block=evidence_block)
            self.samples.append({"prompt": prompt, "gold_id": LABEL2ID[gold_label]})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str | int]:
        return self.samples[idx]


def build_dataloader(
    dataset: Dataset,
    collator: DataCollatorForLanguageModeling,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        shuffle=shuffle,
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
) -> DataLoader:
    def _collate_fn(items: list[dict[str, str | int]]) -> dict[str, torch.Tensor]:
        prompts = [str(x["prompt"]) for x in items]
        gold_ids = torch.tensor([int(x["gold_id"]) for x in items], dtype=torch.long)
        enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
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


def _parse_label_id(raw_text: str) -> int:
    clean = raw_text.strip().lower()
    clean = re.sub(r"[^a-z\- ]", " ", clean)
    tokens = [t for t in clean.split() if t]
    if not tokens:
        return -1
    joined = " ".join(tokens)
    for label in LABELS:
        if re.search(rf"\b{re.escape(label)}\b", joined):
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
    max_new_tokens: int = 24,
) -> dict[str, float]:
    model.eval()
    all_pred_ids: list[torch.Tensor] = []
    all_gold_ids: list[torch.Tensor] = []
    pad_id = -100
    eval_progress = tqdm(
        total=len(dataloader),
        desc="eval",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )

    with torch.no_grad():
        for batch in dataloader:
            gold_ids = batch["gold_ids"]
            if gold_ids.numel() == 0:
                eval_progress.update(1)
                continue

            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                synced_gpus=accelerator.num_processes > 1,
            )
            prompt_lengths = batch["attention_mask"].sum(dim=1)
            pred_ids: list[int] = []
            for i in range(generated.shape[0]):
                gen_ids = generated[i, int(prompt_lengths[i]) :]
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
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            state_dict = accelerator.get_state_dict(model)
        except ValueError as exc:
            if "stage3_gather_16bit_weights_on_model_save" not in str(exc):
                raise
            if not hasattr(model, "save_checkpoint"):
                raise

            ds_ckpt_dir = output_path / "ds_checkpoint"
            model.save_checkpoint(str(ds_ckpt_dir))
            tokenizer.save_pretrained(str(output_path))
            print(
                "[WARN] DeepSpeed ZeRO-3 16-bit gather is disabled; saved a DeepSpeed checkpoint to "
                f"{ds_ckpt_dir}. Convert to fp32 using zero_to_fp32.py or enable "
                "stage3_gather_16bit_weights_on_model_save."
            )
            return

        unwrapped.save_pretrained(
            str(output_path),
            is_main_process=True,
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        tokenizer.save_pretrained(str(output_path))


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

    train_instances = build_sft_instances(train_rows, top_k=top_k, use_context=use_context, context_k=context_k)
    val_instances = build_sft_instances(val_rows, top_k=top_k, use_context=use_context, context_k=context_k)
    val_eval_ds = EvalPromptDataset(val_rows, top_k=top_k, use_context=use_context, context_k=context_k)

    model_name_or_path = str(baseline_cfg.get("model_name_or_path", "/data/models/Qwen3.5-9B"))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    cache_dir = Path(str(cache_dir_cfg)) if cache_dir_cfg else Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline")) / "tokenized_cache"
    builder = SFTDatasetBuilder(tokenizer=tokenizer, max_length=int(train_cfg.get("max_length", 2048)), cache_dir=cache_dir)
    train_ds = builder.build(train_instances, split="train", accelerator=accelerator)
    val_ds = builder.build(val_instances, split="val", accelerator=accelerator)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    num_workers = int(train_cfg.get("dataloader_num_workers", 0))
    train_dl = build_dataloader(
        train_ds,
        collator=data_collator,
        batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        num_workers=num_workers,
        shuffle=True,
    )
    val_dl = build_dataloader(
        val_ds,
        collator=data_collator,
        batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
    )
    val_eval_dl = build_eval_dataloader(
        val_eval_ds,
        tokenizer=tokenizer,
        batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        num_workers=num_workers,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        fused=torch.cuda.is_available(),
    )

    num_epochs = int(math.ceil(float(train_cfg.get("num_train_epochs", 2.0))))
    update_steps_per_epoch = max(1, math.ceil(len(train_dl) / accelerator.gradient_accumulation_steps))
    max_train_steps = num_epochs * update_steps_per_epoch
    warmup_steps = int(max_train_steps * float(train_cfg.get("warmup_ratio", 0.03)))
    scheduler = get_scheduler(
        name=str(train_cfg.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dl, val_dl, val_eval_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, val_eval_dl, scheduler
    )

    output_dir = Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline")) / ("b1" if use_context else "b0")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_steps = int(train_cfg.get("logging_steps", 20))
    eval_steps = int(train_cfg.get("eval_steps", 500))
    save_steps = int(train_cfg.get("save_steps", 500))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
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
                    eval_metrics = evaluate(
                        model,
                        val_eval_dl,
                        tokenizer,
                        accelerator,
                        max_new_tokens=int(baseline_cfg.get("max_new_tokens", 24)),
                    )
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
                        save_model(accelerator, model, tokenizer, output_dir / "best")

                if global_step % save_steps == 0:
                    save_model(accelerator, model, tokenizer, output_dir / f"checkpoint-{global_step}")

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    save_model(accelerator, model, tokenizer, output_dir / "final")
    accelerator.end_training()


if __name__ == "__main__":
    main()
