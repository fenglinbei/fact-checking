from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from liar_raw.baselines.llm_baseline import build_sft_instances, load_jsonl
from liar_raw.config import load_yaml


@dataclass
class SFTDatasetBuilder:
    tokenizer: AutoTokenizer
    max_length: int

    def build(self, instances: list[dict[str, str]]) -> Dataset:
        texts = [f"{x['prompt']} {x['target']}" for x in instances]
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        items = [
            {
                "input_ids": ids,
                "attention_mask": mask,
                "labels": ids[:],
            }
            for ids, mask in zip(enc["input_ids"], enc["attention_mask"])
        ]
        return ListTokenizedDataset(items)


class ListTokenizedDataset(Dataset):
    def __init__(self, items: list[dict[str, list[int]]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.items[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT train for LLM baselines.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    baseline_cfg = cfg["baseline"]
    train_cfg = cfg["sft_train"]
    wandb_cfg = cfg.get("wandb", {})

    train_rows = load_jsonl(data_cfg["train_candidates"])
    val_rows = load_jsonl(data_cfg["val_candidates"])

    use_context = bool(baseline_cfg.get("use_context", False))
    top_k = int(baseline_cfg.get("top_k", 8))
    context_k = int(baseline_cfg.get("context_k", 1))

    train_instances = build_sft_instances(train_rows, top_k=top_k, use_context=use_context, context_k=context_k)
    val_instances = build_sft_instances(val_rows, top_k=top_k, use_context=use_context, context_k=context_k)

    model_name_or_path = str(baseline_cfg.get("model_name_or_path", "/data/models/Qwen3.5-9B"))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    builder = SFTDatasetBuilder(tokenizer=tokenizer, max_length=int(train_cfg.get("max_length", 2048)))
    train_ds = builder.build(train_instances)
    val_ds = builder.build(val_instances)

    output_dir = Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline")) / ("b1" if use_context else "b0")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_name = "b1" if use_context else "b0"

    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    report_to = "wandb" if wandb_enabled else "none"
    run_name = str(wandb_cfg.get("run_name", f"llm_baseline_{baseline_name}_sft"))
    if wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_cfg.get("project", "fact-checking-stage-ab")))
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
        os.environ.setdefault("WANDB_LOG_MODEL", str(wandb_cfg.get("log_model", "false")))
        os.environ.setdefault("WANDB_WATCH", str(wandb_cfg.get("watch", "false")))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-5)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 2.0)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        bf16=bool(train_cfg.get("bf16", True)),
        logging_steps=int(train_cfg.get("logging_steps", 20)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        eval_steps=int(train_cfg.get("eval_steps", 500)),
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to=report_to,
        run_name=run_name,
        deepspeed=str(train_cfg["deepspeed_config"]),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    main()
