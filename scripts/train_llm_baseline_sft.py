from __future__ import annotations

import argparse
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path

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
from liar_raw.config import load_yaml


def _flash_attn2_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _fla_fast_path_available() -> bool:
    return importlib.util.find_spec("fla") is not None and importlib.util.find_spec("causal_conv1d") is not None


@dataclass
class SFTDatasetBuilder:
    tokenizer: AutoTokenizer
    max_length: int

    def build(self, instances: list[dict[str, str]]) -> Dataset:
        return LazyTokenizedDataset(instances=instances, tokenizer=self.tokenizer, max_length=self.max_length)


class LazyTokenizedDataset(Dataset):
    def __init__(self, instances: list[dict[str, str]], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        row = self.instances[idx]
        text = f"{row['prompt']} {row['target']}"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": ids[:],
        }


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


def evaluate(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> float:
    model.eval()
    loss_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss.detach().float()
            gathered = accelerator.gather_for_metrics(loss.unsqueeze(0))
            loss_list.append(gathered)

    if not loss_list:
        return 0.0

    all_losses = torch.cat(loss_list)
    val_loss = all_losses.mean().item()
    model.train()
    return val_loss


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

    use_context = bool(baseline_cfg.get("use_context", False))
    top_k = int(baseline_cfg.get("top_k", 8))
    context_k = int(baseline_cfg.get("context_k", 1))

    train_instances = build_sft_instances(train_rows, top_k=top_k, use_context=use_context, context_k=context_k)
    val_instances = build_sft_instances(val_rows, top_k=top_k, use_context=use_context, context_k=context_k)

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

    builder = SFTDatasetBuilder(tokenizer=tokenizer, max_length=int(train_cfg.get("max_length", 2048)))
    train_ds = builder.build(train_instances)
    val_ds = builder.build(val_instances)

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

    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, val_dl, scheduler)

    output_dir = Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline")) / ("b1" if use_context else "b0")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_steps = int(train_cfg.get("logging_steps", 20))
    eval_steps = int(train_cfg.get("eval_steps", 500))
    save_steps = int(train_cfg.get("save_steps", 500))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    best_val_loss = float("inf")

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
                    val_loss = evaluate(model, val_dl, accelerator)
                    accelerator.log({"eval/loss": val_loss}, step=global_step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
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
