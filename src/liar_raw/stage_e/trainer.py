from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from liar_raw.utils.io import ensure_dir, save_json


class Seq2SeqCollator:
    def __init__(self, tokenizer, max_input_length: int, max_output_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __call__(self, batch):
        sources = [x["source"] for x in batch]
        targets = [x["target"] for x in batch]

        enc = self.tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            dec = self.tokenizer(
                targets,
                padding=True,
                truncation=True,
                max_length=self.max_output_length,
                return_tensors="pt",
            )
        labels = dec["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc


def build_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def train_seq2seq(
    model,
    tokenizer,
    train_loader,
    val_loader,
    device,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader) * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    best_val = 1e18
    best_state = None

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for batch in tqdm(train_loader, desc=f"Train Explainer {epoch}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_loss += float(out.loss.item())
            train_steps += 1

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Explainer {epoch}", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                val_loss += float(out.loss.item())
                val_steps += 1

        train_avg = train_loss / max(train_steps, 1)
        val_avg = val_loss / max(val_steps, 1)
        history.append({"epoch": epoch, "train_loss": train_avg, "val_loss": val_avg})
        if val_avg < best_val:
            best_val = val_avg
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return history, best_state


def save_model_bundle(model, tokenizer, out_dir: str | Path, history: list[dict[str, Any]]) -> None:
    out_dir = ensure_dir(out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    save_json(history, Path(out_dir) / "history.json")
