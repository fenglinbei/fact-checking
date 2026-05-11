from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from fact_checking.data.constants import LABEL2ID, LABELS, LETTER_ORDER
from fact_checking.data.io import load_jsonl
from sft.data.labels import normalize_gold_label
from sft.data.types import PreparedSample


def build_logit_bias(logit_adjust_cfg: dict) -> dict[str, float]:
    """Convert logit_adjust config to OpenAI-compatible logit_bias dict.

    Returns {str(token_id): bias} where bias = -tau * log(prior).
    """
    tau = float(logit_adjust_cfg["tau"])
    letter_token_ids = logit_adjust_cfg["letter_token_ids"]
    log_priors = logit_adjust_cfg["log_priors"]
    return {str(int(tid)): float(-tau * lp) for tid, lp in zip(letter_token_ids, log_priors)}


def compute_priors_from_samples(samples: list[PreparedSample]) -> list[float]:
    """Estimate class priors from a list of PreparedSample."""
    counts = [0] * len(LABELS)
    for sample in samples:
        gid = LABEL2ID.get(getattr(sample, "gold_label", ""), -1)
        if gid >= 0:
            counts[gid] += 1
    total = sum(counts)
    floor = 1.0 / max(total, len(LABELS))
    if total <= 0:
        priors = [1.0 / len(LABELS)] * len(LABELS)
    else:
        priors = [(c / total) if c > 0 else floor for c in counts]
    return priors


def compute_priors_from_jsonl(jsonl_path: str | Path) -> list[float]:
    """Estimate class priors from a build-stage JSONL file."""
    rows = load_jsonl(jsonl_path)
    counts = [0] * len(LABELS)
    total = 0
    for row in rows:
        gold_label = normalize_gold_label(row)
        if not gold_label:
            continue
        gid = LABEL2ID[gold_label]
        counts[gid] += 1
        total += 1
    floor = 1.0 / max(total, len(LABELS))
    if total <= 0:
        return [1.0 / len(LABELS)] * len(LABELS)
    return [(c / total) if c > 0 else floor for c in counts]


def load_logit_adjust_cfg(train_output_dir: str | Path) -> dict | None:
    """Load saved logit_adjust config from training output directory."""
    path = Path(train_output_dir) / "logit_adjust.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_logit_adjust_cfg_from_train_config(
    train_config: dict,
    tokenizer,
) -> dict | None:
    """Rebuild logit_adjust config from train config and tokenizer.

    Used by inference paths when logit_adjust.json is not available (legacy
    checkpoints).  Reads training data to estimate priors.
    """
    sft_cfg = train_config.get("sft_train", {})
    cfg_block = sft_cfg.get("logit_adjust", {}) or {}
    if not bool(cfg_block.get("enabled", False)):
        return None

    tau = float(cfg_block.get("tau", 1.0))

    letter_token_ids: list[int] = []
    for letter in LETTER_ORDER:
        ids = tokenizer(" " + letter, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            return None  # Letter is not a single token; logit_adjust cannot work
        letter_token_ids.append(int(ids[0]))

    prefix_token_ids = list(tokenizer("Label:", add_special_tokens=False)["input_ids"])

    data_cfg = train_config.get("data", {})
    train_path = data_cfg.get("train_candidates", "")
    if train_path and Path(train_path).exists():
        priors = compute_priors_from_jsonl(train_path)
    else:
        priors = [1.0 / len(LABELS)] * len(LABELS)

    log_priors = [math.log(p) for p in priors]
    return {
        "enabled": True,
        "tau": tau,
        "letter_token_ids": letter_token_ids,
        "log_priors": log_priors,
        "prefix_token_ids": prefix_token_ids,
    }


class LabelLogitAdjustProcessor:
    """vLLM LogitsProcessor that applies class-prior correction to letter tokens.

    Equivalent to training-time `_predict_with_logit_adjust` but operating
    inside vLLM's generation loop.
    """

    def __init__(self, letter_token_ids: list[int], biases: list[float]) -> None:
        self._letter_ids = letter_token_ids
        self._biases = biases

    def __call__(self, token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        for tok_id, bias in zip(self._letter_ids, self._biases):
            logits[tok_id] += bias
        return logits


def create_logit_adjust_processor(logit_adjust_cfg: dict):
    """Create a LabelLogitAdjustProcessor from a logit_adjust config dict."""
    tau = float(logit_adjust_cfg["tau"])
    letter_token_ids = logit_adjust_cfg["letter_token_ids"]
    log_priors = logit_adjust_cfg["log_priors"]
    biases = [float(-tau * lp) for lp in log_priors]
    return LabelLogitAdjustProcessor(letter_token_ids, biases)


class LabelChoiceLogitsProcessor:
    """vLLM LogitsProcessor that masks generation to the configured label tokens."""

    def __init__(self, letter_token_ids: list[int], biases: list[float] | None = None) -> None:
        self._letter_ids = [int(x) for x in letter_token_ids]
        self._biases = [float(x) for x in (biases or [0.0] * len(self._letter_ids))]

    def __call__(self, token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        del token_ids
        masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
        letter_ids = torch.as_tensor(self._letter_ids, dtype=torch.long, device=logits.device)
        values = logits.index_select(logits.dim() - 1, letter_ids)
        bias = torch.as_tensor(self._biases, dtype=logits.dtype, device=logits.device)
        masked.index_copy_(logits.dim() - 1, letter_ids, values + bias)
        return masked


def create_label_choice_processor(logit_adjust_cfg: dict):
    """Create a processor equivalent to native eval's restricted A-F argmax."""
    tau = float(logit_adjust_cfg.get("tau", 0.0))
    letter_token_ids = logit_adjust_cfg["letter_token_ids"]
    log_priors = logit_adjust_cfg.get("log_priors", [0.0] * len(letter_token_ids))
    biases = [float(-tau * lp) for lp in log_priors]
    return LabelChoiceLogitsProcessor(letter_token_ids, biases)
