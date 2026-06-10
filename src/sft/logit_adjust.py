from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from fact_checking.data.constants import label2id_for_schema, labels_for_schema, letter_order_for_schema
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


def _choice_text(label_prefix: str, letter: str) -> str:
    return letter if label_prefix.endswith((" ", "\n", "\t")) else " " + letter


def _label_token_ids_for_prefix(
    tokenizer,
    *,
    label_prefix: str,
    letter_order: list[str],
    strict: bool,
) -> list[int] | None:
    letter_token_ids: list[int] = []
    for letter in letter_order:
        ids = tokenizer(_choice_text(label_prefix, letter), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            if strict:
                raise RuntimeError(
                    f"logit_adjust: {_choice_text(label_prefix, letter)!r} is not one tokenizer token "
                    f"for letter={letter!r}; got {ids}."
                )
            return None
        letter_token_ids.append(int(ids[0]))
    return letter_token_ids


def compute_priors_from_samples(samples: list[PreparedSample], *, label_schema: str | None = None) -> list[float]:
    """Estimate class priors from a list of PreparedSample."""
    labels = labels_for_schema(label_schema)
    label2id = label2id_for_schema(label_schema)
    counts = [0] * len(labels)
    for sample in samples:
        gid = label2id.get(getattr(sample, "gold_label", ""), -1)
        if gid >= 0:
            counts[gid] += 1
    total = sum(counts)
    floor = 1.0 / max(total, len(labels))
    if total <= 0:
        priors = [1.0 / len(labels)] * len(labels)
    else:
        priors = [(c / total) if c > 0 else floor for c in counts]
    return priors


def compute_priors_from_jsonl(jsonl_path: str | Path, *, label_schema: str | None = None) -> list[float]:
    """Estimate class priors from a build-stage JSONL file."""
    rows = load_jsonl(jsonl_path)
    labels = labels_for_schema(label_schema)
    label2id = label2id_for_schema(label_schema)
    counts = [0] * len(labels)
    total = 0
    for row in rows:
        gold_label = normalize_gold_label(row, label_schema=label_schema)
        if not gold_label:
            continue
        gid = label2id[gold_label]
        counts[gid] += 1
        total += 1
    floor = 1.0 / max(total, len(labels))
    if total <= 0:
        return [1.0 / len(labels)] * len(labels)
    return [(c / total) if c > 0 else floor for c in counts]


def build_logit_adjust_cfg_from_samples(
    *,
    train_cfg: dict[str, Any],
    tokenizer,
    train_samples: list[PreparedSample],
    label_schema: str | None = None,
    label_prefix: str = "Label:",
) -> dict | None:
    """Build label-prior logit adjustment from in-memory training samples."""
    cfg_block = train_cfg.get("logit_adjust", {}) or {}
    if not bool(cfg_block.get("enabled", False)):
        return None

    resolved_schema = str(label_schema or train_cfg.get("label_schema") or "liar6")
    letter_order = letter_order_for_schema(resolved_schema)
    letter_token_ids = _label_token_ids_for_prefix(
        tokenizer,
        label_prefix=label_prefix,
        letter_order=letter_order,
        strict=True,
    )
    assert letter_token_ids is not None
    prefix_token_ids = list(tokenizer(label_prefix, add_special_tokens=False)["input_ids"])
    priors = compute_priors_from_samples(train_samples, label_schema=resolved_schema)
    log_priors = [math.log(p) for p in priors]
    return {
        "enabled": True,
        "tau": float(cfg_block.get("tau", 1.0)),
        "label_schema": resolved_schema,
        "label_prefix": label_prefix,
        "letter_token_ids": letter_token_ids,
        "log_priors": log_priors,
        "prefix_token_ids": prefix_token_ids,
    }


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
    label_cfg = sft_cfg.get("label_token_ce", {}) or {}
    label_prefix = str(label_cfg.get("label_prefix", "Label:"))
    label_schema = str(
        sft_cfg.get("label_schema")
        or train_config.get("label_schema")
        or (train_config.get("baseline", {}) or {}).get("label_schema")
        or "liar6"
    )
    letter_order = letter_order_for_schema(label_schema)

    letter_token_ids = _label_token_ids_for_prefix(
        tokenizer,
        label_prefix=label_prefix,
        letter_order=letter_order,
        strict=False,
    )
    if letter_token_ids is None:
        return None  # Letter is not a single token; logit_adjust cannot work

    prefix_token_ids = list(tokenizer(label_prefix, add_special_tokens=False)["input_ids"])

    data_cfg = train_config.get("data", {})
    train_path = data_cfg.get("train_candidates", "")
    if train_path and Path(train_path).exists():
        priors = compute_priors_from_jsonl(train_path, label_schema=label_schema)
    else:
        priors = [1.0 / len(letter_order)] * len(letter_order)

    log_priors = [math.log(p) for p in priors]
    return {
        "enabled": True,
        "tau": tau,
        "label_schema": label_schema,
        "label_prefix": label_prefix,
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
