from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from fact_checking.selectors.stage2_oracle import Stage2OracleExample, candidate_text


ACTION_RE = re.compile(r"\bE(\d{2})\b")


@dataclass(frozen=True)
class ChoiceScoreBatch:
    scores: list[torch.Tensor]
    actions: list[list[str]]
    candidate_indices: list[list[int]]


def action_token(candidate_idx: int) -> str:
    idx = int(candidate_idx)
    if idx < 0 or idx > 99:
        raise ValueError(f"candidate_idx must be in [0, 99], got {candidate_idx!r}.")
    return f"E{idx:02d}"


def parse_action(text: str) -> int | None:
    match = ACTION_RE.search(str(text).strip())
    if not match:
        return None
    return int(match.group(1))


def softmax_deltas(deltas: list[float], *, tau: float) -> list[float]:
    if not deltas:
        return []
    scale = max(float(tau), 1.0e-8)
    values = [float(x) / scale for x in deltas]
    offset = max(values)
    exps = [math.exp(x - offset) for x in values]
    denom = sum(exps)
    if denom <= 0.0 or not math.isfinite(denom):
        return [1.0 / len(values)] * len(values)
    return [float(x / denom) for x in exps]


def build_vig_index(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[int, dict[str, Any]]]]:
    out: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        event_id = str(row.get("event_id") or "")
        if not event_id:
            continue
        try:
            step = int(row.get("step"))
            candidate_idx = int(row.get("candidate_idx"))
        except (TypeError, ValueError):
            continue
        out[event_id][step][candidate_idx] = dict(row)
    return out


def build_action_prompt(
    example: Stage2OracleExample,
    *,
    prefix_indices: list[int],
    remaining_indices: list[int],
    max_candidate_chars: int = 180,
    include_retrieval_scores: bool = True,
) -> str:
    lines: list[str] = [
        "You are selecting evidence for fact checking.",
        "Choose exactly one next evidence id from the remaining candidates.",
        f"Claim: {example.claim.strip()}",
        "",
        "Selected prefix:",
    ]
    if prefix_indices:
        for idx in prefix_indices:
            lines.append(f"- {action_token(idx)}: {_trim(candidate_text(example.candidates[idx]), max_candidate_chars)}")
    else:
        lines.append("- None")

    lines.extend(["", "Remaining candidates:"])
    for idx in remaining_indices:
        line = f"- {action_token(idx)}: {_trim(candidate_text(example.candidates[idx]), max_candidate_chars)}"
        if include_retrieval_scores:
            score_row = example.candidate_scores[idx] if idx < len(example.candidate_scores) else {}
            rank = _safe_float(score_row.get("hybrid_rank"), float(idx))
            score = _safe_float(score_row.get("hybrid_score"), float("nan"))
            if math.isfinite(score):
                line += f" [rank={int(rank)}, hybrid={score:.4f}]"
            else:
                line += f" [rank={int(rank)}]"
        lines.append(line)

    lines.extend(["", "Next evidence id:"])
    return "\n".join(lines) + " "


def build_action_samples(
    examples: list[Stage2OracleExample],
    *,
    vig_rows: list[dict[str, Any]],
    split: str,
    top_k: int,
    max_candidate_chars: int = 180,
    include_retrieval_scores: bool = True,
    strict: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vig_index = build_vig_index(vig_rows)
    samples: list[dict[str, Any]] = []
    missing_vig_steps = 0
    missing_vig_candidates = 0
    missing_targets = 0

    for example in examples:
        selected = [int(idx) for idx in example.selected_indices[: int(top_k)]]
        prefix: list[int] = []
        for step, target_idx in enumerate(selected):
            expected_remaining = [idx for idx in range(len(example.candidates)) if idx not in prefix]
            step_rows = vig_index.get(example.event_id, {}).get(step, {})
            if not step_rows:
                missing_vig_steps += 1
                if strict:
                    raise ValueError(f"Missing VIG rows for event_id={example.event_id} step={step}.")
                prefix.append(target_idx)
                continue

            choices: list[dict[str, Any]] = []
            for idx in expected_remaining:
                row = step_rows.get(idx)
                if row is None:
                    missing_vig_candidates += 1
                    if strict:
                        raise ValueError(
                            f"Missing VIG row for event_id={example.event_id} step={step} candidate_idx={idx}."
                        )
                    continue
                score_row = example.candidate_scores[idx] if idx < len(example.candidate_scores) else {}
                choices.append(
                    {
                        "candidate_idx": int(idx),
                        "action": action_token(idx),
                        "delta_margin": _safe_float(row.get("delta_margin"), 0.0),
                        "after_margin": _safe_float(row.get("after_margin"), 0.0),
                        "hybrid_rank": _safe_float(score_row.get("hybrid_rank"), float(idx)),
                        "hybrid_score": _safe_float(score_row.get("hybrid_score"), 0.0),
                    }
                )

            choice_indices = {int(choice["candidate_idx"]) for choice in choices}
            if target_idx not in choice_indices:
                missing_targets += 1
                if strict:
                    raise ValueError(
                        f"Target candidate missing from choices for event_id={example.event_id} "
                        f"step={step}: target_idx={target_idx}."
                    )

            prompt = build_action_prompt(
                example,
                prefix_indices=prefix,
                remaining_indices=[int(choice["candidate_idx"]) for choice in choices],
                max_candidate_chars=int(max_candidate_chars),
                include_retrieval_scores=bool(include_retrieval_scores),
            )
            samples.append(
                {
                    "event_id": example.event_id,
                    "split": str(split),
                    "step": int(step),
                    "claim": example.claim,
                    "gold_label": example.gold_label,
                    "prefix_indices": [int(idx) for idx in prefix],
                    "remaining_indices": [int(choice["candidate_idx"]) for choice in choices],
                    "target_idx": int(target_idx),
                    "target_action": action_token(target_idx),
                    "prompt": prompt,
                    "choices": choices,
                    "fingerprint": example.fingerprint,
                }
            )
            prefix.append(target_idx)

    manifest = {
        "split": str(split),
        "n_examples": int(len(examples)),
        "n_samples": int(len(samples)),
        "top_k": int(top_k),
        "max_candidate_chars": int(max_candidate_chars),
        "include_retrieval_scores": bool(include_retrieval_scores),
        "strict": bool(strict),
        "missing_vig_steps": int(missing_vig_steps),
        "missing_vig_candidates": int(missing_vig_candidates),
        "missing_targets": int(missing_targets),
    }
    return samples, manifest


def score_action_choices(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    *,
    device: torch.device | str,
    max_length: int,
    choice_batch_size: int = 64,
    include_eos: bool = True,
    length_normalize: bool = True,
) -> ChoiceScoreBatch:
    flat: list[tuple[int, str, int, str]] = []
    actions: list[list[str]] = []
    candidate_indices: list[list[int]] = []
    for sample_idx, sample in enumerate(samples):
        sample_actions: list[str] = []
        sample_indices: list[int] = []
        for choice in sample.get("choices") or []:
            action = str(choice["action"])
            idx = int(choice["candidate_idx"])
            flat.append((sample_idx, str(sample["prompt"]), idx, action))
            sample_actions.append(action)
            sample_indices.append(idx)
        actions.append(sample_actions)
        candidate_indices.append(sample_indices)

    per_choice_scores: list[torch.Tensor] = []
    for start in range(0, len(flat), max(int(choice_batch_size), 1)):
        chunk = flat[start : start + max(int(choice_batch_size), 1)]
        tensors = _encode_choice_chunk(
            tokenizer,
            [(prompt, action) for _sample_idx, prompt, _idx, action in chunk],
            max_length=int(max_length),
            include_eos=bool(include_eos),
        )
        tensors = {key: value.to(device) for key, value in tensors.items()}
        outputs = model(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            use_cache=False,
        )
        per_choice_scores.append(
            _continuation_log_likelihood(
                outputs.logits,
                tensors["labels"],
                length_normalize=bool(length_normalize),
            )
        )

    if per_choice_scores:
        flat_scores = torch.cat(per_choice_scores, dim=0)
    else:
        flat_scores = torch.empty((0,), dtype=torch.float32, device=torch.device(device))

    grouped: list[list[torch.Tensor]] = [[] for _ in samples]
    for flat_idx, (sample_idx, _prompt, _idx, _action) in enumerate(flat):
        grouped[sample_idx].append(flat_scores[flat_idx])
    return ChoiceScoreBatch(scores=[torch.stack(items) if items else torch.empty(0, device=device) for items in grouped], actions=actions, candidate_indices=candidate_indices)


def _encode_choice_chunk(
    tokenizer: Any,
    pairs: list[tuple[str, str]],
    *,
    max_length: int,
    include_eos: bool,
) -> dict[str, torch.Tensor]:
    encoded: list[dict[str, list[int]]] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for prompt, action in pairs:
        prompt_ids = tokenizer(str(prompt), add_special_tokens=True, truncation=False)["input_ids"]
        action_ids = tokenizer(str(action).strip(), add_special_tokens=False, truncation=False)["input_ids"]
        if include_eos and tokenizer.eos_token_id is not None:
            action_ids = action_ids + [int(tokenizer.eos_token_id)]
        max_prompt_len = int(max_length) - len(action_ids)
        if max_prompt_len <= 0:
            input_ids = action_ids[: int(max_length)]
            labels = input_ids[:]
        else:
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]
            input_ids = prompt_ids + action_ids
            labels = [-100] * len(prompt_ids) + action_ids
        encoded.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )

    width = max(len(row["input_ids"]) for row in encoded) if encoded else 0
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for row in encoded:
        pad_len = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [int(pad_id)] * pad_len)
        attention_mask.append(row["attention_mask"] + [0] * pad_len)
        labels.append(row["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _continuation_log_likelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    length_normalize: bool,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_scores = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_scores = token_scores * mask.to(token_scores.dtype)
    scores = token_scores.sum(dim=-1)
    if length_normalize:
        denom = mask.sum(dim=-1).clamp_min(1).to(scores.dtype)
        scores = scores / denom
    return scores


def _trim(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text).split())
    limit = max(int(max_chars), 16)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out):
        return float(default)
    return out
