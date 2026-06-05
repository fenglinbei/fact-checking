from __future__ import annotations

from transformers import AutoTokenizer

from sft.runtime.model_loading import is_mistral_common_tokenizer


def _coerce_input_ids(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def tokenize_instances(
    instances: list[dict[str, object]],
    tokenizer: AutoTokenizer,
    max_length: int,
    padding: str,
) -> list[dict[str, list[int]]]:
    tokenized: list[dict[str, list[int]]] = []
    eos_id = tokenizer.eos_token_id
    requires_prompt_input_ids = is_mistral_common_tokenizer(tokenizer)

    for row in instances:
        prompt_add_special_tokens = bool(row.get("prompt_add_special_tokens", True))
        preserve_prompt_prefix = bool(row.get("preserve_prompt_prefix", False))
        prompt_ids = _coerce_input_ids(row.get("prompt_input_ids"))
        if prompt_ids is None:
            if requires_prompt_input_ids:
                raise ValueError(
                    "MistralCommon tokenizers require build rows with prompt_input_ids. "
                    "Rebuild this run with FORCE_BUILD=true so chat prompts are stored from "
                    "apply_chat_template(tokenize=True)."
                )
            prompt_text = str(row["prompt"]).rstrip()
            if prompt_add_special_tokens:
                prompt_text += " "
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=prompt_add_special_tokens,
                truncation=False,
            )["input_ids"]

        target_text = str(row["target"]).strip()
        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if eos_id is not None:
            target_ids = target_ids + [eos_id]

        max_prompt_len = max_length - len(target_ids)
        if max_prompt_len <= 0:
            input_ids = target_ids[:max_length]
            labels = input_ids[:]
        else:
            if len(prompt_ids) > max_prompt_len:
                if preserve_prompt_prefix:
                    raise ValueError(
                        "Protected prompt is longer than the target-aware prompt budget after evidence truncation. "
                        "Increase sft_train.max_length or reduce evidence/context length."
                    )
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
