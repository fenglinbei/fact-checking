from __future__ import annotations

from transformers import AutoTokenizer


def tokenize_instances(
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

        max_prompt_len = max_length - len(target_ids)
        if max_prompt_len <= 0:
            input_ids = target_ids[:max_length]
            labels = input_ids[:]
        else:
            if len(prompt_ids) > max_prompt_len:
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
