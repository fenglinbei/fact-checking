from __future__ import annotations

import hashlib


def build_cache_name(
    split: str,
    max_length: int,
    tokenizer_name: str,
    instances: list[dict[str, object]],
) -> str:
    tok_hash = hashlib.md5(tokenizer_name.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    data_hash = hashlib.md5(usedforsecurity=False)
    data_hash.update(str(len(instances)).encode("utf-8"))
    for row in instances:
        data_hash.update(str(row["prompt"]).encode("utf-8"))
        data_hash.update(str(row["target"]).encode("utf-8"))
        data_hash.update(str(bool(row.get("prompt_add_special_tokens", True))).encode("utf-8"))
        data_hash.update(str(bool(row.get("preserve_prompt_prefix", False))).encode("utf-8"))
    return f"{split}_ml{max_length}_{tok_hash}_{data_hash.hexdigest()[:16]}.pt"
