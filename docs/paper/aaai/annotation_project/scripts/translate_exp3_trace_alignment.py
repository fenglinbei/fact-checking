#!/usr/bin/env python3
"""Fill the content-addressed translation cache for trace-alignment tasks.

The exporter writes one request for each unique authoritative English string.
This helper translates only cache misses and writes a resumable JSONL cache
that can be passed back to ``export_exp3_trace_alignment.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from translate_tasks import API_KEY_ENV, MODEL, call_deepseek


HERE = Path(__file__).resolve().parent
ANNOTATION_ROOT = HERE.parent
DEFAULT_PREPARED_DIR = (
    ANNOTATION_ROOT / "results" / "exp3_trace_alignment_v1"
)
DEFAULT_REQUESTS = DEFAULT_PREPARED_DIR / "translation_inventory.jsonl"
DEFAULT_CACHE = DEFAULT_PREPARED_DIR / "translation_cache.jsonl"
DEFAULT_LOCAL_MODEL = Path("/data/models/Qwen2.5-1.5B-Instruct")


class TranslationError(RuntimeError):
    """Raised when a request/cache contract is invalid."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise TranslationError(f"File does not exist: {path}")
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TranslationError(
                f"Invalid JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise TranslationError(
                f"Expected an object at {path}:{line_number}"
            )
        rows.append(row)
    return rows


def validate_requests(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    requests: dict[str, dict[str, Any]] = {}
    for row in rows:
        text = row.get("text_en")
        key = row.get("translation_key")
        field_types = row.get("field_types")
        if not isinstance(text, str) or not text.strip():
            raise TranslationError("Translation request has empty text_en")
        text = text.strip()
        expected_key = f"sha256:{sha256_text(text)}"
        if key != expected_key:
            raise TranslationError(
                f"Translation key mismatch for {text[:80]!r}"
            )
        if not isinstance(field_types, list) or not all(
            isinstance(item, str) and item for item in field_types
        ):
            raise TranslationError(
                f"Invalid field_types for request {expected_key}"
            )
        previous = requests.get(text)
        normalized = {
            "translation_key": expected_key,
            "text_en": text,
            "field_types": sorted(set(field_types)),
        }
        if previous is not None and previous != normalized:
            raise TranslationError(
                f"Conflicting duplicate request for {expected_key}"
            )
        requests[text] = normalized
    return requests


def _collect_cache_pairs(value: Any, pairs: dict[str, str]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_cache_pairs(item, pairs)
        return
    if not isinstance(value, dict):
        return
    source = (
        value.get("text_en")
        or value.get("source_text")
        or value.get("english")
    )
    target = (
        value.get("text_zh")
        or value.get("translation")
        or value.get("chinese")
    )
    if isinstance(source, str) and isinstance(target, str):
        source = source.strip()
        target = target.strip()
        if source and target:
            previous = pairs.get(source)
            if previous is not None and previous != target:
                raise TranslationError(
                    f"Conflicting cached translations for {source[:80]!r}"
                )
            pairs[source] = target
    elif value and all(
        isinstance(source_text, str) and isinstance(translated_text, str)
        for source_text, translated_text in value.items()
    ):
        for source_text, translated_text in value.items():
            source_text = source_text.strip()
            translated_text = translated_text.strip()
            if not source_text or not translated_text:
                continue
            previous = pairs.get(source_text)
            if previous is not None and previous != translated_text:
                raise TranslationError(
                    f"Conflicting cached translations for {source_text[:80]!r}"
                )
            pairs[source_text] = translated_text
    for nested in value.values():
        if isinstance(nested, (dict, list)):
            _collect_cache_pairs(nested, pairs)


def load_cache_files(paths: Sequence[Path]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".jsonl":
            value: Any = load_jsonl(path)
        else:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise TranslationError(f"Invalid cache JSON: {path}") from exc
        _collect_cache_pairs(value, pairs)
    return pairs


def load_cache_models(paths: Sequence[Path]) -> dict[str, str]:
    """Return per-text provenance when a structured cache records it.

    Later paths take precedence. The CLI passes the destination cache last, so
    manually reviewed rows already present there keep their provenance when a
    complete cache is validated or rewritten.
    """

    models: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        values: Any
        if path.suffix.lower() == ".jsonl":
            values = load_jsonl(path)
        else:
            try:
                values = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise TranslationError(f"Invalid cache JSON: {path}") from exc
        rows = values if isinstance(values, list) else [values]
        for row in rows:
            if not isinstance(row, dict):
                continue
            source = (
                row.get("text_en")
                or row.get("source_text")
                or row.get("english")
            )
            model = row.get("translation_model")
            if (
                isinstance(source, str)
                and source.strip()
                and isinstance(model, str)
                and model.strip()
            ):
                models[source.strip()] = model.strip()
    return models


def cache_rows(
    requests: dict[str, dict[str, Any]],
    translations: dict[str, str],
    *,
    translation_model: str,
    translation_models: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    provenance = translation_models or {}
    for text in sorted(requests, key=sha256_text):
        translated = translations.get(text, "").strip()
        if not translated:
            continue
        request = requests[text]
        rows.append(
            {
                "translation_key": request["translation_key"],
                "text_en": text,
                "text_zh": translated,
                "field_types": request["field_types"],
                "translation_model": provenance.get(text, translation_model),
            }
        )
    return rows


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def translate_missing(
    missing: Sequence[str],
    *,
    api_key: str,
    concurrency: int,
    checkpoint: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, str]:
    if concurrency <= 0:
        raise TranslationError("--concurrency must be positive")
    translated: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(call_deepseek, text, api_key): text
            for text in missing
        }
        for future in as_completed(futures):
            text = futures[future]
            result = future.result().strip()
            if not result:
                raise TranslationError(
                    f"Translation API returned an empty result for {text[:80]!r}"
                )
            translated[text] = result
            if checkpoint is not None and len(translated) % concurrency == 0:
                checkpoint(dict(translated))
    if checkpoint is not None and translated:
        checkpoint(dict(translated))
    return translated


def _clean_local_translation(value: str) -> str:
    translated = value.strip()
    for prefix in (
        "中文翻译：",
        "简体中文翻译：",
        "翻译：",
        "Chinese translation:",
        "Translation:",
    ):
        if translated.lower().startswith(prefix.lower()):
            translated = translated[len(prefix) :].strip()
            break
    if (
        len(translated) >= 2
        and translated[0] == translated[-1]
        and translated[0] in {'"', "'", "“", "”"}
    ):
        translated = translated[1:-1].strip()
    return translated


def translate_missing_local(
    missing: Sequence[str],
    *,
    model_path: Path,
    batch_size: int,
    max_new_tokens: int,
    torch_threads: int,
    checkpoint: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, str]:
    if not model_path.is_dir():
        raise TranslationError(f"Local model does not exist: {model_path}")
    if batch_size <= 0 or max_new_tokens <= 0 or torch_threads <= 0:
        raise TranslationError(
            "Local batch size, max-new-tokens, and torch-threads must be positive"
        )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise TranslationError(
            "The local backend requires torch and transformers"
        ) from exc

    torch.set_num_threads(torch_threads)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()
    translated: dict[str, str] = {}
    system_prompt = (
        "You are a professional English-to-Chinese translator. Translate the "
        "text faithfully into Simplified Chinese. Preserve all names, numbers, "
        "dates, quotations, and factual meaning. Output only the Chinese "
        "translation, without labels, notes, or explanation."
    )
    for offset in range(0, len(missing), batch_size):
        batch = list(missing[offset : offset + batch_size])
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Translate into Simplified Chinese:\n{text}",
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        input_width = int(encoded.input_ids.shape[1])
        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for text, output in zip(batch, outputs, strict=True):
            result = _clean_local_translation(
                tokenizer.decode(
                    output[input_width:],
                    skip_special_tokens=True,
                )
            )
            if not result:
                raise TranslationError(
                    f"Local model returned an empty result for {text[:80]!r}"
                )
            translated[text] = result
        if checkpoint is not None:
            checkpoint(dict(translated))
        print(
            json.dumps(
                {
                    "local_batch_done": min(offset + len(batch), len(missing)),
                    "local_batch_total": len(missing),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return translated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--seed-cache",
        type=Path,
        action="append",
        default=[],
        help="Optional existing JSON/JSONL translation cache; repeatable.",
    )
    parser.add_argument(
        "--api-key-env",
        default=API_KEY_ENV,
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--backend",
        choices=("deepseek-api", "local-hf"),
        default="deepseek-api",
    )
    parser.add_argument(
        "--local-model-path",
        type=Path,
        default=DEFAULT_LOCAL_MODEL,
    )
    parser.add_argument("--local-batch-size", type=int, default=16)
    parser.add_argument("--local-max-new-tokens", type=int, default=192)
    parser.add_argument("--torch-threads", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        help=(
            "Translate at most this many cache misses in one resumable batch; "
            "0 means all remaining requests."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report cache coverage without API calls.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requests = validate_requests(load_jsonl(args.requests))
    cache_paths = [*args.seed_cache, args.cache]
    translations = load_cache_files(cache_paths)
    translation_models = load_cache_models(cache_paths)
    missing = sorted(
        (text for text in requests if not translations.get(text, "").strip()),
        key=sha256_text,
    )
    summary = {
        "requests": len(requests),
        "cache_hits": len(requests) - len(missing),
        "cache_misses": len(missing),
        "cache": str(args.cache.resolve()),
        "translation_model": (
            MODEL
            if args.backend == "deepseek-api"
            else f"local-hf:{args.local_model_path.resolve()}"
        ),
        "backend": args.backend,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if not requests and args.cache.exists() and args.cache.stat().st_size:
        raise TranslationError(
            "Refusing to overwrite a non-empty cache from an empty request "
            "inventory"
        )
    if args.max_new < 0:
        raise TranslationError("--max-new cannot be negative")
    selected_missing = missing[: args.max_new] if args.max_new else missing
    translation_model = (
        MODEL
        if args.backend == "deepseek-api"
        else f"local-hf:{args.local_model_path.resolve()}"
    )

    def checkpoint(new_translations: dict[str, str]) -> None:
        translations.update(new_translations)
        translation_models.update(
            {text: translation_model for text in new_translations}
        )
        atomic_write_jsonl(
            args.cache,
            cache_rows(
                requests,
                translations,
                translation_model=translation_model,
                translation_models=translation_models,
            ),
        )

    if selected_missing and args.backend == "deepseek-api":
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise TranslationError(
                f"Missing API key environment variable: {args.api_key_env}"
            )
        translations.update(
            translate_missing(
                selected_missing,
                api_key=api_key,
                concurrency=args.concurrency,
                checkpoint=checkpoint,
            )
        )
    elif selected_missing:
        translations.update(
            translate_missing_local(
                selected_missing,
                model_path=args.local_model_path,
                batch_size=args.local_batch_size,
                max_new_tokens=args.local_max_new_tokens,
                torch_threads=args.torch_threads,
                checkpoint=checkpoint,
            )
        )
    output_rows = cache_rows(
        requests,
        translations,
        translation_model=translation_model,
        translation_models=translation_models,
    )
    atomic_write_jsonl(args.cache, output_rows)
    remaining = len(requests) - len(output_rows)
    print(
        json.dumps(
            {
                "cache": str(args.cache.resolve()),
                "rows": len(output_rows),
                "new_rows": len(selected_missing),
                "remaining": remaining,
                "complete": remaining == 0,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TranslationError as exc:
        raise SystemExit(f"ERROR: {exc}")
