"""Generate G-Defense decomp+ style claim aspects with local vLLM."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from fact_checking.selectors.aspects import (
    LLM_DECOMP_PLUS_VERSION,
    build_claim_aspect_bundle_from_texts,
)
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    load_stage2_oracle_examples,
    read_jsonl,
    write_json,
)


DEFAULT_MODEL = "/data/models/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = (
    "You are a fake news detection assistant. Your task is to decompose a news claim "
    "into self-contained, atomic, verifiable sub-claims for fact-checking."
)
USER_PROMPT_TEMPLATE = """Analyze the news claim and decompose it into {min_subclaims}-{max_subclaims} clear sub-claims.

Follow these G-Defense decomp+ style requirements:
- Logical decomposition: separate distinct factual assertions that can be verified independently.
- Causal reasoning: identify cause-effect or condition-result relations when present.
- Hierarchical reasoning: distinguish general statements from specific supporting facts.
- Factual reasoning: focus on objectively testable propositions, not opinions.
- News and communication cues: distinguish attribution such as "X said that ..." from factual assertions about reality.
- Capture quantitative, temporal, comparative, and evaluative claims when they affect veracity.
- If a source or attribution is part of the claim, keep it inside the relevant sub-claim.
- If logical dependencies are explicit in the claim, reflect them with words such as "if", "because", or "therefore".
- Cover key fact-checking dimensions when present: who, what, when, where, why/how, and consequences.
- Each sub-claim must be concise, self-contained, and verifiable.
- Avoid redundancy and overlapping information.
- Use only information explicitly stated or clearly implied by the original claim. Do not add outside facts.

Return strictly valid JSON with this exact schema:
{{"sub_claims":["sub-claim 1","sub-claim 2"]}}

News claim:
{claim}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate decomp+ style claim-aspect cache with Qwen/vLLM for selector diagnostics."
    )
    p.add_argument("--oracle-results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--sample-limit", type=int, default=None)

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")))
    p.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    p.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "4096")))
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")

    p.add_argument("--generation-batch-size", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--guided-json", action="store_true", default=True)
    p.add_argument("--no-guided-json", dest="guided_json", action="store_false")

    p.add_argument("--min-subclaims", type=int, default=2)
    p.add_argument("--max-subclaims", type=int, default=5)
    p.add_argument("--min-aspect-tokens", type=int, default=3)
    p.add_argument("--max-aspect-tokens", type=int, default=60)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--store-prompts", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aspects_path = output_dir / "claim_aspects.jsonl"
    raw_path = output_dir / "raw_generations.jsonl"

    examples = load_stage2_oracle_examples(
        args.oracle_results,
        expected_fingerprint=args.expected_chunk_mmr_fingerprint,
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=args.filter_policy,
        min_margin=float(args.min_margin),
        sample_limit=args.sample_limit,
    )
    if not examples:
        raise ValueError("No examples after Stage2 audit/filtering.")

    existing_events = _existing_event_ids(aspects_path) if args.resume else set()
    pending = [example for example in examples if example.event_id not in existing_events]
    if not pending:
        print(f"No pending examples. Reusing existing cache: {aspects_path}")
        manifest = _manifest(args, output_dir, started_at, n_examples=len(examples), n_generated=0)
        manifest["status"] = "completed_noop"
        manifest["aspect_summary"] = _claim_aspect_summary(aspects_path)
        write_json(output_dir / "manifest.json", manifest)
        _write_markdown(output_dir / "analysis.md", manifest)
        return

    llm, tokenizer = _load_vllm(args)
    sampling_params = _build_sampling_params(args)
    prompts = [_format_prompt(tokenizer, example.claim, args=args) for example in pending]

    aspects_mode = "a" if args.resume and aspects_path.exists() else "w"
    raw_mode = "a" if args.resume and raw_path.exists() else "w"
    n_generated = 0
    n_parse_failures = 0
    generate_params = inspect.signature(llm.generate).parameters
    with aspects_path.open(aspects_mode, encoding="utf-8") as aspects_fh, raw_path.open(raw_mode, encoding="utf-8") as raw_fh:
        iterator = list(_batches(list(zip(pending, prompts)), int(args.generation_batch_size)))
        for batch in tqdm(
            iterator,
            desc="llm claim decomposition",
            unit="batch",
            dynamic_ncols=True,
            disable=bool(args.no_progress),
        ):
            batch_examples = [item[0] for item in batch]
            batch_prompts = [item[1] for item in batch]
            generate_kwargs: dict[str, Any] = {
                "prompts": batch_prompts,
                "sampling_params": sampling_params,
            }
            if "use_tqdm" in generate_params:
                generate_kwargs["use_tqdm"] = False
            outputs = llm.generate(**generate_kwargs)
            if len(outputs) != len(batch_examples):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch_examples)} prompts.")

            for example, prompt, output in zip(batch_examples, batch_prompts, outputs):
                raw_text = _output_text(output)
                raw_subclaims, parse_status, parse_error = parse_subclaims_from_generation(raw_text)
                subclaims, rejected_subclaims = filter_valid_subclaims(raw_subclaims)
                if parse_status not in {"empty_generation", "parse_failed"}:
                    if not subclaims:
                        parse_status = "invalid_subclaims"
                    elif len(subclaims) < int(args.min_subclaims):
                        parse_status = "fewer_than_min_valid_subclaims"
                if not subclaims:
                    n_parse_failures += 1
                bundle = build_claim_aspect_bundle_from_texts(
                    example.claim,
                    subclaims,
                    event_id=example.event_id,
                    extraction_version=LLM_DECOMP_PLUS_VERSION,
                    source="llm_decomp_plus",
                    aspect_type="llm_subclaim",
                    max_local_aspects=int(args.max_subclaims),
                    min_tokens=int(args.min_aspect_tokens),
                    max_tokens=int(args.max_aspect_tokens),
                )
                bundle_row = bundle.to_dict()
                bundle_row["llm_decomp"] = {
                    "model": str(args.model),
                    "parse_status": parse_status,
                    "parse_error": parse_error,
                    "n_raw_subclaims": len(raw_subclaims),
                    "n_valid_subclaims": len(subclaims),
                    "n_rejected_subclaims": len(rejected_subclaims),
                    "rejected_subclaims": rejected_subclaims,
                    "min_subclaims": int(args.min_subclaims),
                    "max_subclaims": int(args.max_subclaims),
                }
                raw_row = {
                    "event_id": example.event_id,
                    "claim": example.claim,
                    "gold_label": example.gold_label,
                    "parse_status": parse_status,
                    "parse_error": parse_error,
                    "subclaims": subclaims,
                    "raw_subclaims": raw_subclaims,
                    "rejected_subclaims": rejected_subclaims,
                    "raw_text": raw_text,
                }
                if args.store_prompts:
                    raw_row["prompt"] = prompt
                aspects_fh.write(json.dumps(bundle_row, ensure_ascii=False) + "\n")
                raw_fh.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
                n_generated += 1
            aspects_fh.flush()
            raw_fh.flush()

    manifest = _manifest(args, output_dir, started_at, n_examples=len(examples), n_generated=n_generated)
    manifest["parse_failures"] = int(n_parse_failures)
    manifest["aspect_summary"] = _claim_aspect_summary(aspects_path)
    manifest["raw_generation_summary"] = _raw_generation_summary(raw_path)
    write_json(output_dir / "manifest.json", manifest)
    _write_markdown(output_dir / "analysis.md", manifest)
    print(f"Wrote LLM claim aspects: {aspects_path}")
    print(f"Generated={n_generated}, parse_failures={n_parse_failures}")


def _load_vllm(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed in this environment. Run under the cppo conda env.") from exc

    llm_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "dtype": str(args.dtype),
        "max_model_len": int(args.max_model_len),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.tokenizer:
        llm_kwargs["tokenizer"] = str(args.tokenizer)
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = int(args.max_num_batched_tokens)
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = int(args.max_num_seqs)
    llm = LLM(**llm_kwargs)
    return llm, llm.get_tokenizer()


def _build_sampling_params(args: argparse.Namespace) -> Any:
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed in this environment. Run under the cppo conda env.") from exc

    kwargs: dict[str, Any] = {
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "seed": int(args.seed),
    }
    params = inspect.signature(SamplingParams).parameters
    if args.guided_json:
        schema = _guided_json_schema(args)
        if "guided_json" in params:
            kwargs["guided_json"] = schema
        elif "guided_decoding" in params:
            try:
                from vllm.sampling_params import GuidedDecodingParams
            except ImportError:
                print(
                    "[WARN] SamplingParams supports guided_decoding but GuidedDecodingParams is unavailable; "
                    "falling back to prompt-only JSON parsing.",
                    flush=True,
                )
            else:
                kwargs["guided_decoding"] = GuidedDecodingParams(json=schema)
        else:
            print(
                "[WARN] This vLLM build does not expose guided_json/guided_decoding; "
                "falling back to prompt-only JSON parsing.",
                flush=True,
            )
    return SamplingParams(**kwargs)


def _guided_json_schema(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "sub_claims": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": int(args.min_subclaims),
                "maxItems": int(args.max_subclaims),
            }
        },
        "required": ["sub_claims"],
    }


def _format_prompt(tokenizer: Any, claim: str, *, args: argparse.Namespace) -> str:
    user = USER_PROMPT_TEMPLATE.format(
        claim=str(claim).strip(),
        min_subclaims=int(args.min_subclaims),
        max_subclaims=int(args.max_subclaims),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM_PROMPT}\n\nUser: {user}\n\nAssistant:"


def parse_subclaims_from_generation(text: str) -> tuple[list[str], str, str | None]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return [], "empty_generation", "empty generation"

    candidates = [_strip_code_fence(cleaned)]
    json_slice = _json_slice(cleaned)
    if json_slice and json_slice not in candidates:
        candidates.append(json_slice)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        subclaims = _subclaims_from_payload(payload)
        if subclaims:
            return subclaims, "ok", None

    subclaims = _subclaims_from_numbered_lines(cleaned)
    if subclaims:
        return subclaims, "fallback_numbered_lines", None
    return [], "parse_failed", "could not parse JSON or numbered sub-claims"


def filter_valid_subclaims(subclaims: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    valid: list[str] = []
    rejected: list[dict[str, str]] = []
    for text in subclaims:
        normalized = _normalize_subclaim(text)
        reason = _invalid_subclaim_reason(normalized)
        if reason:
            rejected.append({"text": normalized, "reason": reason})
            continue
        valid.append(normalized)
    return _dedupe_preserve_order(valid), rejected


def _invalid_subclaim_reason(text: str) -> str | None:
    if not text:
        return "empty"
    if re.search(r"[\{\}\[\]<>]", text):
        return "contains_structural_marker"
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        return "contains_control_char"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "contains_non_english_cjk"
    if re.search(
        r"(?i)\b(?:analyze|decompose|decomposition|json|schema|valid json|"
        r"news claim|output requirements|logical decomposition|user:|assistant:|system:)\b",
        text,
    ):
        return "prompt_or_schema_artifact"
    if re.search(r"(?i)^(?:sub[_ -]?(?:claim|claims)?|claim[_ -]?cl|sub[_ -]?cl)\b", text):
        return "subclaim_label_artifact"
    tokens = re.findall(r"[A-Za-z0-9$%][A-Za-z0-9$%'.-]*", text)
    if len(tokens) < 3:
        return "too_few_content_tokens"
    if len(tokens) > 60:
        return "too_many_content_tokens"
    lowered = [token.lower() for token in tokens]
    if len(lowered) >= 12 and len(set(lowered)) / len(lowered) < 0.35:
        return "low_unique_token_ratio"
    for token in set(lowered):
        if token and lowered.count(token) >= 5:
            return "repeated_token_artifact"
    alpha_chars = sum(ch.isalpha() for ch in text)
    if alpha_chars and alpha_chars / max(len(text), 1) < 0.45:
        return "low_alpha_ratio"
    return None


def _subclaims_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        raw_items = (
            payload.get("sub_claims")
            or payload.get("subclaims")
            or payload.get("sub-claims")
            or payload.get("claims")
            or payload.get("aspects")
        )
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = None
    if not isinstance(raw_items, list):
        return []
    output: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("claim")
                or item.get("sub_claim")
                or item.get("subclaim")
                or item.get("content")
            )
        else:
            text = None
        normalized = _normalize_subclaim(text)
        if normalized:
            output.append(normalized)
    return _dedupe_preserve_order(output)


def _subclaims_from_numbered_lines(text: str) -> list[str]:
    output: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(?:[-*]\s*)?(?:\d+[\).\]:-]\s*|sub-?claim\s*\d+\s*[:.)-]\s*)(.+)$", line, flags=re.I)
        if not match:
            continue
        normalized = _normalize_subclaim(match.group(1))
        if normalized:
            output.append(normalized)
    return _dedupe_preserve_order(output)


def _normalize_subclaim(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n\"'`-•")
    value = re.sub(r"^(?:sub-?claim\s*\d+\s*[:.)-]\s*)", "", value, flags=re.I).strip()
    value = re.sub(r"^\d+[\).\]:-]\s*", "", value).strip()
    if not value:
        return ""
    if value.lower().startswith(("news claim:", "claim:")):
        return ""
    return value


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    return stripped


def _json_slice(text: str) -> str:
    start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not start_candidates:
        return ""
    start = min(start_candidates)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        return ""
    return text[start : end + 1].strip()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        sig = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
        if not sig or sig in seen:
            continue
        seen.add(sig)
        output.append(item)
    return output


def _output_text(output: Any) -> str:
    chunks = getattr(output, "outputs", None)
    if not chunks:
        return ""
    first = chunks[0]
    return str(getattr(first, "text", "") or "")


def _existing_event_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    event_ids: set[str] = set()
    for row in read_jsonl(path):
        event_id = str(row.get("event_id") or "")
        if event_id:
            event_ids.add(event_id)
    return event_ids


def _claim_aspect_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path) if path.exists() else []
    local_counts = [len(row.get("aspects") or []) for row in rows]
    dropped_counts = [len(row.get("dropped_aspects") or []) for row in rows]
    type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in rows:
        for aspect in row.get("aspects") or []:
            type_counts[str(aspect.get("type") or "unknown")] += 1
            source_counts[str(aspect.get("source") or "unknown")] += 1
    return {
        "n_claims": len(rows),
        "n_local_aspects": int(sum(local_counts)),
        "n_dropped_aspects": int(sum(dropped_counts)),
        "claims_with_no_local_aspects": int(sum(1 for count in local_counts if count == 0)),
        "local_aspects_per_claim": _numeric_summary(local_counts),
        "dropped_aspects_per_claim": _numeric_summary(dropped_counts),
        "aspect_type_counts": dict(sorted(type_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }


def _raw_generation_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path) if path.exists() else []
    status_counts: Counter[str] = Counter(str(row.get("parse_status") or "unknown") for row in rows)
    subclaim_counts = [len(row.get("subclaims") or []) for row in rows]
    raw_subclaim_counts = [len(row.get("raw_subclaims") or row.get("subclaims") or []) for row in rows]
    rejected_counts = [len(row.get("rejected_subclaims") or []) for row in rows]
    return {
        "n_rows": len(rows),
        "parse_status_counts": dict(sorted(status_counts.items())),
        "valid_subclaims_per_claim": _numeric_summary(subclaim_counts),
        "raw_subclaims_per_claim": _numeric_summary(raw_subclaim_counts),
        "rejected_subclaims_per_claim": _numeric_summary(rejected_counts),
    }


def _numeric_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _manifest(args: argparse.Namespace, out_dir: Path, started_at: float, *, n_examples: int, n_generated: int) -> dict[str, Any]:
    return {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "oracle_results": str(args.oracle_results),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "aspect_extraction_version": LLM_DECOMP_PLUS_VERSION,
        "model": str(args.model),
        "tokenizer": args.tokenizer,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "dtype": str(args.dtype),
        "max_model_len": int(args.max_model_len),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "seed": int(args.seed),
        "guided_json": bool(args.guided_json),
        "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        "max_candidates": int(args.max_candidates),
        "top_k": int(args.top_k),
        "filter_policy": str(args.filter_policy),
        "min_margin": float(args.min_margin),
        "sample_limit": args.sample_limit,
        "min_subclaims": int(args.min_subclaims),
        "max_subclaims": int(args.max_subclaims),
        "n_examples": int(n_examples),
        "n_generated": int(n_generated),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def _write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    aspect = manifest.get("aspect_summary", {})
    raw = manifest.get("raw_generation_summary", {})
    lines = [
        "# LLM Claim Decomposition Aspects",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- model: `{manifest.get('model')}`",
        f"- oracle_results: `{manifest.get('oracle_results')}`",
        f"- n_examples: {manifest.get('n_examples')}",
        f"- n_generated: {manifest.get('n_generated')}",
        f"- n_local_aspects: {aspect.get('n_local_aspects')}",
        f"- claims_with_no_local_aspects: {aspect.get('claims_with_no_local_aspects')}",
        "",
        "## Parse Status",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key, value in (raw.get("parse_status_counts") or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Aspect Types",
            "",
            "| type | count |",
            "|---|---:|",
        ]
    )
    for key, value in (aspect.get("aspect_type_counts") or {}).items():
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _batches(items: list[Any], batch_size: int):
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


if __name__ == "__main__":
    main()
