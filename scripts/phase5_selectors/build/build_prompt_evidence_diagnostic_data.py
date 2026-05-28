#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fact_checking.build.prompts import (
    build_chat_prompt,
    build_system_message,
    build_target,
    build_user_content,
    count_target_tokens,
    count_tokens,
    label_definitions_text,
    load_prompt_tokenizer,
    render_prompt,
    truncate_single_evidence_to_budget,
)
from fact_checking.config import save_yaml
from fact_checking.data.constants import LABEL2ID
from fact_checking.data.io import load_split
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl
from sft.data.labels import normalize_gold_label

from scripts.phase5_selectors.build.build_evidence_map_verifier_data import (
    _render_until_fit as _render_map_full_until_fit,
)
from scripts.phase5_selectors.build.build_trace_verifier_data import (
    _build_train_config,
    _load_experiment_config,
    _resolve_model_path,
)


DEFAULT_OUTPUT_DIR = (
    "outputs/selectors/evidence_map_selector/"
    "v0_5c_val_prompt_evidence_diagnostic/verifier_data"
)
PROMPT_STYLES = ("plain_original", "map_full", "map_minimal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired verifier data for v0.5c prompt x evidence diagnostics."
    )
    parser.add_argument("--selection-trace", required=True)
    parser.add_argument("--expected-selector-name", required=True)
    parser.add_argument("--prompt-style", required=True, choices=PROMPT_STYLES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--raw-path", default="data/raw/LIAR-RAW/val.json")
    parser.add_argument("--config", default="configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml")
    parser.add_argument("--prompt-model-name-or-path", default=None)
    parser.add_argument("--train-model-name-or-path", default=None)
    parser.add_argument("--model-base-path", default=None)
    parser.add_argument("--max-evidence-chars", type=int, default=420)
    parser.add_argument("--max-span-chars", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_experiment_config(str(args.config))
    prompt_cfg = dict((cfg.get("build", {}) or {}).get("prompt", {}) or {})
    if args.prompt_model_name_or_path:
        prompt_cfg["model_name_or_path"] = args.prompt_model_name_or_path
    if args.model_base_path and prompt_cfg.get("model_name_or_path"):
        prompt_cfg["model_name_or_path"] = _resolve_model_path(
            str(prompt_cfg["model_name_or_path"]),
            str(args.model_base_path),
        )
    _validate_model_path(str(prompt_cfg.get("model_name_or_path") or ""), field_name="build.prompt.model_name_or_path")
    tokenizer = load_prompt_tokenizer(str(prompt_cfg["model_name_or_path"]))

    traces = [
        trace
        for trace in read_jsonl(args.selection_trace)
        if str(trace.get("selector_name") or "") == str(args.expected_selector_name)
    ]
    if args.sample_limit is not None:
        traces = traces[: int(args.sample_limit)]
    if not traces:
        raise ValueError(
            f"No traces found for selector={args.expected_selector_name!r} in {args.selection_trace}"
        )

    raw_by_event = {sample.event_id: sample for sample in load_split(args.raw_path)}
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    map_annotation_status: Counter[str] = Counter()
    for trace in traces:
        event_id = str(trace.get("event_id") or "")
        sample = raw_by_event.get(event_id)
        if sample is None:
            skipped["missing_raw_sample"] += 1
            continue
        row = _build_diagnostic_row(
            trace,
            sample=sample,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            prompt_style=str(args.prompt_style),
            max_evidence_chars=int(args.max_evidence_chars),
            max_span_chars=int(args.max_span_chars),
            top_k=int(args.top_k),
        )
        if int(row.get("gold_id", -1)) < 0:
            skipped["invalid_gold"] += 1
            continue
        rows.append(row)
        map_annotation_status[str((row.get("selector_trace") or {}).get("map_annotation_status") or "")] += 1

    if not rows:
        raise ValueError("No verifier rows produced.")

    out_path = out_dir / f"build_{args.split}.jsonl"
    write_jsonl(rows, out_path)
    split_paths = {"train": str(out_path), "val": str(out_path), "test": str(out_path)}
    split_paths[str(args.split)] = str(out_path)
    train_config = _build_train_config(
        cfg=cfg,
        output_dir=out_dir,
        split_paths=split_paths,
        model_base_path=args.model_base_path,
        train_model_name_or_path=args.train_model_name_or_path,
    )
    train_config_path = out_dir / "train.resolved.yaml"
    save_yaml(train_config, train_config_path)

    report = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "selection_trace": str(args.selection_trace),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "raw_path": str(args.raw_path),
        "expected_selector_name": str(args.expected_selector_name),
        "prompt_style": str(args.prompt_style),
        "top_k": int(args.top_k),
        "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
        "n_input_traces": len(traces),
        "n_rows": len(rows),
        "skipped": dict(skipped),
        "label_distribution": dict(Counter(str(row.get("gold_label") or "") for row in rows)),
        "map_annotation_status": dict(map_annotation_status),
        "prompt_token_count": _summary([int(row.get("prompt_token_count") or 0) for row in rows]),
        "target_token_count": _summary([int(row.get("target_token_count") or 0) for row in rows]),
        "evidence_count": _summary([int(row.get("evidence_count") or 0) for row in rows]),
        "evidence_count_before": _summary([int(row.get("evidence_count_before") or 0) for row in rows]),
        "was_truncated_rate": _rate(bool(row.get("was_truncated")) for row in rows),
        "evidence_dropped_rate": _rate(
            int(row.get("evidence_count") or 0) < int(row.get("evidence_count_before") or 0)
            for row in rows
        ),
        "evidence_text_truncated_rate": _rate(bool(row.get("evidence_text_truncated")) for row in rows),
        "overflow_after_rate": _rate(bool(row.get("overflow_after")) for row in rows),
        "selection_metrics": _summarize_trace_metrics(rows),
        "outputs": {"build_jsonl": str(out_path), "train_config": str(train_config_path)},
        "elapsed_seconds": round(time.time() - started_at, 3),
        "notes": [
            "Rows keep selector evidence fixed across prompt styles before prompt-budget truncation.",
            "Gold labels and oracle metadata are used only as dataset targets/offline trace metadata, not rendered into prompts.",
        ],
    }
    save_json(report, out_dir / "build_report.json")
    print(f"Wrote v0.5c diagnostic verifier data: {out_path}")
    print(f"Train config: {train_config_path}")


def _build_diagnostic_row(
    trace: dict[str, Any],
    *,
    sample: Any,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    prompt_style: str,
    max_evidence_chars: int,
    max_span_chars: int,
    top_k: int,
) -> dict[str, Any]:
    gold_label = normalize_gold_label({"label": sample.label})
    output_mode = str(prompt_cfg.get("output_mode", "label_only")).strip().lower()
    label_format = str(prompt_cfg.get("label_format", "name")).strip().lower()
    system_msg = build_system_message(prompt_cfg.get("system_prompt") or None)
    max_length = int(prompt_cfg.get("max_length", 2048))
    target = build_target({"explain": sample.explain}, gold_label, output_mode, label_format)
    target_tokens = count_target_tokens(target, tokenizer)
    budget = max(0, max_length - target_tokens)
    selected_before = _selected_candidates(trace, top_k=top_k)
    if prompt_style == "plain_original":
        prompt, prompt_tokens, evidence_count, evidence_text_truncated, overflow_after, kept = _render_plain_until_fit(
            claim=str(sample.claim),
            selected=selected_before,
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
            budget=budget,
        )
    elif prompt_style == "map_full":
        prompt, prompt_tokens, evidence_count = _render_map_full_until_fit(
            claim=str(sample.claim),
            atoms=list(trace.get("claim_atoms") or []),
            selected=selected_before,
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
            budget=budget,
            max_evidence_chars=max_evidence_chars,
            max_span_chars=max_span_chars,
        )
        kept = selected_before[:evidence_count]
        evidence_text_truncated = False
        overflow_after = bool(prompt_tokens > budget)
    else:
        prompt, prompt_tokens, evidence_count, overflow_after, kept = _render_map_minimal_until_fit(
            claim=str(sample.claim),
            selected=selected_before,
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
            budget=budget,
            max_evidence_chars=max_evidence_chars,
        )
        evidence_text_truncated = False

    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "explain": sample.explain,
        "candidates": kept,
        "prompt": prompt,
        "target": target,
        "gold_label": gold_label,
        "gold_id": LABEL2ID.get(gold_label, -1),
        "gold_explain": str(sample.explain or "").strip(),
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "prompt_token_count": prompt_tokens,
        "target_token_count": target_tokens,
        "evidence_count": evidence_count,
        "evidence_count_before": len(selected_before),
        "was_truncated": bool(evidence_count < len(selected_before) or evidence_text_truncated),
        "evidence_text_truncated": bool(evidence_text_truncated),
        "overflow_after": bool(overflow_after),
        "selection_method": "prompt_evidence_diagnostic_v0_5c",
        "prompt_style": prompt_style,
        "selector_trace": {
            "selector_name": str(trace.get("selector_name") or ""),
            "selected_keys": list(trace.get("selected_keys") or [])[:top_k],
            "selected_keys_before": [str(candidate.get("candidate_key") or "") for candidate in selected_before],
            "selected_texts_before": [str(candidate.get("text") or "") for candidate in selected_before],
            "oracle_ordered_keys": list(trace.get("oracle_ordered_keys") or []),
            "map_annotation_status": _trace_map_annotation_status(trace, selected_before),
            "recall@5": _float_or_default(trace.get("recall@5"), 0.0),
            "jaccard@5": _float_or_default(trace.get("jaccard@5"), 0.0),
            "top1_match": _float_or_default(trace.get("top1_match"), 0.0),
            "oracle_rank_ndcg@5": _float_or_default(trace.get("oracle_rank_ndcg@5"), 0.0),
            "weighted_atom_coverage@5": _float_or_default(trace.get("weighted_atom_coverage@5"), 0.0),
            "direct_or_partial_map_rate@5": _direct_or_partial_rate(selected_before),
            "background_only_map_rate@5": _background_rate(selected_before),
        },
    }


def _render_plain_until_fit(
    *,
    claim: str,
    selected: list[dict[str, Any]],
    tokenizer: Any,
    system_msg: str,
    output_mode: str,
    label_format: str,
    budget: int,
) -> tuple[str, int, int, bool, bool, list[dict[str, Any]]]:
    kept = [dict(candidate) for candidate in selected]
    prompt, prompt_tokens = _render_plain_prompt(
        claim=claim,
        kept=kept,
        tokenizer=tokenizer,
        system_msg=system_msg,
        output_mode=output_mode,
        label_format=label_format,
    )
    while prompt_tokens > budget and len(kept) > 1:
        kept.pop()
        prompt, prompt_tokens = _render_plain_prompt(
            claim=claim,
            kept=kept,
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
        )
    evidence_text_truncated = False
    if prompt_tokens > budget and len(kept) == 1:
        kept_texts, prompt, prompt_tokens, evidence_text_truncated = truncate_single_evidence_to_budget(
            claim=claim,
            evidence_text=str(kept[0].get("text") or ""),
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
            budget=budget,
        )
        if kept_texts:
            kept[0] = {**kept[0], "text": kept_texts[0]}
        else:
            kept = []
    if prompt_tokens > budget and not kept:
        prompt, prompt_tokens = render_prompt(
            claim=claim,
            evidence_texts=[],
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
        )
    return prompt, prompt_tokens, len(kept), evidence_text_truncated, bool(prompt_tokens > budget), kept


def _render_plain_prompt(
    *,
    claim: str,
    kept: list[dict[str, Any]],
    tokenizer: Any,
    system_msg: str,
    output_mode: str,
    label_format: str,
) -> tuple[str, int]:
    evidence_texts = [str(candidate.get("text") or "").strip() for candidate in kept]
    user_content = build_user_content(claim, evidence_texts, output_mode, label_format)
    prompt = build_chat_prompt(tokenizer, system_msg, user_content)
    return prompt, count_tokens(prompt, tokenizer, add_special_tokens=False)


def _render_map_minimal_until_fit(
    *,
    claim: str,
    selected: list[dict[str, Any]],
    tokenizer: Any,
    system_msg: str,
    output_mode: str,
    label_format: str,
    budget: int,
    max_evidence_chars: int,
) -> tuple[str, int, int, bool, list[dict[str, Any]]]:
    kept = [dict(candidate) for candidate in selected]
    for evidence_chars in (max_evidence_chars, 220, 120):
        while kept:
            user_content = _build_map_minimal_user_content(
                claim=claim,
                selected=kept,
                output_mode=output_mode,
                label_format=label_format,
                max_evidence_chars=evidence_chars,
            )
            prompt = build_chat_prompt(tokenizer, system_msg, user_content)
            prompt_tokens = count_tokens(prompt, tokenizer, add_special_tokens=False)
            if prompt_tokens <= budget:
                return prompt, prompt_tokens, len(kept), False, kept
            kept.pop()
    user_content = _build_map_minimal_user_content(
        claim=claim,
        selected=[],
        output_mode=output_mode,
        label_format=label_format,
        max_evidence_chars=0,
    )
    prompt = build_chat_prompt(tokenizer, system_msg, user_content)
    prompt_tokens = count_tokens(prompt, tokenizer, add_special_tokens=False)
    return prompt, prompt_tokens, 0, bool(prompt_tokens > budget), []


def _build_map_minimal_user_content(
    *,
    claim: str,
    selected: list[dict[str, Any]],
    output_mode: str,
    label_format: str,
    max_evidence_chars: int,
) -> str:
    label_placeholder = "<a single letter from A-F>" if label_format == "letter" else "<label>"
    response_rule = (
        "Respond with exactly two lines in this format:\nExplanation: <brief explanation>\n"
        f"Label: {label_placeholder}"
        if output_mode == "explanation_label"
        else f"Respond with exactly one line: Label: {label_placeholder}"
    )
    lines = [
        "Classify the claim into exactly one LIAR-RAW label using the selected evidence.",
        "",
        "Labels:",
        label_definitions_text(label_format),
        "",
        "Rules:",
        "- Use the selected evidence as the primary source.",
        "- Treat relation/directness/atoms as compact evidence metadata.",
        "- Do not invent facts not supported by the evidence.",
        f"- {response_rule}",
        "",
        "Claim:",
        claim.strip(),
        "",
        "Evidence:",
    ]
    if not selected:
        lines.append("(no selected evidence available)")
        return "\n".join(lines)
    for idx, candidate in enumerate(selected, start=1):
        atoms = ",".join(str(atom) for atom in candidate.get("covered_atom_ids") or []) or "none"
        text = str(candidate.get("text") or "").strip()[:max_evidence_chars]
        lines.append(
            "[{idx}] relation={relation}; directness={directness}; atoms={atoms}".format(
                idx=idx,
                relation=str(candidate.get("map_relation") or ""),
                directness=str(candidate.get("map_directness") or ""),
                atoms=atoms,
            )
        )
        lines.append(text)
    return "\n".join(lines)


def _selected_candidates(trace: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    selected = [dict(candidate) for candidate in trace.get("selected_candidates") or []]
    return selected[: int(top_k)]


def _trace_map_annotation_status(trace: dict[str, Any], selected: list[dict[str, Any]]) -> str:
    explicit = str(trace.get("map_annotation_status") or "").strip()
    if explicit:
        return explicit
    if selected and all(
        all(key in candidate for key in ("map_relation", "map_directness", "covered_atom_ids", "key_spans"))
        for candidate in selected
    ):
        return "ok"
    return "missing"


def _summarize_trace_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    traces = [row.get("selector_trace") or {} for row in rows]
    keys = (
        "recall@5",
        "jaccard@5",
        "top1_match",
        "oracle_rank_ndcg@5",
        "weighted_atom_coverage@5",
        "direct_or_partial_map_rate@5",
        "background_only_map_rate@5",
    )
    return {key: _mean(_float_or_default(trace.get(key), 0.0) for trace in traces) for key in keys}


def _direct_or_partial_rate(selected: list[dict[str, Any]]) -> float:
    return _mean(
        1.0 if str(candidate.get("map_directness") or "") in {"direct", "partial"} else 0.0
        for candidate in selected
    )


def _background_rate(selected: list[dict[str, Any]]) -> float:
    return _mean(
        1.0 if str(candidate.get("map_relation") or "") in {"background", "irrelevant"} else 0.0
        for candidate in selected
    )


def _validate_model_path(model_name_or_path: str, *, field_name: str) -> None:
    value = str(model_name_or_path or "").strip()
    if not value:
        raise ValueError(f"{field_name} is empty; pass --prompt-model-name-or-path or fix the config.")
    if value.startswith("/"):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(
                f"{field_name} points to a local model path that does not exist on this server: {value}. "
                "Set MODEL_BASE_PATH/PROMPT_MODEL_NAME_OR_PATH to an existing tokenizer path."
            )


def _summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _rate(values: Any) -> float:
    items = [bool(value) for value in values]
    return float(sum(1 for value in items if value) / max(len(items), 1))


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return float(sum(items) / len(items)) if items else 0.0


def _float_or_default(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(out):
        return float(default)
    return float(out)


if __name__ == "__main__":
    main()
