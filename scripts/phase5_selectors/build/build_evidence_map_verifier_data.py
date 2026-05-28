#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fact_checking.build.prompts import (
    build_chat_prompt,
    build_system_message,
    build_target,
    count_target_tokens,
    count_tokens,
    label_definitions_text,
    load_prompt_tokenizer,
)
from fact_checking.config import save_yaml
from fact_checking.data.constants import LABEL2ID, LABEL_LETTERS
from fact_checking.data.io import load_split
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl
from sft.data.labels import normalize_gold_label

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase5_selectors.build.build_trace_verifier_data import (
    _build_train_config,
    _load_experiment_config,
    _resolve_model_path,
)


DEFAULT_OUTPUT_DIR = "outputs/selectors/evidence_map_selector/v0_5a_val"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build label-token verifier data with v0.5a evidence-map prompts.")
    p.add_argument("--selection-trace", required=True)
    p.add_argument("--output-dir", default=f"{DEFAULT_OUTPUT_DIR}/verifier_data")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--raw-path", default="data/raw/LIAR-RAW/val.json")
    p.add_argument("--expected-selector-name", default="v0_5a_evidence_map_top5")
    p.add_argument("--config", default="configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml")
    p.add_argument("--prompt-model-name-or-path", default=None)
    p.add_argument("--train-model-name-or-path", default=None)
    p.add_argument("--model-base-path", default=None)
    p.add_argument("--max-evidence-chars", type=int, default=420)
    p.add_argument("--max-span-chars", type=int, default=160)
    p.add_argument("--sample-limit", type=int, default=None)
    return p.parse_args()


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
        prompt_cfg["model_name_or_path"] = _resolve_model_path(str(prompt_cfg["model_name_or_path"]), str(args.model_base_path))
    tokenizer = load_prompt_tokenizer(str(prompt_cfg["model_name_or_path"]))

    traces = [
        trace
        for trace in read_jsonl(args.selection_trace)
        if str(trace.get("selector_name") or "") == str(args.expected_selector_name)
    ]
    if args.sample_limit is not None:
        traces = traces[: int(args.sample_limit)]
    raw_by_event = {sample.event_id: sample for sample in load_split(args.raw_path)}
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for trace in traces:
        event_id = str(trace.get("event_id") or "")
        sample = raw_by_event.get(event_id)
        if sample is None:
            skipped["missing_raw_sample"] = skipped.get("missing_raw_sample", 0) + 1
            continue
        row = _build_map_verifier_row(trace, sample=sample, tokenizer=tokenizer, prompt_cfg=prompt_cfg, args=args)
        if int(row.get("gold_id", -1)) < 0:
            skipped["invalid_gold"] = skipped.get("invalid_gold", 0) + 1
            continue
        rows.append(row)
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
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "selection_trace": str(args.selection_trace),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "raw_path": str(args.raw_path),
        "expected_selector_name": str(args.expected_selector_name),
        "n_input_traces": len(traces),
        "n_rows": len(rows),
        "skipped": skipped,
        "prompt_token_count": _summary([int(row.get("prompt_token_count") or 0) for row in rows]),
        "evidence_count": _summary([int(row.get("evidence_count") or 0) for row in rows]),
        "outputs": {"build_jsonl": str(out_path), "train_config": str(train_config_path)},
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    save_json(manifest, out_dir / "build_report.json")
    print(f"Wrote evidence-map verifier data: {out_path}")
    print(f"Train config: {train_config_path}")


def _build_map_verifier_row(trace: dict[str, Any], *, sample: Any, tokenizer: Any, prompt_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    gold_label = normalize_gold_label({"label": sample.label})
    output_mode = str(prompt_cfg.get("output_mode", "label_only")).strip().lower()
    label_format = str(prompt_cfg.get("label_format", "name")).strip().lower()
    system_msg = build_system_message(prompt_cfg.get("system_prompt") or None)
    max_length = int(prompt_cfg.get("max_length", 2048))
    target = build_target({"explain": sample.explain}, gold_label, output_mode, label_format)
    target_tokens = count_target_tokens(target, tokenizer)
    budget = max(0, max_length - target_tokens)
    selected = list(trace.get("selected_candidates") or [])
    atoms = list(trace.get("claim_atoms") or [])
    prompt, prompt_tokens, evidence_count = _render_until_fit(
        claim=str(sample.claim),
        atoms=atoms,
        selected=selected,
        tokenizer=tokenizer,
        system_msg=system_msg,
        output_mode=output_mode,
        label_format=label_format,
        budget=budget,
        max_evidence_chars=int(args.max_evidence_chars),
        max_span_chars=int(args.max_span_chars),
    )
    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "explain": sample.explain,
        "candidates": selected[:evidence_count],
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
        "evidence_count_before": len(selected),
        "was_truncated": evidence_count < len(selected),
        "selection_method": "evidence_map_v0_5a",
        "selector_trace": {
            "selector_name": str(trace.get("selector_name") or ""),
            "selected_keys": list(trace.get("selected_keys") or []),
            "weighted_atom_coverage@5": float(trace.get("weighted_atom_coverage@5") or 0.0),
        },
    }


def _render_until_fit(
    *,
    claim: str,
    atoms: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    tokenizer: Any,
    system_msg: str,
    output_mode: str,
    label_format: str,
    budget: int,
    max_evidence_chars: int,
    max_span_chars: int,
) -> tuple[str, int, int]:
    kept = list(selected)
    for evidence_chars, span_chars, include_spans in (
        (max_evidence_chars, max_span_chars, True),
        (220, 96, True),
        (120, 0, False),
    ):
        while kept:
            user_content = _build_map_user_content(
                claim=claim,
                atoms=atoms,
                selected=kept,
                output_mode=output_mode,
                label_format=label_format,
                max_evidence_chars=evidence_chars,
                max_span_chars=span_chars,
                include_spans=include_spans,
            )
            prompt = build_chat_prompt(tokenizer, system_msg, user_content)
            prompt_tokens = count_tokens(prompt, tokenizer, add_special_tokens=False)
            if prompt_tokens <= budget:
                return prompt, prompt_tokens, len(kept)
            kept.pop()
    user_content = _build_map_user_content(
        claim=claim,
        atoms=[],
        selected=[],
        output_mode=output_mode,
        label_format=label_format,
        max_evidence_chars=0,
        max_span_chars=0,
        include_spans=False,
    )
    prompt = build_chat_prompt(tokenizer, system_msg, user_content)
    prompt_tokens = count_tokens(prompt, tokenizer, add_special_tokens=False)
    if prompt_tokens <= budget:
        return prompt, prompt_tokens, 0
    minimal = _build_minimal_user_content(claim=claim, label_format=label_format)
    prompt = build_chat_prompt(tokenizer, system_msg, minimal)
    return prompt, count_tokens(prompt, tokenizer, add_special_tokens=False), 0


def _build_map_user_content(
    *,
    claim: str,
    atoms: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    output_mode: str,
    label_format: str,
    max_evidence_chars: int,
    max_span_chars: int,
    include_spans: bool,
) -> str:
    label_placeholder = "<a single letter from A-F>" if label_format == "letter" else "<label>"
    response_rule = (
        "Respond with exactly two lines in this format:\nExplanation: <brief explanation>\n"
        f"Label: {label_placeholder}"
        if output_mode == "explanation_label"
        else f"Respond with exactly one line: Label: {label_placeholder}"
    )
    lines = [
        "Classify the claim into exactly one LIAR-RAW label using the evidence map.",
        "",
        "Labels:",
        label_definitions_text(label_format),
        "",
        "Rules:",
        "- Use the selected evidence and atom map as the primary source.",
        "- Do not invent facts not supported by the evidence.",
        "- Treat background/context evidence as weaker than direct support or refutation.",
        f"- {response_rule}",
        "",
        "Claim:",
        claim.strip(),
        "",
        "Claim Atoms:",
    ]
    if atoms:
        for atom in atoms:
            lines.append(f"- {atom.get('atom_id')}: {str(atom.get('text') or '')[:240]}")
    else:
        lines.append("(no atom map available)")
    lines.extend(["", "Selected Evidence Map:"])
    if selected:
        for idx, candidate in enumerate(selected, start=1):
            text = str(candidate.get("text") or "").strip()[:max_evidence_chars]
            atoms_text = ",".join(str(atom) for atom in candidate.get("covered_atom_ids") or []) or "none"
            lines.append(
                "[{idx}] relation={relation}; directness={directness}; atoms={atoms}".format(
                    idx=idx,
                    relation=str(candidate.get("map_relation") or ""),
                    directness=str(candidate.get("map_directness") or ""),
                    atoms=atoms_text,
                )
            )
            if include_spans:
                spans = [str(span).strip()[:max_span_chars] for span in candidate.get("key_spans") or [] if str(span).strip()]
                if spans:
                    lines.append("    spans: " + " | ".join(spans[:2]))
            lines.append(f"    evidence: {text}")
    else:
        lines.append("(no selected evidence available)")
    return "\n".join(lines)


def _build_minimal_user_content(*, claim: str, label_format: str) -> str:
    label_placeholder = "<a single letter from A-F>" if label_format == "letter" else "<label>"
    return "\n".join(
        [
            "Classify the claim into exactly one LIAR-RAW label.",
            f"Respond with exactly one line: Label: {label_placeholder}",
            "Claim:",
            claim.strip(),
        ]
    )


def _summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": float(sum(values) / len(values)), "max": float(max(values))}


if __name__ == "__main__":
    main()
