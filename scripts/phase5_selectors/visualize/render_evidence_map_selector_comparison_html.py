#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from fact_checking.utils.io import read_json, read_jsonl, save_json
except ModuleNotFoundError as exc:
    if exc.name != "yaml":
        raise

    def read_json(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def save_json(payload: dict[str, Any], path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

import render_evidence_map_claim_html as map_html
import render_evidence_chain_graph_html as chain_html


DEFAULT_CANDIDATE_FEATURES = (
    "outputs/selectors/evidence_map_selector/v0_6b_val/"
    "candidate_evidence_map_features_val.jsonl"
)
DEFAULT_LEFT_TRACE = (
    "outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_val/"
    "selection_trace_val.jsonl"
)
DEFAULT_RIGHT_TRACE = (
    "outputs/selectors/evidence_chain_graph/v0_7_budgeted_marginal_adaptive5_10_val/"
    "selection_trace_val.jsonl"
)
DEFAULT_RAW_DATA = "data/raw/LIAR-RAW/val.json"
DEFAULT_COVERAGE_DIFF = (
    "outputs/data_quality/source_coverage_flash/liar_raw/original_diff/"
    "case_coverage_diff_val.jsonl"
)
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_OUTPUT_DIR = "outputs/analysis/map/v0.7"
DEFAULT_LEFT_LABEL = "v0.6c RuleStep"
DEFAULT_RIGHT_LABEL = "v0.7 BudgetedMarginal adaptive5_10"
DEFAULT_TRANSLATION_BASE_URL = "https://api.deepseek.com"
DEFAULT_TRANSLATION_MODEL = "deepseek-v4-flash"

LIAR_RAW_V07_BUILD_COMMAND = """SPLIT=val \\
INPUT=outputs/selectors/evidence_map_selector/v0_6b_val/candidate_evidence_map_features_val.jsonl \\
OUTPUT_DIR=outputs/selectors/evidence_chain_graph/v0_7_budgeted_marginal_adaptive5_10_val \\
bash scripts/phase5_selectors/run/run_evidence_chain_graph_v0_7.sh"""


@dataclass
class ResolvedInputs:
    split: str
    row: dict[str, Any]
    left_trace: dict[str, Any]
    right_trace: dict[str, Any]
    left_graph_row: dict[str, Any] | None
    right_graph_row: dict[str, Any] | None
    raw_row: dict[str, Any] | None
    coverage_diff: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side evidence-map selector comparison for one claim."
    )
    parser.add_argument(
        "--candidate-features",
        default="",
        help="Optional candidate_evidence_map_features_*.jsonl override. Defaults to scanning train/val/test.",
    )
    parser.add_argument("--left-trace", default="", help="Optional v0.6c selection_trace_*.jsonl override.")
    parser.add_argument("--right-trace", default="", help="Optional v0.7 selection_trace_*.jsonl override.")
    parser.add_argument("--left-chain-graph", default="", help="Optional v0.6c chain_graph_*.jsonl override for rich graph edges.")
    parser.add_argument("--right-chain-graph", default="", help="Optional v0.7 chain_graph_*.jsonl override for rich graph edges.")
    parser.add_argument(
        "--raw-data",
        default="",
        help="Optional original split JSON with gold explain/explanation. Defaults to the matched split.",
    )
    parser.add_argument(
        "--coverage-diff",
        default="",
        help="Optional case_coverage_diff_*.jsonl from compare_coverage_to_original.py.",
    )
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma-separated default splits to scan.")
    parser.add_argument("--event-id", default="", help="Event id to render. Accepts both 10004 and 10004.json.")
    parser.add_argument("--claim-contains", default="")
    parser.add_argument("--left-label", default=DEFAULT_LEFT_LABEL)
    parser.add_argument("--right-label", default=DEFAULT_RIGHT_LABEL)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--output", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--translate-zh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call DeepSeek-compatible API and embed Chinese translations. Default: enabled.",
    )
    parser.add_argument("--translation-cache", default="", help="Optional translation cache JSON path. Defaults beside the HTML.")
    parser.add_argument("--force-translate", action="store_true", help="Ignore existing cached translations and call the API again.")
    parser.add_argument(
        "--translation-base-url",
        default=os.environ.get("TRANSLATION_BASE_URL", os.environ.get("TEACHER_BASE_URL", DEFAULT_TRANSLATION_BASE_URL)),
    )
    parser.add_argument(
        "--translation-model",
        default=os.environ.get("TRANSLATION_MODEL", os.environ.get("TEACHER_MODEL", DEFAULT_TRANSLATION_MODEL)),
    )
    parser.add_argument(
        "--translation-api-key-env",
        default=os.environ.get("TRANSLATION_API_KEY_ENV", os.environ.get("TEACHER_API_KEY_ENV", "DEEPSEEK_API_KEY")),
    )
    parser.add_argument("--translation-timeout", type=float, default=120.0)
    parser.add_argument("--translation-max-tokens", type=int, default=4096)
    parser.add_argument("--translation-batch-chars", type=int, default=7000)
    parser.add_argument("--translation-max-retries", type=int, default=3)
    parser.add_argument("--translation-retry-base-sleep", type=float, default=2.0)
    parser.add_argument(
        "--translation-thinking-type",
        default=os.environ.get("THINKING_TYPE", "disabled"),
        choices=["disabled", "enabled", "none"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_inputs(args)
    row = resolved.row
    output_path = Path(args.output) if args.output else default_output_path(args, row, split=resolved.split)
    translations = load_or_build_translations(
        row,
        raw_row=resolved.raw_row,
        coverage_diff=resolved.coverage_diff,
        left_trace=resolved.left_trace,
        right_trace=resolved.right_trace,
        left_graph_row=resolved.left_graph_row,
        right_graph_row=resolved.right_graph_row,
        args=args,
        output_path=output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(
            row,
            raw_row=resolved.raw_row,
            coverage_diff=resolved.coverage_diff,
            left_trace=resolved.left_trace,
            right_trace=resolved.right_trace,
            left_graph_row=resolved.left_graph_row,
            right_graph_row=resolved.right_graph_row,
            args=args,
            translations=translations,
        ),
        encoding="utf-8",
    )
    print(f"Wrote evidence-map selector comparison HTML: {output_path}")


def resolve_inputs(args: argparse.Namespace) -> ResolvedInputs:
    search_errors: list[str] = []
    explicit_features = str(getattr(args, "candidate_features", "") or "").strip()
    split_candidates = [(infer_split_from_path(explicit_features) or first_split(args), explicit_features)] if explicit_features else [
        (split, default_candidate_features_path(split)) for split in split_list(args)
    ]
    for split, candidate_features in split_candidates:
        features_path = Path(candidate_features)
        if not features_path.exists():
            search_errors.append(f"{split}: missing features {candidate_features}")
            continue
        feature_rows = read_jsonl(features_path)
        row = find_feature_row_or_none(feature_rows, event_id=str(getattr(args, "event_id", "") or ""), claim_contains=str(getattr(args, "claim_contains", "") or ""))
        if not row:
            search_errors.append(f"{split}: no matching feature row in {candidate_features}")
            continue
        event_id = str(row.get("event_id") or getattr(args, "event_id", "") or "")
        left_trace_path = str(getattr(args, "left_trace", "") or default_left_trace_path(split))
        right_trace_path = str(getattr(args, "right_trace", "") or default_right_trace_path(split))
        left_chain_graph_path = str(getattr(args, "left_chain_graph", "") or default_left_chain_graph_path(split))
        right_chain_graph_path = str(getattr(args, "right_chain_graph", "") or default_right_chain_graph_path(split))
        raw_data_path = str(getattr(args, "raw_data", "") or default_raw_data_path(split))
        coverage_diff_path = str(getattr(args, "coverage_diff", "") or default_coverage_diff_path(split))
        left_trace = find_trace_row(left_trace_path, event_id=event_id, role="left")
        right_trace = find_trace_row(right_trace_path, event_id=event_id, role="right")
        left_graph_row = load_chain_graph_row(left_chain_graph_path, event_id=event_id)
        right_graph_row = load_chain_graph_row(right_chain_graph_path, event_id=event_id)
        raw_row = load_raw_row(raw_data_path, event_id=event_id)
        coverage_diff = load_coverage_diff_row(coverage_diff_path, event_id=event_id)
        args.candidate_features = candidate_features
        args.left_trace = left_trace_path
        args.right_trace = right_trace_path
        args.left_chain_graph = left_chain_graph_path
        args.right_chain_graph = right_chain_graph_path
        args.raw_data = raw_data_path
        args.coverage_diff = coverage_diff_path
        args.resolved_split = split
        return ResolvedInputs(
            split=split,
            row=row,
            left_trace=left_trace,
            right_trace=right_trace,
            left_graph_row=left_graph_row,
            right_graph_row=right_graph_row,
            raw_row=raw_row,
            coverage_diff=coverage_diff,
        )
    target = getattr(args, "event_id", "") or getattr(args, "claim_contains", "") or "<first row>"
    details = "\n".join(f"- {item}" for item in search_errors) if search_errors else "- no candidate feature paths were searched"
    raise ValueError(f"No default split inputs matched {target!r}.\nSearched:\n{details}")


def find_feature_row_or_none(rows: list[dict[str, Any]], *, event_id: str, claim_contains: str) -> dict[str, Any] | None:
    try:
        return map_html.find_feature_row(rows, event_id=event_id, claim_contains=claim_contains)
    except ValueError:
        return None


def split_list(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "splits", "") or "")
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    return splits or list(DEFAULT_SPLITS)


def first_split(args: argparse.Namespace) -> str:
    return split_list(args)[0]


def infer_split_from_path(path: str) -> str:
    text = str(path or "")
    for split in DEFAULT_SPLITS:
        if f"_{split}/" in text or f"_{split}." in text or f"_{split}_" in text or f"/{split}." in text:
            return split
    return ""


def default_candidate_features_path(split: str) -> str:
    return (
        f"outputs/selectors/evidence_map_selector/v0_6b_{split}/"
        f"candidate_evidence_map_features_{split}.jsonl"
    )


def default_left_trace_path(split: str) -> str:
    return (
        f"outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_{split}/"
        f"selection_trace_{split}.jsonl"
    )


def default_right_trace_path(split: str) -> str:
    return (
        f"outputs/selectors/evidence_chain_graph/v0_7_budgeted_marginal_adaptive5_10_{split}/"
        f"selection_trace_{split}.jsonl"
    )


def default_left_chain_graph_path(split: str) -> str:
    return (
        f"outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_{split}/"
        f"chain_graph_{split}.jsonl"
    )


def default_right_chain_graph_path(split: str) -> str:
    return (
        f"outputs/selectors/evidence_chain_graph/v0_7_budgeted_marginal_adaptive5_10_{split}/"
        f"chain_graph_{split}.jsonl"
    )


def default_raw_data_path(split: str) -> str:
    return f"data/raw/LIAR-RAW/{split}.json"


def default_coverage_diff_path(split: str) -> str:
    return f"outputs/data_quality/source_coverage_flash/liar_raw/original_diff/case_coverage_diff_{split}.jsonl"


def find_trace_row(path: str, *, event_id: str, role: str, expected_selector_name: str = "") -> dict[str, Any]:
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(missing_trace_message(path, role=role))
    wanted_event = map_html.canonical_event_id(event_id)
    selector_mismatches: list[str] = []
    for trace in read_jsonl(trace_path):
        if map_html.canonical_event_id(str(trace.get("event_id") or "")) != wanted_event:
            continue
        selector_name = str(trace.get("selector_name") or "")
        if expected_selector_name and selector_name != expected_selector_name:
            selector_mismatches.append(selector_name or "<missing>")
            continue
        return trace
    if expected_selector_name and selector_mismatches:
        raise ValueError(
            f"No {role} trace row matched event_id={event_id!r} and selector_name={expected_selector_name!r}; "
            f"matched event with selector(s): {sorted(set(selector_mismatches))}."
        )
    raise ValueError(f"No {role} trace row matched event_id={event_id!r} in {path}.")


def load_chain_graph_row(path: str, *, event_id: str) -> dict[str, Any] | None:
    if not path:
        return None
    graph_path = Path(path)
    if not graph_path.exists():
        return None
    wanted_event = map_html.canonical_event_id(event_id)
    for row in read_jsonl(graph_path):
        if map_html.canonical_event_id(str(row.get("event_id") or "")) == wanted_event:
            return row
    return None


def missing_trace_message(path: str, *, role: str) -> str:
    message = f"Missing {role} trace file: {path}"
    if str(path) == DEFAULT_RIGHT_TRACE or "v0_7" in str(path):
        message += "\nGenerate the LIAR-RAW v0.7 trace with:\n\n" + liar_raw_v07_build_command(infer_split_from_path(path) or "val")
    return message


def liar_raw_v07_build_command(split: str) -> str:
    return f"""SPLIT={split} \\
INPUT={default_candidate_features_path(split)} \\
OUTPUT_DIR=outputs/selectors/evidence_chain_graph/v0_7_budgeted_marginal_adaptive5_10_{split} \\
bash scripts/phase5_selectors/run/run_evidence_chain_graph_v0_7.sh"""


def load_raw_row(path: str, *, event_id: str) -> dict[str, Any] | None:
    if not path:
        return None
    raw_path = Path(path)
    if not raw_path.exists():
        return None
    payload = read_json(raw_path)
    if isinstance(payload, dict):
        rows = list(payload.values()) if not looks_like_sample(payload) else [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        return None
    wanted = map_html.canonical_event_id(event_id)
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("event_id", "id", "uid", "filename", "json_id"):
            if map_html.canonical_event_id(str(row.get(key) or "")) == wanted:
                return row
    return None


def load_coverage_diff_row(path: str, *, event_id: str) -> dict[str, Any] | None:
    if not path:
        return None
    diff_path = Path(path)
    if not diff_path.exists():
        return None
    wanted = map_html.canonical_event_id(event_id)
    for row in read_jsonl(diff_path):
        if not isinstance(row, dict):
            continue
        for key in ("event_id", "id", "uid", "filename", "json_id"):
            if map_html.canonical_event_id(str(row.get(key) or "")) == wanted:
                return row
    return None


def looks_like_sample(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("claim", "explain", "explanation", "event_id", "id"))


def default_output_path(args: argparse.Namespace, row: dict[str, Any], *, split: str = "") -> Path:
    event = map_html.slug(map_html.canonical_event_id(str(row.get("event_id") or "claim")))
    left = map_html.slug(str(args.left_label or "left"))
    right = map_html.slug(str(args.right_label or "right"))
    split_prefix = f"{map_html.slug(split)}_" if split else ""
    return Path(getattr(args, "output_dir", "") or DEFAULT_OUTPUT_DIR) / f"{split_prefix}evidence_map_compare_{event}_{left}_vs_{right}.html"


def load_or_build_translations(
    row: dict[str, Any],
    *,
    raw_row: dict[str, Any] | None,
    coverage_diff: dict[str, Any] | None,
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
    left_graph_row: dict[str, Any] | None = None,
    right_graph_row: dict[str, Any] | None = None,
    args: argparse.Namespace,
    output_path: Path,
) -> dict[str, str]:
    cache_path = Path(args.translation_cache) if args.translation_cache else output_path.with_suffix(".zh.json")
    translations: dict[str, str] = {}
    if cache_path.exists() and not bool(args.force_translate):
        payload = read_json(cache_path)
        translations.update({str(k): str(v) for k, v in (payload.get("translations") or {}).items() if str(v).strip()})
    if not bool(args.translate_zh):
        return translations

    items = collect_translation_items(
        row,
        raw_row=raw_row,
        coverage_diff=coverage_diff,
        left_trace=left_trace,
        right_trace=right_trace,
        left_graph_row=left_graph_row,
        right_graph_row=right_graph_row,
        max_candidates=int(args.max_candidates),
    )
    missing = {key: text for key, text in items.items() if key not in translations}
    if not missing:
        return translations
    api_key = os.environ.get(str(args.translation_api_key_env) or "")
    if not api_key:
        raise RuntimeError(
            f"--translate-zh requires API key env {args.translation_api_key_env}. "
            "Set it before rendering translated HTML."
        )
    new_translations, usage_totals = map_html.translate_items_zh(missing, args=args, api_key=api_key)
    translations.update(new_translations)
    save_json(
        {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event_id": str(row.get("event_id") or ""),
            "model": str(args.translation_model),
            "base_url": str(args.translation_base_url),
            "n_items": len(items),
            "n_cached_before": len(items) - len(missing),
            "n_translated_now": len(new_translations),
            "usage_totals": usage_totals,
            "translations": translations,
        },
        cache_path,
    )
    print(f"Wrote zh translation cache: {cache_path}")
    return translations


def collect_translation_items(
    row: dict[str, Any],
    *,
    raw_row: dict[str, Any] | None = None,
    coverage_diff: dict[str, Any] | None = None,
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
    left_graph_row: dict[str, Any] | None = None,
    right_graph_row: dict[str, Any] | None = None,
    max_candidates: int,
) -> dict[str, str]:
    items = map_html.collect_translation_items(row, trace=left_trace, max_candidates=max_candidates)
    items.update(map_html.collect_translation_items(row, trace=right_trace, max_candidates=max_candidates))
    for graph_row in (left_graph_row, right_graph_row):
        if graph_row:
            items.update(
                chain_html.collect_translation_items(
                    graph_row,
                    max_candidates=max_candidates,
                    max_chains=max_candidates,
                )
            )
    map_html.add_translation_item(items, "gold_explain", gold_explain_text(row, raw_row=raw_row))
    if coverage_diff:
        for idx, preview in enumerate(coverage_diff.get("top_evidence_preview") or [], start=1):
            if not isinstance(preview, dict):
                continue
            map_html.add_translation_item(items, f"coverage_preview:{idx}:text", str(preview.get("text") or ""))
    return items


def render_html(
    row: dict[str, Any],
    *,
    raw_row: dict[str, Any] | None = None,
    coverage_diff: dict[str, Any] | None = None,
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
    left_graph_row: dict[str, Any] | None = None,
    right_graph_row: dict[str, Any] | None = None,
    args: argparse.Namespace,
    translations: dict[str, str],
) -> str:
    event_id = str(row.get("event_id") or "")
    title = f"Evidence map selector comparison: {event_id}"
    candidates = comparison_candidates(row, left_trace, right_trace, max_candidates=int(args.max_candidates))
    atoms = list((row.get("evidence_map") or {}).get("claim_atoms") or row.get("claim_atoms") or [])
    left_label = str(args.left_label or DEFAULT_LEFT_LABEL)
    right_label = str(args.right_label or DEFAULT_RIGHT_LABEL)
    gold_label = str(row.get("gold_label") or gold_label_text(raw_row) or "")
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "candidate_features": args.candidate_features,
        "left_trace": args.left_trace,
        "right_trace": args.right_trace,
        "left_chain_graph": getattr(args, "left_chain_graph", ""),
        "right_chain_graph": getattr(args, "right_chain_graph", ""),
        "raw_data": args.raw_data,
        "coverage_diff": getattr(args, "coverage_diff", ""),
        "resolved_split": getattr(args, "resolved_split", ""),
        "event_id": event_id,
        "created_at": created,
    }
    translation_items = collect_translation_items(
        row,
        raw_row=raw_row,
        coverage_diff=coverage_diff,
        left_trace=left_trace,
        right_trace=right_trace,
        left_graph_row=left_graph_row,
        right_graph_row=right_graph_row,
        max_candidates=int(args.max_candidates),
    )
    missing_translation_count = sum(
        1 for key in translation_items if not str(translations.get(key) or "").strip()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <h1>{esc(title)}</h1>
    <div class="meta">
      gold_label={esc(gold_label)} |
      left={esc(str(left_trace.get("selector_name") or left_label))} |
      right={esc(str(right_trace.get("selector_name") or right_label))}
    </div>
    <p class="claim text-wrap-safe">{trans_html("claim", row.get("claim"), translations)}</p>
    <div class="path-grid">
      <div><b>split</b><span class="path-value">{esc(getattr(args, "resolved_split", ""))}</span></div>
      <div><b>features</b><span class="path-value">{esc(args.candidate_features)}</span></div>
      <div><b>{esc(left_label)}</b><span class="path-value">{esc(args.left_trace)}</span></div>
      <div><b>{esc(right_label)}</b><span class="path-value">{esc(args.right_trace)}</span></div>
      <div><b>{esc(left_label)} graph</b><span class="path-value">{esc(getattr(args, "left_chain_graph", ""))}</span></div>
      <div><b>{esc(right_label)} graph</b><span class="path-value">{esc(getattr(args, "right_chain_graph", ""))}</span></div>
      <div><b>coverage diff</b><span class="path-value">{esc(getattr(args, "coverage_diff", ""))}</span></div>
    </div>
    {render_translation_toolbar(translations, args=args, missing_count=missing_translation_count)}
  </header>
  <main>
    <section class="section">
      <h2>Overview</h2>
      {render_overview(left_trace, right_trace, left_label=left_label, right_label=right_label)}
    </section>
    <section class="section">
      <h2>Gold Explanation</h2>
      {render_gold_explanation(row, raw_row=raw_row, args=args, translations=translations)}
    </section>
    <section class="section">
      <h2>Coverage Diff</h2>
      {render_coverage_diff(coverage_diff, args=args, translations=translations)}
    </section>
    <section class="section">
      <h2>Atom Coverage Comparison</h2>
      {render_atom_coverage(atoms, candidates, left_trace, right_trace, left_label=left_label, right_label=right_label)}
    </section>
    <section class="section">
      <h2>Evidence Map Graphs</h2>
      {render_evidence_map_graphs(candidates, atoms=atoms, left_trace=left_trace, right_trace=right_trace, left_graph_row=left_graph_row, right_graph_row=right_graph_row, left_label=left_label, right_label=right_label, translations=translations, max_candidates=int(args.max_candidates))}
    </section>
    <section class="section">
      <h2>Selected Flow</h2>
      <div class="two-col">
        {render_selected_flow(left_trace, candidates, label=left_label, side="left", translations=translations)}
        {render_selected_flow(right_trace, candidates, label=right_label, side="right", translations=translations)}
      </div>
    </section>
    <section class="section">
      <h2>Candidate Comparison</h2>
      {render_candidate_table(candidates, left_trace, right_trace, translations=translations)}
    </section>
    <section class="section">
      <h2>Render Metadata</h2>
      <pre class="raw">{esc(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>
    </section>
  </main>
  {translation_toggle_script(translations, args=args, missing_count=missing_translation_count)}
  {graph_switcher_script()}
</body>
</html>
"""


def render_overview(
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
    *,
    left_label: str,
    right_label: str,
) -> str:
    return (
        '<div class="overview-grid">'
        + render_overview_card(left_trace, label=left_label, side="left")
        + render_overview_card(right_trace, label=right_label, side="right")
        + "</div>"
    )


def render_overview_card(trace: dict[str, Any], *, label: str, side: str) -> str:
    components = trace.get("objective_final_components") or {}
    metrics = [
        ("selected", len(trace.get("selected_evidence_ids") or [])),
        ("precision@5", trace.get("precision@5")),
        ("recall@5", trace.get("recall@5")),
        ("jaccard@5", trace.get("jaccard@5")),
        ("weighted_atom_coverage@5", trace.get("weighted_atom_coverage@5")),
        ("missing_atom_rate@5", trace.get("missing_atom_rate@5")),
        ("stop", trace.get("adaptive_stop_reason")),
    ]
    if "objective_final_score" in trace:
        metrics.extend(
            [
                ("objective score", trace.get("objective_final_score")),
                ("coverage", components.get("coverage")),
                ("node quality", components.get("node_quality")),
                ("pair utility", components.get("pair_utility")),
                ("penalty total", objective_penalty_total(components)),
            ]
        )
    cells = "".join(metric_cell(name, value) for name, value in metrics if value is not None and value != "")
    if not components:
        objective_html = '<div class="small">No objective components on this trace.</div>'
    else:
        objective_html = render_component_strip(components)
    return (
        f'<article class="overview-card {esc(side)}">'
        f"<h3>{esc(label)}</h3>"
        f'<div class="selector text-wrap-safe">{esc(str(trace.get("selector_name") or ""))}</div>'
        f'<div class="metric-grid">{cells}</div>'
        f"{objective_html}"
        "</article>"
    )


def render_gold_explanation(
    row: dict[str, Any],
    *,
    raw_row: dict[str, Any] | None,
    args: argparse.Namespace,
    translations: dict[str, str],
) -> str:
    explain = gold_explain_text(row, raw_row=raw_row)
    if not explain:
        raw_note = f" Raw source checked: {args.raw_data}." if str(getattr(args, "raw_data", "") or "") else ""
        return f'<div class="small">No gold explanation found on this feature row or raw sample.{esc(raw_note)}</div>'
    raw_label = gold_label_text(raw_row) if raw_row else ""
    label_bits = []
    if raw_label:
        label_bits.append(f"raw_label={raw_label}")
    if str(getattr(args, "raw_data", "") or ""):
        label_bits.append(f"raw_data={args.raw_data}")
    meta = f'<div class="small">{" | ".join(esc(bit) for bit in label_bits)}</div>' if label_bits else ""
    return (
        '<article class="gold-explain">'
        f"{meta}"
        f'<p class="text-wrap-safe">{trans_html("gold_explain", explain, translations)}</p>'
        "</article>"
    )


def render_coverage_diff(
    coverage_diff: dict[str, Any] | None,
    *,
    args: argparse.Namespace,
    translations: dict[str, str],
) -> str:
    diff_path = str(getattr(args, "coverage_diff", "") or "")
    if not coverage_diff:
        note = f" Checked coverage diff source: {diff_path}." if diff_path else ""
        return f'<div class="small">No coverage diff row found for this event.{esc(note)}</div>'
    label = str(coverage_diff.get("coverage_label") or "")
    metrics = [
        ("raw label", coverage_diff.get("label")),
        ("coverage_label", label),
        ("coverage_score", coverage_diff.get("coverage_score")),
        ("weak_score", coverage_diff.get("weak_score")),
        ("decision_source", coverage_diff.get("decision_source")),
        ("in_all", bool_text(coverage_diff.get("in_all"))),
        ("in_covered", bool_text(coverage_diff.get("in_covered"))),
        ("in_covered_weak", bool_text(coverage_diff.get("in_covered_weak"))),
    ]
    metric_html = "".join(metric_cell(name, value) for name, value in metrics if value is not None and value != "")
    missing = [str(item) for item in coverage_diff.get("critical_missing") or [] if str(item).strip()]
    missing_html = (
        "".join(map_html.badge(item, class_name="critical-missing") for item in missing)
        if missing
        else '<span class="small">No critical missing anchors recorded.</span>'
    )
    sidecar = str(coverage_diff.get("source_sidecar") or "")
    sidecar_html = f'<div class="small">source_sidecar={esc(sidecar)}</div>' if sidecar else ""
    return (
        f'<article class="coverage-diff {esc(label_class(label))}">'
        '<div class="coverage-head">'
        f'<span class="coverage-label {esc(label_class(label))}">{esc(label or "unknown")}</span>'
        f"{sidecar_html}"
        "</div>"
        f'<div class="metric-grid">{metric_html}</div>'
        '<div class="coverage-missing">'
        "<h3>critical_missing</h3>"
        f'<div class="badges">{missing_html}</div>'
        "</div>"
        '<div class="coverage-preview">'
        "<h3>Top Evidence Preview</h3>"
        f"{render_coverage_preview_table(coverage_diff, translations=translations)}"
        "</div>"
        "</article>"
    )


def render_coverage_preview_table(coverage_diff: dict[str, Any], *, translations: dict[str, str]) -> str:
    rows: list[str] = []
    for idx, preview in enumerate(coverage_diff.get("top_evidence_preview") or [], start=1):
        if not isinstance(preview, dict):
            continue
        text = str(preview.get("text") or "")
        source_bits = [
            str(preview.get(key) or "")
            for key in ("report_id", "source_id", "doc_id", "url")
            if str(preview.get(key) or "").strip()
        ]
        sent_idx = preview.get("sent_idx")
        if sent_idx is not None and sent_idx != "":
            source_bits.append(f"sent={sent_idx}")
        score_bits = [
            f"{key}={fmt(preview.get(key))}"
            for key in (
                "bm25",
                "lexical",
                "lexical_coverage",
                "embedding",
                "embedding_score",
                "hybrid",
                "hybrid_score",
                "source_coverage",
                "all_report_coverage",
                "top_evidence_coverage",
            )
            if preview.get(key) is not None
        ]
        rows.append(
            "<tr>"
            f"<td>{esc(preview.get('rank') or idx)}</td>"
            f"<td>{esc(' | '.join(source_bits))}</td>"
            f"<td>{esc(' | '.join(score_bits))}</td>"
            f"<td>{render_anchor_hits(preview.get('anchor_hits'))}</td>"
            f'<td class="text-cell text-wrap-safe">{trans_html(f"coverage_preview:{idx}:text", text, translations)}</td>'
            "</tr>"
        )
    return table(["rank", "source", "scores", "anchor hits", "text"], rows)


def render_anchor_hits(anchor_hits: Any) -> str:
    if not anchor_hits:
        return '<span class="small">-</span>'
    badges: list[str] = []
    if isinstance(anchor_hits, dict):
        for key, value in anchor_hits.items():
            if isinstance(value, (list, tuple, set)):
                values = [str(item) for item in value if str(item).strip()]
            elif isinstance(value, dict):
                values = [json.dumps(value, ensure_ascii=False, sort_keys=True)]
            else:
                values = [str(value)] if str(value).strip() else []
            badges.extend(f"{key}:{value}" for value in values)
    elif isinstance(anchor_hits, (list, tuple, set)):
        badges.extend(str(item) for item in anchor_hits if str(item).strip())
    else:
        badges.append(str(anchor_hits))
    if not badges:
        return '<span class="small">-</span>'
    return '<div class="badges">' + "".join(map_html.badge(item, class_name="anchor-hit") for item in badges) + "</div>"


def gold_explain_text(row: dict[str, Any], *, raw_row: dict[str, Any] | None = None) -> str:
    for source in (row, raw_row or {}):
        for key in ("gold_explain", "explain", "explanation"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return ""


def gold_label_text(raw_row: dict[str, Any] | None) -> str:
    if not raw_row:
        return ""
    for key in ("gold_label", "label"):
        value = str(raw_row.get(key) or "").strip()
        if value:
            return value
    return ""


def render_atom_coverage(
    atoms: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
    *,
    left_label: str,
    right_label: str,
) -> str:
    if not atoms:
        return '<div class="small">No claim atoms available.</div>'
    by_id = candidate_by_id(candidates)
    left_ids = selected_ids(left_trace)
    right_ids = selected_ids(right_trace)
    rows = []
    for atom in atoms:
        atom_id = str(atom.get("atom_id") or "")
        left_hits = evidence_covering_atom(atom_id, left_ids, by_id)
        right_hits = evidence_covering_atom(atom_id, right_ids, by_id)
        rows.append(
            "<tr>"
            f"<td>{esc(atom_id)}</td>"
            f"<td class='text-cell text-wrap-safe'>{esc(str(atom.get('text') or ''))}</td>"
            f"<td>{render_hit_badges(left_hits)}</td>"
            f"<td>{render_hit_badges(right_hits)}</td>"
            "</tr>"
        )
    return table(["atom", "text", left_label, right_label], rows)


def render_evidence_map_graphs(
    candidates: list[dict[str, Any]],
    *,
    atoms: list[dict[str, Any]],
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
    left_graph_row: dict[str, Any] | None = None,
    right_graph_row: dict[str, Any] | None = None,
    left_label: str,
    right_label: str,
    translations: dict[str, str],
    max_candidates: int,
) -> str:
    if not candidates or not atoms:
        return '<div class="small">No graphable atom/evidence links available.</div>'
    left_selected = selected_index_for_candidates(left_trace, candidates)
    right_selected = selected_index_for_candidates(right_trace, candidates)
    left_count = len(selected_ids(left_trace))
    right_count = len(selected_ids(right_trace))
    left_coverage = fmt(left_trace.get("weighted_atom_coverage@5"))
    right_coverage = fmt(right_trace.get("weighted_atom_coverage@5"))
    return f"""
<div class="map-graph-shell" data-graph-switcher>
  <div class="graph-switcher" role="tablist" aria-label="Selector graph">
    <button class="graph-option active left" type="button" data-graph-option="left" role="tab" aria-selected="true">
      <b>{esc(left_label)}</b>
      <span class="text-wrap-safe">{esc(str(left_trace.get("selector_name") or ""))}</span>
      <small>{left_count} selected · coverage {left_coverage}</small>
    </button>
    <button class="graph-option right" type="button" data-graph-option="right" role="tab" aria-selected="false">
      <b>{esc(right_label)}</b>
      <span class="text-wrap-safe">{esc(str(right_trace.get("selector_name") or ""))}</span>
      <small>{right_count} selected · coverage {right_coverage}</small>
    </button>
  </div>
  <div class="graph-controls">
    <label>relation
      <select data-graph-relation-filter>
        <option value="">all</option>
        <option value="support">support</option>
        <option value="refute">refute</option>
        <option value="qualify">qualify</option>
        <option value="mixed">mixed</option>
        <option value="background">background</option>
        <option value="irrelevant">irrelevant</option>
      </select>
    </label>
    <label>edge
      <select data-graph-edge-filter>
        <option value="">all</option>
        <option value="complements">complements</option>
        <option value="corroborates">corroborates</option>
        <option value="tension">tension</option>
        <option value="bridge_context">bridge_context</option>
        <option value="duplicate">duplicate</option>
        <option value="same_source_context">same_source_context</option>
        <option value="evidence_covers_atom">evidence_covers_atom</option>
      </select>
    </label>
    <label class="checkbox-label"><input type="checkbox" data-graph-selected-only> selected only</label>
    <button type="button" data-graph-fit-toggle>Natural width</button>
  </div>
  {render_graph_detail_panel()}
  <article class="map-graph-panel left graph-panel-active" data-graph-panel="left">
    <h3>{esc(left_label)}</h3>
    <div class="selector small text-wrap-safe">{esc(str(left_trace.get("selector_name") or ""))}</div>
    {render_selector_graph_panel(left_graph_row, candidates=candidates, atoms=atoms, trace=left_trace, selected_index=left_selected, side="left", translations=translations, max_candidates=max_candidates)}
  </article>
  <article class="map-graph-panel right graph-panel-hidden" data-graph-panel="right">
    <h3>{esc(right_label)}</h3>
    <div class="selector small text-wrap-safe">{esc(str(right_trace.get("selector_name") or ""))}</div>
    {render_selector_graph_panel(right_graph_row, candidates=candidates, atoms=atoms, trace=right_trace, selected_index=right_selected, side="right", translations=translations, max_candidates=max_candidates)}
  </article>
</div>
"""


def render_selector_graph_panel(
    graph_row: dict[str, Any] | None,
    *,
    candidates: list[dict[str, Any]],
    atoms: list[dict[str, Any]],
    trace: dict[str, Any],
    selected_index: dict[str, int],
    side: str,
    translations: dict[str, str],
    max_candidates: int,
) -> str:
    if graph_row:
        evidence_nodes = list(graph_row.get("evidence_nodes") or [])[: max(int(max_candidates), 1)]
        selected_ids_for_graph = {str(eid) for eid in graph_row.get("selected_evidence_ids") or []}
        graph_html = chain_html.render_graph(
            graph_row,
            evidence_nodes=evidence_nodes,
            selected_ids=selected_ids_for_graph,
            translations=translations,
        )
        graph_html = normalize_chain_graph_html(graph_html, side=side)
        return graph_html + render_chain_graph_edge_relationships(graph_row, evidence_nodes=evidence_nodes)
    return (
        map_html.render_evidence_graph(candidates, atoms=atoms, selected_index=selected_index, translations=translations)
        + render_evidence_relationships(trace, candidates)
    )


def normalize_chain_graph_html(graph_html: str, *, side: str) -> str:
    safe_side = map_html.slug(side) or "graph"
    html = graph_html.replace('id="chainGraphSvg"', f'id="chainGraphSvg-{safe_side}"')
    html = html.replace('class="graph-svg"', 'class="graph-svg chain-graph-svg"')
    return html


def render_chain_graph_edge_relationships(graph_row: dict[str, Any], *, evidence_nodes: list[dict[str, Any]]) -> str:
    displayed_ids = {str(node.get("node_id") or node.get("evidence_id") or "") for node in evidence_nodes}
    selected_ids_for_graph = {str(eid) for eid in graph_row.get("selected_evidence_ids") or []}
    rows: list[str] = []
    for edge in graph_row.get("edges") or []:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source not in displayed_ids or target not in displayed_ids:
            continue
        edge_type = str(edge.get("edge_type") or "")
        selected = source in selected_ids_for_graph and target in selected_ids_for_graph
        atom_ids = [str(atom_id) for atom_id in edge.get("atom_ids") or [] if str(atom_id).strip()]
        rows.append(
            f'<tr class="edge-row" data-graph-relationship="1" data-from-evidence-id="{esc(source)}" '
            f'data-to-evidence-id="{esc(target)}" data-source="{esc(source)}" data-target="{esc(target)}" '
            f'data-edge-type="{esc(edge_type)}" data-selected="{"1" if selected else "0"}">'
            f"<td>{map_html.badge(edge_type, class_name=edge_type)}</td>"
            f"<td><b>{esc(source)}</b> → <b>{esc(target)}</b></td>"
            f"<td>{fmt(edge.get('weight'))}</td>"
            f"<td>{esc(', '.join(atom_ids) or '-')}</td>"
            f"<td>{esc(str(edge.get('reason') or edge.get('relation') or ''))}</td>"
            "</tr>"
        )
    if not rows:
        return '<section class="evidence-relationships"><h3>Evidence-Evidence Edges</h3><div class="small">No evidence-evidence graph edges recorded for the displayed candidates.</div></section>'
    return (
        '<section class="evidence-relationships">'
        "<h3>Evidence-Evidence Edges</h3>"
        + table(["type", "edge", "weight", "atoms", "note"], rows)
        + "</section>"
    )


def render_graph_detail_panel() -> str:
    return """
  <aside class="graph-detail" data-graph-detail>
    <div>
      <h3 data-graph-detail-title>Evidence detail</h3>
      <div class="small" data-graph-detail-meta>Click an evidence node in the map.</div>
    </div>
    <div class="graph-detail-body text-wrap-safe" data-graph-detail-body>
      Select an evidence card to inspect relation, directness, covered atoms, source, and linked evidence relationships.
    </div>
  </aside>
"""


def render_evidence_relationships(trace: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    by_id = candidate_by_id(candidates)
    rows: list[str] = []
    for step in trace.get("selection_steps") or []:
        eid = str(step.get("evidence_id") or "")
        anchors = [str(anchor) for anchor in step.get("anchor_evidence_ids") or [] if str(anchor).strip()]
        if not anchors:
            continue
        candidate = by_id.get(eid, {})
        for anchor_id in anchors:
            anchor = by_id.get(anchor_id, {})
            relation = relationship_label(step)
            shared_atoms = shared_atom_ids(candidate, anchor)
            rows.append(
                f'<tr data-graph-relationship="1" data-from-evidence-id="{esc(eid)}" data-to-evidence-id="{esc(anchor_id)}">'
                f"<td><b>{esc(eid)}</b> → <b>{esc(anchor_id)}</b></td>"
                f"<td>{esc(relation)}</td>"
                f"<td>{esc(step.get('relation') or candidate.get('map_relation') or '')}/"
                f"{esc(step.get('directness') or candidate.get('map_directness') or '')}</td>"
                f"<td>{esc(', '.join(shared_atoms) or '-')}</td>"
                f"<td>{render_relationship_delta_summary(step.get('component_deltas') or {})}</td>"
                "</tr>"
            )
    if not rows:
        return '<section class="evidence-relationships"><h3>Evidence Relationships</h3><div class="small">No anchor or pair relationships recorded for this selector trace.</div></section>'
    return (
        '<section class="evidence-relationships">'
        "<h3>Evidence Relationships</h3>"
        + table(["edge", "relationship", "map", "shared atoms", "delta summary"], rows)
        + "</section>"
    )


def relationship_label(step: dict[str, Any]) -> str:
    deltas = step.get("component_deltas") or {}
    labels: list[str] = []
    for key, label in (
        ("complements_gain", "complements"),
        ("corroborates_gain", "corroborates"),
        ("conditional_tension_gain", "tension"),
        ("bridge_context_gain", "bridge context"),
    ):
        try:
            value = float(deltas.get(key) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            labels.append(label)
    try:
        pair_utility = float(deltas.get("pair_utility") or 0.0)
    except (TypeError, ValueError):
        pair_utility = 0.0
    if pair_utility and "pair utility" not in labels:
        labels.append("pair utility")
    if not labels:
        labels.append("anchors")
    return " + ".join(labels)


def render_relationship_delta_summary(deltas: Mapping[str, Any]) -> str:
    wanted = [
        ("pair utility", deltas.get("pair_utility")),
        ("complements", deltas.get("complements_gain")),
        ("corroborates", deltas.get("corroborates_gain")),
        ("tension", deltas.get("conditional_tension_gain")),
        ("bridge", deltas.get("bridge_context_gain")),
        ("redundancy", deltas.get("redundancy_penalty")),
    ]
    parts = [f"{label}={fmt(value)}" for label, value in wanted if value not in (None, "")]
    return esc(" | ".join(parts) or "-")


def shared_atom_ids(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    left = {str(atom_id) for atom_id in a.get("covered_atom_ids") or [] if str(atom_id).strip()}
    right = {str(atom_id) for atom_id in b.get("covered_atom_ids") or [] if str(atom_id).strip()}
    return sorted(left & right)


def render_selected_flow(
    trace: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    label: str,
    side: str,
    translations: dict[str, str],
) -> str:
    by_id = candidate_by_id(candidates)
    steps = list(trace.get("selection_steps") or [])
    if not steps:
        ids = list(trace.get("selected_evidence_ids") or [])
        steps = [{"step": idx, "evidence_id": evidence_id} for idx, evidence_id in enumerate(ids, start=1)]
    items: list[str] = []
    for step in steps:
        eid = str(step.get("evidence_id") or "")
        candidate = by_id.get(eid, {})
        text_key = f"{map_html.candidate_translation_base(candidate)}:text" if candidate else ""
        text = candidate.get("text") or ""
        body = [
            f'<div class="flow-head"><span class="rank">{esc(step.get("step"))}</span><b>{esc(eid)}</b></div>',
            '<div class="badges">'
            + map_html.badge(step.get("rule"))
            + map_html.badge(step.get("relation") or candidate.get("map_relation"), class_name=str(step.get("relation") or candidate.get("map_relation") or ""))
            + map_html.badge(step.get("directness") or candidate.get("map_directness"), class_name=str(step.get("directness") or candidate.get("map_directness") or ""))
            + map_html.badge("fallback" if step.get("fallback_used") else "", class_name="fallback")
            + "</div>",
            f'<div class="small">new atoms={esc(", ".join(str(x) for x in step.get("covered_new_atom_ids") or [])) or "-"} | anchors={esc(", ".join(str(x) for x in step.get("anchor_evidence_ids") or [])) or "-"}</div>',
        ]
        if "marginal_gain" in step:
            body.append(
                f'<div class="gain-line">marginal gain <b>{fmt(step.get("marginal_gain"))}</b> · '
                f'coverage after step <b>{fmt(step.get("coverage_after_step"))}</b></div>'
            )
            body.append(render_delta_strip(step.get("component_deltas") or {}))
        if text:
            body.append(f'<div class="flow-text text-wrap-safe">{trans_html(text_key, text, translations, original_html=esc(map_html.truncate(text, 260)))}</div>')
        items.append(f'<article class="flow-card">{"".join(body)}</article>')
    return (
        f'<div class="flow-panel {esc(side)}">'
        f"<h3>{esc(label)}</h3>"
        f'<div class="selector small text-wrap-safe">{esc(str(trace.get("selector_name") or ""))}</div>'
        f'<div class="stop small">stop: {esc(str(trace.get("adaptive_stop_reason") or ""))}</div>'
        f'{"".join(items)}'
        "</div>"
    )


def render_candidate_table(
    candidates: list[dict[str, Any]],
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
    *,
    translations: dict[str, str],
) -> str:
    left_rank = trace_rank_index(left_trace)
    right_rank = trace_rank_index(right_trace)
    left_steps = step_index(left_trace)
    right_steps = step_index(right_trace)
    left_ids = set(left_rank)
    right_ids = set(right_rank)
    rows: list[str] = []
    for candidate in candidates:
        eid = candidate_id(candidate)
        status = comparison_status(eid, left_ids, right_ids)
        left_step = left_steps.get(eid, {})
        right_step = right_steps.get(eid, {})
        text_key = f"{map_html.candidate_translation_base(candidate)}:text"
        spans = [str(span) for span in candidate.get("key_spans") or [] if str(span).strip()]
        score_bits = [
            f"quality={fmt(candidate.get('evidence_map_quality_score'))}",
            f"base={fmt(candidate.get('evidence_map_base_score'))}",
            f"retrieval={fmt(candidate.get('retrieval_score'))}",
        ]
        rows.append(
            f'<tr data-status="{esc(status)}" class="{esc(status)}">'
            f"<td>{esc(eid)}<div>{status_badge(status)}</div></td>"
            f"<td>{rank_label(left_rank.get(eid))}</td>"
            f"<td>{rank_label(right_rank.get(eid))}</td>"
            f"<td>{esc(str(left_step.get('rule') or ''))}</td>"
            f"<td>{fmt(right_step.get('marginal_gain'))}</td>"
            f"<td>{esc(candidate.get('map_relation'))}/{esc(candidate.get('map_directness'))}</td>"
            f"<td>{esc(', '.join(str(atom_id) for atom_id in candidate.get('covered_atom_ids') or []))}</td>"
            f"<td>{esc(' | '.join(score_bits))}</td>"
            f"<td>{esc(candidate.get('source_group'))}</td>"
            f'<td class="text-cell text-wrap-safe">{trans_html(text_key, candidate.get("text"), translations, original_html=map_html.highlight_text(str(candidate.get("text") or ""), spans))}{render_spans(candidate, translations)}</td>'
            "</tr>"
        )
    return table(
        ["eid", "left rank", "right rank", "left rule", "right gain", "map", "atoms", "scores", "source", "text"],
        rows,
    )


def comparison_candidates(
    row: dict[str, Any],
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates = list(row.get("candidates") or [])
    by_id = candidate_by_id(candidates)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates[: max(max_candidates, 1)]:
        eid = candidate_id(candidate)
        if eid:
            seen.add(eid)
        out.append(candidate)
    for eid in list(left_trace.get("selected_evidence_ids") or []) + list(right_trace.get("selected_evidence_ids") or []):
        eid = str(eid)
        if eid and eid not in seen and eid in by_id:
            out.append(by_id[eid])
            seen.add(eid)
    return out


def selected_ids(trace: dict[str, Any]) -> set[str]:
    ids = {str(eid) for eid in trace.get("selected_evidence_ids") or [] if str(eid)}
    if ids:
        return ids
    return {candidate_id(candidate) for candidate in trace.get("selected_candidates") or [] if candidate_id(candidate)}


def trace_rank_index(trace: dict[str, Any]) -> dict[str, int]:
    ids = list(trace.get("selected_evidence_ids") or [])
    if not ids:
        ids = [candidate_id(candidate) for candidate in trace.get("selected_candidates") or []]
    return {str(eid): idx for idx, eid in enumerate(ids, start=1) if str(eid)}


def selected_index_for_candidates(trace: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = candidate_by_id(candidates)
    selected_payloads = {candidate_id(candidate): dict(candidate) for candidate in trace.get("selected_candidates") or [] if candidate_id(candidate)}
    out: dict[str, dict[str, Any]] = {}
    for evidence_id, rank in trace_rank_index(trace).items():
        base = dict(by_id.get(evidence_id) or {})
        payload = {**base, **selected_payloads.get(evidence_id, {})}
        if not payload:
            payload = {"evidence_id": evidence_id}
        payload["selection_rank"] = rank
        key_source = by_id.get(evidence_id) or payload
        key = map_html.key_for_candidate(key_source)
        if key:
            out[key] = payload
    return out


def step_index(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(step.get("evidence_id") or ""): dict(step) for step in trace.get("selection_steps") or [] if step.get("evidence_id")}


def candidate_by_id(candidates: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {candidate_id(candidate): candidate for candidate in candidates if candidate_id(candidate)}


def candidate_id(candidate: dict[str, Any]) -> str:
    for key in ("evidence_id", "candidate_uid", "candidate_key"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return ""


def comparison_status(eid: str, left_ids: set[str], right_ids: set[str]) -> str:
    if eid in left_ids and eid in right_ids:
        return "common"
    if eid in left_ids:
        return "left-only"
    if eid in right_ids:
        return "right-only"
    return "unselected"


def evidence_covering_atom(atom_id: str, evidence_ids: set[str], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for evidence_id in sorted(evidence_ids):
        candidate = by_id.get(evidence_id)
        if not candidate:
            continue
        if str(atom_id) in {str(item) for item in candidate.get("covered_atom_ids") or []}:
            hits.append(candidate)
    return hits


def render_hit_badges(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return '<span class="small">-</span>'
    return "".join(
        map_html.badge(
            f"{candidate_id(candidate)} {candidate.get('map_relation')}/{candidate.get('map_directness')}",
            class_name=str(candidate.get("map_relation") or ""),
        )
        for candidate in candidates
    )


def render_component_strip(components: dict[str, Any]) -> str:
    keys = [
        "coverage",
        "node_quality",
        "pair_utility",
        "redundancy_penalty",
        "background_penalty",
        "source_concentration_penalty",
        "length_penalty",
    ]
    cells = [metric_cell(key.replace("_", " "), components.get(key)) for key in keys if key in components]
    return f'<div class="component-strip">{"".join(cells)}</div>' if cells else '<div class="small">No objective components available.</div>'


def render_delta_strip(deltas: dict[str, Any]) -> str:
    top = top_component_deltas(deltas, limit=5)
    if not top:
        return '<div class="small">No component deltas.</div>'
    return '<div class="delta-strip">' + "".join(metric_cell(name.replace("_", " "), value) for name, value in top) + "</div>"


def top_component_deltas(deltas: dict[str, Any], *, limit: int) -> list[tuple[str, Any]]:
    rows: list[tuple[float, str, Any]] = []
    for key, value in deltas.items():
        try:
            magnitude = abs(float(value))
        except (TypeError, ValueError):
            continue
        if magnitude <= 1e-12:
            continue
        rows.append((magnitude, str(key), value))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [(key, value) for _magnitude, key, value in rows[: max(limit, 1)]]


def objective_penalty_total(components: dict[str, Any]) -> float:
    total = 0.0
    for key, value in components.items():
        if not str(key).endswith("_penalty"):
            continue
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total


def metric_cell(name: str, value: Any) -> str:
    return f'<div class="metric"><b>{esc(name)}</b><span class="metric-value">{fmt(value)}</span></div>'


def bool_text(value: Any) -> Any:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return value


def label_class(label: str) -> str:
    normalized = str(label or "unknown").strip().lower().replace("_", "-").replace(" ", "-")
    return normalized or "unknown"


def render_spans(candidate: dict[str, Any], translations: dict[str, str]) -> str:
    spans = [str(span) for span in candidate.get("key_spans") or [] if str(span).strip()]
    if not spans:
        return ""
    cbase = map_html.candidate_translation_base(candidate)
    return (
        '<div class="span-list">'
        + "".join(
            f'<span class="span">{trans_html(f"{cbase}:span:{idx}", span, translations)}</span>'
            for idx, span in enumerate(spans)
        )
        + "</div>"
    )


def status_badge(status: str) -> str:
    return f'<span class="status-badge {esc(status)}">{esc(status)}</span>'


def rank_label(rank: int | None) -> str:
    return f"#{rank}" if rank else ""


def table(headers: list[str], rows: list[str]) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = "".join(rows) if rows else f"<tr><td colspan='{len(headers)}' class='small'>No rows.</td></tr>"
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def trans_html(key: str, text: Any, translations: dict[str, str], *, original_html: str | None = None) -> str:
    return map_html.trans_html(key, text, translations, original_html=original_html)


def render_translation_toolbar(translations: dict[str, str], *, args: argparse.Namespace, missing_count: int = 0) -> str:
    count = len(translations)
    web_enabled = bool(getattr(args, "web_translation_enabled", False))
    disabled = "" if count or web_enabled else " disabled"
    if count and missing_count > 0 and web_enabled:
        status = f"{count} zh translations embedded; {missing_count} missing, click to update."
    elif count and missing_count > 0:
        status = f"{count} zh translations embedded; {missing_count} missing. Re-render with --translate-zh."
    elif count:
        status = f"{count} zh translations embedded"
    elif web_enabled:
        status = "尚未翻译，点击后调用 API。未缓存 case 需要服务环境提供 DEEPSEEK_API_KEY。"
    else:
        status = "No zh translations embedded. Re-render with --translate-zh or enable live translation."
    return (
        '<div class="translation-toolbar" data-translation-toolbar>'
        f'<button type="button" data-translation-toggle{disabled}>显示中文</button>'
        f'<span class="small" data-translation-status>{esc(status)}</span>'
        "</div>"
    )


def translation_toggle_script(
    translations: dict[str, str],
    *,
    args: argparse.Namespace,
    missing_count: int = 0,
) -> str:
    return _translation_toggle_script(translations, args=args, missing_count=missing_count)


def _translation_toggle_script(translations: dict[str, str], *, args: argparse.Namespace, missing_count: int) -> str:
    payload = json.dumps(translations, ensure_ascii=False).replace("</", "<\\/")
    request = translation_request_payload(args)
    request_payload = json.dumps(request, ensure_ascii=False).replace("</", "<\\/")
    return f"""<script>
window.EVIDENCE_TRANSLATIONS = {payload};
window.EVIDENCE_TRANSLATION_REQUEST = {request_payload};
window.EVIDENCE_TRANSLATION_MISSING_COUNT = {int(max(missing_count, 0))};
(() => {{
  const translations = window.EVIDENCE_TRANSLATIONS || {{}};
  const request = window.EVIDENCE_TRANSLATION_REQUEST || {{}};
  const missingTranslationCount = Number(window.EVIDENCE_TRANSLATION_MISSING_COUNT || 0);
  const button = document.querySelector("[data-translation-toggle]");
  const status = document.querySelector("[data-translation-status]");
  const svgTexts = Array.from(document.querySelectorAll("[data-i18n-svg-key]"));
  const reloadKey = "evidence-map-show-zh-after-translation";
  const compact = (value, maxChars) => {{
    const text = String(value || "");
    if (!maxChars || text.length <= maxChars) return text;
    return text.slice(0, Math.max(maxChars - 3, 0)).trimEnd() + "...";
  }};
  const setSvgLanguage = (lang) => {{
    svgTexts.forEach((el) => {{
      const original = el.dataset.i18nOriginal || "";
      const zh = el.dataset.i18nZh || "";
      const maxChars = Number(el.dataset.i18nMax || "0");
      el.textContent = compact(lang === "zh" && zh ? zh : original, maxChars);
    }});
  }};
  const setLanguage = (lang) => {{
    document.body.classList.toggle("zh-mode", lang === "zh");
    setSvgLanguage(lang);
    if (button) button.textContent = lang === "zh" ? "Show English" : "显示中文";
    if (status) {{
      const cachedCount = Object.keys(translations).length;
      const missingText = missingTranslationCount > 0 ? `; ${{missingTranslationCount}} missing` : "";
      status.textContent = lang === "zh" ? `中文翻译已显示${{missingText}}` : `${{cachedCount}} zh translations embedded${{missingText}}`;
    }}
  }};
  const fetchTranslations = async () => {{
    if (!request.enabled || !request.url) return;
    if (status) status.textContent = "正在翻译，请稍候...";
    if (button) {{
      button.disabled = true;
      button.textContent = "翻译中...";
    }}
    try {{
      const response = await fetch(request.url, {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify(request.payload || {{}})
      }});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${{response.status}}`);
      if (status) status.textContent = `翻译完成，已缓存 ${{data.translation_count || 0}} 条，正在刷新...`;
      sessionStorage.setItem(reloadKey, "1");
      window.location.reload();
    }} catch (error) {{
      if (status) status.textContent = `翻译失败：${{error.message || error}}`;
      if (button) {{
        button.disabled = false;
        button.textContent = "显示中文";
      }}
    }}
  }};
  if (!button) return;
  let lang = "en";
  const hasTranslations = Object.keys(translations).length > 0;
  button.addEventListener("click", () => {{
    if (!hasTranslations || (missingTranslationCount > 0 && request.enabled)) {{
      fetchTranslations();
      return;
    }}
    lang = lang === "en" ? "zh" : "en";
    setLanguage(lang);
  }});
  if (hasTranslations && sessionStorage.getItem(reloadKey) === "1") {{
    sessionStorage.removeItem(reloadKey);
    lang = "zh";
    setLanguage(lang);
  }} else {{
    setSvgLanguage(lang);
  }}
}})();
</script>"""


def translation_request_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(getattr(args, "web_translation_enabled", False)):
        return {"enabled": False}
    base_path = str(getattr(args, "web_base_path", "") or "").rstrip("/")
    if not base_path:
        return {"enabled": False}
    token = str(getattr(args, "web_token", "") or "")
    url = f"{base_path}/api/translate"
    if token:
        url += f"?token={quote(token, safe='')}"
    return {
        "enabled": True,
        "url": url,
        "payload": {
            "split": str(getattr(args, "web_split", "") or getattr(args, "resolved_split", "") or ""),
            "event_id": str(getattr(args, "web_event_id", "") or ""),
            "left_label": str(getattr(args, "left_label", "") or DEFAULT_LEFT_LABEL),
            "right_label": str(getattr(args, "right_label", "") or DEFAULT_RIGHT_LABEL),
        },
    }


def graph_switcher_script() -> str:
    return """<script>
(() => {
  const root = document.querySelector("[data-graph-switcher]");
  if (!root) return;
  const options = Array.from(root.querySelectorAll("[data-graph-option]"));
  const panels = Array.from(root.querySelectorAll("[data-graph-panel]"));
  const relationFilter = root.querySelector("[data-graph-relation-filter]");
  const edgeFilter = root.querySelector("[data-graph-edge-filter]");
  const selectedOnly = root.querySelector("[data-graph-selected-only]");
  const fitToggle = root.querySelector("[data-graph-fit-toggle]");
  const detail = root.querySelector("[data-graph-detail]");
  const detailTitle = root.querySelector("[data-graph-detail-title]");
  const detailMeta = root.querySelector("[data-graph-detail-meta]");
  const detailBody = root.querySelector("[data-graph-detail-body]");

  const escapeHtml = (value) => String(value || "").replace(/[&<>"']/g, ch => (
    {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]
  ));

  const activePanel = () => panels.find((panel) => panel.classList.contains("graph-panel-active")) || panels[0];

  const setActive = (side) => {
    options.forEach((option) => {
      const active = option.dataset.graphOption === side;
      option.classList.toggle("active", active);
      option.setAttribute("aria-selected", active ? "true" : "false");
    });
    panels.forEach((panel) => {
      const active = panel.dataset.graphPanel === side;
      panel.classList.toggle("graph-panel-active", active);
      panel.classList.toggle("graph-panel-hidden", !active);
    });
    applyFilters();
  };

  const evidenceRelationships = (panel, evidenceId) => {
    return Array.from(panel.querySelectorAll("[data-graph-relationship='1']")).filter((row) => (
      row.dataset.fromEvidenceId === evidenceId || row.dataset.toEvidenceId === evidenceId
    ));
  };

  const setEvidenceDetail = (node) => {
    if (!node || !detail) return;
    const panel = node.closest("[data-graph-panel]") || activePanel();
    const evidenceId = node.dataset.graphEvidenceId || "";
    const relation = node.dataset.relation || "";
    const directness = node.dataset.directness || "";
    const selected = node.dataset.selected === "1" ? "selected" : "not selected";
    const atoms = (node.dataset.coveredAtoms || "").split("|").filter(Boolean).join(", ") || "-";
    const source = node.dataset.sourceGroup || "-";
    const role = node.dataset.role || "-";
    const relationships = evidenceRelationships(panel, evidenceId);
    const relationshipText = relationships.length
      ? relationships.map((row) => row.innerText.trim().replace(/\\s+/g, " ")).join("\\n")
      : "No recorded anchor/pair relationship for this evidence in the active selector trace.";
    if (detailTitle) detailTitle.textContent = evidenceId ? `Evidence ${evidenceId}` : "Evidence detail";
    if (detailMeta) detailMeta.textContent = `${relation}/${directness} · ${selected} · atoms ${atoms}`;
    if (detailBody) {
      detailBody.innerHTML = [
        `<div><b>role</b>: ${escapeHtml(role)}</div>`,
        `<div><b>source</b>: ${escapeHtml(source)}</div>`,
        `<div><b>relationships</b>:<pre class="graph-detail-relationships">${escapeHtml(relationshipText)}</pre></div>`,
        `<div><b>text</b>: ${escapeHtml(node.dataset.graphText || "")}</div>`,
      ].join("");
    }
  };

  const highlightEvidence = (node) => {
    const panel = node.closest("[data-graph-panel]") || activePanel();
    const evidenceId = node.dataset.graphEvidenceId || "";
    panels.forEach((item) => {
      item.querySelectorAll("[data-graph-node='evidence']").forEach((el) => {
        el.classList.toggle("graph-node-active", el === node);
      });
      item.querySelectorAll("[data-graph-edge], .graph-edge[data-source][data-target]").forEach((el) => {
        const connected = el.dataset.evidenceId === evidenceId ||
          el.dataset.source === evidenceId ||
          el.dataset.target === evidenceId;
        el.classList.toggle("graph-edge-active", Boolean(evidenceId && connected && item === panel));
      });
      item.querySelectorAll("[data-graph-relationship='1']").forEach((row) => {
        const active = item === panel && evidenceId && (
          row.dataset.fromEvidenceId === evidenceId || row.dataset.toEvidenceId === evidenceId
        );
        row.classList.toggle("relationship-active", Boolean(active));
      });
    });
    setEvidenceDetail(node);
  };

  const applyFilters = () => {
    const relation = relationFilter ? relationFilter.value : "";
    const edgeType = edgeFilter ? edgeFilter.value : "";
    const onlySelected = Boolean(selectedOnly && selectedOnly.checked);
    panels.forEach((panel) => {
      panel.querySelectorAll("[data-graph-node='evidence'], [data-graph-edge]").forEach((el) => {
        const itemRelation = el.dataset.relation || "";
        const itemSelected = el.dataset.selected === "1";
        const itemEdgeType = el.dataset.edgeType || "";
        const hiddenByRelation = relation && itemRelation !== relation;
        const hiddenByEdge = edgeType && itemEdgeType && itemEdgeType !== edgeType;
        const hiddenBySelection = onlySelected && !itemSelected;
        el.classList.toggle("graph-filter-hidden", Boolean(hiddenByRelation || hiddenByEdge || hiddenBySelection));
      });
      panel.querySelectorAll(".graph-edge[data-source][data-target], .edge-row[data-edge-type]").forEach((el) => {
        const itemEdgeType = el.dataset.edgeType || "";
        const itemSelected = el.dataset.selected === "1";
        const hiddenByEdge = edgeType && itemEdgeType !== edgeType;
        const hiddenBySelection = onlySelected && !itemSelected;
        el.classList.toggle("graph-filter-hidden", Boolean(hiddenByEdge || hiddenBySelection));
      });
    });
  };

  options.forEach((option) => {
    option.addEventListener("click", () => setActive(option.dataset.graphOption || "left"));
  });
  if (relationFilter) relationFilter.addEventListener("change", applyFilters);
  if (edgeFilter) edgeFilter.addEventListener("change", applyFilters);
  if (selectedOnly) selectedOnly.addEventListener("change", applyFilters);
  if (fitToggle) {
    fitToggle.addEventListener("click", () => {
      root.classList.toggle("graph-fit-natural");
      fitToggle.textContent = root.classList.contains("graph-fit-natural") ? "Fit width" : "Natural width";
    });
  }
  root.addEventListener("click", (event) => {
    const node = event.target.closest("[data-graph-node='evidence']");
    if (node && root.contains(node)) highlightEvidence(node);
  });
  root.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const node = event.target.closest("[data-graph-node='evidence']");
    if (!node || !root.contains(node)) return;
    event.preventDefault();
    highlightEvidence(node);
  });
  setActive("left");
  initChainGraphDragging(root);
})();

function initChainGraphDragging(root) {
  const svgs = Array.from(root.querySelectorAll("svg.chain-graph-svg"));
  for (const svg of svgs) {
    const nodes = new Map();
    const refreshNodes = () => {
      nodes.clear();
      svg.querySelectorAll(".graph-node[data-node-id]").forEach((node) => {
        const id = node.dataset.nodeId;
        if (id) nodes.set(id, node);
      });
    };
    const centerOf = (id) => {
      const node = nodes.get(id);
      if (!node) return null;
      const x = Number(node.dataset.x || 0);
      const y = Number(node.dataset.y || 0);
      const w = Number(node.dataset.w || 0);
      const h = Number(node.dataset.h || 0);
      return {x: x + w / 2, y: y + h / 2};
    };
    const updateEdges = () => {
      refreshNodes();
      svg.querySelectorAll(".graph-edge[data-source][data-target]").forEach((edge) => {
        const source = centerOf(edge.dataset.source);
        const target = centerOf(edge.dataset.target);
        if (!source || !target) return;
        const dx = target.x - source.x;
        const curve = Math.max(70, Math.min(230, Math.abs(dx) * 0.45));
        const d = `M ${source.x.toFixed(1)} ${source.y.toFixed(1)} C ${(source.x + curve).toFixed(1)} ${source.y.toFixed(1)}, ${(target.x - curve).toFixed(1)} ${target.y.toFixed(1)}, ${target.x.toFixed(1)} ${target.y.toFixed(1)}`;
        edge.setAttribute("d", d);
      });
    };
    const svgPoint = (evt) => {
      const point = svg.createSVGPoint();
      point.x = evt.clientX;
      point.y = evt.clientY;
      return point.matrixTransform(svg.getScreenCTM().inverse());
    };
    let drag = null;
    svg.querySelectorAll(".graph-node.draggable").forEach((node) => {
      node.addEventListener("pointerdown", (evt) => {
        evt.preventDefault();
        node.setPointerCapture(evt.pointerId);
        const point = svgPoint(evt);
        drag = {
          node,
          pointerId: evt.pointerId,
          offsetX: point.x - Number(node.dataset.x || 0),
          offsetY: point.y - Number(node.dataset.y || 0),
        };
        node.classList.add("dragging");
      });
      node.addEventListener("pointermove", (evt) => {
        if (!drag || drag.node !== node) return;
        const point = svgPoint(evt);
        const x = point.x - drag.offsetX;
        const y = point.y - drag.offsetY;
        node.dataset.x = x.toFixed(1);
        node.dataset.y = y.toFixed(1);
        node.setAttribute("transform", `translate(${x.toFixed(1)},${y.toFixed(1)})`);
        updateEdges();
      });
      node.addEventListener("pointerup", (evt) => {
        if (!drag || drag.node !== node) return;
        node.releasePointerCapture(evt.pointerId);
        node.classList.remove("dragging");
        drag = null;
      });
      node.addEventListener("pointercancel", () => {
        node.classList.remove("dragging");
        drag = null;
      });
    });
    updateEdges();
  }
}
</script>"""


def fmt(value: Any) -> str:
    return map_html.fmt(value)


def esc(value: Any) -> str:
    return map_html.esc(value)


def css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #202733;
  --muted: #637084;
  --line: #d9e0e8;
  --left: #7a5a18;
  --right: #1f6f6d;
  --common: #315fba;
  --soft: #eef2f7;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
  line-height: 1.45;
}
header {
  background: #fff;
  border-bottom: 1px solid var(--line);
  padding: 22px 28px 18px;
  position: sticky;
  top: 0;
  z-index: 5;
}
main {
  max-width: 1560px;
  margin: 0 auto;
  padding: 22px 28px 44px;
}
h1 { font-size: 22px; line-height: 1.2; margin: 0 0 8px; letter-spacing: 0; }
h2 { font-size: 16px; margin: 0 0 12px; letter-spacing: 0; }
h3 { font-size: 14px; margin: 0 0 8px; letter-spacing: 0; }
.claim { max-width: 1180px; font-size: 16px; margin: 8px 0 0; }
.meta, .small { color: var(--muted); font-size: 12px; }
.section {
  margin-bottom: 18px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(25, 33, 46, 0.06);
  padding: 16px;
}
.path-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 8px;
  margin-top: 12px;
  font-size: 12px;
}
.path-grid div { border: 1px solid var(--line); border-radius: 6px; padding: 7px 9px; background: #fbfcfd; }
.path-grid b { display: block; color: #384459; margin-bottom: 2px; }
.path-value {
  display: block;
  color: var(--muted);
  overflow-wrap: anywhere;
  max-height: 4.8em;
  overflow: auto;
}
.text-wrap-safe {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: normal;
  min-width: 0;
}
.translation-toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
.translation-toolbar button {
  min-height: 32px;
  border: 1px solid #b8c6d7;
  border-radius: 6px;
  background: #ffffff;
  color: #29435f;
  padding: 5px 10px;
  font: inherit;
  font-size: 13px;
  font-weight: 650;
  cursor: pointer;
}
.translation-toolbar button:disabled { cursor: not-allowed; opacity: 0.55; }
.i18n-zh { display: none; }
body.zh-mode .i18n-original { display: none; }
body.zh-mode .i18n-zh { display: inline; }
.overview-grid, .two-col {
  display: grid;
  grid-template-columns: repeat(2, minmax(320px, 1fr));
  gap: 14px;
  align-items: start;
}
.map-graph-shell { display: grid; gap: 12px; min-width: 0; }
.graph-switcher {
  display: grid;
  grid-template-columns: repeat(2, minmax(220px, 1fr));
  gap: 10px;
}
.graph-option {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  padding: 10px 12px;
  text-align: left;
  color: var(--ink);
  cursor: pointer;
  min-width: 0;
}
.graph-option.active { outline: 2px solid var(--common); outline-offset: -2px; }
.graph-option.left { box-shadow: inset 4px 0 0 var(--left); }
.graph-option.right { box-shadow: inset 4px 0 0 var(--right); }
.graph-option b, .graph-option span, .graph-option small { display: block; min-width: 0; }
.graph-option span { margin-top: 3px; color: var(--muted); font-size: 12px; }
.graph-option small { margin-top: 5px; color: #46576e; }
.graph-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfd;
  padding: 9px 10px;
}
.graph-controls label {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  color: var(--muted);
  font-size: 12px;
  font-weight: 650;
}
.graph-controls select, .graph-controls button {
  min-height: 30px;
  border: 1px solid #b8c6d7;
  border-radius: 6px;
  background: #fff;
  color: #29435f;
  padding: 4px 8px;
  font: inherit;
  font-size: 12px;
}
.checkbox-label input { margin: 0; }
.graph-detail {
  display: grid;
  grid-template-columns: minmax(180px, 0.32fr) minmax(260px, 1fr);
  gap: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  padding: 11px 12px;
  min-width: 0;
}
.graph-detail h3 { margin-bottom: 4px; }
.graph-detail-body {
  font-size: 12px;
  color: #344155;
}
.graph-detail-body b { color: #253044; }
.graph-detail-relationships {
  margin: 4px 0 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font: inherit;
  color: #46576e;
}
.map-graph-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfd;
  padding: 13px;
  min-width: 0;
  overflow: hidden;
}
.map-graph-panel.graph-panel-hidden { display: none; }
.map-graph-panel.graph-panel-active { display: block; }
.map-graph-panel.left { box-shadow: inset 4px 0 0 var(--left); }
.map-graph-panel.right { box-shadow: inset 4px 0 0 var(--right); }
.overview-card, .flow-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfd;
  padding: 13px;
}
.gold-explain {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfd;
  padding: 13px;
}
.gold-explain p {
  margin: 8px 0 0;
  max-width: 1180px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.coverage-diff {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfd;
  padding: 13px;
}
.coverage-head {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.coverage-label {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 12px;
  font-weight: 780;
  background: #edf1f5;
  color: #536171;
}
.coverage-label.covered { background: #e6f6ee; color: #247a52; }
.coverage-label.weak-covered { background: #fff2d9; color: #a5681f; }
.coverage-label.uncovered { background: #fdebea; color: #b8443e; }
.coverage-missing, .coverage-preview { margin-top: 12px; }
.badge.critical-missing { background: #fdebea; color: #b8443e; }
.badge.anchor-hit { background: #e7efff; color: var(--common); }
.overview-card.left, .flow-panel.left { box-shadow: inset 4px 0 0 var(--left); }
.overview-card.right, .flow-panel.right { box-shadow: inset 4px 0 0 var(--right); }
.selector { color: var(--muted); overflow-wrap: anywhere; margin-bottom: 10px; }
.metric-grid, .component-strip, .delta-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
  gap: 8px;
  margin-top: 8px;
}
.metric {
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  padding: 7px 8px;
  min-width: 0;
}
.metric b { display: block; color: var(--muted); font-size: 12px; font-weight: 620; }
.metric-value { display: block; font-size: 16px; font-weight: 720; margin-top: 2px; overflow-wrap: anywhere; min-width: 0; }
.flow-card {
  border: 1px solid var(--line);
  border-radius: 7px;
  background: #fff;
  padding: 10px;
  margin-top: 10px;
}
.flow-head { display: flex; gap: 8px; align-items: center; }
.rank {
  display: inline-flex;
  min-width: 26px;
  height: 24px;
  border-radius: 999px;
  align-items: center;
  justify-content: center;
  background: #e6edf7;
  color: #28405f;
  font-size: 12px;
  font-weight: 750;
}
.gain-line { margin-top: 7px; font-size: 13px; color: #29435f; }
.flow-text { margin-top: 8px; font-size: 13px; }
.badges, .legend, .span-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  margin-top: 7px;
}
.badge, .status-badge {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 650;
  background: var(--soft);
  color: #354155;
  white-space: nowrap;
}
.badge.support { background: #e6f6ee; color: #247a52; }
.badge.refute { background: #fdebea; color: #b8443e; }
.badge.qualify, .badge.mixed { background: #fff2d9; color: #a5681f; }
.badge.background, .badge.context { background: #edf1f5; color: #536171; }
.badge.irrelevant, .badge.none { background: #f3e9ee; color: #8a4361; }
.badge.fallback { background: #fff2d9; color: #7a5a18; }
.badge.complements, .badge.corroborates { background: #e6f6ee; color: #247a52; }
.badge.tension { background: #fdebea; color: #b8443e; }
.badge.bridge_context, .badge.bridge-context { background: #fff2d9; color: #a5681f; }
.badge.duplicate { background: #f3e9ee; color: #8a4361; }
.badge.same_source_context, .badge.same-source-context { background: #edf1f5; color: #536171; }
.badge.evidence_covers_atom, .badge.evidence-covers-atom, .badge.selected_chain_step, .badge.selected-chain-step { background: #e7efff; color: #2f6fcf; }
.status-badge.common, .swatch.common { background: #e7efff; color: var(--common); }
.status-badge.left-only, .swatch.left-only { background: #fff2d9; color: var(--left); }
.status-badge.right-only, .swatch.right-only { background: #ddf4f1; color: var(--right); }
.status-badge.unselected, .swatch.unselected { background: #edf1f5; color: #536171; }
.swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 3px;
  margin-right: 5px;
}
.graph-wrap {
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  margin-top: 8px;
}
.graph-svg { display: block; width: 100%; min-width: 1080px; }
.graph-fit-natural .graph-svg { width: auto; }
.graph-filter-hidden { display: none; }
.graph-node rect { fill: #fff; stroke: #bdc7d3; stroke-width: 1.1; }
.graph-node.atom rect { fill: #f6f9fd; }
.graph-node.selected rect { fill: #eef4ff; stroke: #2f6fcf; stroke-width: 1.8; }
.graph-node.claim rect { fill: #f7faff; stroke: #a9c6f7; }
.graph-node.side rect { fill: #fff; stroke: #d8e0ea; }
.graph-node.evidence.common rect { fill: #eef4ff; stroke: var(--common); stroke-width: 1.8; }
.graph-node.evidence.left-only rect { fill: #fff7e8; stroke: var(--left); stroke-width: 1.8; }
.graph-node.evidence.right-only rect { fill: #e8f8f5; stroke: var(--right); stroke-width: 1.8; }
.graph-node.evidence.unselected rect { fill: #fbfcfd; stroke: #cbd5e1; }
.graph-node.evidence { cursor: pointer; }
.graph-node.draggable { cursor: grab; touch-action: none; }
.graph-node.dragging { cursor: grabbing; }
.graph-node.dragging rect { stroke: #2f6fcf; stroke-width: 2.4; }
.graph-node.evidence:focus rect,
.graph-node.evidence.graph-node-active rect {
  stroke: #172f66;
  stroke-width: 2.6;
  filter: drop-shadow(0 2px 4px rgba(23, 47, 102, 0.22));
}
.graph-title { font-size: 12px; font-weight: 730; fill: #253044; }
.graph-subtitle { font-size: 11px; fill: #667386; }
.graph-evidence-text {
  color: #667386;
  font-size: 11px;
  line-height: 1.22;
  overflow-wrap: anywhere;
  word-break: normal;
}
.graph-rank { font-size: 10px; font-weight: 800; fill: #2f6fcf; }
.graph-oracle { font-size: 9px; font-weight: 800; fill: #8a4361; }
.graph-edge { fill: none; }
.graph-edge.selected_chain_step { stroke-width: 4.2; opacity: .94; }
.graph-edge.graph-edge-active {
  opacity: 1;
  stroke-width: 4.2;
}
.svg-badge { font-size: 10px; font-weight: 800; fill: #b8443e; }
.svg-badge.selected { fill: #2f6fcf; }
.graph-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin: 8px 0;
  color: var(--muted);
  font-size: 12px;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.legend-swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 3px;
}
.evidence-relationships {
  margin-top: 12px;
}
.evidence-relationships .table-wrap {
  background: #fff;
}
tr.relationship-active {
  background: #fff7d8;
  outline: 2px solid #d4a626;
  outline-offset: -2px;
}
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: left; vertical-align: top; }
th { color: #3f4b5f; background: #f8fafc; font-weight: 720; position: sticky; top: 0; z-index: 1; }
.table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
tr.common { background: #f6f9ff; }
tr.left-only { background: #fffaf0; }
tr.right-only { background: #f1fbf9; }
.text-cell { min-width: 320px; max-width: 640px; }
.span {
  border-left: 3px solid #d4a626;
  background: #fff9df;
  border-radius: 4px;
  padding: 4px 7px;
  font-size: 12px;
}
mark { background: #fff0a8; color: inherit; padding: 0 2px; border-radius: 2px; }
.raw { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }
@media (max-width: 900px) {
  header { position: static; padding: 18px; }
  main { padding: 18px; }
  .overview-grid, .two-col, .graph-switcher, .graph-detail { grid-template-columns: 1fr; }
}
"""


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
