#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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


DEFAULT_CANDIDATE_FEATURES = (
    "outputs/selectors/evidence_map_selector/liar_raw_dense_v0_6b_val/"
    "candidate_evidence_map_features_val.jsonl"
)
DEFAULT_LEFT_TRACE = (
    "outputs/selectors/evidence_chain_graph/liar_raw_dense_v0_6c_adaptive5_10_val/"
    "selection_trace_val.jsonl"
)
DEFAULT_RIGHT_TRACE = (
    "outputs/selectors/evidence_chain_graph/liar_raw_dense_v0_7_budgeted_marginal_adaptive3_10_val/"
    "selection_trace_val.jsonl"
)
DEFAULT_RAW_DATA = "data/raw/LIAR-RAW/val.json"
DEFAULT_LEFT_LABEL = "v0.6c RuleStep"
DEFAULT_RIGHT_LABEL = "v0.7 BudgetedMarginal"
DEFAULT_TRANSLATION_BASE_URL = "https://api.deepseek.com"
DEFAULT_TRANSLATION_MODEL = "deepseek-v4-flash"

LIAR_RAW_V07_BUILD_COMMAND = """SPLIT=val \\
INPUT=outputs/selectors/evidence_map_selector/liar_raw_dense_v0_6b_val/candidate_evidence_map_features_val.jsonl \\
OUTPUT_DIR=outputs/selectors/evidence_chain_graph/liar_raw_dense_v0_7_budgeted_marginal_adaptive3_10_val \\
bash scripts/phase5_selectors/run/run_evidence_chain_graph_v0_7.sh"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side evidence-map selector comparison for one claim."
    )
    parser.add_argument("--candidate-features", default=DEFAULT_CANDIDATE_FEATURES)
    parser.add_argument("--left-trace", default=DEFAULT_LEFT_TRACE)
    parser.add_argument("--right-trace", default=DEFAULT_RIGHT_TRACE)
    parser.add_argument("--raw-data", default=DEFAULT_RAW_DATA, help="Optional original split JSON with gold explain/explanation.")
    parser.add_argument("--event-id", default="", help="Event id to render. Accepts both 10004 and 10004.json.")
    parser.add_argument("--claim-contains", default="")
    parser.add_argument("--left-label", default=DEFAULT_LEFT_LABEL)
    parser.add_argument("--right-label", default=DEFAULT_RIGHT_LABEL)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--output", default="")
    parser.add_argument("--translate-zh", action="store_true", help="Call DeepSeek-compatible API and embed Chinese translations.")
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
    feature_rows = read_jsonl(args.candidate_features)
    row = map_html.find_feature_row(feature_rows, event_id=args.event_id, claim_contains=args.claim_contains)
    event_id = str(row.get("event_id") or "")
    left_trace = find_trace_row(args.left_trace, event_id=event_id, role="left")
    right_trace = find_trace_row(args.right_trace, event_id=event_id, role="right")
    raw_row = load_raw_row(args.raw_data, event_id=event_id)
    output_path = Path(args.output) if args.output else default_output_path(args, row)
    translations = load_or_build_translations(
        row,
        raw_row=raw_row,
        left_trace=left_trace,
        right_trace=right_trace,
        args=args,
        output_path=output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(row, raw_row=raw_row, left_trace=left_trace, right_trace=right_trace, args=args, translations=translations),
        encoding="utf-8",
    )
    print(f"Wrote evidence-map selector comparison HTML: {output_path}")


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


def missing_trace_message(path: str, *, role: str) -> str:
    message = f"Missing {role} trace file: {path}"
    if str(path) == DEFAULT_RIGHT_TRACE or "v0_7" in str(path):
        message += "\nGenerate the LIAR-RAW v0.7 trace with:\n\n" + LIAR_RAW_V07_BUILD_COMMAND
    return message


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


def looks_like_sample(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("claim", "explain", "explanation", "event_id", "id"))


def default_output_path(args: argparse.Namespace, row: dict[str, Any]) -> Path:
    event = map_html.slug(map_html.canonical_event_id(str(row.get("event_id") or "claim")))
    left = map_html.slug(str(args.left_label or "left"))
    right = map_html.slug(str(args.right_label or "right"))
    return Path(args.candidate_features).parent / "visualizations" / f"evidence_map_compare_{event}_{left}_vs_{right}.html"


def load_or_build_translations(
    row: dict[str, Any],
    *,
    raw_row: dict[str, Any] | None,
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
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
        left_trace=left_trace,
        right_trace=right_trace,
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
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
    max_candidates: int,
) -> dict[str, str]:
    items = map_html.collect_translation_items(row, trace=left_trace, max_candidates=max_candidates)
    items.update(map_html.collect_translation_items(row, trace=right_trace, max_candidates=max_candidates))
    map_html.add_translation_item(items, "gold_explain", gold_explain_text(row, raw_row=raw_row))
    return items


def render_html(
    row: dict[str, Any],
    *,
    raw_row: dict[str, Any] | None = None,
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
    args: argparse.Namespace,
    translations: dict[str, str],
) -> str:
    event_id = str(row.get("event_id") or "")
    title = f"Evidence map selector comparison: {event_id}"
    candidates = comparison_candidates(row, left_trace, right_trace, max_candidates=int(args.max_candidates))
    atoms = list((row.get("evidence_map") or {}).get("claim_atoms") or row.get("claim_atoms") or [])
    left_label = str(args.left_label or DEFAULT_LEFT_LABEL)
    right_label = str(args.right_label or DEFAULT_RIGHT_LABEL)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "candidate_features": args.candidate_features,
        "left_trace": args.left_trace,
        "right_trace": args.right_trace,
        "raw_data": args.raw_data,
        "event_id": event_id,
        "created_at": created,
    }
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
      gold_label={esc(str(row.get("gold_label") or ""))} |
      left={esc(str(left_trace.get("selector_name") or left_label))} |
      right={esc(str(right_trace.get("selector_name") or right_label))}
    </div>
    <p class="claim">{trans_html("claim", row.get("claim"), translations)}</p>
    <div class="path-grid">
      <div><b>features</b><span>{esc(args.candidate_features)}</span></div>
      <div><b>{esc(left_label)}</b><span>{esc(args.left_trace)}</span></div>
      <div><b>{esc(right_label)}</b><span>{esc(args.right_trace)}</span></div>
    </div>
    {map_html.render_translation_toolbar(translations)}
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
      <h2>Atom Coverage Comparison</h2>
      {render_atom_coverage(atoms, candidates, left_trace, right_trace, left_label=left_label, right_label=right_label)}
    </section>
    <section class="section">
      <h2>Evidence Map Graphs</h2>
      {render_evidence_map_graphs(candidates, atoms=atoms, left_trace=left_trace, right_trace=right_trace, left_label=left_label, right_label=right_label, translations=translations)}
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
  {map_html.translation_toggle_script(translations)}
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
        f'<div class="selector">{esc(str(trace.get("selector_name") or ""))}</div>'
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
        f'<p>{trans_html("gold_explain", explain, translations)}</p>'
        "</article>"
    )


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
            f"<td class='text-cell'>{esc(str(atom.get('text') or ''))}</td>"
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
    left_label: str,
    right_label: str,
    translations: dict[str, str],
) -> str:
    if not candidates or not atoms:
        return '<div class="small">No graphable atom/evidence links available.</div>'
    left_selected = selected_index_for_candidates(left_trace, candidates)
    right_selected = selected_index_for_candidates(right_trace, candidates)
    return f"""
<div class="map-graph-grid">
  <article class="map-graph-panel left">
    <h3>{esc(left_label)}</h3>
    <div class="selector small">{esc(str(left_trace.get("selector_name") or ""))}</div>
    {map_html.render_evidence_graph(candidates, atoms=atoms, selected_index=left_selected, translations=translations)}
  </article>
  <article class="map-graph-panel right">
    <h3>{esc(right_label)}</h3>
    <div class="selector small">{esc(str(right_trace.get("selector_name") or ""))}</div>
    {map_html.render_evidence_graph(candidates, atoms=atoms, selected_index=right_selected, translations=translations)}
  </article>
</div>
"""


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
            body.append(f'<div class="flow-text">{trans_html(text_key, text, translations, original_html=esc(map_html.truncate(text, 260)))}</div>')
        items.append(f'<article class="flow-card">{"".join(body)}</article>')
    return (
        f'<div class="flow-panel {esc(side)}">'
        f"<h3>{esc(label)}</h3>"
        f'<div class="selector small">{esc(str(trace.get("selector_name") or ""))}</div>'
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
            f'<td class="text-cell">{trans_html(text_key, candidate.get("text"), translations, original_html=map_html.highlight_text(str(candidate.get("text") or ""), spans))}{render_spans(candidate, translations)}</td>'
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
    return f'<div class="metric"><b>{esc(name)}</b><span>{fmt(value)}</span></div>'


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
.path-grid span { color: var(--muted); overflow-wrap: anywhere; }
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
.map-graph-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(420px, 1fr));
  gap: 14px;
  align-items: start;
}
.map-graph-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfd;
  padding: 13px;
  min-width: 0;
}
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
}
.metric b { display: block; color: var(--muted); font-size: 12px; font-weight: 620; }
.metric span { display: block; font-size: 16px; font-weight: 720; margin-top: 2px; overflow-wrap: anywhere; }
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
.flow-text { margin-top: 8px; font-size: 13px; white-space: pre-wrap; overflow-wrap: anywhere; }
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
.graph-node rect { fill: #fff; stroke: #bdc7d3; stroke-width: 1.1; }
.graph-node.atom rect { fill: #f6f9fd; }
.graph-node.selected rect { fill: #eef4ff; stroke: #2f6fcf; stroke-width: 1.8; }
.graph-node.evidence.common rect { fill: #eef4ff; stroke: var(--common); stroke-width: 1.8; }
.graph-node.evidence.left-only rect { fill: #fff7e8; stroke: var(--left); stroke-width: 1.8; }
.graph-node.evidence.right-only rect { fill: #e8f8f5; stroke: var(--right); stroke-width: 1.8; }
.graph-node.evidence.unselected rect { fill: #fbfcfd; stroke: #cbd5e1; }
.graph-title { font-size: 12px; font-weight: 730; fill: #253044; }
.graph-subtitle { font-size: 11px; fill: #667386; }
.graph-rank { font-size: 10px; font-weight: 800; fill: #2f6fcf; }
.graph-oracle { font-size: 9px; font-weight: 800; fill: #8a4361; }
.graph-edge { fill: none; }
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
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: left; vertical-align: top; }
th { color: #3f4b5f; background: #f8fafc; font-weight: 720; position: sticky; top: 0; z-index: 1; }
.table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
tr.common { background: #f6f9ff; }
tr.left-only { background: #fffaf0; }
tr.right-only { background: #f1fbf9; }
.text-cell { min-width: 320px; max-width: 640px; overflow-wrap: anywhere; }
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
  .overview-grid, .two-col, .map-graph-grid { grid-template-columns: 1fr; }
}
"""


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
