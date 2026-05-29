#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.utils.io import read_json, read_jsonl


DEFAULT_DIAGNOSTIC_DIR = "outputs/selectors/evidence_map_selector/v0_5c_val_prompt_evidence_diagnostic"
DEFAULT_EVIDENCE_SOURCE = "v0_5a_evidence_map_top5"
DEFAULT_PLAIN_STYLE = "plain_original"
DEFAULT_MAP_STYLE = "map_full"
DEFAULT_CHECKPOINT = "checkpoint-600"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a single-claim plain-prompt vs map-prompt verifier diagnostic HTML."
    )
    parser.add_argument("--diagnostic-dir", default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--evidence-source", default=DEFAULT_EVIDENCE_SOURCE)
    parser.add_argument("--plain-style", default=DEFAULT_PLAIN_STYLE)
    parser.add_argument("--map-style", default=DEFAULT_MAP_STYLE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--event-id", default="", help="Event id to render. Accepts both 10004 and 10004.json.")
    parser.add_argument("--claim-contains", default="", help="Case-insensitive claim substring fallback.")
    parser.add_argument("--output", default="", help="HTML output path. Defaults under diagnostic-dir/visualizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostic_dir = Path(args.diagnostic_dir)
    plain_build_path = build_path(diagnostic_dir, args.evidence_source, args.plain_style, args.split)
    map_build_path = build_path(diagnostic_dir, args.evidence_source, args.map_style, args.split)
    plain_rows = read_jsonl(plain_build_path)
    map_rows = read_jsonl(map_build_path)
    plain_idx, plain_row = find_row(plain_rows, event_id=args.event_id, claim_contains=args.claim_contains)
    map_idx, map_row = matching_row(map_rows, event_id=str(plain_row.get("event_id") or ""))
    plain_pred = load_prediction(
        diagnostic_dir,
        args.evidence_source,
        args.plain_style,
        args.checkpoint,
        args.split,
        sample_idx=plain_idx,
    )
    map_pred = load_prediction(
        diagnostic_dir,
        args.evidence_source,
        args.map_style,
        args.checkpoint,
        args.split,
        sample_idx=map_idx,
    )
    plain_metrics = load_metrics(diagnostic_dir, args.evidence_source, args.plain_style, args.checkpoint)
    map_metrics = load_metrics(diagnostic_dir, args.evidence_source, args.map_style, args.checkpoint)
    delta = load_paired_delta(
        diagnostic_dir,
        event_id=str(plain_row.get("event_id") or ""),
        evidence_source=str(args.evidence_source),
        prompt_style=str(args.map_style),
    )
    output_path = Path(args.output) if args.output else default_output_path(diagnostic_dir, plain_row, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(
            plain_row=plain_row,
            map_row=map_row,
            plain_pred=plain_pred,
            map_pred=map_pred,
            plain_metrics=plain_metrics,
            map_metrics=map_metrics,
            delta=delta,
            args=args,
        ),
        encoding="utf-8",
    )
    print(f"Wrote prompt comparison HTML: {output_path}")


def build_path(diagnostic_dir: Path, evidence_source: str, prompt_style: str, split: str) -> Path:
    return diagnostic_dir / "verifier_data" / slug(evidence_source) / slug(prompt_style) / f"build_{split}.jsonl"


def prediction_path(diagnostic_dir: Path, evidence_source: str, prompt_style: str, checkpoint: str, split: str) -> Path:
    return diagnostic_dir / "eval" / slug(evidence_source) / slug(prompt_style) / slug(checkpoint) / f"{split}_predictions.jsonl"


def metrics_path(diagnostic_dir: Path, evidence_source: str, prompt_style: str, checkpoint: str) -> Path:
    return diagnostic_dir / "eval" / slug(evidence_source) / slug(prompt_style) / slug(checkpoint) / "metrics.json"


def find_row(rows: list[dict[str, Any]], *, event_id: str, claim_contains: str) -> tuple[int, dict[str, Any]]:
    if not rows:
        raise ValueError("No build rows loaded.")
    if event_id:
        wanted = canonical_event_id(event_id)
        for idx, row in enumerate(rows):
            if canonical_event_id(str(row.get("event_id") or "")) == wanted:
                return idx, row
        raise ValueError(f"No build row matched event id: {event_id}")
    if claim_contains:
        needle = claim_contains.lower().strip()
        for idx, row in enumerate(rows):
            if needle in str(row.get("claim") or "").lower():
                return idx, row
        raise ValueError(f"No build row matched claim substring: {claim_contains}")
    return 0, rows[0]


def matching_row(rows: list[dict[str, Any]], *, event_id: str) -> tuple[int, dict[str, Any]]:
    wanted = canonical_event_id(event_id)
    for idx, row in enumerate(rows):
        if canonical_event_id(str(row.get("event_id") or "")) == wanted:
            return idx, row
    raise ValueError(f"No paired map row matched event id: {event_id}")


def load_prediction(
    diagnostic_dir: Path,
    evidence_source: str,
    prompt_style: str,
    checkpoint: str,
    split: str,
    *,
    sample_idx: int,
) -> dict[str, Any]:
    path = prediction_path(diagnostic_dir, evidence_source, prompt_style, checkpoint, split)
    if not path.exists():
        return {}
    for row in read_jsonl(path):
        row_sample_idx = row.get("sample_idx")
        if row_sample_idx is not None and int(row_sample_idx) == int(sample_idx):
            return row
    return {}


def load_metrics(diagnostic_dir: Path, evidence_source: str, prompt_style: str, checkpoint: str) -> dict[str, Any]:
    path = metrics_path(diagnostic_dir, evidence_source, prompt_style, checkpoint)
    return read_json(path) if path.exists() else {}


def load_paired_delta(
    diagnostic_dir: Path,
    *,
    event_id: str,
    evidence_source: str,
    prompt_style: str,
) -> dict[str, Any]:
    path = diagnostic_dir / "analysis" / "paired_prompt_delta_by_event.jsonl"
    if not path.exists():
        return {}
    wanted_event = canonical_event_id(event_id)
    for row in read_jsonl(path):
        if (
            canonical_event_id(str(row.get("event_id") or "")) == wanted_event
            and str(row.get("evidence_source") or "") == evidence_source
            and str(row.get("prompt_style") or "") == prompt_style
        ):
            return row
    return {}


def default_output_path(diagnostic_dir: Path, row: dict[str, Any], args: argparse.Namespace) -> Path:
    event = slug(canonical_event_id(str(row.get("event_id") or "claim")))
    source = slug(str(args.evidence_source))
    checkpoint = slug(str(args.checkpoint))
    return diagnostic_dir / "visualizations" / f"prompt_compare_{event}_{source}_{checkpoint}.html"


def render_html(
    *,
    plain_row: dict[str, Any],
    map_row: dict[str, Any],
    plain_pred: dict[str, Any],
    map_pred: dict[str, Any],
    plain_metrics: dict[str, Any],
    map_metrics: dict[str, Any],
    delta: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    claim = str(plain_row.get("claim") or map_row.get("claim") or "")
    event_id = str(plain_row.get("event_id") or map_row.get("event_id") or "")
    gold = str(plain_row.get("gold_label") or plain_row.get("label") or "")
    title = f"Prompt Comparison: {event_id}"
    outcome_html = render_outcome(plain_row, map_row, plain_pred, map_pred, delta)
    metrics_html = render_metric_pair(plain_metrics, map_metrics, args)
    evidence_html = render_evidence_table(plain_row, map_row)
    prompt_cards = render_prompt_cards(plain_row, map_row, args)
    raw_html = render_raw_payload(plain_row, map_row, plain_pred, map_pred, delta)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #1d2633;
      --muted: #647386;
      --line: #d8e0ea;
      --blue: #2f6fcf;
      --green: #247a52;
      --red: #b8443e;
      --amber: #a5681f;
      --violet: #6d58b8;
      --chip: #eef2f7;
      --shadow: 0 1px 2px rgba(22, 31, 44, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 24px 28px 18px;
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    main {{
      max-width: 1540px;
      margin: 0 auto;
      padding: 22px 28px 44px;
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 16px;
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    h3 {{
      font-size: 14px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    .claim {{
      max-width: 1120px;
      margin: 6px 0 0;
      font-size: 16px;
    }}
    .meta, .small {{
      color: var(--muted);
      font-size: 12px;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
      margin-bottom: 18px;
    }}
    .outcome-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
    }}
    .tile {{
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfd;
      padding: 10px;
      min-height: 64px;
    }}
    .tile .value {{
      font-size: 18px;
      font-weight: 780;
      margin-top: 4px;
    }}
    .good {{ color: var(--green); }}
    .bad {{ color: var(--red); }}
    .neutral {{ color: #44566d; }}
    .prompt-grid {{
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
      gap: 16px;
      align-items: start;
    }}
    .prompt-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }}
    .prompt-head {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }}
    .prompt-body {{
      max-height: 780px;
      overflow: auto;
      background: #111820;
      color: #eaf2f8;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      padding: 12px 14px;
    }}
    .line {{
      display: block;
      min-height: 1.45em;
      border-radius: 3px;
      padding: 0 3px;
    }}
    .line.claim-line {{ background: rgba(47, 111, 207, 0.22); }}
    .line.evidence-line {{ background: rgba(255, 255, 255, 0.045); }}
    .line.map-line {{ background: rgba(109, 88, 184, 0.24); }}
    .line.span-line {{ background: rgba(165, 104, 31, 0.22); }}
    .line.rule-line {{ color: #cbd7e4; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 700;
      background: var(--chip);
      color: #344258;
      white-space: nowrap;
    }}
    .badge.good {{ background: #e6f6ee; color: var(--green); }}
    .badge.bad {{ background: #fdebea; color: var(--red); }}
    .badge.map {{ background: #eeeafd; color: var(--violet); }}
    .badge.plain {{ background: #e7f0ff; color: var(--blue); }}
    .badge.warn {{ background: #fff2d9; color: var(--amber); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      padding: 8px;
    }}
    th {{
      color: var(--muted);
      background: #fbfcfd;
      font-weight: 750;
    }}
    .text-cell {{
      min-width: 250px;
      max-width: 520px;
    }}
    .span {{
      display: inline-block;
      border-left: 3px solid #d4a626;
      background: #fff9df;
      border-radius: 4px;
      padding: 3px 6px;
      margin: 2px 4px 2px 0;
    }}
    pre.raw {{
      white-space: pre-wrap;
      overflow: auto;
      background: #101820;
      color: #ecf3f9;
      padding: 12px;
      border-radius: 6px;
      font-size: 12px;
    }}
    @media (max-width: 960px) {{
      header {{ position: static; }}
      main {{ padding: 16px; }}
      .prompt-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="meta">event_id={esc(event_id)} | gold={esc(gold)} | evidence_source={esc(args.evidence_source)} | checkpoint={esc(args.checkpoint)}</div>
    <h1>{esc(title)}</h1>
    <p class="claim">{esc(claim)}</p>
  </header>
  <main>
    <section class="section">
      <h2>Prediction And Prompt Delta</h2>
      {outcome_html}
    </section>
    <section class="section">
      <h2>Overall Checkpoint Metrics</h2>
      {metrics_html}
    </section>
    <section class="section">
      <h2>Prompt Side By Side</h2>
      {prompt_cards}
    </section>
    <section class="section">
      <h2>Selected Evidence Comparison</h2>
      {evidence_html}
    </section>
    <section class="section">
      <h2>Raw Pair Payload</h2>
      {raw_html}
    </section>
  </main>
</body>
</html>
"""


def render_outcome(
    plain_row: dict[str, Any],
    map_row: dict[str, Any],
    plain_pred: dict[str, Any],
    map_pred: dict[str, Any],
    delta: dict[str, Any],
) -> str:
    gold = str(plain_row.get("gold_label") or plain_pred.get("gold_label") or "")
    plain_label = str(plain_pred.get("pred_label") or "missing")
    map_label = str(map_pred.get("pred_label") or "missing")
    shifted = bool(plain_label and map_label and plain_label != map_label)
    cells = [
        tile("Gold", gold, "neutral"),
        tile("Plain Pred", plain_label, correctness_class(plain_label, gold)),
        tile("Map Pred", map_label, correctness_class(map_label, gold)),
        tile("Label Shift", "yes" if shifted else "no", "warn" if shifted else "neutral"),
        tile("Plain Tokens", plain_row.get("prompt_token_count"), "neutral"),
        tile("Map Tokens", map_row.get("prompt_token_count"), "neutral"),
        tile("Token Delta", signed(delta.get("prompt_token_delta_vs_plain"), int(map_row.get("prompt_token_count") or 0) - int(plain_row.get("prompt_token_count") or 0)), "warn"),
        tile("Evidence Delta", signed(delta.get("evidence_count_delta_vs_plain"), int(map_row.get("evidence_count") or 0) - int(plain_row.get("evidence_count") or 0)), "neutral"),
        tile("Plain Trunc", yes_no(plain_row.get("was_truncated")), "warn" if plain_row.get("was_truncated") else "neutral"),
        tile("Map Trunc", yes_no(map_row.get("was_truncated")), "warn" if map_row.get("was_truncated") else "neutral"),
    ]
    return f'<div class="outcome-grid">{"".join(cells)}</div>'


def render_metric_pair(plain_metrics: dict[str, Any], map_metrics: dict[str, Any], args: argparse.Namespace) -> str:
    fields = ["accuracy", "macro_f1", "true_side_macro_f1", "selection_score", "eval_loss", "parse_error_rate"]
    cells = []
    for field in fields:
        plain = as_float(plain_metrics.get(field))
        mapped = as_float(map_metrics.get(field))
        delta = None if plain is None or mapped is None else mapped - plain
        value = f"{fmt(plain)} → {fmt(mapped)}"
        if delta is not None:
            value += f" ({signed(delta)})"
        cells.append(tile(field, value, "neutral"))
    cells.append(tile("Plain Style", args.plain_style, "plain"))
    cells.append(tile("Map Style", args.map_style, "map"))
    return f'<div class="metric-grid">{"".join(cells)}</div>'


def render_prompt_cards(plain_row: dict[str, Any], map_row: dict[str, Any], args: argparse.Namespace) -> str:
    plain_meta = prompt_meta(plain_row)
    map_meta = prompt_meta(map_row)
    return (
        '<div class="prompt-grid">'
        + render_prompt_card("Plain Prompt", args.plain_style, plain_row.get("prompt"), plain_meta, "plain")
        + render_prompt_card("Map Prompt", args.map_style, map_row.get("prompt"), map_meta, "map")
        + "</div>"
    )


def render_prompt_card(title: str, style: str, prompt: Any, meta: str, badge_class: str) -> str:
    return f"""
<article class="prompt-card">
  <div class="prompt-head">
    <div>
      <h3>{esc(title)}</h3>
      <div class="small">{esc(meta)}</div>
    </div>
    <div><span class="badge {badge_class}">{esc(style)}</span></div>
  </div>
  <div class="prompt-body">{render_prompt_lines(str(prompt or ""))}</div>
</article>
"""


def render_prompt_lines(prompt: str) -> str:
    lines = []
    for raw_line in prompt.splitlines():
        cls = "line"
        stripped = raw_line.strip()
        if stripped in {"Claim:", "Evidence:", "Claim Atoms:", "Selected Evidence Map:"}:
            cls += " claim-line" if stripped == "Claim:" else " map-line" if "Map" in stripped or "Atoms" in stripped else "evidence-line"
        elif re.match(r"^\[\d+\]", stripped):
            cls += " map-line" if "relation=" in stripped else "evidence-line"
        elif stripped.startswith("spans:") or stripped.startswith("spans"):
            cls += " span-line"
        elif stripped.startswith("evidence:"):
            cls += " evidence-line"
        elif stripped.startswith("- Use") or stripped.startswith("- Treat") or stripped.startswith("- Respond") or stripped.startswith("- Do not"):
            cls += " rule-line"
        elif re.match(r"^- A\d+:", stripped):
            cls += " map-line"
        lines.append(f'<span class="{cls}">{esc(raw_line)}</span>')
    return "\n".join(lines)


def prompt_meta(row: dict[str, Any]) -> str:
    return (
        f"tokens={row.get('prompt_token_count')} | evidence={row.get('evidence_count')}/"
        f"{row.get('evidence_count_before')} | truncated={yes_no(row.get('was_truncated'))} | "
        f"text_truncated={yes_no(row.get('evidence_text_truncated'))}"
    )


def render_evidence_table(plain_row: dict[str, Any], map_row: dict[str, Any]) -> str:
    plain_candidates = list(plain_row.get("candidates") or [])
    map_candidates = list(map_row.get("candidates") or [])
    rows = []
    max_len = max(len(plain_candidates), len(map_candidates))
    for idx in range(max_len):
        plain = plain_candidates[idx] if idx < len(plain_candidates) else {}
        mapped = map_candidates[idx] if idx < len(map_candidates) else {}
        candidate = mapped or plain
        spans = "".join(f'<span class="span">{esc(span)}</span>' for span in candidate.get("key_spans") or [])
        rows.append(
            "<tr>"
            f"<td>{idx + 1}</td>"
            f"<td>{esc(candidate.get('evidence_id') or plain.get('evidence_id'))}</td>"
            f"<td>{esc(candidate.get('map_relation'))}</td>"
            f"<td>{esc(candidate.get('map_directness'))}</td>"
            f"<td>{esc(', '.join(str(atom) for atom in candidate.get('covered_atom_ids') or []))}</td>"
            f"<td>{fmt(candidate.get('evidence_map_quality_score'))}</td>"
            f"<td>{esc(yes_no(candidate.get('oracle_selected')))}</td>"
            f'<td class="text-cell">{esc(plain.get("text") or "")}</td>'
            f'<td class="text-cell">{spans}<div>{esc(mapped.get("text") or "")}</div></td>'
            "</tr>"
        )
    if not rows:
        return '<div class="small">No candidates available.</div>'
    return (
        "<table>"
        "<thead><tr><th>#</th><th>EID</th><th>relation</th><th>directness</th><th>atoms</th><th>quality</th><th>oracle</th><th>plain evidence text</th><th>map evidence + spans</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_raw_payload(
    plain_row: dict[str, Any],
    map_row: dict[str, Any],
    plain_pred: dict[str, Any],
    map_pred: dict[str, Any],
    delta: dict[str, Any],
) -> str:
    payload = {
        "plain": compact_row(plain_row, plain_pred),
        "map": compact_row(map_row, map_pred),
        "paired_delta": delta,
    }
    return f'<pre class="raw">{esc(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>'


def compact_row(row: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("selector_trace") or {}
    return {
        "event_id": row.get("event_id"),
        "prompt_style": row.get("prompt_style"),
        "prompt_token_count": row.get("prompt_token_count"),
        "evidence_count": row.get("evidence_count"),
        "evidence_count_before": row.get("evidence_count_before"),
        "was_truncated": row.get("was_truncated"),
        "map_annotation_status": trace.get("map_annotation_status"),
        "weighted_atom_coverage@5": trace.get("weighted_atom_coverage@5"),
        "direct_or_partial_map_rate@5": trace.get("direct_or_partial_map_rate@5"),
        "pred_label": pred.get("pred_label"),
        "raw_output": pred.get("raw_output"),
        "gold_label": row.get("gold_label"),
    }


def tile(label: str, value: Any, tone: str) -> str:
    return f'<div class="tile"><div class="small">{esc(label)}</div><div class="value {esc(tone)}">{esc(value)}</div></div>'


def correctness_class(pred: str, gold: str) -> str:
    if not pred or pred == "missing":
        return "neutral"
    return "good" if pred == gold else "bad"


def canonical_event_id(value: str) -> str:
    text = str(value).strip()
    return text[:-5] if text.endswith(".json") else text


def slug(value: str) -> str:
    out = str(value or "").replace("/", "_").replace(" ", "_").replace(",", "").replace(":", "_").replace(".", "_")
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", out)
    return out.strip("._") or "item"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def signed(value: Any, fallback: Any | None = None) -> str:
    raw = value if value is not None else fallback
    number = as_float(raw)
    if number is None:
        return str(raw or "")
    sign = "+" if number > 0 else ""
    return sign + fmt(number)


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return str(value or "")
    if number.is_integer() and abs(number) < 10000:
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


if __name__ == "__main__":
    main()
