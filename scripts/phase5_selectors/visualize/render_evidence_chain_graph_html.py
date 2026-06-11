#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from fact_checking.selectors.evidence_chain_graph import GRAPH_VERSION
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    GRAPH_VERSION = "evidence_chain_graph_v0_6b"
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


DEFAULT_CHAIN_GRAPH = "outputs/selectors/evidence_chain_graph/v0_6a_val/chain_graph_val.jsonl"
DEFAULT_TRANSLATION_BASE_URL = "https://api.deepseek.com"
DEFAULT_TRANSLATION_MODEL = "deepseek-v4-flash"
SVG_TITLE_LINES = 2
SVG_TITLE_ZH_LINES = 5
SVG_SUBTITLE_LINES = 4
SVG_SUBTITLE_ZH_LINES = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one v0.6a evidence-chain graph row as standalone HTML.")
    parser.add_argument("--chain-graph", default=DEFAULT_CHAIN_GRAPH)
    parser.add_argument("--event-id", default="", help="Event id to render. Accepts both 10004 and 10004.json.")
    parser.add_argument("--claim-contains", default="")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-chains", type=int, default=8)
    parser.add_argument("--output", default="")
    parser.add_argument("--translate-zh", action="store_true", help="Call DeepSeek-compatible API and embed Chinese translations.")
    parser.add_argument("--translation-cache", default="", help="Optional translation cache JSON path. Defaults beside the HTML.")
    parser.add_argument("--force-translate", action="store_true", help="Ignore existing cached translations and call the API again.")
    parser.add_argument("--translation-base-url", default=os.environ.get("TRANSLATION_BASE_URL", os.environ.get("TEACHER_BASE_URL", DEFAULT_TRANSLATION_BASE_URL)))
    parser.add_argument("--translation-model", default=os.environ.get("TRANSLATION_MODEL", os.environ.get("TEACHER_MODEL", DEFAULT_TRANSLATION_MODEL)))
    parser.add_argument("--translation-api-key-env", default=os.environ.get("TRANSLATION_API_KEY_ENV", os.environ.get("TEACHER_API_KEY_ENV", "DEEPSEEK_API_KEY")))
    parser.add_argument("--translation-timeout", type=float, default=120.0)
    parser.add_argument("--translation-max-tokens", type=int, default=4096)
    parser.add_argument("--translation-batch-chars", type=int, default=7000)
    parser.add_argument("--translation-max-retries", type=int, default=3)
    parser.add_argument("--translation-retry-base-sleep", type=float, default=2.0)
    parser.add_argument("--translation-thinking-type", default=os.environ.get("THINKING_TYPE", "disabled"), choices=["disabled", "enabled", "none"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.chain_graph)
    row = find_graph_row(rows, event_id=args.event_id, claim_contains=args.claim_contains)
    output_path = Path(args.output) if args.output else default_output_path(args, row)
    translations = load_or_build_translations(row, args=args, output_path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(row, args=args, translations=translations), encoding="utf-8")
    print(f"Wrote evidence-chain HTML: {output_path}")


def find_graph_row(rows: list[dict[str, Any]], *, event_id: str, claim_contains: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("No chain-graph rows loaded.")
    if event_id:
        wanted = canonical_event_id(event_id)
        for row in rows:
            if canonical_event_id(str(row.get("event_id") or "")) == wanted:
                return row
        raise ValueError(f"No chain-graph row matched event id: {event_id}")
    if claim_contains:
        needle = claim_contains.strip().lower()
        for row in rows:
            if needle in str(row.get("claim") or "").lower():
                return row
        raise ValueError(f"No chain-graph row matched claim substring: {claim_contains}")
    return rows[0]


def default_output_path(args: argparse.Namespace, row: dict[str, Any]) -> Path:
    graph_path = Path(args.chain_graph)
    event = slug(canonical_event_id(str(row.get("event_id") or "claim")))
    return graph_path.parent / "visualizations" / f"evidence_chain_{event}_v0_6a.html"


def load_or_build_translations(row: dict[str, Any], *, args: argparse.Namespace, output_path: Path) -> dict[str, str]:
    cache_path = Path(args.translation_cache) if args.translation_cache else output_path.with_suffix(".zh.json")
    translations: dict[str, str] = {}
    if cache_path.exists() and not bool(args.force_translate):
        payload = read_json(cache_path)
        translations.update({str(k): str(v) for k, v in (payload.get("translations") or {}).items() if str(v).strip()})
    if not bool(args.translate_zh):
        return translations
    items = collect_translation_items(row, max_candidates=int(args.max_candidates), max_chains=int(args.max_chains))
    missing = {key: text for key, text in items.items() if key not in translations}
    if not missing:
        return translations
    api_key = os.environ.get(str(args.translation_api_key_env) or "")
    if not api_key:
        raise RuntimeError(f"--translate-zh requires API key env {args.translation_api_key_env}. Set it before rendering translated HTML.")
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


def collect_translation_items(row: dict[str, Any], *, max_candidates: int, max_chains: int) -> dict[str, str]:
    items: dict[str, str] = {}
    map_html.add_translation_item(items, "claim", row.get("claim"))
    for atom in row.get("atom_nodes") or []:
        atom_id = str(atom.get("atom_id") or atom.get("node_id") or "")
        if atom_id:
            map_html.add_translation_item(items, f"atom:{atom_id}:text", atom.get("text"))
    for evidence in list(row.get("evidence_nodes") or [])[: max(max_candidates, 1)]:
        evidence_id = str(evidence.get("evidence_id") or evidence.get("node_id") or "")
        if not evidence_id:
            continue
        map_html.add_translation_item(items, f"evidence:{evidence_id}:text", evidence.get("text"))
        for idx, span in enumerate(evidence.get("key_spans") or []):
            map_html.add_translation_item(items, f"evidence:{evidence_id}:span:{idx}", span)
    for chain in list(row.get("chains") or [])[: max(max_chains, 1)]:
        chain_id = str(chain.get("chain_id") or "")
        if chain_id:
            map_html.add_translation_item(items, f"chain:{chain_id}:summary", chain_summary_text(chain))
    return items


def render_html(row: dict[str, Any], *, args: argparse.Namespace, translations: dict[str, str]) -> str:
    event_id = str(row.get("event_id") or "")
    title = f"Evidence chain graph: {event_id}"
    selected_ids = {str(eid) for eid in row.get("selected_evidence_ids") or []}
    evidence_nodes = list(row.get("evidence_nodes") or [])[: max(int(args.max_candidates), 1)]
    chain_rows = list(row.get("chains") or [])[: max(int(args.max_chains), 1)]
    graph_html = render_graph(row, evidence_nodes=evidence_nodes, selected_ids=selected_ids, translations=translations)
    chains_html = render_chains(chain_rows, selected_chain_id=str(row.get("selected_chain_id") or ""), translations=translations)
    candidates_html = render_candidates(row, evidence_nodes, selected_ids=selected_ids, translations=translations)
    edges_html = render_edges(row, displayed_evidence_ids={str(node.get("node_id") or "") for node in evidence_nodes})
    filters_html = render_filters(row, evidence_nodes)
    translation_ui = render_translation_toolbar(translations)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "chain_graph": args.chain_graph,
        "event_id": event_id,
        "graph_version": str(row.get("graph_version") or GRAPH_VERSION),
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
    <div class="meta">graph_version={esc(str(row.get("graph_version") or ""))} | selector={esc(str(row.get("selector_name") or ""))} | gold_label={esc(str(row.get("gold_label") or ""))}</div>
    <p class="claim">{trans_html("claim", row.get("claim"), translations)}</p>
    {translation_ui}
  </header>
  <main>
    <section class="section">
      <h2>Selected Chain</h2>
      {render_selected_summary(row, translations=translations)}
    </section>
    <section class="section">
      <h2>Chain Graph</h2>
      {filters_html}
      {graph_html}
    </section>
    <div class="grid">
      <section class="section">
        <h2>Chains</h2>
        {chains_html}
      </section>
      <section class="section">
        <h2>Candidates</h2>
        {candidates_html}
      </section>
    </div>
    <section class="section">
      <h2>Edges</h2>
      {edges_html}
    </section>
    <section class="section">
      <h2>Render Metadata</h2>
      <pre class="raw">{esc(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>
    </section>
  </main>
  <script>{filter_script()}</script>
  <script>{graph_interaction_script()}</script>
  <script>{translation_script(translations)}</script>
</body>
</html>
"""


def render_selected_summary(row: dict[str, Any], *, translations: dict[str, str]) -> str:
    selected_id = str(row.get("selected_chain_id") or "")
    chain = next((item for item in row.get("chains") or [] if str(item.get("chain_id") or "") == selected_id), {})
    if not chain:
        return '<div class="small">No selected chain.</div>'
    return (
        '<div class="summary-grid">'
        f'<div class="metric"><b>chain</b><span>{esc(selected_id)}</span></div>'
        f'<div class="metric"><b>score</b><span>{fmt(chain.get("chain_score"))}</span></div>'
        f'<div class="metric"><b>coverage</b><span>{fmt(chain.get("weighted_atom_coverage"))}</span></div>'
        f'<div class="metric"><b>direct rate</b><span>{fmt(chain.get("direct_or_partial_rate"))}</span></div>'
        f'<div class="metric wide"><b>summary</b><span>{trans_html(f"chain:{selected_id}:summary", chain_summary_text(chain), translations)}</span></div>'
        "</div>"
    )


def render_chains(chains: list[dict[str, Any]], *, selected_chain_id: str, translations: dict[str, str]) -> str:
    rows = []
    for chain in chains:
        chain_id = str(chain.get("chain_id") or "")
        css_class = "selected-row" if chain_id == selected_chain_id else ""
        rows.append(
            f"<tr class='{css_class}'>"
            f"<td>{esc(chain_id)}</td>"
            f"<td>{fmt(chain.get('chain_score'))}</td>"
            f"<td>{fmt(chain.get('weighted_atom_coverage'))}</td>"
            f"<td>{fmt(chain.get('direct_or_partial_rate'))}</td>"
            f"<td>{fmt(chain.get('positive_pair_edge_density'))}</td>"
            f"<td>{esc(', '.join(str(eid) for eid in chain.get('evidence_ids') or []))}</td>"
            f"<td class='text-cell'>{trans_html(f'chain:{chain_id}:summary', chain_summary_text(chain), translations)}</td>"
            "</tr>"
        )
    return table(["chain", "score", "coverage", "direct", "edge density", "evidence", "summary"], rows)


def render_candidates(row: dict[str, Any], evidence_nodes: list[dict[str, Any]], *, selected_ids: set[str], translations: dict[str, str]) -> str:
    rows = []
    for node in evidence_nodes:
        evidence_id = str(node.get("evidence_id") or node.get("node_id") or "")
        selected = evidence_id in selected_ids
        oracle = bool(node.get("oracle_selected"))
        oracle_label = oracle_badge_label(node, row=row)
        atoms = "|".join(str(atom_id) for atom_id in node.get("covered_atom_ids") or [])
        row_class = "candidate-row selected-row" if selected else "candidate-row"
        rows.append(
            f"<tr class='{row_class}' data-selected='{yn(selected)}' data-oracle='{yn(oracle)}' "
            f"data-relation='{esc(str(node.get('relation') or ''))}' data-directness='{esc(str(node.get('directness') or ''))}' data-atoms='{esc(atoms)}'>"
            f"<td>{badge(evidence_id, 'selected-badge' if selected else '')}{' ' + badge(oracle_label, 'oracle') if oracle else ''}</td>"
            f"<td>{esc(str(node.get('relation') or ''))}/{esc(str(node.get('directness') or ''))}</td>"
            f"<td>{fmt(node.get('base_score'))}</td>"
            f"<td>{esc(', '.join(str(atom_id) for atom_id in node.get('covered_atom_ids') or []))}</td>"
            f"<td>{esc(str(node.get('source_group') or ''))}</td>"
            f"<td>{esc(str(node.get('duplicate_group') or ''))}</td>"
            f"<td class='text-cell'>{trans_html(f'evidence:{evidence_id}:text', node.get('text'), translations)}{render_spans(node, translations)}</td>"
            "</tr>"
        )
    return table(["id", "map", "base", "atoms", "source", "duplicate", "text"], rows)


def render_edges(row: dict[str, Any], *, displayed_evidence_ids: set[str]) -> str:
    atom_ids = {str(atom.get("node_id") or "") for atom in row.get("atom_nodes") or []}
    selected_ids = {str(eid) for eid in row.get("selected_evidence_ids") or []}
    rows = []
    for edge in row.get("edges") or []:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source.startswith("E") and source not in displayed_evidence_ids:
            continue
        if target.startswith("E") and target not in displayed_evidence_ids:
            continue
        atom_data = "|".join(str(atom_id) for atom_id in edge.get("atom_ids") or ([] if target not in atom_ids else [target]))
        selected = source in selected_ids and target in selected_ids
        rows.append(
            f"<tr class='edge-row' data-edge-type='{esc(str(edge.get('edge_type') or ''))}' data-atoms='{esc(atom_data)}' "
            f"data-source='{esc(source)}' data-target='{esc(target)}' data-selected='{'1' if selected else '0'}'>"
            f"<td>{badge(edge.get('edge_type'), str(edge.get('edge_type') or ''))}</td>"
            f"<td>{esc(source)} -> {esc(target)}</td>"
            f"<td>{fmt(edge.get('weight'))}</td>"
            f"<td>{esc(', '.join(str(atom_id) for atom_id in edge.get('atom_ids') or []))}</td>"
            f"<td>{esc(str(edge.get('reason') or edge.get('relation') or ''))}</td>"
            "</tr>"
        )
    return table(["type", "nodes", "weight", "atoms", "note"], rows)


def render_graph(row: dict[str, Any], *, evidence_nodes: list[dict[str, Any]], selected_ids: set[str], translations: dict[str, str]) -> str:
    atoms = list(row.get("atom_nodes") or [])
    atom_ids = [str(atom.get("node_id") or "") for atom in atoms]
    selected_order = [str(eid) for eid in row.get("selected_evidence_ids") or [] if str(eid) in {str(node.get("node_id") or "") for node in evidence_nodes}]
    selected_set = set(selected_order)
    selected_nodes = [node for eid in selected_order for node in evidence_nodes if str(node.get("node_id") or "") == eid]
    side_nodes = [node for node in evidence_nodes if str(node.get("node_id") or "") not in selected_set]
    width = 1560
    top = 92
    atom_gap = 18
    selected_gap = 22
    side_gap = 14
    claim_x, claim_y, claim_w = 40, 26, 430
    atom_x, atom_w = 70, 380
    chain_x, chain_w = 540, 520
    side_x, side_w = 1110, 390
    claim_h = node_height("C0 · Claim", None, truncate(row.get("claim"), 120), translations.get("claim"), claim_w)
    atom_heights = {}
    for atom in atoms:
        atom_id = str(atom.get("node_id") or "")
        atom_heights[atom_id] = node_height(
            f"{atom_id} · {truncate(atom.get('text'), 64)}",
            f"{atom_id} · {translations.get(f'atom:{atom_id}:text')}" if translations.get(f"atom:{atom_id}:text") else None,
            f"importance={fmt(atom.get('importance'))} · type={atom.get('atom_type')}",
            None,
            atom_w,
        )
    evidence_heights = {}
    for node in evidence_nodes:
        evidence_id = str(node.get("node_id") or "")
        selected = evidence_id in selected_set
        oracle = bool(node.get("oracle_selected"))
        node_w = chain_w if selected else side_w
        reserve = 96 if selected or oracle else 0
        evidence_heights[evidence_id] = node_height(
            f"{evidence_id} · {node.get('relation')}/{node.get('directness')}",
            None,
            truncate(node.get("text"), 150 if selected else 110),
            translations.get(f"evidence:{evidence_id}:text"),
            node_w,
            reserve_right=reserve,
        )
    atom_top = _stack_tops(atom_ids, heights=atom_heights, top=top + 34, gap=atom_gap)
    selected_top = _stack_tops(selected_order, heights=evidence_heights, top=top + 14, gap=selected_gap)
    side_ids = [str(node.get("node_id") or "") for node in side_nodes]
    side_top = _stack_tops(side_ids, heights=evidence_heights, top=top, gap=side_gap)
    content_bottom = max(
        [claim_y + claim_h]
        + [atom_top.get(atom_id, top) + atom_heights.get(atom_id, 54) for atom_id in atom_ids]
        + [selected_top.get(eid, top) + evidence_heights.get(eid, 64) for eid in selected_order]
        + [side_top.get(eid, top) + evidence_heights.get(eid, 54) for eid in side_ids]
    )
    height = max(720, int(content_bottom + 80))
    evidence_pos: dict[str, tuple[float, float, float, float]] = {}
    for evidence_id, y in selected_top.items():
        evidence_pos[evidence_id] = (chain_x, y, chain_w, evidence_heights.get(evidence_id, 64))
    for evidence_id, y in side_top.items():
        evidence_pos[evidence_id] = (side_x, y, side_w, evidence_heights.get(evidence_id, 54))
    defs = svg_defs()
    edge_paths: list[str] = []
    for atom_id in atom_ids:
        edge_paths.append(svg_edge("claim_has_atom", "C0", atom_id, selected=False, label=f"C0 -> {atom_id}"))
    for edge in row.get("edges") or []:
        edge_type = str(edge.get("edge_type") or "")
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if edge_type == "evidence_covers_atom" and source in evidence_pos and target in atom_top:
            selected = source in selected_ids
            edge_paths.append(svg_edge(edge_type, source, target, selected=selected, label=f"{source} covers {target}"))
        elif source in evidence_pos and target in evidence_pos and show_pair_edge_in_graph(edge_type, source, target, selected_set):
            selected = source in selected_set and target in selected_set and edge_type in {"complements", "corroborates", "tension", "bridge_context"}
            edge_paths.append(svg_edge(edge_type, source, target, selected=selected, label=f"{edge_type} {source}-{target}"))
    for idx, left in enumerate(selected_order[:-1], start=1):
        right = selected_order[idx]
        edge_paths.append(svg_edge("selected_chain_step", left, right, selected=True, label=f"chain step {idx}: {left} -> {right}"))
    atom_nodes = []
    for atom in atoms:
        atom_id = str(atom.get("node_id") or "")
        y = atom_top.get(atom_id, top)
        h = atom_heights.get(atom_id, 54)
        atom_nodes.append(
            svg_node(
                atom_x,
                y,
                atom_w,
                h,
                f"{atom_id} · {truncate(atom.get('text'), 64)}",
                f"importance={fmt(atom.get('importance'))} · type={atom.get('atom_type')}",
                "atom",
                node_id=atom_id,
                title_key=f"atom:{atom_id}:text",
                title_zh=f"{atom_id} · {translations.get(f'atom:{atom_id}:text')}" if translations.get(f"atom:{atom_id}:text") else None,
            )
        )
    evidence_svg = []
    for node in evidence_nodes:
        evidence_id = str(node.get("node_id") or "")
        if evidence_id not in evidence_pos:
            continue
        x, y, w, h = evidence_pos[evidence_id]
        selected = evidence_id in selected_ids
        oracle = bool(node.get("oracle_selected"))
        css = "evidence selected" if selected else "evidence side"
        label = f"{evidence_id} · {node.get('relation')}/{node.get('directness')}"
        sub = truncate(node.get("text"), 150 if selected else 110)
        text_key = f"evidence:{evidence_id}:text"
        sub_zh = translations.get(text_key)
        badges = ""
        if oracle:
            badges += f'<text x="{w - 74}" y="16" class="svg-badge oracle">{esc(oracle_badge_label(node, row=row))}</text>'
        if selected:
            rank = selected_order.index(evidence_id) + 1 if evidence_id in selected_order else ""
            badges += f'<text x="{w - 44}" y="34" class="svg-badge selected">TOP {esc(rank)}</text>'
        evidence_svg.append(
            svg_node(
                x,
                y,
                w,
                h,
                label,
                sub,
                css,
                extra=badges,
                node_id=evidence_id,
                subtitle_key=text_key,
                subtitle_zh=sub_zh,
                reserve_right=96 if selected or oracle else 0,
                data_attrs=(
                    f'data-graph-node="evidence" data-graph-evidence-id="{esc(evidence_id)}" '
                    f'role="button" tabindex="0" data-selected="{"1" if selected else "0"}" '
                    f'data-relation="{esc(str(node.get("relation") or ""))}" '
                    f'data-directness="{esc(str(node.get("directness") or ""))}" '
                    f'data-covered-atoms="{esc("|".join(str(atom_id) for atom_id in node.get("covered_atom_ids") or []))}" '
                    f'data-source-group="{esc(str(node.get("source_group") or ""))}" '
                    f'data-role="{esc(str(node.get("role") or node.get("evidence_role") or ""))}" '
                    f'data-graph-title="{esc(label)}" '
                    f'data-graph-text="{esc(str(node.get("text") or ""))}"'
                ),
            )
        )
    claim_node = svg_node(
        claim_x,
        claim_y,
        claim_w,
        claim_h,
        "C0 · Claim",
        truncate(row.get("claim"), 54),
        "claim",
        node_id="C0",
        subtitle_key="claim",
        subtitle_zh=translations.get("claim"),
    )
    return f"""
<div class="graph-wrap">
  {render_graph_legend()}
  <div class="graph-help small">Drag nodes to inspect the chain. Blue thick edges are the selected chain order; colored evidence-evidence edges show relation structure around the chain.</div>
  <svg id="chainGraphSvg" class="graph-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Evidence chain graph">
    {defs}
    <text x="{atom_x}" y="{top - 18}" class="graph-title">claim atoms</text>
    <text x="{chain_x}" y="{top - 18}" class="graph-title">selected evidence chain</text>
    <text x="{side_x}" y="{top - 18}" class="graph-title">other candidates</text>
    {claim_node}
    {''.join(edge_paths)}
    {''.join(atom_nodes)}
    {''.join(evidence_svg)}
  </svg>
</div>
"""


def render_filters(row: dict[str, Any], evidence_nodes: list[dict[str, Any]]) -> str:
    relations = sorted({str(node.get("relation") or "") for node in evidence_nodes if node.get("relation")})
    directness = sorted({str(node.get("directness") or "") for node in evidence_nodes if node.get("directness")})
    edge_types = sorted({str(edge.get("edge_type") or "") for edge in row.get("edges") or [] if edge.get("edge_type")})
    atoms = [str(atom.get("node_id") or "") for atom in row.get("atom_nodes") or [] if atom.get("node_id")]
    return (
        '<div class="filters">'
        + select_filter("selected", "Selected", ["yes", "no"])
        + select_filter("oracle", "Oracle", ["yes", "no"])
        + select_filter("relation", "Relation", relations)
        + select_filter("directness", "Directness", directness)
        + select_filter("edge-type", "Edge Type", edge_types)
        + select_filter("atom", "Atom", atoms)
        + '<button type="button" id="resetFilters">Reset</button><span id="filterStatus" class="small"></span>'
        + "</div>"
    )


def render_spans(node: dict[str, Any], translations: dict[str, str]) -> str:
    spans = []
    evidence_id = str(node.get("evidence_id") or node.get("node_id") or "")
    for idx, span in enumerate(node.get("key_spans") or []):
        spans.append(f'<span class="span">{trans_html(f"evidence:{evidence_id}:span:{idx}", span, translations)}</span>')
    return f'<div class="span-list">{"".join(spans)}</div>' if spans else ""


def table(headers: list[str], rows: list[str]) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = "".join(rows) if rows else f"<tr><td colspan='{len(headers)}' class='small'>No rows.</td></tr>"
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def chain_summary_text(chain: dict[str, Any]) -> str:
    evidence_ids = ", ".join(str(eid) for eid in chain.get("evidence_ids") or [])
    atoms = ", ".join(str(atom_id) for atom_id in chain.get("covered_atom_ids") or [])
    return (
        f"Chain {chain.get('chain_id') or ''} selects evidence {evidence_ids}; "
        f"covers atoms {atoms or 'none'}; direct rate {fmt(chain.get('direct_or_partial_rate'))}; "
        f"positive edge density {fmt(chain.get('positive_pair_edge_density'))}."
    )


def svg_defs() -> str:
    return """
<defs>
  <marker id="arrow-default" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 Z" fill="#6d7786" />
  </marker>
</defs>
"""


def svg_edge(edge_type: str, source: str, target: str, *, selected: bool, label: str = "") -> str:
    return (
        f'<path class="graph-edge {esc(edge_type)}" data-source="{esc(source)}" data-target="{esc(target)}" '
        f'data-edge-type="{esc(edge_type)}" data-selected="{"1" if selected else "0"}" stroke="{edge_color(edge_type)}" stroke-width="{"3.4" if selected else "1.4"}" '
        f'opacity="{"0.88" if selected else "0.30"}" marker-end="url(#arrow-default)">'
        f"<title>{esc(label or edge_type)}</title></path>"
    )


def svg_path(edge_type: str, x1: float, y1: float, x2: float, y2: float, *, selected: bool, label: str = "") -> str:
    color = edge_color(edge_type)
    d = f"M {x1:.1f} {y1:.1f} C {(x1 + x2) / 2:.1f} {y1:.1f}, {(x1 + x2) / 2:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}"
    return (
        f'<path class="graph-edge {esc(edge_type)}" d="{d}" stroke="{color}" stroke-width="{"3.0" if selected else "1.3"}" '
        f'opacity="{"0.82" if selected else "0.28"}" marker-end="url(#arrow-default)"><title>{esc(label or edge_type)}</title></path>'
    )


def svg_node(
    x: float,
    y: float,
    width: float,
    height: float,
    title: Any,
    subtitle: Any,
    css_class: str,
    *,
    extra: str = "",
    node_id: str = "",
    title_key: str = "",
    title_zh: str | None = None,
    subtitle_key: str = "",
    subtitle_zh: str | None = None,
    reserve_right: int = 0,
    data_attrs: str = "",
) -> str:
    text_width = max(int(width) - 24 - int(reserve_right), 90)
    title_lines = wrap_text_for_svg(str(title or ""), svg_line_units(text_width, kind="title", zh=False), max_lines=SVG_TITLE_LINES)
    title_zh_lines = (
        wrap_text_for_svg(str(title_zh or ""), svg_line_units(text_width, kind="title", zh=True), max_lines=SVG_TITLE_ZH_LINES)
        if title_zh
        else []
    )
    title_block_lines = max(len(title_lines), len(title_zh_lines), 1)
    subtitle_y = 16 + title_block_lines * 15 + 2
    subtitle_lines = wrap_text_for_svg(str(subtitle or ""), svg_line_units(text_width, kind="subtitle", zh=False), max_lines=SVG_SUBTITLE_LINES)
    subtitle_zh_lines = (
        wrap_text_for_svg(str(subtitle_zh or ""), svg_line_units(text_width, kind="subtitle", zh=True), max_lines=SVG_SUBTITLE_ZH_LINES)
        if subtitle_zh
        else []
    )
    title_html = svg_i18n_text(12, 17, "graph-title", title_lines, key=title_key, zh_lines=title_zh_lines)
    subtitle_html = svg_i18n_text(12, subtitle_y, "graph-subtitle", subtitle_lines, key=subtitle_key, zh_lines=subtitle_zh_lines)
    return f"""
<g class="graph-node {esc(css_class)} draggable" data-node-id="{esc(node_id)}" {data_attrs} data-x="{x:.1f}" data-y="{y:.1f}" data-w="{width:.1f}" data-h="{height:.1f}" transform="translate({x:.1f},{y:.1f})">
  <rect width="{width:.1f}" height="{height:.1f}" rx="7" />
  {title_html}
  {subtitle_html}
  {extra}
</g>"""


def render_graph_legend() -> str:
    items = ["selected_chain_step", "evidence_covers_atom", "complements", "corroborates", "tension", "bridge_context", "duplicate", "same_source_context"]
    return '<div class="legend">' + "".join(f'<span><i style="background:{edge_color(item)}"></i>{esc(item)}</span>' for item in items) + "</div>"


def oracle_badge_label(node: dict[str, Any], *, row: dict[str, Any] | None = None) -> str:
    idx = oracle_display_index(node, row=row)
    return f"ORACLE {idx}" if idx is not None else "ORACLE"


def oracle_display_index(node: dict[str, Any], *, row: dict[str, Any] | None = None) -> int | None:
    if not bool(node.get("oracle_selected")):
        return None
    for value in (node.get("oracle_step"), (node.get("candidate") or {}).get("oracle_step")):
        parsed = parse_nonnegative_int(value)
        if parsed is not None:
            return parsed + 1
    if not row:
        return None
    candidates = {
        norm_text_for_match(node.get("candidate_key")),
        norm_text_for_match(node.get("text")),
        norm_text_for_match((node.get("candidate") or {}).get("candidate_key")),
        norm_text_for_match((node.get("candidate") or {}).get("text")),
    }
    candidates.discard("")
    for idx, key in enumerate(row.get("oracle_ordered_keys") or [], start=1):
        if norm_text_for_match(key) in candidates:
            return idx
    return None


def parse_nonnegative_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def norm_text_for_match(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def svg_i18n_text(x: int, y: int, css_class: str, original_lines: list[str], *, key: str = "", zh_lines: list[str] | None = None) -> str:
    original = svg_multiline_text(x, y, css_class, original_lines)
    if not zh_lines:
        return original
    return (
        f'<g class="i18n-original" data-i18n-key="{esc(key)}">{original}</g>'
        f'<g class="i18n-zh" data-i18n-key="{esc(key)}">{svg_multiline_text(x, y, css_class, zh_lines)}</g>'
    )


def svg_multiline_text(x: int, y: int, css_class: str, lines: list[str]) -> str:
    safe_lines = lines or [""]
    tspans = []
    for idx, line in enumerate(safe_lines):
        dy = 0 if idx == 0 else 15
        tspans.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    return f'<text x="{x}" y="{y}" class="{esc(css_class)}">{"".join(tspans)}</text>'


def show_pair_edge_in_graph(edge_type: str, source: str, target: str, selected_set: set[str]) -> bool:
    if source in selected_set and target in selected_set:
        return edge_type != "same_source_context" or len(selected_set) <= 6
    if source in selected_set or target in selected_set:
        return edge_type in {"tension", "bridge_context", "duplicate"}
    return False


def node_height(
    title: Any,
    title_zh: str | None,
    subtitle: Any,
    subtitle_zh: str | None,
    width: float,
    *,
    reserve_right: int = 0,
) -> int:
    text_width = max(int(width) - 24 - int(reserve_right), 90)
    title_lines = wrap_text_for_svg(str(title or ""), svg_line_units(text_width, kind="title", zh=False), max_lines=SVG_TITLE_LINES)
    title_zh_lines = (
        wrap_text_for_svg(str(title_zh or ""), svg_line_units(text_width, kind="title", zh=True), max_lines=SVG_TITLE_ZH_LINES)
        if title_zh
        else []
    )
    subtitle_lines = wrap_text_for_svg(str(subtitle or ""), svg_line_units(text_width, kind="subtitle", zh=False), max_lines=SVG_SUBTITLE_LINES)
    subtitle_zh_lines = (
        wrap_text_for_svg(str(subtitle_zh or ""), svg_line_units(text_width, kind="subtitle", zh=True), max_lines=SVG_SUBTITLE_ZH_LINES)
        if subtitle_zh
        else []
    )
    title_count = max(len(title_lines), len(title_zh_lines), 1)
    subtitle_count = max(len(subtitle_lines), len(subtitle_zh_lines), 1)
    return int(max(52, 16 + title_count * 15 + 3 + subtitle_count * 15 + 12))


def svg_line_units(text_width: int, *, kind: str, zh: bool) -> int:
    if kind == "title":
        px_per_unit = 7.0 if not zh else 7.4
        minimum = 10
    else:
        px_per_unit = 6.6 if not zh else 7.0
        minimum = 12
    return max(int(text_width / px_per_unit), minimum)


def wrap_text_for_svg(value: str, max_units: int, *, max_lines: int) -> list[str]:
    text = " ".join(str(value or "").split())
    if not text:
        return [""]
    limit = max(int(max_units), 6)
    if " " in text and visual_width(text) > limit:
        lines: list[str] = []
        current = ""
        for word in text.split():
            parts = split_token_for_svg(word, limit) if visual_width(word) > limit else [word]
            for part in parts:
                candidate = part if not current else f"{current} {part}"
                if visual_width(candidate) <= limit:
                    current = candidate
                    continue
                if current:
                    lines.append(current)
                current = part
        if current:
            lines.append(current)
    else:
        lines = []
        current = ""
        current_width = 0.0
        for char in text:
            width = char_width(char)
            if current and current_width + width > limit:
                lines.append(current)
                current = char
                current_width = width
            else:
                current += char
                current_width += width
        if current:
            lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = trim_to_units(lines[-1], max(limit - 1, 5)) + "..."
    return lines or [""]


def split_token_for_svg(value: str, limit: int) -> list[str]:
    parts: list[str] = []
    current = ""
    current_width = 0.0
    for char in str(value or ""):
        width = char_width(char)
        if current and current_width + width > limit:
            parts.append(current)
            current = char
            current_width = width
        else:
            current += char
            current_width += width
    if current:
        parts.append(current)
    return parts or [""]


def visual_width(value: str) -> float:
    return sum(char_width(char) for char in str(value or ""))


def char_width(char: str) -> float:
    return 1.55 if unicodedata.east_asian_width(char) in {"W", "F"} else 1.0


def trim_to_units(value: str, max_units: int) -> str:
    out = ""
    total = 0.0
    for char in str(value or ""):
        width = char_width(char)
        if total + width > max_units:
            break
        out += char
        total += width
    return out


def select_filter(key: str, label: str, options: list[str]) -> str:
    opts = '<option value="">All</option>' + "".join(f'<option value="{esc(option)}">{esc(option)}</option>' for option in options)
    return f'<label>{esc(label)}<select data-filter="{esc(key)}">{opts}</select></label>'


def render_translation_toolbar(translations: dict[str, str]) -> str:
    if not translations:
        return '<div class="translation-toolbar small">No embedded Chinese translation. Re-render with <code>--translate-zh</code> to add one.</div>'
    return '<div class="translation-toolbar"><button type="button" id="showOriginal">English</button><button type="button" id="showZh">中文</button><span class="small">Chinese translation embedded from generation-time cache.</span></div>'


def translation_script(translations: dict[str, str]) -> str:
    if not translations:
        return ""
    return """
document.getElementById("showOriginal")?.addEventListener("click", () => document.body.classList.remove("zh-mode"));
document.getElementById("showZh")?.addEventListener("click", () => document.body.classList.add("zh-mode"));
"""


def graph_interaction_script() -> str:
    return """
function initDraggableChainGraph() {
  const svg = document.getElementById("chainGraphSvg");
  if (!svg) return;
  const nodes = new Map();
  function refreshNodes() {
    nodes.clear();
    svg.querySelectorAll(".graph-node[data-node-id]").forEach(node => {
      const id = node.dataset.nodeId;
      if (!id) return;
      nodes.set(id, node);
    });
  }
  function centerOf(id) {
    const node = nodes.get(id);
    if (!node) return null;
    const x = Number(node.dataset.x || 0);
    const y = Number(node.dataset.y || 0);
    const w = Number(node.dataset.w || 0);
    const h = Number(node.dataset.h || 0);
    return { x: x + w / 2, y: y + h / 2 };
  }
  function updateEdges() {
    refreshNodes();
    svg.querySelectorAll(".graph-edge[data-source][data-target]").forEach(edge => {
      const source = centerOf(edge.dataset.source);
      const target = centerOf(edge.dataset.target);
      if (!source || !target) return;
      const dx = target.x - source.x;
      const curve = Math.max(70, Math.min(230, Math.abs(dx) * 0.45));
      const d = `M ${source.x.toFixed(1)} ${source.y.toFixed(1)} C ${(source.x + curve).toFixed(1)} ${source.y.toFixed(1)}, ${(target.x - curve).toFixed(1)} ${target.y.toFixed(1)}, ${target.x.toFixed(1)} ${target.y.toFixed(1)}`;
      edge.setAttribute("d", d);
    });
  }
  function svgPoint(evt) {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }
  let drag = null;
  svg.querySelectorAll(".graph-node.draggable").forEach(node => {
    node.addEventListener("pointerdown", evt => {
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
    node.addEventListener("pointermove", evt => {
      if (!drag || drag.node !== node) return;
      const point = svgPoint(evt);
      const x = point.x - drag.offsetX;
      const y = point.y - drag.offsetY;
      node.dataset.x = x.toFixed(1);
      node.dataset.y = y.toFixed(1);
      node.setAttribute("transform", `translate(${x.toFixed(1)},${y.toFixed(1)})`);
      updateEdges();
    });
    node.addEventListener("pointerup", evt => {
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
initDraggableChainGraph();
"""


def filter_script() -> str:
    return """
const filters = Array.from(document.querySelectorAll("[data-filter]"));
function activeFilters() {
  const out = {};
  for (const el of filters) out[el.dataset.filter] = el.value || "";
  return out;
}
function matchCommon(row, f) {
  if (f.selected && row.dataset.selected !== f.selected) return false;
  if (f.oracle && row.dataset.oracle !== f.oracle) return false;
  if (f.relation && row.dataset.relation !== f.relation) return false;
  if (f.directness && row.dataset.directness !== f.directness) return false;
  if (f.atom) {
    const atoms = (row.dataset.atoms || "").split("|").filter(Boolean);
    if (!atoms.includes(f.atom)) return false;
  }
  return true;
}
function applyFilters() {
  const f = activeFilters();
  let shownCandidates = 0;
  let shownEdges = 0;
  for (const row of document.querySelectorAll(".candidate-row")) {
    const show = matchCommon(row, f);
    row.classList.toggle("is-hidden", !show);
    if (show) shownCandidates += 1;
  }
  for (const row of document.querySelectorAll(".edge-row")) {
    let show = true;
    if (f["edge-type"] && row.dataset.edgeType !== f["edge-type"]) show = false;
    if (f.atom) {
      const atoms = (row.dataset.atoms || "").split("|").filter(Boolean);
      if (!atoms.includes(f.atom)) show = false;
    }
    row.classList.toggle("is-hidden", !show);
    if (show) shownEdges += 1;
  }
  const status = document.getElementById("filterStatus");
  if (status) status.textContent = `${shownCandidates} candidates · ${shownEdges} edges`;
}
filters.forEach(el => el.addEventListener("change", applyFilters));
document.getElementById("resetFilters")?.addEventListener("click", () => {
  filters.forEach(el => { el.value = ""; });
  applyFilters();
});
applyFilters();
"""


def trans_html(key: str, value: Any, translations: dict[str, str]) -> str:
    original = str(value or "")
    zh = translations.get(key)
    if not zh:
        return esc(original)
    return f'<span class="i18n-original">{esc(original)}</span><span class="i18n-zh">{esc(zh)}</span>'


def badge(value: Any, class_name: str = "") -> str:
    text = str(value or "")
    return f'<span class="badge {esc(class_name)}">{esc(text)}</span>'


def yn(value: bool) -> str:
    return "yes" if value else "no"


def edge_color(edge_type: str) -> str:
    return {
        "selected_chain_step": "#2f6fcf",
        "claim_has_atom": "#6d7786",
        "evidence_covers_atom": "#2f6fcf",
        "complements": "#247a52",
        "corroborates": "#46906b",
        "tension": "#b8443e",
        "bridge_context": "#a5681f",
        "duplicate": "#8a4361",
        "same_source_context": "#69778a",
    }.get(str(edge_type), "#6d7786")


def _stack_tops(ids: list[str], *, heights: dict[str, int], top: int, gap: int) -> dict[str, float]:
    out: dict[str, float] = {}
    cursor = float(top)
    for item in ids:
        out[item] = cursor
        cursor += float(heights.get(item, 54) + gap)
    return out


def canonical_event_id(value: str) -> str:
    text = str(value or "").strip()
    return text if text.endswith(".json") else f"{text}.json"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_") or "item"


def truncate(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(limit - 1, 0)] + "…"


def fmt(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ""
    if parsed != parsed:
        return ""
    return f"{parsed:.4f}"


def esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def css() -> str:
    return """
:root { color-scheme: light; --bg:#f7f8fa; --panel:#fff; --ink:#1c2430; --muted:#617085; --line:#d9e0e8; --blue:#2f6fcf; --green:#247a52; --red:#b8443e; --amber:#a5681f; --chip:#eef2f7; }
* { box-sizing: border-box; }
body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); line-height:1.45; }
header { background:#fff; border-bottom:1px solid var(--line); padding:24px 28px 18px; position:sticky; top:0; z-index:10; }
main { max-width:1500px; margin:0 auto; padding:22px 28px 44px; }
h1 { font-size:22px; margin:0 0 10px; letter-spacing:0; }
h2 { font-size:16px; margin:0 0 12px; letter-spacing:0; }
.claim { max-width:1120px; font-size:16px; margin:8px 0 0; }
.meta,.small { color:var(--muted); font-size:12px; }
.section { margin-bottom:18px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 1px 2px rgba(25,33,46,.08); }
.grid { display:grid; grid-template-columns:minmax(420px,.9fr) minmax(560px,1.1fr); gap:18px; align-items:start; }
.summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.metric { border:1px solid var(--line); border-radius:7px; padding:10px; background:#fbfcfd; }
.metric b { display:block; color:var(--muted); font-size:12px; }
.metric span { display:block; margin-top:4px; font-weight:700; }
.metric.wide { grid-column:span 2; }
.filters { display:flex; flex-wrap:wrap; gap:10px; align-items:end; padding:12px; border:1px solid var(--line); border-radius:8px; background:#fbfcfd; margin-bottom:12px; }
.filters label { display:grid; gap:4px; color:var(--muted); font-size:12px; font-weight:650; }
.filters select,.filters button { min-height:34px; border:1px solid #b8c6d7; border-radius:6px; background:#fff; color:#29435f; padding:5px 8px; font:inherit; font-size:13px; }
.graph-wrap { border:1px solid var(--line); border-radius:8px; background:#fbfcfd; overflow-x:auto; }
.graph-help { padding:8px 12px; border-bottom:1px solid var(--line); background:#fff; }
.graph-svg { display:block; min-width:1500px; width:100%; height:auto; }
.graph-node rect { fill:#fff; stroke:#cbd7e4; stroke-width:1.2; }
.graph-node.claim rect { fill:#f7faff; stroke:#a9c6f7; }
.graph-node.atom rect { fill:#fffdfa; stroke:#e4c98f; }
.graph-node.evidence.selected rect { fill:#edf5ff; stroke:#6ba3f2; stroke-width:2; }
.graph-node.side rect { fill:#ffffff; stroke:#d8e0ea; }
.graph-node.draggable { cursor:grab; touch-action:none; }
.graph-node.dragging { cursor:grabbing; }
.graph-node.dragging rect { stroke:#2f6fcf; stroke-width:2.4; }
.graph-title { font-size:13px; font-weight:750; fill:#273447; }
.graph-subtitle { font-size:11px; fill:#617085; }
.graph-edge { fill:none; }
.graph-edge.selected_chain_step { stroke-width:4.2; opacity:.94; }
.svg-badge { font-size:10px; font-weight:800; fill:#b8443e; }
.svg-badge.selected { fill:#2f6fcf; }
.legend { display:flex; flex-wrap:wrap; gap:10px; padding:10px 12px; border-bottom:1px solid var(--line); }
.legend span { display:inline-flex; align-items:center; gap:5px; color:var(--muted); font-size:12px; }
.legend i { display:inline-block; width:16px; height:3px; border-radius:999px; }
.table-wrap { overflow:auto; border:1px solid var(--line); border-radius:8px; }
table { width:100%; border-collapse:collapse; font-size:12px; background:#fff; }
th,td { border-bottom:1px solid var(--line); padding:8px 9px; vertical-align:top; text-align:left; }
th { position:sticky; top:0; background:#fbfcfd; color:#526177; font-weight:750; z-index:1; }
tr.selected-row { background:#f0f7ff; }
tr.is-hidden { display:none; }
.text-cell { min-width:260px; max-width:560px; }
.badge { display:inline-flex; align-items:center; min-height:21px; border-radius:999px; padding:2px 8px; font-size:11px; font-weight:700; background:var(--chip); color:#354155; white-space:nowrap; }
.badge.selected-badge { background:#e7f0ff; color:var(--blue); }
.badge.oracle { background:#fdebea; color:var(--red); }
.badge.complements,.badge.corrobates { background:#e6f6ee; color:var(--green); }
.badge.tension { background:#fdebea; color:var(--red); }
.badge.bridge_context { background:#fff2d9; color:var(--amber); }
.span-list { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.span { border-left:3px solid #d4a626; background:#fff9df; border-radius:4px; padding:4px 7px; font-size:12px; }
pre.raw { white-space:pre-wrap; overflow:auto; background:#101820; color:#ecf3f9; padding:12px; border-radius:6px; font-size:12px; }
.translation-toolbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:12px; }
.translation-toolbar button { min-height:32px; border:1px solid #b8c6d7; border-radius:6px; background:#fff; color:#29435f; padding:5px 10px; font:inherit; font-size:13px; font-weight:650; cursor:pointer; }
.i18n-zh { display:none; }
body.zh-mode .i18n-original { display:none; }
body.zh-mode .i18n-zh { display:inline; }
body.zh-mode text.i18n-zh { display:block; }
@media (max-width: 900px) { header { position:static; } main { padding:16px; } .grid { grid-template-columns:1fr; } .metric.wide { grid-column:auto; } }
"""


if __name__ == "__main__":
    main()
