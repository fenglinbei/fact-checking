#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
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

import render_prompt_comparison_html as prompt_compare


DEFAULT_OUTPUT_DIR = "outputs/selectors/evidence_map_selector/v0_5a_val"
DEFAULT_SELECTOR = "v0_5a_evidence_map_top5"
DEFAULT_PROMPT_DIAGNOSTIC_DIR = "outputs/selectors/evidence_map_selector/v0_5c_val_prompt_evidence_diagnostic"
DEFAULT_PROMPT_CHECKPOINT = "checkpoint-600"
DEFAULT_TRANSLATION_BASE_URL = "https://api.deepseek.com"
DEFAULT_TRANSLATION_MODEL = "deepseek-v4-flash"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a single claim's evidence-map selector artifact as standalone HTML."
    )
    p.add_argument(
        "--candidate-features",
        default=f"{DEFAULT_OUTPUT_DIR}/candidate_evidence_map_features_val.jsonl",
        help="candidate_evidence_map_features_*.jsonl from postprocess_evidence_maps.py.",
    )
    p.add_argument(
        "--selection-trace",
        default=f"{DEFAULT_OUTPUT_DIR}/selection_trace_val.jsonl",
        help="selection_trace_*.jsonl from eval_evidence_map_selector_v0_5a.py.",
    )
    p.add_argument("--event-id", default="", help="Event id to render. Accepts both 10004 and 10004.json.")
    p.add_argument("--claim-contains", default="", help="Case-insensitive substring fallback when event id is unknown.")
    p.add_argument("--selector-name", default=DEFAULT_SELECTOR)
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--output", default="", help="HTML output path. Defaults beside the features file.")
    p.add_argument("--prompt-comparison", dest="prompt_comparison", action="store_true", default=True)
    p.add_argument("--no-prompt-comparison", dest="prompt_comparison", action="store_false")
    p.add_argument("--prompt-diagnostic-dir", default=DEFAULT_PROMPT_DIAGNOSTIC_DIR)
    p.add_argument("--prompt-evidence-source", default=DEFAULT_SELECTOR)
    p.add_argument("--plain-prompt-style", default="plain_original")
    p.add_argument("--map-prompt-style", default="map_full")
    p.add_argument("--prompt-checkpoint", default=DEFAULT_PROMPT_CHECKPOINT)
    p.add_argument("--prompt-split", default="val")
    p.add_argument("--translate-zh", action="store_true", help="Call DeepSeek-compatible API and embed Chinese translations.")
    p.add_argument("--translation-cache", default="", help="Optional translation cache JSON path. Defaults beside the HTML.")
    p.add_argument("--force-translate", action="store_true", help="Ignore existing cached translations and call the API again.")
    p.add_argument("--translation-base-url", default=os.environ.get("TRANSLATION_BASE_URL", os.environ.get("TEACHER_BASE_URL", DEFAULT_TRANSLATION_BASE_URL)))
    p.add_argument("--translation-model", default=os.environ.get("TRANSLATION_MODEL", os.environ.get("TEACHER_MODEL", DEFAULT_TRANSLATION_MODEL)))
    p.add_argument("--translation-api-key-env", default=os.environ.get("TRANSLATION_API_KEY_ENV", os.environ.get("TEACHER_API_KEY_ENV", "DEEPSEEK_API_KEY")))
    p.add_argument("--translation-timeout", type=float, default=120.0)
    p.add_argument("--translation-max-tokens", type=int, default=4096)
    p.add_argument("--translation-batch-chars", type=int, default=7000)
    p.add_argument("--translation-max-retries", type=int, default=3)
    p.add_argument("--translation-retry-base-sleep", type=float, default=2.0)
    p.add_argument("--translation-thinking-type", default=os.environ.get("THINKING_TYPE", "disabled"), choices=["disabled", "enabled", "none"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feature_rows = read_jsonl(args.candidate_features)
    row = find_feature_row(feature_rows, event_id=args.event_id, claim_contains=args.claim_contains)
    trace = find_trace(args.selection_trace, event_id=str(row.get("event_id") or ""), selector_name=args.selector_name)
    output_path = Path(args.output) if args.output else default_output_path(args, row)
    try:
        translations = load_or_build_translations(row, trace=trace, args=args, output_path=output_path)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(row, trace=trace, args=args, translations=translations), encoding="utf-8")
    print(f"Wrote evidence-map HTML: {output_path}")


def find_feature_row(rows: list[dict[str, Any]], *, event_id: str, claim_contains: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("No feature rows loaded.")
    if event_id:
        wanted = canonical_event_id(event_id)
        for row in rows:
            if canonical_event_id(str(row.get("event_id") or "")) == wanted:
                return row
        raise ValueError(f"No feature row matched event id: {event_id}")
    if claim_contains:
        needle = claim_contains.strip().lower()
        for row in rows:
            if needle in str(row.get("claim") or "").lower():
                return row
        raise ValueError(f"No feature row matched claim substring: {claim_contains}")
    return rows[0]


def find_trace(path: str, *, event_id: str, selector_name: str) -> dict[str, Any] | None:
    trace_path = Path(path)
    if not path or not trace_path.exists():
        return None
    wanted_event = canonical_event_id(event_id)
    wanted_selector = str(selector_name or "")
    for trace in read_jsonl(trace_path):
        if canonical_event_id(str(trace.get("event_id") or "")) != wanted_event:
            continue
        if wanted_selector and str(trace.get("selector_name") or "") != wanted_selector:
            continue
        return trace
    return None


def default_output_path(args: argparse.Namespace, row: dict[str, Any]) -> Path:
    features_path = Path(args.candidate_features)
    event = slug(canonical_event_id(str(row.get("event_id") or "claim")))
    selector = slug(str(args.selector_name or "selector"))
    return features_path.parent / "visualizations" / f"evidence_map_{event}_{selector}.html"


def load_or_build_translations(
    row: dict[str, Any],
    *,
    trace: dict[str, Any] | None,
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

    items = collect_translation_items(row, trace=trace, max_candidates=int(args.max_candidates))
    missing = {key: text for key, text in items.items() if key not in translations}
    if not missing:
        return translations

    api_key = os.environ.get(str(args.translation_api_key_env) or "")
    if not api_key:
        raise RuntimeError(
            f"--translate-zh requires API key env {args.translation_api_key_env}. "
            "Set it before rendering translated HTML."
        )
    new_translations, usage_totals = translate_items_zh(missing, args=args, api_key=api_key)
    translations.update(new_translations)
    cache_payload = {
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
    }
    save_json(cache_payload, cache_path)
    print(f"Wrote zh translation cache: {cache_path}")
    return translations


def collect_translation_items(
    row: dict[str, Any],
    *,
    trace: dict[str, Any] | None,
    max_candidates: int,
) -> dict[str, str]:
    items: dict[str, str] = {}
    add_translation_item(items, "claim", row.get("claim"))
    atoms = list((row.get("evidence_map") or {}).get("claim_atoms") or row.get("claim_atoms") or [])
    for atom in atoms:
        atom_id = str(atom.get("atom_id") or "")
        if atom_id:
            add_translation_item(items, atom_translation_key(atom_id), atom.get("text"))

    candidates = list(row.get("candidates") or [])[: max(max_candidates, 1)]
    for candidate in candidates:
        ckey = candidate_translation_base(candidate)
        title = candidate.get("candidate_key") or candidate.get("canonical_text") or candidate.get("text") or ""
        add_translation_item(items, f"{ckey}:title", title)
        add_translation_item(items, f"{ckey}:text", candidate.get("text"))
        for idx, span in enumerate(candidate.get("key_spans") or []):
            add_translation_item(items, f"{ckey}:span:{idx}", span)

    for candidate in (trace or {}).get("selected_candidates") or []:
        ckey = candidate_translation_base(candidate)
        add_translation_item(items, f"{ckey}:text", candidate.get("text"))
    return items


def add_translation_item(items: dict[str, str], key: str, value: Any) -> None:
    text = str(value or "").strip()
    if not key or not text:
        return
    if len(text) < 3 or not re.search(r"[A-Za-z]", text):
        return
    items.setdefault(key, text)


def translate_items_zh(items: dict[str, str], *, args: argparse.Namespace, api_key: str) -> tuple[dict[str, str], dict[str, int]]:
    translations: dict[str, str] = {}
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for batch in batch_translation_items(items, batch_chars=int(args.translation_batch_chars)):
        data = call_translation_api(batch, args=args, api_key=api_key)
        usage = data.get("usage") or {}
        for key in usage_totals:
            usage_totals[key] += int(usage.get(key) or 0)
        parsed = parse_translation_response(response_content(data))
        for key, original in batch.items():
            value = str(parsed.get(key) or "").strip()
            if value and value != original:
                translations[key] = value
    return translations, usage_totals


def batch_translation_items(items: dict[str, str], *, batch_chars: int) -> list[dict[str, str]]:
    batches: list[dict[str, str]] = []
    current: dict[str, str] = {}
    current_chars = 0
    limit = max(int(batch_chars), 1000)
    for key, text in items.items():
        size = len(key) + len(text)
        if current and current_chars + size > limit:
            batches.append(current)
            current = {}
            current_chars = 0
        current[key] = text
        current_chars += size
    if current:
        batches.append(current)
    return batches


def call_translation_api(batch: dict[str, str], *, args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    system_prompt = (
        "You are a precise English-to-Simplified-Chinese translator for a fact-checking evidence-map UI. "
        "Translate faithfully and concisely. Preserve names, numbers, labels like A1/E01, URLs, and factual uncertainty. "
        "Return strictly valid JSON only."
    )
    user_prompt = json.dumps(
        {
            "task": "Translate each text value into Simplified Chinese.",
            "output_schema": {"translations": {"same_id": "Chinese translation"}},
            "items": [{"id": key, "text": text} for key, text in batch.items()],
        },
        ensure_ascii=False,
    )
    payload: dict[str, Any] = {
        "model": str(args.translation_model),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": int(args.translation_max_tokens),
        "user": "evidence_map_visualization_translation",
        "stream": False,
    }
    thinking_type = str(args.translation_thinking_type or "disabled")
    if thinking_type != "none":
        payload["thinking"] = {"type": thinking_type}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    attempts = max(int(args.translation_max_retries), 0) + 1
    last_error = ""
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            url=f"{str(args.translation_base_url).rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(args.translation_timeout)) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:500]}"
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(last_error) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = str(exc)
        if attempt < attempts:
            time.sleep(float(args.translation_retry_base_sleep) * (2 ** (attempt - 1)))
    raise RuntimeError(f"Translation API failed after {attempts} attempts: {last_error}")


def response_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Translation API response has no choices.")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def parse_translation_response(content: str) -> dict[str, str]:
    payload = json.loads(strip_json_fence(content))
    raw = payload.get("translations") if isinstance(payload, dict) else {}
    if raw is None and isinstance(payload, dict):
        raw = payload
    if not isinstance(raw, dict):
        raise ValueError("Translation response must contain a translations object.")
    return {str(key): str(value).strip() for key, value in raw.items() if str(value).strip()}


def strip_json_fence(content: str) -> str:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def render_html(row: dict[str, Any], *, trace: dict[str, Any] | None, args: argparse.Namespace, translations: dict[str, str]) -> str:
    event_id = str(row.get("event_id") or "")
    claim = str(row.get("claim") or "")
    title = f"Evidence map: {event_id}"
    candidates = list(row.get("candidates") or [])[: max(int(args.max_candidates), 1)]
    atoms = list((row.get("evidence_map") or {}).get("claim_atoms") or row.get("claim_atoms") or [])
    selected_index = build_selected_index(trace)
    rows_html = "\n".join(render_candidate(candidate, selected_index=selected_index, atoms=atoms, translations=translations) for candidate in candidates)
    selected_html = render_selected_flow(trace, selected_index=selected_index, translations=translations)
    graph_html = render_evidence_graph(candidates, atoms=atoms, selected_index=selected_index, translations=translations)
    matrix_html = render_matrix(candidates, atoms=atoms, selected_index=selected_index)
    filter_html = render_candidate_filters(candidates, atoms=atoms, selected_index=selected_index)
    metrics_html = render_metrics(row, trace)
    atoms_html = render_atoms(atoms, translations=translations)
    prompt_comparison_html = render_prompt_comparison_section(row, args=args)
    filter_script = candidate_filter_script()
    translation_ui = render_translation_toolbar(translations)
    translation_script = translation_toggle_script(translations)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "candidate_features": args.candidate_features,
        "selection_trace": args.selection_trace,
        "selector_name": args.selector_name,
        "prompt_diagnostic_dir": args.prompt_diagnostic_dir,
        "prompt_evidence_source": args.prompt_evidence_source,
        "prompt_checkpoint": args.prompt_checkpoint,
        "created_at": created,
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #1c2430;
      --muted: #617085;
      --line: #d9e0e8;
      --blue: #2f6fcf;
      --green: #247a52;
      --red: #b8443e;
      --amber: #a5681f;
      --violet: #6d58b8;
      --chip: #eef2f7;
      --shadow: 0 1px 2px rgba(25, 33, 46, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 24px 28px 18px;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    main {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 22px 28px 44px;
    }}
    h1 {{
      font-size: 22px;
      line-height: 1.2;
      margin: 0 0 10px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 16px;
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    h3 {{
      font-size: 14px;
      margin: 0;
      letter-spacing: 0;
    }}
    .claim {{
      max-width: 1100px;
      font-size: 16px;
      margin: 6px 0 0;
    }}
    .translation-toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .translation-toolbar button {{
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
    }}
    .translation-toolbar button:disabled {{
      cursor: not-allowed;
      opacity: 0.55;
    }}
    .i18n-zh {{
      display: none;
    }}
    body.zh-mode .i18n-original {{
      display: none;
    }}
    body.zh-mode .i18n-zh {{
      display: inline;
    }}
    .meta, .small {{
      color: var(--muted);
      font-size: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(300px, 0.75fr) minmax(420px, 1.25fr);
      gap: 18px;
      align-items: start;
    }}
    .section {{
      margin-bottom: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fbfcfd;
    }}
    .metric .value {{
      font-size: 18px;
      font-weight: 700;
      margin-top: 3px;
    }}
    .atom-list {{
      display: grid;
      gap: 10px;
    }}
    .atom {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfd;
    }}
    .atom-id {{
      display: inline-flex;
      min-width: 34px;
      justify-content: center;
      border-radius: 999px;
      padding: 2px 8px;
      background: #e7f0ff;
      color: #1d5cad;
      font-weight: 700;
      font-size: 12px;
      margin-right: 8px;
    }}
    .candidate {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      margin-bottom: 12px;
      overflow: hidden;
    }}
    .candidate.selected {{
      border-color: #87b5ff;
      box-shadow: inset 3px 0 0 var(--blue);
    }}
    .candidate.is-hidden, tr.is-hidden {{
      display: none;
    }}
    .candidate-head {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }}
    .candidate-body {{
      padding: 13px 14px 14px;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 650;
      background: var(--chip);
      color: #354155;
      white-space: nowrap;
    }}
    .badge.support, .badge.direct {{ background: #e6f6ee; color: var(--green); }}
    .badge.refute {{ background: #fdebea; color: var(--red); }}
    .badge.qualify, .badge.mixed {{ background: #fff2d9; color: var(--amber); }}
    .badge.background, .badge.context {{ background: #edf1f5; color: #536171; }}
    .badge.irrelevant, .badge.none {{ background: #f3e9ee; color: #8a4361; }}
    .badge.selected-badge {{ background: #e7f0ff; color: var(--blue); }}
    .score-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(92px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}
    .score {{
      background: #f7f9fb;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
    }}
    .score b {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
    }}
    .evidence {{
      margin: 10px 0 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 13px;
    }}
    mark {{
      background: #fff0a8;
      color: inherit;
      padding: 0 2px;
      border-radius: 2px;
    }}
    .span-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }}
    .span {{
      border-left: 3px solid #d4a626;
      background: #fff9df;
      border-radius: 4px;
      padding: 4px 7px;
      font-size: 12px;
    }}
    .prompt-compare {{
      margin-top: 18px;
    }}
    .outcome-grid, .metric-grid {{
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
    .warn {{ color: var(--amber); }}
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
      max-height: 640px;
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
    .badge.good {{ background: #e6f6ee; color: var(--green); }}
    .badge.bad {{ background: #fdebea; color: var(--red); }}
    .badge.map {{ background: #eeeafd; color: var(--violet); }}
    .badge.plain {{ background: #e7f0ff; color: var(--blue); }}
    .badge.warn {{ background: #fff2d9; color: var(--amber); }}
    .prompt-compare-block {{
      display: grid;
      gap: 14px;
    }}
    .text-cell {{
      min-width: 240px;
      max-width: 520px;
    }}
    .prompt-compare pre.raw {{
      white-space: pre-wrap;
      overflow: auto;
      background: #101820;
      color: #ecf3f9;
      padding: 12px;
      border-radius: 6px;
      font-size: 12px;
    }}
    .candidate-filters {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      padding: 12px;
      margin-bottom: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }}
    .filter-field {{
      display: grid;
      gap: 4px;
    }}
    .filter-field label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    .filter-field select {{
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      padding: 5px 8px;
      font: inherit;
      font-size: 13px;
    }}
    .filter-actions {{
      display: flex;
      align-items: end;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .filter-actions button {{
      min-height: 34px;
      border: 1px solid #b8c6d7;
      border-radius: 6px;
      background: #ffffff;
      color: #29435f;
      padding: 5px 10px;
      font: inherit;
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
    }}
    .filter-status {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      padding-bottom: 8px;
    }}
    .graph-wrap {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      overflow-x: auto;
    }}
    .graph-svg {{
      display: block;
      min-width: 920px;
      width: 100%;
      height: auto;
    }}
    .graph-node rect {{
      fill: #ffffff;
      stroke: #ccd6e2;
      stroke-width: 1.2;
    }}
    .graph-node.selected rect {{
      stroke: var(--blue);
      stroke-width: 2.4;
      fill: #f5f9ff;
    }}
    .graph-node.atom rect {{
      fill: #eef6ff;
      stroke: #9ac1ef;
    }}
    .graph-title {{
      font-size: 12px;
      font-weight: 750;
      fill: #1f2a38;
    }}
    .graph-subtitle {{
      font-size: 10px;
      fill: #5f6d7f;
    }}
    .graph-rank {{
      fill: var(--blue);
      font-size: 11px;
      font-weight: 800;
    }}
    .graph-oracle {{
      fill: var(--green);
      font-size: 11px;
      font-weight: 800;
    }}
    .graph-edge {{
      fill: none;
      stroke-linecap: round;
    }}
    .graph-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend-swatch {{
      width: 18px;
      height: 3px;
      border-radius: 999px;
      display: inline-block;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: #fbfcfd;
      position: sticky;
      top: 86px;
      z-index: 3;
    }}
    td.covered {{
      background: #eaf7ef;
      color: var(--green);
      font-weight: 700;
      text-align: center;
    }}
    td.not-covered {{
      color: #a5afbc;
      text-align: center;
    }}
    .flow {{
      display: grid;
      gap: 8px;
    }}
    .flow-item {{
      display: grid;
      grid-template-columns: 38px 1fr;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fbfcfd;
    }}
    .rank {{
      width: 30px;
      height: 30px;
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      color: #ffffff;
      background: var(--blue);
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      color: var(--blue);
      cursor: pointer;
      font-size: 12px;
      font-weight: 650;
    }}
    pre {{
      white-space: pre-wrap;
      overflow: auto;
      background: #101820;
      color: #ecf3f9;
      padding: 12px;
      border-radius: 6px;
      font-size: 12px;
    }}
    @media (max-width: 900px) {{
      header {{
        position: static;
      }}
      main {{
        padding: 16px;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
      .prompt-grid {{
        grid-template-columns: 1fr;
      }}
      .candidate-head {{
        grid-template-columns: 1fr;
      }}
      th {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="meta">event_id={esc(event_id)} | gold_label={esc(str(row.get("gold_label") or ""))} | parse_status={esc(str(row.get("evidence_map_parse_status") or ""))} | selector={esc(str(args.selector_name or ""))}</div>
    <h1>{esc(title)}</h1>
    <p class="claim">{trans_html("claim", claim, translations)}</p>
    {translation_ui}
  </header>
  <main>
    <div class="grid">
      <aside>
        <section class="section">
          <h2>Claim Atoms</h2>
          {atoms_html}
        </section>
        <section class="section">
          <h2>Selector Metrics</h2>
          {metrics_html}
        </section>
        <section class="section">
          <h2>Selected Top 5</h2>
          {selected_html}
        </section>
        <section class="section">
          <h2>Artifact Inputs</h2>
          <pre>{esc(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>
        </section>
      </aside>
      <div>
        <section class="section">
          <h2>Evidence Graph</h2>
          {graph_html}
        </section>
        <section class="section">
          <h2>Atom Coverage Matrix</h2>
          {matrix_html}
        </section>
        <section class="section">
          <h2>Candidates</h2>
          {filter_html}
          <div id="candidate-list">
            {rows_html}
          </div>
        </section>
      </div>
    </div>
    {prompt_comparison_html}
  </main>
  {filter_script}
  {translation_script}
</body>
</html>
"""


def render_prompt_comparison_section(row: dict[str, Any], *, args: argparse.Namespace) -> str:
    if not bool(args.prompt_comparison):
        return ""
    event_id = str(row.get("event_id") or "")
    diagnostic_dir = Path(args.prompt_diagnostic_dir)
    plain_style = str(args.plain_prompt_style)
    map_style = str(args.map_prompt_style)
    evidence_source = str(args.prompt_evidence_source)
    checkpoint = str(args.prompt_checkpoint)
    split = str(args.prompt_split)
    plain_build_path = prompt_compare.build_path(diagnostic_dir, evidence_source, plain_style, split)
    map_build_path = prompt_compare.build_path(diagnostic_dir, evidence_source, map_style, split)
    if not plain_build_path.exists() or not map_build_path.exists():
        return (
            '<section class="section prompt-compare">'
            "<h2>Plain vs Map Prompt</h2>"
            '<div class="small">Prompt diagnostic artifacts not found. '
            f'plain={esc(plain_build_path)} map={esc(map_build_path)}</div>'
            "</section>"
        )
    try:
        plain_rows = read_jsonl(plain_build_path)
        map_rows = read_jsonl(map_build_path)
        plain_idx, plain_row = prompt_compare.find_row(plain_rows, event_id=event_id, claim_contains="")
        map_idx, map_row = prompt_compare.matching_row(map_rows, event_id=event_id)
        plain_pred = prompt_compare.load_prediction(
            diagnostic_dir,
            evidence_source,
            plain_style,
            checkpoint,
            split,
            sample_idx=plain_idx,
        )
        map_pred = prompt_compare.load_prediction(
            diagnostic_dir,
            evidence_source,
            map_style,
            checkpoint,
            split,
            sample_idx=map_idx,
        )
        plain_metrics = prompt_compare.load_metrics(diagnostic_dir, evidence_source, plain_style, checkpoint)
        map_metrics = prompt_compare.load_metrics(diagnostic_dir, evidence_source, map_style, checkpoint)
        delta = prompt_compare.load_paired_delta(
            diagnostic_dir,
            event_id=event_id,
            evidence_source=evidence_source,
            prompt_style=map_style,
        )
        pc_args = argparse.Namespace(
            plain_style=plain_style,
            map_style=map_style,
            evidence_source=evidence_source,
            checkpoint=checkpoint,
        )
        return (
            '<section class="section prompt-compare">'
            "<h2>Plain vs Map Prompt</h2>"
            '<div class="prompt-compare-block">'
            "<div>"
            '<h3>Prediction And Prompt Delta</h3>'
            f"{prompt_compare.render_outcome(plain_row, map_row, plain_pred, map_pred, delta)}"
            "</div>"
            "<div>"
            '<h3>Overall Checkpoint Metrics</h3>'
            f"{prompt_compare.render_metric_pair(plain_metrics, map_metrics, pc_args)}"
            "</div>"
            "<div>"
            '<h3>Prompt Side By Side</h3>'
            f"{prompt_compare.render_prompt_cards(plain_row, map_row, pc_args)}"
            "</div>"
            "<div>"
            '<h3>Selected Evidence Comparison</h3>'
            f"{prompt_compare.render_evidence_table(plain_row, map_row)}"
            "</div>"
            "<details>"
            "<summary>raw paired prompt payload</summary>"
            f"{prompt_compare.render_raw_payload(plain_row, map_row, plain_pred, map_pred, delta)}"
            "</details>"
            "</div>"
            "</section>"
        )
    except Exception as exc:
        return (
            '<section class="section prompt-compare">'
            "<h2>Plain vs Map Prompt</h2>"
            f'<div class="small">Prompt comparison could not be rendered: {esc(type(exc).__name__)}: {esc(exc)}</div>'
            "</section>"
        )


def render_translation_toolbar(translations: dict[str, str]) -> str:
    count = len(translations)
    disabled = "" if count else " disabled"
    status = f"{count} zh translations embedded" if count else "No zh translations embedded. Re-render with --translate-zh."
    return (
        '<div class="translation-toolbar">'
        f'<button type="button" data-translation-toggle{disabled}>显示中文</button>'
        f'<span class="small" data-translation-status>{esc(status)}</span>'
        "</div>"
    )


def translation_toggle_script(translations: dict[str, str]) -> str:
    payload = json.dumps(translations, ensure_ascii=False).replace("</", "<\\/")
    return f"""<script>
window.EVIDENCE_TRANSLATIONS = {payload};
(() => {{
  const translations = window.EVIDENCE_TRANSLATIONS || {{}};
  const button = document.querySelector("[data-translation-toggle]");
  const status = document.querySelector("[data-translation-status]");
  const svgTexts = Array.from(document.querySelectorAll("[data-i18n-svg-key]"));
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
  if (!button) return;
  let lang = "en";
  button.addEventListener("click", () => {{
    lang = lang === "en" ? "zh" : "en";
    document.body.classList.toggle("zh-mode", lang === "zh");
    setSvgLanguage(lang);
    button.textContent = lang === "zh" ? "Show English" : "显示中文";
    if (status) {{
      status.textContent = lang === "zh" ? "中文翻译已显示" : `${{Object.keys(translations).length}} zh translations embedded`;
    }}
  }});
  setSvgLanguage(lang);
}})();
</script>"""


def trans_html(key: str, text: Any, translations: dict[str, str], *, original_html: str | None = None) -> str:
    original = str(text or "")
    zh = str(translations.get(key) or "").strip()
    if not zh:
        return original_html if original_html is not None else esc(original)
    original_rendered = original_html if original_html is not None else esc(original)
    return (
        f'<span class="i18n-original" data-i18n-key="{esc(key)}">{original_rendered}</span>'
        f'<span class="i18n-zh" data-i18n-key="{esc(key)}">{esc(zh)}</span>'
    )


def trans_svg_attrs(
    key: str,
    text: Any,
    translations: dict[str, str],
    *,
    max_chars: int,
    zh_text: str | None = None,
) -> str:
    original = str(text or "")
    zh = str(zh_text if zh_text is not None else translations.get(key) or "").strip()
    attrs = {
        "data-i18n-svg-key": key,
        "data-i18n-original": original,
        "data-i18n-max": str(max_chars),
    }
    if zh:
        attrs["data-i18n-zh"] = zh
    return " ".join(f'{name}="{esc(value)}"' for name, value in attrs.items())


def render_atoms(atoms: list[dict[str, Any]], *, translations: dict[str, str]) -> str:
    if not atoms:
        return '<div class="small">No claim atoms available.</div>'
    items = []
    for atom in atoms:
        atom_id = str(atom.get("atom_id") or "")
        items.append(
            '<div class="atom">'
            f'<span class="atom-id">{esc(atom_id)}</span>'
            f'{trans_html(atom_translation_key(atom_id), atom.get("text"), translations)}'
            f'<div class="small">type={esc(atom.get("type"))} | importance={fmt(atom.get("importance"))}</div>'
            "</div>"
        )
    return f'<div class="atom-list">{"".join(items)}</div>'


def render_metrics(row: dict[str, Any], trace: dict[str, Any] | None) -> str:
    fields = [
        ("recall@5", trace),
        ("precision@5", trace),
        ("jaccard@5", trace),
        ("weighted_atom_coverage@5", trace),
        ("missing_atom_rate@5", trace),
        ("source_entropy@5", trace),
        ("stance_bucket_entropy@5", trace),
        ("candidate_top_n", row),
        ("oracle_selected_count", row),
    ]
    cells = []
    for name, source in fields:
        if not source or name not in source:
            continue
        cells.append(f'<div class="metric"><div class="small">{esc(name)}</div><div class="value">{fmt(source.get(name))}</div></div>')
    if not cells:
        return '<div class="small">No selector metrics available.</div>'
    return f'<div class="metrics">{"".join(cells)}</div>'


def render_selected_flow(
    trace: dict[str, Any] | None,
    *,
    selected_index: dict[str, dict[str, Any]],
    translations: dict[str, str],
) -> str:
    selected = list((trace or {}).get("selected_candidates") or [])
    if not selected:
        return '<div class="small">No selection trace matched this claim and selector.</div>'
    slot_by_uid = {
        key_for_candidate(slot): slot
        for slot in (trace or {}).get("slot_trace") or []
        if key_for_candidate(slot)
    }
    items = []
    for candidate in selected:
        key = key_for_candidate(candidate)
        slot = slot_by_uid.get(key, {})
        rank = candidate.get("selection_rank") or len(items) + 1
        bits = [
            badge(candidate.get("evidence_id")),
            badge(candidate.get("map_relation"), class_name=str(candidate.get("map_relation") or "")),
            badge(candidate.get("map_directness"), class_name=str(candidate.get("map_directness") or "")),
            badge("oracle" if candidate.get("oracle_selected") else "non-oracle"),
        ]
        score = slot.get("slot_score", candidate.get("slot_score"))
        new_cov = slot.get("new_weighted_atom_coverage")
        text = str(candidate.get("text") or "")
        text_key = f"{candidate_translation_base(candidate)}:text"
        body = (
            f'<div class="badges">{"".join(bits)}</div>'
            f'<div class="small">slot_score={fmt(score)} | new_atom_coverage={fmt(new_cov)} | source={esc(candidate.get("source_group"))}</div>'
            f'<div class="small">{trans_html(text_key, text, translations, original_html=esc(truncate(text, 220)))}</div>'
        )
        items.append(f'<div class="flow-item"><div class="rank">{esc(rank)}</div><div>{body}</div></div>')
    return f'<div class="flow">{"".join(items)}</div>'


def render_candidate_filters(
    candidates: list[dict[str, Any]],
    *,
    atoms: list[dict[str, Any]],
    selected_index: dict[str, dict[str, Any]],
) -> str:
    if not candidates:
        return ""
    relations = sorted({str(candidate.get("map_relation") or "") for candidate in candidates if candidate.get("map_relation")})
    directness = sorted({str(candidate.get("map_directness") or "") for candidate in candidates if candidate.get("map_directness")})
    roles = sorted({str(candidate.get("map_evidence_role") or "") for candidate in candidates if candidate.get("map_evidence_role")})
    atom_ids = [str(atom.get("atom_id") or "") for atom in atoms if atom.get("atom_id")]
    selected_values = ["selected", "not-selected"] if selected_index else []
    oracle_values = sorted({"oracle" if candidate.get("oracle_selected") else "non-oracle" for candidate in candidates})
    fields = [
        ("relation", "Relation", relations),
        ("directness", "Directness", directness),
        ("role", "Role", roles),
        ("selected", "Selected", selected_values),
        ("oracle", "Oracle", oracle_values),
        ("atom", "Atom", atom_ids),
    ]
    controls = [render_filter_select(name, label, values) for name, label, values in fields if values]
    controls.append(
        '<div class="filter-actions">'
        '<button type="button" data-filter-reset>Reset</button>'
        '<span class="filter-status" data-filter-status></span>'
        "</div>"
    )
    return f'<div class="candidate-filters">{"".join(controls)}</div>'


def render_filter_select(name: str, label: str, values: list[str]) -> str:
    options = ['<option value="">All</option>']
    for value in values:
        options.append(f'<option value="{esc(value)}">{esc(value)}</option>')
    return (
        '<div class="filter-field">'
        f'<label for="candidate-filter-{esc(name)}">{esc(label)}</label>'
        f'<select id="candidate-filter-{esc(name)}" data-candidate-filter="{esc(name)}">'
        f'{"".join(options)}'
        "</select>"
        "</div>"
    )


def candidate_filter_script() -> str:
    return """<script>
(() => {
  const controls = Array.from(document.querySelectorAll("[data-candidate-filter]"));
  const reset = document.querySelector("[data-filter-reset]");
  const status = document.querySelector("[data-filter-status]");
  const candidates = Array.from(document.querySelectorAll('[data-filter-target="candidate"]'));
  const matrixRows = Array.from(document.querySelectorAll('[data-filter-target="matrix-row"]'));

  const matches = (el, filters) => {
    for (const [key, value] of Object.entries(filters)) {
      if (!value) continue;
      if (key === "atom") {
        const atoms = (el.dataset.atoms || "").split("|").filter(Boolean);
        if (!atoms.includes(value)) return false;
      } else if ((el.dataset[key] || "") !== value) {
        return false;
      }
    }
    return true;
  };

  const readFilters = () => {
    const filters = {};
    controls.forEach((control) => {
      filters[control.dataset.candidateFilter] = control.value || "";
    });
    return filters;
  };

  const applyFilters = () => {
    const filters = readFilters();
    let visible = 0;
    candidates.forEach((el) => {
      const keep = matches(el, filters);
      el.classList.toggle("is-hidden", !keep);
      if (keep) visible += 1;
    });
    matrixRows.forEach((el) => {
      el.classList.toggle("is-hidden", !matches(el, filters));
    });
    if (status) {
      status.textContent = `${visible} / ${candidates.length} candidates`;
    }
  };

  controls.forEach((control) => control.addEventListener("change", applyFilters));
  if (reset) {
    reset.addEventListener("click", () => {
      controls.forEach((control) => {
        control.value = "";
      });
      applyFilters();
    });
  }
  applyFilters();
})();
</script>"""


def render_evidence_graph(
    candidates: list[dict[str, Any]],
    *,
    atoms: list[dict[str, Any]],
    selected_index: dict[str, dict[str, Any]],
    translations: dict[str, str],
) -> str:
    if not candidates or not atoms:
        return '<div class="small">No graphable atom/evidence links available.</div>'

    atom_ids = [str(atom.get("atom_id") or "") for atom in atoms]
    atom_by_id = {str(atom.get("atom_id") or ""): atom for atom in atoms}
    candidate_gap = 48
    atom_gap = 74
    width = 1120
    top = 58
    bottom = 46
    height = max(320, top + bottom + max(len(candidates) * candidate_gap, len(atoms) * atom_gap))
    atom_x = 34
    atom_w = 290
    evidence_x = 700
    evidence_w = 376
    node_h = 36
    y_min = top
    y_max = height - bottom

    if len(atoms) == 1:
        atom_y = {atom_ids[0]: (y_min + y_max) / 2.0}
    else:
        span = max(y_max - y_min, 1)
        atom_y = {
            atom_id: y_min + span * idx / float(max(len(atom_ids) - 1, 1))
            for idx, atom_id in enumerate(atom_ids)
        }
    evidence_y = {
        key_for_candidate(candidate): top + idx * candidate_gap
        for idx, candidate in enumerate(candidates)
    }

    defs = """
<defs>
  <marker id="arrow-support" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 Z" fill="#247a52" />
  </marker>
  <marker id="arrow-refute" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 Z" fill="#b8443e" />
  </marker>
  <marker id="arrow-qualify" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 Z" fill="#a5681f" />
  </marker>
  <marker id="arrow-background" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 Z" fill="#69778a" />
  </marker>
  <marker id="arrow-irrelevant" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 Z" fill="#8a4361" />
  </marker>
</defs>
"""

    edges: list[str] = []
    for candidate in candidates:
        candidate_key = key_for_candidate(candidate)
        cy = evidence_y.get(candidate_key, top)
        selected = selected_index.get(candidate_key)
        relation = str(candidate.get("map_relation") or "irrelevant")
        color = relation_color(relation)
        opacity = "0.86" if selected else "0.34"
        stroke_width = "3.0" if selected else "1.5"
        marker = f"arrow-{relation_marker(relation)}"
        covered = [str(atom_id) for atom_id in candidate.get("covered_atom_ids") or [] if str(atom_id) in atom_y]
        for atom_id in covered:
            ay = atom_y[atom_id]
            d = (
                f"M {atom_x + atom_w} {ay:.1f} "
                f"C 456 {ay:.1f}, 542 {cy:.1f}, {evidence_x} {cy:.1f}"
            )
            title = (
                f"{candidate.get('evidence_id')} {relation}/{candidate.get('map_directness')} "
                f"covers {atom_id}"
            )
            edges.append(
                f'<path class="graph-edge" d="{d}" stroke="{color}" stroke-width="{stroke_width}" '
                f'opacity="{opacity}" marker-end="url(#{marker})"><title>{esc(title)}</title></path>'
            )

    atom_nodes: list[str] = []
    for atom_id in atom_ids:
        atom = atom_by_id.get(atom_id, {})
        y = atom_y[atom_id]
        text = str(atom.get("text") or "")
        label = f"{atom_id} · {truncate(text, 38)}"
        label_key = atom_translation_key(atom_id)
        zh_label = f"{atom_id} · {translations[label_key]}" if translations.get(label_key) else None
        label_attrs = trans_svg_attrs(label_key, label, translations, max_chars=44, zh_text=zh_label)
        atom_nodes.append(
            f"""
<g class="graph-node atom" transform="translate({atom_x},{y - node_h / 2:.1f})">
  <rect width="{atom_w}" height="{node_h}" rx="7" />
  <title>{esc(text)}</title>
  <text x="12" y="15" class="graph-title" {label_attrs}>{esc(label)}</text>
  <text x="12" y="29" class="graph-subtitle">importance={fmt(atom.get("importance"))} · type={esc(atom.get("type"))}</text>
</g>"""
        )

    evidence_nodes: list[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        candidate_key = key_for_candidate(candidate)
        y = evidence_y.get(candidate_key, top)
        selected = selected_index.get(candidate_key)
        selected_rank = selected.get("selection_rank") if selected else ""
        relation = str(candidate.get("map_relation") or "")
        directness = str(candidate.get("map_directness") or "")
        oracle_selected = bool(candidate.get("oracle_selected"))
        label = f"{candidate.get('evidence_id') or idx}"
        if selected_rank:
            label = f"{label} · top {selected_rank}"
        css = "graph-node selected" if selected else "graph-node"
        text = str(candidate.get("text") or "")
        text_key = f"{candidate_translation_base(candidate)}:text"
        graph_text = truncate(text, 58)
        graph_text_attrs = trans_svg_attrs(text_key, text, translations, max_chars=58)
        evidence_nodes.append(
            f"""
<g class="{css}" transform="translate({evidence_x},{y - node_h / 2:.1f})">
  <rect width="{evidence_w}" height="{node_h}" rx="7" />
  <title>{esc(text)}</title>
  <text x="12" y="15" class="graph-title">{esc(label)} · {esc(relation)}/{esc(directness)}</text>
  <text x="12" y="29" class="graph-subtitle" {graph_text_attrs}>{esc(graph_text)}</text>
  {render_svg_oracle(oracle_selected, evidence_w)}
  {render_svg_rank(selected_rank, evidence_w)}
</g>"""
        )

    legend = render_graph_legend()
    svg = f"""
<div class="graph-wrap">
  <svg class="graph-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Evidence map graph">
    {defs}
    <text x="{atom_x}" y="24" class="graph-title">claim atoms</text>
    <text x="{evidence_x}" y="24" class="graph-title">evidence candidates</text>
    {''.join(edges)}
    {''.join(atom_nodes)}
    {''.join(evidence_nodes)}
  </svg>
</div>
"""
    return legend + svg


def render_graph_legend() -> str:
    items = [
        ("support", relation_color("support")),
        ("refute", relation_color("refute")),
        ("qualify/mixed", relation_color("qualify")),
        ("background", relation_color("background")),
        ("irrelevant", relation_color("irrelevant")),
    ]
    return (
        '<div class="graph-legend">'
        + "".join(
            f'<span class="legend-item"><span class="legend-swatch" style="background:{color}"></span>{esc(label)}</span>'
            for label, color in items
        )
        + "</div>"
    )


def render_svg_rank(selected_rank: Any, evidence_w: int) -> str:
    if not selected_rank:
        return ""
    return f'<text x="{evidence_w - 50}" y="23" class="graph-rank">TOP {esc(selected_rank)}</text>'


def render_svg_oracle(oracle_selected: bool, evidence_w: int) -> str:
    label = "ORACLE" if oracle_selected else "NON-ORACLE"
    x = evidence_w - (112 if oracle_selected else 145)
    return f'<text x="{x}" y="13" class="graph-oracle">{esc(label)}</text>'


def render_matrix(
    candidates: list[dict[str, Any]],
    *,
    atoms: list[dict[str, Any]],
    selected_index: dict[str, dict[str, Any]],
) -> str:
    if not candidates:
        return '<div class="small">No candidates available.</div>'
    atom_ids = [str(atom.get("atom_id") or "") for atom in atoms]
    head_atoms = "".join(f"<th>{esc(atom_id)}</th>" for atom_id in atom_ids)
    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        selected = selected_index.get(key_for_candidate(candidate))
        covered = {str(atom_id) for atom_id in candidate.get("covered_atom_ids") or []}
        filter_attrs = candidate_filter_attrs(candidate, selected=bool(selected), target="matrix-row")
        atom_cells = "".join(
            '<td class="covered">covered</td>' if atom_id in covered else '<td class="not-covered">-</td>'
            for atom_id in atom_ids
        )
        selected_text = f"top {selected.get('selection_rank')}" if selected else ""
        rows.append(
            f"<tr {filter_attrs}>"
            f"<td>{idx}</td>"
            f"<td>{esc(candidate.get('evidence_id'))}</td>"
            f"<td>{esc(selected_text)}</td>"
            f"<td>{esc(candidate.get('map_relation'))}</td>"
            f"<td>{esc(candidate.get('map_directness'))}</td>"
            f"<td>{fmt(candidate.get('evidence_map_quality_score'))}</td>"
            f"{atom_cells}"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>#</th><th>EID</th><th>selected</th><th>relation</th><th>directness</th><th>quality</th>"
        f"{head_atoms}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_candidate(
    candidate: dict[str, Any],
    *,
    selected_index: dict[str, dict[str, Any]],
    atoms: list[dict[str, Any]],
    translations: dict[str, str],
) -> str:
    selected = selected_index.get(key_for_candidate(candidate))
    selected_rank = selected.get("selection_rank") if selected else None
    css = "candidate selected" if selected else "candidate"
    relation = str(candidate.get("map_relation") or "")
    directness = str(candidate.get("map_directness") or "")
    spans = [str(span) for span in candidate.get("key_spans") or [] if str(span).strip()]
    badges = [
        badge(candidate.get("evidence_id")),
        badge(f"top {selected_rank}", class_name="selected-badge") if selected else "",
        badge(relation, class_name=relation),
        badge(directness, class_name=directness),
        badge(candidate.get("map_evidence_role")),
        badge("oracle" if candidate.get("oracle_selected") else "non-oracle"),
    ]
    atom_badges = [badge(atom_id, class_name="selected-badge") for atom_id in candidate.get("covered_atom_ids") or []]
    title = candidate.get("candidate_key") or candidate.get("canonical_text") or candidate.get("text") or ""
    cbase = candidate_translation_base(candidate)
    title_key = f"{cbase}:title"
    text_key = f"{cbase}:text"
    meta = (
        f'evidence_id={esc(candidate.get("evidence_id"))} | '
        f'source={esc(candidate.get("source_group"))} | '
        f'union_rank={esc(candidate.get("union_pool_rank"))} | '
        f'baseline_rank={esc(candidate.get("baseline_rank"))} | '
        f'qd_rank={esc(candidate.get("qd_pool_rank"))}'
    )
    scores = render_score_grid(
        [
            ("slot", selected.get("slot_score") if selected else candidate.get("slot_score")),
            ("quality", candidate.get("evidence_map_quality_score")),
            ("coverage", candidate.get("atom_coverage_score")),
            ("base", candidate.get("evidence_map_base_score")),
            ("fusion", candidate.get("fusion_refit_score")),
            ("direct_ce", candidate.get("direct_ce_score")),
            ("oracle_lik", candidate.get("oracle_likelihood_score")),
            ("retrieval", candidate.get("retrieval_score")),
        ]
    )
    span_html = (
        '<div class="span-list">'
        + "".join(
            f'<span class="span">{trans_html(f"{cbase}:span:{idx}", span, translations)}</span>'
            for idx, span in enumerate(spans)
        )
        + "</div>"
        if spans
        else ""
    )
    filter_attrs = candidate_filter_attrs(candidate, selected=bool(selected), target="candidate")
    debug = compact_candidate_json(candidate)
    return f"""
<article class="{css}" {filter_attrs}>
  <div class="candidate-head">
    <div>
      <h3>{trans_html(title_key, title, translations, original_html=esc(truncate(title, 180)))}</h3>
      <div class="small">{meta}</div>
    </div>
    <div class="badges">{''.join(badges)}</div>
  </div>
  <div class="candidate-body">
    <div class="badges">{''.join(atom_badges) or '<span class="small">No atom coverage</span>'}</div>
    {scores}
    {span_html}
    <div class="evidence">{trans_html(text_key, candidate.get("text"), translations, original_html=highlight_text(str(candidate.get("text") or ""), spans))}</div>
    <details>
      <summary>candidate fields</summary>
      <pre>{esc(json.dumps(debug, ensure_ascii=False, indent=2))}</pre>
    </details>
  </div>
</article>
"""


def render_score_grid(items: Iterable[tuple[str, Any]]) -> str:
    cells = []
    for name, value in items:
        if value is None or value == "":
            continue
        cells.append(f'<div class="score"><b>{esc(name)}</b>{fmt(value)}</div>')
    return f'<div class="score-grid">{"".join(cells)}</div>' if cells else ""


def compact_candidate_json(candidate: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "candidate_uid",
        "candidate_key",
        "evidence_id",
        "covered_atom_ids",
        "map_relation",
        "map_directness",
        "map_evidence_role",
        "key_spans",
        "duplicate_group",
        "map_confidence",
        "evidence_map_quality_score",
        "evidence_map_base_score",
        "fusion_refit_score",
        "direct_ce_score",
        "oracle_likelihood_score",
        "oracle_selected",
        "oracle_step",
        "source_group",
        "source_pools",
        "source_domain",
        "report_id",
        "sent_idx",
    ]
    return {key: candidate.get(key) for key in keep if key in candidate}


def build_selected_index(trace: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for candidate in (trace or {}).get("selected_candidates") or []:
        key = key_for_candidate(candidate)
        if key:
            index[key] = dict(candidate)
    return index


def key_for_candidate(candidate: dict[str, Any]) -> str:
    for key in ("candidate_uid", "evidence_id", "candidate_key"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    return ""


def candidate_translation_base(candidate: dict[str, Any]) -> str:
    return "candidate:" + key_for_candidate(candidate)


def atom_translation_key(atom_id: str) -> str:
    return f"atom:{atom_id}:text"


def candidate_filter_attrs(candidate: dict[str, Any], *, selected: bool, target: str) -> str:
    atoms = "|".join(str(atom_id) for atom_id in candidate.get("covered_atom_ids") or [] if str(atom_id).strip())
    attrs = {
        "data-filter-target": target,
        "data-relation": str(candidate.get("map_relation") or ""),
        "data-directness": str(candidate.get("map_directness") or ""),
        "data-role": str(candidate.get("map_evidence_role") or ""),
        "data-selected": "selected" if selected else "not-selected",
        "data-oracle": "oracle" if candidate.get("oracle_selected") else "non-oracle",
        "data-atoms": atoms,
    }
    return " ".join(f'{name}="{esc(value)}"' for name, value in attrs.items())


def highlight_text(text: str, spans: list[str]) -> str:
    if not text:
        return ""
    intervals: list[tuple[int, int]] = []
    lowered = text.lower()
    for span in spans[:5]:
        needle = span.strip()
        if not needle:
            continue
        start = lowered.find(needle.lower())
        if start < 0:
            continue
        end = start + len(needle)
        if any(not (end <= a or start >= b) for a, b in intervals):
            continue
        intervals.append((start, end))
    if not intervals:
        return esc(text)
    intervals.sort()
    out = []
    pos = 0
    for start, end in intervals:
        out.append(esc(text[pos:start]))
        out.append(f"<mark>{esc(text[start:end])}</mark>")
        pos = end
    out.append(esc(text[pos:]))
    return "".join(out)


def badge(value: Any, *, class_name: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    cls = "badge"
    safe_class = slug(class_name or text).replace("_", "-")
    if safe_class:
        cls += f" {safe_class}"
    return f'<span class="{cls}">{esc(text)}</span>'


def relation_color(relation: str) -> str:
    return {
        "support": "#247a52",
        "refute": "#b8443e",
        "qualify": "#a5681f",
        "mixed": "#a5681f",
        "background": "#69778a",
        "irrelevant": "#8a4361",
    }.get(str(relation or "").strip().lower(), "#69778a")


def relation_marker(relation: str) -> str:
    value = str(relation or "").strip().lower()
    if value in {"support", "refute", "background", "irrelevant"}:
        return value
    if value in {"qualify", "mixed"}:
        return "qualify"
    return "background"


def canonical_event_id(value: str) -> str:
    text = str(value).strip()
    return text[:-5] if text.endswith(".json") else text


def slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return out.strip("._") or "item"


def truncate(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer() and abs(number) < 10_000:
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
