#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_evidence_map_claim_html as map_html
import render_evidence_map_selector_comparison_html as comparison


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_BASE_PATH = "/evidence-map"


@dataclass
class SplitIndex:
    split: str
    candidate_features_path: Path
    left_trace_path: Path
    right_trace_path: Path
    left_chain_graph_path: Path
    right_chain_graph_path: Path
    raw_data_path: Path
    coverage_diff_path: Path
    rows: dict[str, dict[str, Any]]
    raw_rows: dict[str, dict[str, Any]]
    coverage_rows: dict[str, dict[str, Any]]
    left_traces: dict[str, dict[str, Any]]
    right_traces: dict[str, dict[str, Any]]
    left_graph_rows: dict[str, dict[str, Any]]
    right_graph_rows: dict[str, dict[str, Any]]


class ComparisonStore:
    def __init__(self, root: Path, split_indexes: dict[str, SplitIndex], max_candidates: int) -> None:
        self.root = root
        self.split_indexes = split_indexes
        self.max_candidates = int(max_candidates)

    @classmethod
    def load(cls, root: str | Path, splits: list[str], max_candidates: int) -> "ComparisonStore":
        root_path = Path(root).resolve()
        indexes: dict[str, SplitIndex] = {}
        for split in splits:
            index = load_split_index(root_path, split)
            if index is not None:
                indexes[split] = index
        return cls(root=root_path, split_indexes=indexes, max_candidates=max_candidates)

    def search_cases(self, query: str = "", split: str = "", limit: int = 50) -> list[dict[str, Any]]:
        needle = str(query or "").strip().lower()
        wanted_split = str(split or "").strip()
        max_rows = max(int(limit or 50), 1)
        results: list[dict[str, Any]] = []
        for split_name, index in self.split_indexes.items():
            if wanted_split and split_name != wanted_split:
                continue
            for event_key in sorted(index.rows):
                if event_key not in index.left_traces or event_key not in index.right_traces:
                    continue
                row = index.rows[event_key]
                raw_row = index.raw_rows.get(event_key)
                coverage_row = index.coverage_rows.get(event_key)
                haystack = " ".join(
                    str(value or "")
                    for value in (
                        row.get("event_id"),
                        row.get("claim"),
                        row.get("gold_label"),
                        _gold_label(row, raw_row),
                        coverage_row.get("coverage_label") if coverage_row else "",
                    )
                ).lower()
                if needle and needle not in haystack:
                    continue
                results.append(
                    {
                        "split": split_name,
                        "event_id": str(row.get("event_id") or event_key),
                        "claim": str(row.get("claim") or ""),
                        "gold_label": _gold_label(row, raw_row),
                        "coverage_label": str((coverage_row or {}).get("coverage_label") or ""),
                    }
                )
                if len(results) >= max_rows:
                    return results
        return results

    def render_case(
        self,
        split: str,
        event_id: str,
        left_label: str,
        right_label: str,
        *,
        translate_zh: bool = False,
        force_translate: bool = False,
        web_translation_enabled: bool = False,
        web_base_path: str = "",
        web_token: str = "",
    ) -> str:
        index, event_key, row, args, output_path = self._render_context(
            split=split,
            event_id=event_id,
            left_label=left_label,
            right_label=right_label,
            translate_zh=translate_zh,
            force_translate=force_translate,
            web_translation_enabled=web_translation_enabled,
            web_base_path=web_base_path,
            web_token=web_token,
        )
        translations = comparison.load_or_build_translations(
            row,
            raw_row=index.raw_rows.get(event_key),
            coverage_diff=index.coverage_rows.get(event_key),
            left_trace=index.left_traces[event_key],
            right_trace=index.right_traces[event_key],
            left_graph_row=index.left_graph_rows.get(event_key),
            right_graph_row=index.right_graph_rows.get(event_key),
            args=args,
            output_path=output_path,
        )
        return comparison.render_html(
            row,
            raw_row=index.raw_rows.get(event_key),
            coverage_diff=index.coverage_rows.get(event_key),
            left_trace=index.left_traces[event_key],
            right_trace=index.right_traces[event_key],
            left_graph_row=index.left_graph_rows.get(event_key),
            right_graph_row=index.right_graph_rows.get(event_key),
            args=args,
            translations=translations,
        )

    def translate_case(
        self,
        split: str,
        event_id: str,
        left_label: str,
        right_label: str,
        *,
        enabled: bool,
        force_translate: bool = False,
    ) -> dict[str, Any]:
        if not enabled:
            raise PermissionError("Live translation is disabled for this server.")
        index, event_key, row, args, output_path = self._render_context(
            split=split,
            event_id=event_id,
            left_label=left_label,
            right_label=right_label,
            translate_zh=False,
            force_translate=False,
        )
        cache_path = Path(args.translation_cache)
        args.translate_zh = True
        args.force_translate = force_translate
        translations = comparison.load_or_build_translations(
            row,
            raw_row=index.raw_rows.get(event_key),
            coverage_diff=index.coverage_rows.get(event_key),
            left_trace=index.left_traces[event_key],
            right_trace=index.right_traces[event_key],
            left_graph_row=index.left_graph_rows.get(event_key),
            right_graph_row=index.right_graph_rows.get(event_key),
            args=args,
            output_path=output_path,
        )
        return {
            "status": "ok",
            "event_id": str(row.get("event_id") or event_key),
            "cache_path": str(cache_path),
            "translation_count": len(translations),
        }

    def _render_context(
        self,
        *,
        split: str,
        event_id: str,
        left_label: str,
        right_label: str,
        translate_zh: bool,
        force_translate: bool,
        web_translation_enabled: bool = False,
        web_base_path: str = "",
        web_token: str = "",
    ) -> tuple[SplitIndex, str, dict[str, Any], argparse.Namespace, Path]:
        split_name = str(split or "").strip()
        if split_name not in self.split_indexes:
            raise KeyError(f"Unknown split: {split}")
        index = self.split_indexes[split_name]
        event_key = map_html.canonical_event_id(event_id)
        row = index.rows[event_key]
        args = argparse.Namespace(
            candidate_features=str(index.candidate_features_path),
            left_trace=str(index.left_trace_path),
            right_trace=str(index.right_trace_path),
            left_chain_graph=str(index.left_chain_graph_path),
            right_chain_graph=str(index.right_chain_graph_path),
            raw_data=str(index.raw_data_path),
            coverage_diff=str(index.coverage_diff_path),
            resolved_split=split_name,
            output_dir=str(self.root / comparison.DEFAULT_OUTPUT_DIR),
            left_label=str(left_label or comparison.DEFAULT_LEFT_LABEL),
            right_label=str(right_label or comparison.DEFAULT_RIGHT_LABEL),
            max_candidates=self.max_candidates,
            translation_cache="",
            force_translate=bool(force_translate),
            translate_zh=bool(translate_zh),
            translation_base_url=os.environ.get(
                "TRANSLATION_BASE_URL",
                os.environ.get("TEACHER_BASE_URL", comparison.DEFAULT_TRANSLATION_BASE_URL),
            ),
            translation_model=os.environ.get(
                "TRANSLATION_MODEL",
                os.environ.get("TEACHER_MODEL", comparison.DEFAULT_TRANSLATION_MODEL),
            ),
            translation_api_key_env=os.environ.get(
                "TRANSLATION_API_KEY_ENV",
                os.environ.get("TEACHER_API_KEY_ENV", "DEEPSEEK_API_KEY"),
            ),
            translation_timeout=120.0,
            translation_max_tokens=4096,
            translation_batch_chars=7000,
            translation_max_retries=3,
            translation_retry_base_sleep=2.0,
            translation_thinking_type="disabled",
            web_translation_enabled=bool(web_translation_enabled),
            web_base_path=str(web_base_path or ""),
            web_token=str(web_token or ""),
            web_split=split_name,
            web_event_id=str(row.get("event_id") or event_id),
        )
        output_path = comparison.default_output_path(args, row, split=split_name)
        args.translation_cache = str(output_path.with_suffix(".zh.json"))
        return index, event_key, row, args, output_path


def load_split_index(root: Path, split: str) -> SplitIndex | None:
    candidate_features_path = root / comparison.default_candidate_features_path(split)
    if not candidate_features_path.exists():
        return None
    left_trace_path = root / comparison.default_left_trace_path(split)
    right_trace_path = root / comparison.default_right_trace_path(split)
    left_chain_graph_path = root / comparison.default_left_chain_graph_path(split)
    right_chain_graph_path = root / comparison.default_right_chain_graph_path(split)
    raw_data_path = root / comparison.default_raw_data_path(split)
    coverage_diff_path = root / comparison.default_coverage_diff_path(split)
    rows = _index_rows(comparison.read_jsonl(candidate_features_path))
    return SplitIndex(
        split=split,
        candidate_features_path=candidate_features_path,
        left_trace_path=left_trace_path,
        right_trace_path=right_trace_path,
        left_chain_graph_path=left_chain_graph_path,
        right_chain_graph_path=right_chain_graph_path,
        raw_data_path=raw_data_path,
        coverage_diff_path=coverage_diff_path,
        rows=rows,
        raw_rows=_load_raw_index(raw_data_path),
        coverage_rows=_load_jsonl_index(coverage_diff_path),
        left_traces=_load_jsonl_index(left_trace_path),
        right_traces=_load_jsonl_index(right_trace_path),
        left_graph_rows=_load_jsonl_index(left_chain_graph_path),
        right_graph_rows=_load_jsonl_index(right_chain_graph_path),
    )


def is_authorized(token: str, query: Mapping[str, Any], headers: Mapping[str, str]) -> bool:
    expected = str(token or "")
    if not expected:
        return True
    query_token = query.get("token")
    if isinstance(query_token, str) and query_token == expected:
        return True
    if isinstance(query_token, list) and expected in [str(item) for item in query_token]:
        return True
    header_token = headers.get("X-Access-Token") or headers.get("x-access-token")
    return str(header_token or "") == expected


def strip_base_path(path: str, base_path: str) -> str:
    if not base_path:
        return path or "/"
    base = "/" + str(base_path).strip("/")
    request_path = path or "/"
    if request_path == base:
        return "/"
    if request_path.startswith(base + "/"):
        stripped = request_path[len(base) :]
        return stripped or "/"
    return request_path


def make_handler(
    store: ComparisonStore,
    *,
    base_path: str,
    token: str,
    translation_enabled: bool,
) -> type[BaseHTTPRequestHandler]:
    class EvidenceMapHandler(BaseHTTPRequestHandler):
        server_version = "EvidenceMapComparisonHTTP/1.0"

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            if not is_authorized(token, query, self.headers):
                self._send_json({"error": "unauthorized"}, status=401)
                return
            route = strip_base_path(parsed.path, base_path)
            try:
                if route in {"/", "/index.html"}:
                    self._send_html(
                        index_html(
                            store,
                            base_path=base_path,
                            query=query,
                            translation_enabled=translation_enabled,
                        )
                    )
                elif route == "/api/cases":
                    self._send_json(
                        store.search_cases(
                            query=_first(query, "q") or _first(query, "query"),
                            split=_first(query, "split"),
                            limit=int(_first(query, "limit") or "50"),
                        )
                    )
                elif route == "/render":
                    self._send_html(
                        store.render_case(
                            split=_first(query, "split"),
                            event_id=_first(query, "event_id"),
                            left_label=_first(query, "left_label") or comparison.DEFAULT_LEFT_LABEL,
                            right_label=_first(query, "right_label") or comparison.DEFAULT_RIGHT_LABEL,
                            web_translation_enabled=translation_enabled,
                            web_base_path=base_path,
                            web_token=_first(query, "token"),
                        )
                    )
                elif route == "/healthz":
                    self._send_json({"status": "ok", "splits": sorted(store.split_indexes)})
                else:
                    self._send_json({"error": "not found"}, status=404)
            except (KeyError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=404)

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            if not is_authorized(token, query, self.headers):
                self._send_json({"error": "unauthorized"}, status=401)
                return
            route = strip_base_path(parsed.path, base_path)
            if route != "/api/translate":
                self._send_json({"error": "method not allowed"}, status=405)
                return
            try:
                payload = self._read_json_body()
                result = store.translate_case(
                    split=str(payload.get("split") or ""),
                    event_id=str(payload.get("event_id") or ""),
                    left_label=str(payload.get("left_label") or comparison.DEFAULT_LEFT_LABEL),
                    right_label=str(payload.get("right_label") or comparison.DEFAULT_RIGHT_LABEL),
                    enabled=translation_enabled,
                    force_translate=bool(payload.get("force_translate")),
                )
                self._send_json(result)
            except PermissionError as exc:
                self._send_json({"error": str(exc)}, status=403)
            except (KeyError, ValueError, RuntimeError) as exc:
                self._send_json({"error": str(exc)}, status=400)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_html(self, body: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def _send_json(self, payload: Any, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            if length > 65536:
                raise ValueError("Request body is too large.")
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("JSON request body must be an object.")
            return payload

    return EvidenceMapHandler


def index_html(
    store: ComparisonStore,
    *,
    base_path: str,
    query: Mapping[str, Any],
    translation_enabled: bool,
) -> str:
    token = _first(query, "token")
    split_options = "\n".join(
        f'<option value="{html.escape(split)}">{html.escape(split)}</option>' for split in sorted(store.split_indexes)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evidence Map Selector Comparison</title>
  <style>
    :root {{ --line: #d8dee6; --ink: #1f2933; --muted: #64748b; --active: #e8f1ff; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #f7f8fa; }}
    header {{ display: flex; gap: 8px; align-items: center; min-height: 56px; padding: 10px 12px; background: #ffffff; border-bottom: 1px solid var(--line); }}
    input, select, button {{ font: inherit; min-height: 34px; border: 1px solid #c8d1dc; border-radius: 6px; background: #ffffff; color: var(--ink); }}
    button {{ padding: 5px 10px; cursor: pointer; }}
    button:disabled {{ cursor: not-allowed; opacity: 0.55; }}
    #q {{ width: min(38vw, 380px); padding: 5px 9px; }}
    #layout {{ display: grid; grid-template-columns: 340px minmax(0, 1fr); height: calc(100vh - 56px); transition: grid-template-columns 160ms ease; }}
    body[data-sidebar-state="collapsed"] #layout {{ grid-template-columns: 44px minmax(0, 1fr); }}
    #sidebar {{ min-width: 0; overflow: hidden; border-right: 1px solid var(--line); background: #ffffff; }}
    #sidebarRail {{ display: none; height: 100%; align-items: start; justify-content: center; padding-top: 8px; }}
    body[data-sidebar-state="collapsed"] #sidebarRail {{ display: flex; }}
    body[data-sidebar-state="collapsed"] #casePane {{ display: none; }}
    #casePane {{ display: grid; grid-template-rows: auto 1fr; height: 100%; min-width: 0; }}
    #caseStatus {{ padding: 8px 10px; color: var(--muted); font-size: 12px; border-bottom: 1px solid #edf1f4; }}
    #cases {{ overflow: auto; min-height: 0; }}
    #cases button {{ display: block; width: 100%; height: auto; border: 0; border-bottom: 1px solid #eef1f4; border-radius: 0; background: #ffffff; text-align: left; cursor: pointer; padding: 9px 10px; }}
    #cases button.active {{ background: var(--active); }}
    #cases b {{ display: block; font-size: 12px; margin-bottom: 3px; overflow-wrap: anywhere; }}
    #cases span {{ display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; color: var(--muted); font-size: 12px; line-height: 1.35; }}
    iframe {{ width: 100%; height: 100%; border: 0; background: #ffffff; }}
    .status {{ color: var(--muted); font-size: 12px; margin-left: auto; }}
    .toolbar-spacer {{ flex: 1 1 auto; }}
    @media (max-width: 760px) {{
      #layout {{ grid-template-columns: minmax(260px, 42vw) minmax(0, 1fr); }}
      #q {{ width: min(42vw, 260px); }}
    }}
  </style>
</head>
<body data-sidebar-state="expanded">
  <header>
    <button id="sidebarToggle" type="button" data-sidebar-toggle aria-label="Toggle case browser">Cases</button>
    <select id="split">{split_options}</select>
    <input id="q" placeholder="Search claims" autocomplete="off">
    <button id="prev" type="button">Prev</button>
    <button id="next" type="button">Next</button>
  </header>
  <div id="layout">
    <aside id="sidebar">
      <div id="sidebarRail">
        <button type="button" data-sidebar-toggle aria-label="Expand case browser">></button>
      </div>
      <div id="casePane">
        <div id="caseStatus">Loading cases...</div>
        <div id="cases"></div>
      </div>
    </aside>
    <iframe id="frame" title="comparison"></iframe>
  </div>
  <script>
    const basePath = {json.dumps(base_path.rstrip("/") or "")};
    const inheritedToken = {json.dumps(token)};
    const sidebarKey = "evidence-map-selector-sidebar-state";
    let cases = [];
    let activeIndex = -1;
    let loadTimer = 0;

    function withToken(params) {{
      if (inheritedToken) params.set("token", inheritedToken);
      return params;
    }}

    async function loadCases() {{
      const params = withToken(new URLSearchParams({{split: split.value, q: q.value, limit: "100"}}));
      const response = await fetch(`${{basePath}}/api/cases?${{params}}`);
      cases = await response.json();
      activeIndex = cases.length ? 0 : -1;
      caseStatus.textContent = `${{cases.length}} cases`;
      renderList();
      renderActive();
    }}

    function renderList() {{
      casesEl.innerHTML = cases.map((item, idx) =>
        `<button type="button" data-idx="${{idx}}" class="${{idx === activeIndex ? "active" : ""}}">` +
        `<b>${{escapeHtml(item.event_id)}}</b><span>${{escapeHtml(item.claim || "")}}</span></button>`
      ).join("");
    }}

    function renderActive() {{
      if (activeIndex < 0 || !cases[activeIndex]) {{
        frame.removeAttribute("src");
        return;
      }}
      const item = cases[activeIndex];
      const params = withToken(new URLSearchParams({{split: item.split, event_id: item.event_id}}));
      frame.src = `${{basePath}}/render?${{params}}`;
      renderList();
    }}

    function setSidebarState(state) {{
      const next = state === "collapsed" ? "collapsed" : "expanded";
      document.body.dataset.sidebarState = next;
      localStorage.setItem(sidebarKey, next);
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[ch]));
    }}

    const split = document.getElementById("split");
    const q = document.getElementById("q");
    const casesEl = document.getElementById("cases");
    const caseStatus = document.getElementById("caseStatus");
    const frame = document.getElementById("frame");
    document.getElementById("prev").addEventListener("click", () => {{ if (activeIndex > 0) {{ activeIndex--; renderActive(); }} }});
    document.getElementById("next").addEventListener("click", () => {{ if (activeIndex + 1 < cases.length) {{ activeIndex++; renderActive(); }} }});
    split.addEventListener("change", loadCases);
    q.addEventListener("input", () => {{
      clearTimeout(loadTimer);
      loadTimer = setTimeout(loadCases, 180);
    }});
    document.querySelectorAll("[data-sidebar-toggle]").forEach(button => {{
      button.addEventListener("click", () => {{
        setSidebarState(document.body.dataset.sidebarState === "collapsed" ? "expanded" : "collapsed");
      }});
    }});
    casesEl.addEventListener("click", event => {{
      const button = event.target.closest("button[data-idx]");
      if (!button) return;
      activeIndex = Number(button.dataset.idx);
      renderActive();
    }});
    setSidebarState(localStorage.getItem(sidebarKey) || "expanded");
    loadCases();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve evidence-map selector comparison HTML locally.")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--splits", default=",".join(comparison.DEFAULT_SPLITS))
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--token", default="")
    parser.add_argument("--enable-live-translation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [item.strip() for item in str(args.splits or "").split(",") if item.strip()]
    store = ComparisonStore.load(args.root, splits=splits, max_candidates=args.max_candidates)
    handler = make_handler(
        store,
        base_path=args.base_path,
        token=args.token,
        translation_enabled=bool(args.enable_live_translation),
    )
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    url = f"http://{args.host}:{args.port}{str(args.base_path).rstrip('/') or '/'}"
    print(f"Serving evidence-map selector comparison at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _index_by_event_id(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for key in ("event_id", "id", "uid", "filename", "json_id"):
        value = str(row.get(key) or "")
        if value:
            return map_html.canonical_event_id(value), row
    return None


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = _index_by_event_id(row)
        if item is not None:
            indexed[item[0]] = item[1]
    return indexed


def _load_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return _index_rows(comparison.read_jsonl(path))


def _load_raw_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = comparison.read_json(path)
    if isinstance(payload, dict):
        rows = [payload] if comparison.looks_like_sample(payload) else list(payload.values())
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    return _index_rows([row for row in rows if isinstance(row, dict)])


def _load_translation_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = comparison.read_json(path)
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in (payload.get("translations") or {}).items() if str(value).strip()}


def _gold_label(row: dict[str, Any], raw_row: dict[str, Any] | None) -> str:
    return str(row.get("gold_label") or comparison.gold_label_text(raw_row) or "")


def _first(query: Mapping[str, Any], key: str) -> str:
    value = query.get(key)
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


if __name__ == "__main__":
    main()
