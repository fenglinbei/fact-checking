# Evidence Map Selector Web Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a private, read-only web UI for switching evidence-map selector comparison cases in real time, while using the public server `165.22.48.237` only as a reverse proxy/forwarder.

**Architecture:** The visualization service runs on the data/workstation machine that has `/data/liaozijie/fact-checking` and binds to `127.0.0.1:8765`. It loads the existing JSONL artifacts once into an in-memory index, renders cases through the existing `render_evidence_map_selector_comparison_html.py` renderer, and serves an index page with search plus an iframe-backed case view. The public server exposes only a new reverse-proxy route to a server-local SSH reverse tunnel port, so existing web services on ports 80/443 keep ownership of their current domains and routes.

**Tech Stack:** Python 3 stdlib `http.server`, existing renderer helpers, existing unittest style, SSH reverse tunnel, Nginx or Caddy reverse proxy.

---

## File Structure

- Create `scripts/phase5_selectors/visualize/serve_evidence_map_selector_comparison.py`
  - Private HTTP service.
  - Loads split inputs into indexes keyed by canonical event id.
  - Serves `/`, `/api/cases`, `/render`, and `/healthz`.
  - Enforces optional token access through query string or `X-Access-Token`.
  - Supports `--base-path /evidence-map` for path-based reverse proxying.

- Create `scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh`
  - Repo-local launcher with conservative defaults.
  - Binds to `127.0.0.1:8765`.
  - Requires the operator to pass `EVIDENCE_MAP_TOKEN`.

- Create `src/fact_checking/selectors/test_evidence_map_selector_comparison_server.py`
  - Focused unit tests for indexing, search, token checks, base-path routing, and rendering through temp fixture files.

- Create `docs/E-selectors/evidence-map-selector-comparison-web.md`
  - Operator runbook for local service, SSH reverse tunnel, Nginx/Caddy snippets, health checks, rollback, and non-interference checks.

- Modify no existing reverse-proxy config in this repo.
  - Server config is an operator step on `165.22.48.237`, not committed here.

## Safety Constraints

- The Python app must bind to `127.0.0.1` by default.
- The SSH reverse tunnel must bind the remote port to `127.0.0.1` on `165.22.48.237`.
- The public server must proxy only a new specific route such as `/evidence-map/`.
- The app must be read-only: no write routes, no arbitrary path input, no shell command execution.
- Live translation API calls must be disabled in the web service by default. Cached translations may be read.
- Generated HTML may show research paths inside the rendered metadata because this is private access, but proxy config and app logs must not print API keys.

---

### Task 1: Add Server Fixture Tests

**Files:**
- Create: `src/fact_checking/selectors/test_evidence_map_selector_comparison_server.py`

- [ ] **Step 1: Write failing tests for split indexing, search, and render args**

Create the test file with these imports and fixtures:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase5_selectors.visualize import serve_evidence_map_selector_comparison as server
from scripts.phase5_selectors.visualize.render_evidence_map_selector_comparison_html import (
    default_candidate_features_path,
    default_coverage_diff_path,
    default_left_trace_path,
    default_raw_data_path,
    default_right_trace_path,
)


class EvidenceMapSelectorComparisonServerTest(unittest.TestCase):
    def test_store_loads_cases_and_searches_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_split(root, "val")

            store = server.ComparisonStore.load(root=root, splits=["val"], max_candidates=20)
            cases = store.search_cases(query="city budget", split="val", limit=10)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["event_id"], "case.json")
        self.assertEqual(cases[0]["split"], "val")
        self.assertEqual(cases[0]["gold_label"], "mostly-true")
        self.assertEqual(cases[0]["coverage_label"], "uncovered")

    def test_store_renders_existing_comparison_html_without_live_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_split(root, "val")

            store = server.ComparisonStore.load(root=root, splits=["val"], max_candidates=20)
            html = store.render_case(split="val", event_id="case", left_label="left", right_label="right")

        self.assertIn("Evidence map selector comparison: case.json", html)
        self.assertIn("The city budget increased.", html)
        self.assertIn("Coverage Diff", html)

    def test_token_authorization_accepts_header_or_query(self) -> None:
        self.assertTrue(server.is_authorized(token="", query={}, headers={}))
        self.assertTrue(server.is_authorized(token="secret", query={"token": ["secret"]}, headers={}))
        self.assertTrue(server.is_authorized(token="secret", query={}, headers={"X-Access-Token": "secret"}))
        self.assertFalse(server.is_authorized(token="secret", query={}, headers={}))
        self.assertFalse(server.is_authorized(token="secret", query={"token": ["wrong"]}, headers={}))

    def test_base_path_stripping(self) -> None:
        self.assertEqual(server.strip_base_path("/evidence-map/api/cases", "/evidence-map"), "/api/cases")
        self.assertEqual(server.strip_base_path("/evidence-map/", "/evidence-map"), "/")
        self.assertEqual(server.strip_base_path("/api/cases", ""), "/api/cases")
```

Add fixture helpers in the same file:

```python
def _write_split(root: Path, split: str) -> None:
    _write_jsonl(root / default_candidate_features_path(split), [_row()])
    _write_jsonl(root / default_left_trace_path(split), [_left_trace()])
    _write_jsonl(root / default_right_trace_path(split), [_right_trace()])
    _write_json(root / default_raw_data_path(split), [{"event_id": "case.json", "explain": "Raw explanation"}])
    _write_jsonl(root / default_coverage_diff_path(split), [_coverage_diff()])


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row() -> dict:
    return {
        "event_id": "case.json",
        "claim": "The city budget increased.",
        "gold_label": "mostly-true",
        "evidence_map": {"claim_atoms": [{"atom_id": "A1", "text": "The budget increased.", "importance": 1.0}]},
        "candidates": [
            {"evidence_id": "E01", "text": "The budget rose last year.", "covered_atom_ids": ["A1"], "map_relation": "support"}
        ],
    }


def _left_trace() -> dict:
    return {
        "event_id": "case.json",
        "selector_name": "v0_6c_rule_step_adaptive5_10",
        "selected_evidence_ids": ["E01"],
        "selected_candidates": [{"evidence_id": "E01", "text": "The budget rose last year.", "selection_rank": 1}],
        "recall@5": 1.0,
        "precision@5": 1.0,
    }


def _right_trace() -> dict:
    trace = _left_trace()
    trace["selector_name"] = "v0_7_budgeted_marginal_adaptive3_10"
    trace["objective_final_score"] = 0.5
    trace["objective_final_components"] = {"coverage": 0.5, "pair_utility": 0.0}
    return trace


def _coverage_diff() -> dict:
    return {
        "event_id": "case.json",
        "coverage_label": "uncovered",
        "top_evidence_preview": [{"rank": 1, "text": "Top source evidence mentions the budget."}],
    }


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail because the server module is missing**

Run:

```bash
PYTHONPATH=.:src python -m unittest src.fact_checking.selectors.test_evidence_map_selector_comparison_server -v
```

Expected: FAIL with an import error for `serve_evidence_map_selector_comparison`.

- [ ] **Step 3: Commit test scaffold after implementation is ready**

Do not commit the failing-only state. Commit this file together with Task 2.

---

### Task 2: Implement the Private HTTP Service

**Files:**
- Create: `scripts/phase5_selectors/visualize/serve_evidence_map_selector_comparison.py`
- Test: `src/fact_checking/selectors/test_evidence_map_selector_comparison_server.py`

- [ ] **Step 1: Create imports, constants, and dataclasses**

Implement the top of the file:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_evidence_map_selector_comparison_html as comparison
import render_evidence_map_claim_html as map_html


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_BASE_PATH = "/evidence-map"
```

Add data models:

```python
@dataclass
class SplitIndex:
    split: str
    candidate_features_path: str
    left_trace_path: str
    right_trace_path: str
    raw_data_path: str
    coverage_diff_path: str
    rows_by_event: dict[str, dict[str, Any]]
    left_by_event: dict[str, dict[str, Any]]
    right_by_event: dict[str, dict[str, Any]]
    raw_by_event: dict[str, dict[str, Any]]
    coverage_by_event: dict[str, dict[str, Any]]


@dataclass
class ComparisonStore:
    root: Path
    splits: dict[str, SplitIndex]
    max_candidates: int

    @classmethod
    def load(cls, *, root: Path, splits: list[str], max_candidates: int) -> "ComparisonStore":
        indexes = {split: load_split_index(root=root, split=split) for split in splits}
        return cls(root=root, splits=indexes, max_candidates=max_candidates)
```

- [ ] **Step 2: Implement JSON loading and event-id indexes**

Add these functions:

```python
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def canonical_event_id(value: Any) -> str:
    return map_html.canonical_event_id(str(value or ""))


def event_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for field in ("event_id", "id", "uid", "filename", "json_id"):
        key = canonical_event_id(row.get(field))
        if key and key not in keys:
            keys.append(key)
    return keys


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in event_keys(row):
            index.setdefault(key, row)
    return index


def read_raw_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if comparison.looks_like_sample(payload):
            return [payload]
        return [row for row in payload.values() if isinstance(row, dict)]
    return []


def load_split_index(*, root: Path, split: str) -> SplitIndex:
    candidate_features_path = comparison.default_candidate_features_path(split)
    left_trace_path = comparison.default_left_trace_path(split)
    right_trace_path = comparison.default_right_trace_path(split)
    raw_data_path = comparison.default_raw_data_path(split)
    coverage_diff_path = comparison.default_coverage_diff_path(split)

    rows_by_event = index_rows(read_jsonl(root / candidate_features_path))
    left_by_event = index_rows(read_jsonl(root / left_trace_path))
    right_by_event = index_rows(read_jsonl(root / right_trace_path))
    raw_by_event = index_rows(read_raw_rows(root / raw_data_path))
    coverage_by_event = index_rows(read_jsonl(root / coverage_diff_path))

    return SplitIndex(
        split=split,
        candidate_features_path=candidate_features_path,
        left_trace_path=left_trace_path,
        right_trace_path=right_trace_path,
        raw_data_path=raw_data_path,
        coverage_diff_path=coverage_diff_path,
        rows_by_event=rows_by_event,
        left_by_event=left_by_event,
        right_by_event=right_by_event,
        raw_by_event=raw_by_event,
        coverage_by_event=coverage_by_event,
    )
```

- [ ] **Step 3: Implement search and rendering methods**

Add methods inside `ComparisonStore`:

```python
    def search_cases(self, *, query: str, split: str, limit: int) -> list[dict[str, Any]]:
        split_names = [split] if split else sorted(self.splits)
        needle = query.strip().lower()
        results: list[dict[str, Any]] = []
        for split_name in split_names:
            index = self.splits.get(split_name)
            if not index:
                continue
            for event_id, row in index.rows_by_event.items():
                claim = str(row.get("claim") or "")
                if needle and needle not in event_id.lower() and needle not in claim.lower():
                    continue
                coverage = index.coverage_by_event.get(event_id) or {}
                results.append(
                    {
                        "split": split_name,
                        "event_id": str(row.get("event_id") or event_id),
                        "claim": claim,
                        "gold_label": str(row.get("gold_label") or row.get("label") or ""),
                        "coverage_label": str(coverage.get("coverage_label") or ""),
                    }
                )
                if len(results) >= limit:
                    return results
        return results

    def render_case(self, *, split: str, event_id: str, left_label: str, right_label: str) -> str:
        split_name = split or next(iter(sorted(self.splits)))
        index = self.splits[split_name]
        key = canonical_event_id(event_id)
        row = index.rows_by_event.get(key)
        if not row:
            raise KeyError(f"No case matched split={split_name!r} event_id={event_id!r}")
        left_trace = index.left_by_event.get(key)
        right_trace = index.right_by_event.get(key)
        if not left_trace:
            raise KeyError(f"No left trace matched split={split_name!r} event_id={event_id!r}")
        if not right_trace:
            raise KeyError(f"No right trace matched split={split_name!r} event_id={event_id!r}")

        args = argparse.Namespace(
            candidate_features=index.candidate_features_path,
            left_trace=index.left_trace_path,
            right_trace=index.right_trace_path,
            raw_data=index.raw_data_path,
            coverage_diff=index.coverage_diff_path,
            resolved_split=split_name,
            left_label=left_label,
            right_label=right_label,
            max_candidates=self.max_candidates,
            output_dir=comparison.DEFAULT_OUTPUT_DIR,
            translation_cache="",
            force_translate=False,
            translate_zh=False,
        )
        output_path = comparison.default_output_path(args, row, split=split_name)
        args.translation_cache = str(output_path.with_suffix(".zh.json"))
        translations = comparison.load_or_build_translations(
            row,
            raw_row=index.raw_by_event.get(key),
            coverage_diff=index.coverage_by_event.get(key),
            left_trace=left_trace,
            right_trace=right_trace,
            args=args,
            output_path=output_path,
        )
        return comparison.render_html(
            row,
            raw_row=index.raw_by_event.get(key),
            coverage_diff=index.coverage_by_event.get(key),
            left_trace=left_trace,
            right_trace=right_trace,
            args=args,
            translations=translations,
        )
```

- [ ] **Step 4: Implement authorization and base-path helpers**

Add:

def is_authorized(*, token: str, query: dict[str, list[str]], headers: dict[str, str]) -> bool:
    if not token:
        return True
    if (query.get("token") or [""])[0] == token:
        return True
    if headers.get("X-Access-Token", "") == token:
        return True
    return False


def strip_base_path(path: str, base_path: str) -> str:
    normalized = "/" + base_path.strip("/") if base_path else ""
    if normalized and path == normalized:
        return "/"
    if normalized and path.startswith(normalized + "/"):
        return path[len(normalized):] or "/"
    return path
```

- [ ] **Step 5: Implement the index HTML**

Add:

```python
def render_index_html(*, base_path: str, splits: list[str], token_in_query: str) -> str:
    base = "/" + base_path.strip("/") if base_path else ""
    split_options = "".join(f'<option value="{html_escape(split)}">{html_escape(split)}</option>' for split in splits)
    token_query = f"token={quote(token_in_query)}" if token_in_query else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evidence Map Selector Cases</title>
  <style>
    body {{ margin: 0; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #17202a; }}
    header {{ position: sticky; top: 0; z-index: 5; display: grid; grid-template-columns: minmax(140px, 220px) 1fr auto auto; gap: 8px; align-items: center; padding: 10px 12px; border-bottom: 1px solid #d9dee5; background: #fff; }}
    select, input, button {{ height: 34px; border: 1px solid #c7ced8; border-radius: 6px; background: #fff; color: #17202a; font: inherit; }}
    input {{ min-width: 260px; padding: 0 10px; }}
    button {{ padding: 0 12px; cursor: pointer; }}
    main {{ display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: calc(100vh - 55px); }}
    aside {{ border-right: 1px solid #d9dee5; background: #fff; overflow: auto; max-height: calc(100vh - 55px); }}
    .case {{ display: block; width: 100%; text-align: left; border: 0; border-bottom: 1px solid #edf0f3; border-radius: 0; height: auto; padding: 10px 12px; }}
    .case b {{ display: block; font-size: 12px; margin-bottom: 3px; }}
    .case span {{ display: block; color: #5e6b78; font-size: 12px; line-height: 1.35; }}
    .case.active {{ background: #e8f0ff; }}
    iframe {{ width: 100%; height: calc(100vh - 55px); border: 0; background: #fff; }}
    .status {{ color: #5e6b78; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <select id="split">{split_options}</select>
    <input id="query" type="search" placeholder="event id or claim text" autocomplete="off">
    <button id="prev" type="button">Prev</button>
    <button id="next" type="button">Next</button>
  </header>
  <main>
    <aside>
      <div class="status" id="status">Loading cases...</div>
      <div id="cases"></div>
    </aside>
    <iframe id="frame" title="Evidence map selector comparison"></iframe>
  </main>
  <script>
    const BASE = {json.dumps(base)};
    const TOKEN_QUERY = {json.dumps(token_query)};
    const split = document.getElementById("split");
    const query = document.getElementById("query");
    const statusEl = document.getElementById("status");
    const casesEl = document.getElementById("cases");
    const frame = document.getElementById("frame");
    let cases = [];
    let selected = -1;
    const suffix = () => TOKEN_QUERY ? "&" + TOKEN_QUERY : "";
    const apiUrl = () => `${{BASE}}/api/cases?split=${{encodeURIComponent(split.value)}}&q=${{encodeURIComponent(query.value)}}${{suffix()}}`;
    const renderUrl = (item) => `${{BASE}}/render?split=${{encodeURIComponent(item.split)}}&event_id=${{encodeURIComponent(item.event_id)}}${{suffix()}}`;
    async function loadCases() {{
      const res = await fetch(apiUrl(), {{ credentials: "same-origin" }});
      if (!res.ok) {{
        statusEl.textContent = `Failed to load cases: ${{res.status}}`;
        return;
      }}
      cases = await res.json();
      selected = cases.length ? 0 : -1;
      drawCases();
      openSelected();
    }}
    function drawCases() {{
      statusEl.textContent = `${{cases.length}} cases`;
      casesEl.innerHTML = "";
      cases.forEach((item, idx) => {{
        const button = document.createElement("button");
        button.className = "case" + (idx === selected ? " active" : "");
        button.type = "button";
        button.innerHTML = `<b>${{item.split}} / ${{item.event_id}}</b><span>${{item.gold_label || ""}} ${{item.coverage_label || ""}}</span><span>${{item.claim || ""}}</span>`;
        button.addEventListener("click", () => {{ selected = idx; drawCases(); openSelected(); }});
        casesEl.appendChild(button);
      }});
    }}
    function openSelected() {{
      if (selected < 0 || !cases[selected]) {{
        frame.removeAttribute("src");
        return;
      }}
      frame.src = renderUrl(cases[selected]);
    }}
    document.getElementById("prev").addEventListener("click", () => {{
      if (!cases.length) return;
      selected = (selected + cases.length - 1) % cases.length;
      drawCases();
      openSelected();
    }});
    document.getElementById("next").addEventListener("click", () => {{
      if (!cases.length) return;
      selected = (selected + 1) % cases.length;
      drawCases();
      openSelected();
    }});
    split.addEventListener("change", loadCases);
    query.addEventListener("input", () => {{
      clearTimeout(query._timer);
      query._timer = setTimeout(loadCases, 180);
    }});
    loadCases();
  </script>
</body>
</html>"""
```

Add the escape helper used above:

```python
def html_escape(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
```

- [ ] **Step 6: Implement request handler and server entry point**

Add:

```python
class EvidenceMapHandler(BaseHTTPRequestHandler):
    store: ComparisonStore
    base_path: str
    token: str
    left_label: str
    right_label: str

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = strip_base_path(parsed.path, self.base_path)
        query = parse_qs(parsed.query)
        headers = {key: value for key, value in self.headers.items()}
        if not is_authorized(token=self.token, query=query, headers=headers):
            self.send_text("unauthorized\n", status=HTTPStatus.UNAUTHORIZED)
            return

        try:
            if route in ("/", "/index.html"):
                token_for_links = (query.get("token") or [""])[0] if self.token else ""
                self.send_html(render_index_html(base_path=self.base_path, splits=sorted(self.store.splits), token_in_query=token_for_links))
            elif route == "/api/cases":
                rows = self.store.search_cases(
                    query=(query.get("q") or [""])[0],
                    split=(query.get("split") or [""])[0],
                    limit=int((query.get("limit") or ["200"])[0]),
                )
                self.send_json(rows)
            elif route == "/render":
                html = self.store.render_case(
                    split=(query.get("split") or [""])[0],
                    event_id=unquote((query.get("event_id") or [""])[0]),
                    left_label=self.left_label,
                    right_label=self.right_label,
                )
                self.send_html(html)
            elif route == "/healthz":
                self.send_json({"status": "ok", "splits": sorted(self.store.splits)})
            else:
                self.send_text("not found\n", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_text(f"error: {exc}\n", status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
```

Add CLI:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve private evidence-map selector comparison UI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--splits", default="val")
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--token", default=os.environ.get("EVIDENCE_MAP_TOKEN", ""))
    parser.add_argument("--left-label", default=comparison.DEFAULT_LEFT_LABEL)
    parser.add_argument("--right-label", default=comparison.DEFAULT_RIGHT_LABEL)
    parser.add_argument("--max-candidates", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    store = ComparisonStore.load(root=Path(args.root), splits=splits, max_candidates=int(args.max_candidates))

    class Handler(EvidenceMapHandler):
        pass

    Handler.store = store
    Handler.base_path = "/" + str(args.base_path).strip("/") if str(args.base_path).strip("/") else ""
    Handler.token = str(args.token or "")
    Handler.left_label = str(args.left_label or comparison.DEFAULT_LEFT_LABEL)
    Handler.right_label = str(args.right_label or comparison.DEFAULT_RIGHT_LABEL)

    server = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    print(f"Serving evidence-map selector UI at http://{args.host}:{args.port}{Handler.base_path or '/'}")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run tests**

Run:

```bash
PYTHONPATH=.:src python -m unittest src.fact_checking.selectors.test_evidence_map_selector_comparison_server -v
```

Expected: PASS for all tests in the new server test file.

- [ ] **Step 8: Commit**

Run:

```bash
git add scripts/phase5_selectors/visualize/serve_evidence_map_selector_comparison.py src/fact_checking/selectors/test_evidence_map_selector_comparison_server.py
git commit -m "add private evidence map comparison web server"
```

---

### Task 3: Add a Repo-Local Launcher

**Files:**
- Create: `scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh`

- [ ] **Step 1: Write the launcher**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
BASE_PATH="${BASE_PATH:-/evidence-map}"
SPLITS="${SPLITS:-val}"
MAX_CANDIDATES="${MAX_CANDIDATES:-20}"

if [[ -z "${EVIDENCE_MAP_TOKEN:-}" ]]; then
  echo "ERROR: set EVIDENCE_MAP_TOKEN before starting the private web UI." >&2
  exit 2
fi

PYTHONPATH=. python scripts/phase5_selectors/visualize/serve_evidence_map_selector_comparison.py \
  --host "$HOST" \
  --port "$PORT" \
  --base-path "$BASE_PATH" \
  --splits "$SPLITS" \
  --max-candidates "$MAX_CANDIDATES"
```

- [ ] **Step 2: Make it executable**

Run:

```bash
chmod +x scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh
```

- [ ] **Step 3: Smoke test locally**

Run:

```bash
EVIDENCE_MAP_TOKEN=local-test-token timeout 5s bash scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh
```

Expected: command starts, prints the serving URL, then exits due to `timeout`.

- [ ] **Step 4: Commit**

Run:

```bash
git add scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh
git commit -m "add evidence map web launcher"
```

---

### Task 4: Add Operator Runbook for Forward-Only Public Server

**Files:**
- Create: `docs/E-selectors/evidence-map-selector-comparison-web.md`

- [ ] **Step 1: Write the runbook**

Create this document:

```markdown
# Evidence Map Selector Comparison Web UI

This UI is private and read-only. The Python app runs on the data machine and binds to `127.0.0.1:8765`. The public server `165.22.48.237` is used only as a reverse proxy through a server-local SSH reverse tunnel port.

## Start the app on the data machine

```bash
cd /data/liaozijie/fact-checking
export EVIDENCE_MAP_TOKEN="$(openssl rand -hex 24)"
HOST=127.0.0.1 PORT=8765 BASE_PATH=/evidence-map SPLITS=val \
  bash scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh
```

Health check from the data machine:

```bash
curl -fsS "http://127.0.0.1:8765/evidence-map/healthz?token=${EVIDENCE_MAP_TOKEN}"
```

## Create the reverse tunnel to `165.22.48.237`

Run this from the data machine. It exposes only `127.0.0.1:18765` on the public server:

```bash
ssh -N -R 127.0.0.1:18765:127.0.0.1:8765 165.22.48.237
```

Health check from `165.22.48.237`:

```bash
curl -fsS "http://127.0.0.1:18765/evidence-map/healthz?token=${EVIDENCE_MAP_TOKEN}"
```

## Nginx route on `165.22.48.237`

Add only this location inside the existing HTTPS `server {}` block for the already-bound domain:

```nginx
location /evidence-map/ {
    proxy_pass http://127.0.0.1:18765;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 120s;
}
```

Validate and reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

This does not change existing root, domain, certificate, or upstream service ownership. It only reserves the `/evidence-map/` path.

## Caddy route on `165.22.48.237`

If the server uses Caddy, add this handle inside the existing site block:

```caddyfile
handle /evidence-map/* {
    reverse_proxy 127.0.0.1:18765
}
```

Validate and reload:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

## Non-interference checklist

- The Python service listens on `127.0.0.1:8765`, not `0.0.0.0`.
- The reverse tunnel listens on `127.0.0.1:18765` on `165.22.48.237`, not `0.0.0.0:18765`.
- Existing 80/443 services keep their current `server {}` or site block.
- Only a new `/evidence-map/` path is added.
- `nginx -t` or `caddy validate` passes before reload.
- External access goes through HTTPS on the existing domain.
- The app token is required unless the proxy is protected by stronger authentication.

## Rollback

Remove the new `/evidence-map/` route from Nginx or Caddy and reload the proxy:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

or:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

Then stop the SSH reverse tunnel and the local Python process. Existing services are unaffected because no existing location, upstream, certificate, or port binding was changed.
```

- [ ] **Step 2: Commit runbook**

Run:

```bash
git add docs/E-selectors/evidence-map-selector-comparison-web.md
git commit -m "document evidence map web forwarding"
```

---

### Task 5: End-to-End Verification

**Files:**
- Verify: `scripts/phase5_selectors/visualize/serve_evidence_map_selector_comparison.py`
- Verify: `scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh`
- Verify: `docs/E-selectors/evidence-map-selector-comparison-web.md`

- [ ] **Step 1: Run server unit tests**

Run:

```bash
PYTHONPATH=.:src python -m unittest src.fact_checking.selectors.test_evidence_map_selector_comparison_server -v
```

Expected: all tests pass.

- [ ] **Step 2: Run existing comparison renderer tests**

Run:

```bash
PYTHONPATH=. python -m unittest src.fact_checking.selectors.test_evidence_map_selector_comparison_html -v
```

Expected: all tests pass. This guards against breaking the existing static HTML renderer.

- [ ] **Step 3: Run compile check**

Run:

```bash
PYTHONPATH=src python -m compileall src scripts/phase5_selectors/visualize
```

Expected: compile succeeds.

- [ ] **Step 4: Local manual check**

Start the app:

```bash
cd /data/liaozijie/fact-checking
export EVIDENCE_MAP_TOKEN=local-test-token
HOST=127.0.0.1 PORT=8765 BASE_PATH=/evidence-map SPLITS=val \
  bash scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh
```

In another shell:

```bash
curl -fsS "http://127.0.0.1:8765/evidence-map/healthz?token=local-test-token"
curl -fsS "http://127.0.0.1:8765/evidence-map/api/cases?token=local-test-token&split=val&limit=3"
```

Expected: health JSON returns `status=ok`; cases JSON returns at least one case for val when the artifacts exist.

- [ ] **Step 5: Public-server forwarding check**

Start the reverse tunnel from the data machine:

```bash
ssh -N -R 127.0.0.1:18765:127.0.0.1:8765 165.22.48.237
```

From `165.22.48.237`, verify:

```bash
curl -fsS "http://127.0.0.1:18765/evidence-map/healthz?token=local-test-token"
```

Expected: same health JSON as the local app. If this fails, fix the SSH tunnel before touching Nginx or Caddy.

- [ ] **Step 6: Proxy non-interference check**

Before reload on `165.22.48.237`, run:

```bash
sudo nginx -t
```

or:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
```

Expected: validation passes. Reload only after validation passes.

- [ ] **Step 7: Commit final verification updates**

If verification required documentation edits, run:

```bash
git add docs/E-selectors/evidence-map-selector-comparison-web.md
git commit -m "clarify evidence map web verification"
```

---

## Self-Review

- Spec coverage: The plan builds the private app, supports real-time case switching, avoids new Python dependencies, and uses `165.22.48.237` only as a forwarding server.
- Non-interference: The proxy route is path-scoped to `/evidence-map/`; Python and tunnel ports bind only to loopback; proxy reload is gated by validation.
- Security: The app has optional token auth, the launcher requires a token, and live translation is disabled by default.
- Testing: Unit tests cover indexing, search, render integration, token auth, and base-path routing; existing renderer tests remain in the verification set.
