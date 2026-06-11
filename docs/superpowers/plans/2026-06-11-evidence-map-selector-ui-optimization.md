# Evidence Map Selector UI Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the private evidence-map selector comparison web UI so the case browser can collapse, Chinese translation can be requested from the page when explicitly enabled, comparison graphs show one map at a time, and long English/Chinese labels stay inside their panels.

**Architecture:** Keep the existing stdlib HTTP server and iframe-backed renderer. The outer server page owns case search, sidebar layout, navigation, and the controlled translation API route. The inner comparison renderer owns the visible translation button/progress state, evidence-map comparison sections, selector graph switching, text wrapping, and reuse of the existing single-map graph renderer from `render_evidence_map_claim_html.py`.

**Tech Stack:** Python 3 stdlib `http.server`, existing HTML string renderers, existing unittest tests, cached DeepSeek-compatible translation helper already present in the renderer.

---

## File Structure

- Modify `scripts/phase5_selectors/visualize/serve_evidence_map_selector_comparison.py`
  - Add collapsible case sidebar markup, CSS, and JavaScript.
  - Add optional `--enable-live-translation`.
  - Add `/api/translate` as a POST-only route that only translates the currently loaded case and only when live translation is enabled.
  - Pass translation options through `ComparisonStore.render_case`.

- Modify `scripts/phase5_selectors/visualize/render_evidence_map_selector_comparison_html.py`
  - Replace side-by-side graph panels with selector cards/tabs and a single active graph.
  - Add graph controls for selected-only, relation filter, and fit/natural width.
  - Harden CSS for long paths, metric values, selector labels, table cells, SVG labels, and mixed Chinese/English content.

- Modify `scripts/phase5_selectors/visualize/render_evidence_map_claim_html.py`
  - Make evidence candidate SVG cards height-adaptive.
  - Render long evidence text in SVG `foreignObject` blocks so English/Chinese content wraps inside the card.

- Modify `src/fact_checking/selectors/test_evidence_map_selector_comparison_server.py`
  - Add tests for index HTML sidebar controls.
  - Add tests for translation being disabled by default.
  - Add tests for live translation routing using a preseeded cache path and without accepting arbitrary text/path input.

- Modify `src/fact_checking/selectors/test_evidence_map_selector_comparison_html.py`
  - Add tests for single-map graph switch markup.
  - Add tests that both selector graph cards exist but only one graph viewport is shown by default.
  - Add tests for overflow-safe classes on long labels and graph controls.
  - Add tests for adaptive wrapped evidence candidate graph cards.

## Safety Constraints

- Live translation stays disabled unless `--enable-live-translation` is passed.
- The translate route must not accept raw text, arbitrary cache paths, arbitrary output paths, or shell commands.
- Translation cache writes must go to the renderer's existing default per-case `.zh.json` path under the configured project root/output directory.
- The web service remains read-mostly; the only write path is the explicitly enabled translation cache.
- Existing route auth via query token or `X-Access-Token` continues to protect `/api/translate`.

---

### Task 1: Add Failing Server UI And Translation Tests

**Files:**
- Modify: `src/fact_checking/selectors/test_evidence_map_selector_comparison_server.py`

- [ ] **Step 1: Add a test for the collapsible sidebar shell without a duplicate translation action**

Add this test method to `EvidenceMapSelectorComparisonServerTest`:

```python
    def test_index_html_has_collapsible_sidebar_without_duplicate_translation_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val")
                store = ComparisonStore.load(Path(tmp), splits=["val"], max_candidates=20)
                html = index_html(store, base_path="/evidence-map", query={"token": ["secret"]}, translation_enabled=False)
            finally:
                os.chdir(old_cwd)

        self.assertIn('data-sidebar-toggle', html)
        self.assertIn('data-sidebar-state', html)
        self.assertNotIn('id="translate"', html)
        self.assertNotIn("translateStatus", html)
        self.assertIn("localStorage", html)
```

Update the imports:

```python
from scripts.phase5_selectors.visualize.serve_evidence_map_selector_comparison import (
    ComparisonStore,
    index_html,
    is_authorized,
    strip_base_path,
)
```

- [ ] **Step 2: Add a test for default-disabled live translation**

Add:

```python
    def test_translate_case_requires_live_translation_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val")
                store = ComparisonStore.load(Path(tmp), splits=["val"], max_candidates=20)

                with self.assertRaisesRegex(PermissionError, "Live translation is disabled"):
                    store.translate_case(split="val", event_id="case", left_label="left", right_label="right", enabled=False)
            finally:
                os.chdir(old_cwd)
```

- [ ] **Step 3: Add a test that enabled translation uses the existing cache path**

Add:

```python
    def test_translate_case_returns_cached_translation_without_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val")
                store = ComparisonStore.load(Path(tmp), splits=["val"], max_candidates=20)

                result = store.translate_case(split="val", event_id="case", left_label="left", right_label="right", enabled=True)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["translation_count"], 1)
        self.assertTrue(str(result["cache_path"]).endswith("val_evidence_map_compare_case_left_vs_right.zh.json"))
```

- [ ] **Step 4: Run server tests and verify RED**

Run:

```bash
PYTHONPATH=.:src python -m unittest src.fact_checking.selectors.test_evidence_map_selector_comparison_server -v
```

Expected: FAIL because `index_html` does not accept `translation_enabled` and `ComparisonStore.translate_case` is missing.

---

### Task 2: Add Failing Renderer Graph And Overflow Tests

**Files:**
- Modify: `src/fact_checking/selectors/test_evidence_map_selector_comparison_html.py`

- [ ] **Step 1: Replace the side-by-side graph expectation**

Add this test method:

```python
    def test_renderer_shows_one_switchable_evidence_map_graph(self) -> None:
        html = render_html(_row(), left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("Evidence Map Graphs", html)
        self.assertIn("data-graph-switcher", html)
        self.assertIn('data-graph-option="left"', html)
        self.assertIn('data-graph-option="right"', html)
        self.assertIn('data-graph-panel="left"', html)
        self.assertIn('data-graph-panel="right"', html)
        self.assertIn("graph-panel-hidden", html)
        self.assertNotIn("map-graph-grid", html)
```

- [ ] **Step 2: Add a test for graph controls and overflow-safe classes**

Add:

```python
    def test_renderer_includes_graph_controls_and_overflow_safe_text_classes(self) -> None:
        row = _row()
        row["claim"] = "A very long claim " + ("with repeated text " * 30)
        row["candidates"][0]["text"] = "A very long evidence passage " + ("with repeated supporting context " * 30)

        html = render_html(row, left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("data-graph-relation-filter", html)
        self.assertIn("data-graph-selected-only", html)
        self.assertIn("data-graph-fit-toggle", html)
        self.assertIn("text-wrap-safe", html)
        self.assertIn("metric-value", html)
        self.assertIn("path-value", html)
```

- [ ] **Step 3: Run renderer tests and verify RED**

Run:

```bash
PYTHONPATH=.:src python -m unittest src.fact_checking.selectors.test_evidence_map_selector_comparison_html -v
```

Expected: FAIL because graph switcher/control markup and the new overflow class names do not exist yet.

---

### Task 3: Implement Server Shell Sidebar And Translation Route

**Files:**
- Modify: `scripts/phase5_selectors/visualize/serve_evidence_map_selector_comparison.py`
- Test: `src/fact_checking/selectors/test_evidence_map_selector_comparison_server.py`

- [ ] **Step 1: Add translation args and render options**

Change `ComparisonStore.render_case` to accept `translate_zh: bool = False` and `force_translate: bool = False`, then set those fields in the `argparse.Namespace` passed to the renderer.

- [ ] **Step 2: Implement `ComparisonStore.translate_case`**

Add a method that calls `render_case(..., translate_zh=True, force_translate=False)` when enabled, then reads the default cache path and returns:

```python
{
    "status": "ok",
    "event_id": row_event_id,
    "cache_path": str(cache_path),
    "translation_count": len(translations),
}
```

If `enabled` is false, raise `PermissionError("Live translation is disabled for this server.")`.

- [ ] **Step 3: Add `--enable-live-translation`**

Add to `parse_args()`:

```python
parser.add_argument("--enable-live-translation", action="store_true")
```

Pass the flag into `make_handler`.

- [ ] **Step 4: Add `/api/translate` as POST-only**

Update `make_handler` to accept `translation_enabled: bool`. In `do_POST`, route only `/api/translate`; parse JSON body with `split`, `event_id`, optional `left_label`, optional `right_label`; call `store.translate_case`; return JSON. Keep all other POST routes as 405.

- [ ] **Step 5: Redesign `index_html` shell**

Update `index_html(store, base_path, query, translation_enabled)` with:

- collapsible sidebar button using `data-sidebar-toggle`
- root layout marker `data-sidebar-state`
- no outer `Translate` button, so the renderer's lower `显示中文` button remains the only translation control
- JavaScript storing sidebar state in `localStorage`
- render iframe URLs that pass enough context for the inner page to call `/api/translate` for the active case only when enabled

- [ ] **Step 6: Run server tests and verify GREEN**

Run:

```bash
PYTHONPATH=.:src python -m unittest src.fact_checking.selectors.test_evidence_map_selector_comparison_server -v
```

Expected: PASS.

---

### Task 4: Implement Single-Graph Switcher And Overflow-Safe Renderer

**Files:**
- Modify: `scripts/phase5_selectors/visualize/render_evidence_map_selector_comparison_html.py`
- Test: `src/fact_checking/selectors/test_evidence_map_selector_comparison_html.py`

- [ ] **Step 1: Update metric/path markup**

Change `metric_cell` to render the value as `<span class="metric-value">...</span>`. Change path values in the header to `<span class="path-value">...</span>`. Add `text-wrap-safe` to claim, gold explanation, flow text, selector labels, and table text cells.

- [ ] **Step 2: Replace `render_evidence_map_graphs` output**

Render a `data-graph-switcher` block with two selector cards/buttons. Render one graph panel per side, but mark the right panel hidden by default with `graph-panel-hidden`. Keep both graph DOMs available so switching is instant.

- [ ] **Step 3: Add graph controls**

Add controls above the active graph:

- relation filter select with `data-graph-relation-filter`
- selected-only checkbox with `data-graph-selected-only`
- fit toggle button with `data-graph-fit-toggle`

Use data attributes on panels and nodes so the script can hide nonmatching evidence nodes and edges.

- [ ] **Step 4: Add graph switch/filter script**

Append a small script after the translation toggle script. It switches active graph panels, updates selected card state, filters graph nodes by relation, toggles selected-only mode, and toggles a `graph-fit-natural` class.

- [ ] **Step 5: Make graph evidence candidate cards adaptive**

In `render_evidence_map_claim_html.py`, compute each evidence node height from its text length and render the text in an SVG `foreignObject` with a `.graph-evidence-text` wrapper. Keep graph height based on the cumulative evidence card heights so long evidence candidates do not overlap or overflow.

- [ ] **Step 6: Harden CSS**

Update CSS so:

- `.text-wrap-safe` uses `white-space: pre-wrap; overflow-wrap: anywhere; word-break: normal;`
- `.metric` has `min-width: 0`
- `.metric-value` can wrap inside its cell
- `.path-value` is smaller, wraps anywhere, and has a max block size with scroll for extremely long paths
- `.map-graph-panel` uses stable width and `overflow: hidden`
- only `.map-graph-panel.graph-panel-active` is displayed
- `.graph-svg` supports fit-width by default and natural width when toggled
- `.graph-evidence-text` wraps long English/Chinese content inside the SVG evidence card

- [ ] **Step 7: Run renderer tests and verify GREEN**

Run:

```bash
PYTHONPATH=.:src python -m unittest src.fact_checking.selectors.test_evidence_map_selector_comparison_html -v
```

Expected: PASS.

---

### Task 5: Full Verification

**Files:**
- Verify: `scripts/phase5_selectors/visualize/serve_evidence_map_selector_comparison.py`
- Verify: `scripts/phase5_selectors/visualize/render_evidence_map_selector_comparison_html.py`
- Verify: `src/fact_checking/selectors/test_evidence_map_selector_comparison_server.py`
- Verify: `src/fact_checking/selectors/test_evidence_map_selector_comparison_html.py`

- [ ] **Step 1: Run both focused test modules**

Run:

```bash
PYTHONPATH=.:src python -m unittest src.fact_checking.selectors.test_evidence_map_selector_comparison_html src.fact_checking.selectors.test_evidence_map_selector_comparison_server -v
```

Expected: PASS.

- [ ] **Step 2: Run compile check**

Run:

```bash
PYTHONPATH=src python -m compileall src scripts/phase5_selectors/visualize
```

Expected: exit 0.

- [ ] **Step 3: Optional local smoke check**

If the real selector artifacts are available, run:

```bash
EVIDENCE_MAP_TOKEN=local-test-token ENABLE_LIVE_TRANSLATION=1 HOST=127.0.0.1 PORT=8765 BASE_PATH=/evidence-map SPLITS=val bash scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh
```

Then check:

```bash
curl -fsS "http://127.0.0.1:8765/evidence-map/healthz?token=local-test-token"
curl -fsS "http://127.0.0.1:8765/evidence-map/api/cases?token=local-test-token&split=val&limit=3"
```

Expected: health JSON and at least one case if artifacts exist.

---

## Self-Review Notes

- The five confirmed UI requirements are covered by Tasks 1-4.
- Live API translation is guarded by an explicit server flag and token-protected route.
- The plan keeps the existing renderer/server boundaries and does not introduce a frontend framework.
- The plan uses test-first steps before production code changes.
