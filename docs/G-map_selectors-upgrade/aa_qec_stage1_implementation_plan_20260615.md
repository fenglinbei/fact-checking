# AA-QEC Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage 1 `AA-QEC-View`, which keeps the v0.7 selected evidence set fixed while changing only atom-anchored order and cue assignment.

**Architecture:** Add a focused `atom_anchored_qec.py` selector module that emits standard `selection_trace` rows. Add a build wrapper to materialize Stage 1 traces and a sentence-trace wrapper to stage/run RAWFC O1/O2/O3 cases through the existing LoRA pipeline. Update `build_trace_verifier_data.py` so `qec_min/qec_map` uses `trace.chain_steps` when present and preserves the current fallback path otherwise.

**Tech Stack:** Python 3.10+, pytest/unittest style tests, bash wrappers, existing `sentence_trace_method` LoRA launcher, existing selector trace schema.

---

### Task 1: AA-QEC-View Selector Core

**Files:**
- Create: `src/fact_checking/selectors/atom_anchored_qec.py`
- Create: `src/fact_checking/selectors/test_atom_anchored_qec.py`

- [ ] **Step 1: Write failing tests for Stage 1 ordering**

Add tests that assert:

```python
def test_view_reorders_same_selected_set_by_atom_order() -> None:
    row = _row_with_selected_indices([1, 0, 2])
    trace = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(selection_policy="keep_all_reorder"),
    )
    assert trace["selector_ordered_indices"] == [0, 1, 2]
    assert sorted(trace["selector_ordered_indices"]) == [0, 1, 2]
    assert [step["role"] for step in trace["chain_steps"]] == ["primary", "primary", "fallback"]
```

```python
def test_view_shuffled_order_is_seeded_and_preserves_selected_set() -> None:
    row = _row_with_selected_indices([0, 1, 2, 3])
    first = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(selection_policy="shuffled", random_seed=7),
    )
    second = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(selection_policy="shuffled", random_seed=7),
    )
    assert first["selector_ordered_indices"] == second["selector_ordered_indices"]
    assert sorted(first["selector_ordered_indices"]) == [0, 1, 2, 3]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=.:src /data/liaozijie/conda/accelerate-fc/bin/python -m pytest src/fact_checking/selectors/test_atom_anchored_qec.py -v
```

Expected: FAIL because `fact_checking.selectors.atom_anchored_qec` does not exist.

- [ ] **Step 3: Implement minimal selector core**

Implement:

```python
@dataclass(frozen=True)
class AtomAnchoredQECParams:
    candidate_top_n: int = 20
    min_chain_steps: int = 5
    max_chain_steps: int = 10
    cue_policy: str = "qd_prefer"
    candidate_scope: str = "selected"
    selection_policy: str = "keep_all_reorder"
    source_selector_name: str = "v0_7_budgeted_marginal_chain_adaptive5_10"
    random_seed: int = 0


def build_atom_anchored_qec_trace_row(row: dict[str, Any], *, params: AtomAnchoredQECParams) -> dict[str, Any]:
    ...
```

Support only Stage 1 policies in this task:

```text
keep_all_reorder
primary_secondary_order
shuffled
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=.:src /data/liaozijie/conda/accelerate-fc/bin/python -m pytest src/fact_checking/selectors/test_atom_anchored_qec.py -v
```

Expected: PASS.

### Task 2: Prompt Builder Chain-Step Support

**Files:**
- Modify: `scripts/phase5_selectors/build/build_trace_verifier_data.py`
- Modify: `scripts/phase5_selectors/build/test_build_trace_verifier_data.py`

- [ ] **Step 1: Write failing test for chain_steps cue usage**

Add a test that builds a trace with `chain_steps` and asserts:

```python
assert "Check: Atom-step cue" in row["prompt"]
assert "Check: route question" not in row["prompt"]
assert "role=primary" not in row["prompt"]
assert "map_confidence" not in row["prompt"]
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
PYTHONPATH=.:src /data/liaozijie/conda/accelerate-fc/bin/python -m pytest scripts/phase5_selectors/build/test_build_trace_verifier_data.py::test_qec_min_prefers_trace_chain_steps_when_present -v
```

Expected: FAIL because current qec_min ignores `trace.chain_steps`.

- [ ] **Step 3: Implement chain_steps-first prompt rendering**

Update `_apply_qec_prompt_fields()` to accept `chain_steps` and use `cue_text` / `evidence_text` from the step when available. Keep current QD route / atom fallback behavior when `chain_steps` is missing.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=.:src /data/liaozijie/conda/accelerate-fc/bin/python -m pytest scripts/phase5_selectors/build/test_build_trace_verifier_data.py::test_qec_min_prefers_trace_chain_steps_when_present scripts/phase5_selectors/build/test_build_trace_verifier_data.py::test_qec_min_prompt_uses_question_then_atom_then_fallback_cues scripts/phase5_selectors/build/test_build_trace_verifier_data.py::test_qec_map_prompt_adds_compact_map_tags -v
```

Expected: PASS.

### Task 3: Stage 1 Build Script

**Files:**
- Create: `scripts/phase5_selectors/build/build_atom_anchored_qec.py`
- Create: `scripts/phase5_selectors/run/run_atom_anchored_qec.sh`

- [ ] **Step 1: Write/import-level tests through selector tests**

Extend selector tests to assert `build_atom_anchored_qec_trace_row()` returns:

```python
assert trace["selector_name"].startswith("aa_qec_view_")
assert trace["graph_version"] == "atom_anchored_qec_v1"
assert trace["candidate_pool_metadata"]["adaptive_policy"] == "aa_qec_view"
assert len(trace["chain_steps"]) == len(trace["selector_ordered_indices"])
```

- [ ] **Step 2: Implement build script and shell wrapper**

`build_atom_anchored_qec.py` reads JSONL rows, calls `build_atom_anchored_qec_trace_row()`, writes:

```text
chain_graph_{split}.jsonl
selection_trace_{split}.jsonl
graph_diagnostics.json
manifest.json
```

`run_atom_anchored_qec.sh` exposes:

```bash
SPLIT
INPUT
OUTPUT_DIR
SELECTION_POLICY
SOURCE_SELECTOR_NAME
CANDIDATE_TOP_N
MIN_CHAIN_STEPS
MAX_CHAIN_STEPS
CUE_POLICY
RANDOM_SEED
SAMPLE_LIMIT
```

- [ ] **Step 3: Run syntax checks**

Run:

```bash
bash -n scripts/phase5_selectors/run/run_atom_anchored_qec.sh
PYTHONPATH=src /data/liaozijie/conda/accelerate-fc/bin/python -m compileall src/fact_checking/selectors/atom_anchored_qec.py scripts/phase5_selectors/build/build_atom_anchored_qec.py
```

Expected: exit 0.

### Task 4: Sentence-Trace Stage 1 Wrapper

**Files:**
- Create: `scripts/sentence_trace_method/prepare_aa_qec_sources.sh`
- Create: `scripts/sentence_trace_method/run_aa_qec_stage1_ministral3.sh`
- Modify: `scripts/sentence_trace_method/test_experiment_matrix_scripts.py`

- [ ] **Step 1: Write failing dry-run test**

Add a test that runs `run_aa_qec_stage1_ministral3.sh` with `DRY_RUN=true MODE=build` and asserts output contains:

```text
rawfc__ministral3_8b__aa_qec_o1_view_atom_order
rawfc__ministral3_8b__aa_qec_o2_view_primary_secondary_order
rawfc__ministral3_8b__aa_qec_o3_view_shuffled
TRACE_PROMPT_STYLE=qec_min
SFT_LEARNING_RATE=1e-5
SFT_NUM_TRAIN_EPOCHS=10
SFT_EVAL_STEPS=50
REQUIRE_PROMPT_INPUT_IDS=true
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
PYTHONPATH=.:src /data/liaozijie/conda/accelerate-fc/bin/python -m pytest scripts/sentence_trace_method/test_experiment_matrix_scripts.py::test_aa_qec_stage1_ministral3_dry_run_expands_rawfc_view_cases -v
```

Expected: FAIL because wrapper does not exist.

- [ ] **Step 3: Implement wrappers**

`prepare_aa_qec_sources.sh` builds and stages Stage 1 O1/O2/O3 sources.

`run_aa_qec_stage1_ministral3.sh` expands RAWFC O1/O2/O3 cases and reuses `run_lora_matrix.sh` with:

```bash
DATASETS=rawfc
MODELS=ministral3_8b
TRACE_PROMPT_STYLE=qec_min
DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz1_ga4.json
SFT_GRADIENT_ACCUMULATION_STEPS=4
SFT_LEARNING_RATE=1e-5
SFT_NUM_TRAIN_EPOCHS=10
SFT_EVAL_STEPS=50
SFT_SAVE_STEPS=50
SFT_EARLY_STOPPING_PATIENCE=8
REQUIRE_PROMPT_INPUT_IDS=true
```

- [ ] **Step 4: Run dry-run test to verify GREEN**

Run:

```bash
PYTHONPATH=.:src /data/liaozijie/conda/accelerate-fc/bin/python -m pytest scripts/sentence_trace_method/test_experiment_matrix_scripts.py::test_aa_qec_stage1_ministral3_dry_run_expands_rawfc_view_cases -v
```

Expected: PASS.

### Task 5: Final Verification

**Files:**
- All files above

- [ ] **Step 1: Run targeted tests**

Run:

```bash
PYTHONPATH=.:src /data/liaozijie/conda/accelerate-fc/bin/python -m pytest \
  src/fact_checking/selectors/test_atom_anchored_qec.py \
  scripts/phase5_selectors/build/test_build_trace_verifier_data.py \
  scripts/sentence_trace_method/test_experiment_matrix_scripts.py -v
```

- [ ] **Step 2: Run shell and compile checks**

Run:

```bash
bash -n scripts/phase5_selectors/run/run_atom_anchored_qec.sh
bash -n scripts/sentence_trace_method/prepare_aa_qec_sources.sh
bash -n scripts/sentence_trace_method/run_aa_qec_stage1_ministral3.sh
PYTHONPATH=src /data/liaozijie/conda/accelerate-fc/bin/python -m compileall src scripts/phase5_selectors/build scripts/sentence_trace_method
git diff --check
```

- [ ] **Step 3: Report status**

Report files changed, verification commands, and any skipped GPU/full training commands.

