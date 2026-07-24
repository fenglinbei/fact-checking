#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/models/Ministral-3-8B-Instruct-2512}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_minmax5_10.yaml}"
ABC_ROOT="${ABC_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1}"
LEARNED_TRACE_ROOT="${LEARNED_TRACE_ROOT:-${ABC_ROOT}/05_mrec_v0_2_learned_marginal_proxy_fullpool}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${ABC_ROOT}/11_typed_role_rescue_v0_1}"
INTRINSIC_ROOT="${INTRINSIC_ROOT:-${ARTIFACT_ROOT}/intrinsic}"
PROMPT_FEASIBLE_ROOT="${PROMPT_FEASIBLE_ROOT:-${ARTIFACT_ROOT}/prompt_feasible}"
DIAGNOSTIC_BUILD_ROOT="${DIAGNOSTIC_BUILD_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__typed_role_rescue_diagnostic_v0_1}"
FORMAL_BUILD_ROOT="${FORMAL_BUILD_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__typed_role_rescue_v0_1}"
REFERENCE_CONTRACT="${REFERENCE_CONTRACT:-configs/validation/baces_native_label_token_reference_v0_1.json}"
K="${K:-5}"
SEED="${SEED:-20260715}"

CELLS=(r_only learned_fixed5 cor opp ctx retr random full)
EXPECTED_SPLITS=(train val test)

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

timestamp() { date '+%F %T %Z %z'; }
status() { printf '[%s] %s\n' "$(timestamp)" "$1"; }
raw_path() { printf '%s\n' "data/raw/LIAR-RAW/$1.json"; }

[[ -x "$PYTHON_BIN" ]] || { printf 'Python environment is unavailable: %s\n' "$PYTHON_BIN" >&2; exit 2; }
[[ -f "$REFERENCE_CONTRACT" ]] || { printf 'Missing reference contract: %s\n' "$REFERENCE_CONTRACT" >&2; exit 2; }
NATIVE_GATE_BUILD="$(jq -er '.artifacts.build.path' "$REFERENCE_CONTRACT")"
[[ -f "$NATIVE_GATE_BUILD" ]] || { printf 'Missing native gate build: %s\n' "$NATIVE_GATE_BUILD" >&2; exit 2; }

status "phase 1/6: materializing intrinsic typed role-rescue slates"
for split in "${EXPECTED_SPLITS[@]}"; do
  "$PYTHON_BIN" scripts/phase5_selectors/build/build_role_rescue_traces.py \
    --input "${LEARNED_TRACE_ROOT}/selection_trace_${split}.jsonl" \
    --output-dir "${INTRINSIC_ROOT}/${split}" \
    --split "$split" \
    --k "$K" \
    --seed "$SEED"
done

status "phase 2/6: rendering diagnostic prompt surfaces"
for cell in "${CELLS[@]}"; do
  "$PYTHON_BIN" scripts/phase5_selectors/build/build_trace_verifier_data.py \
    --config "$CONFIG_PATH" \
    --train-trace "${INTRINSIC_ROOT}/train/${cell}/selection_trace_train.jsonl" \
    --val-trace "${INTRINSIC_ROOT}/val/${cell}/selection_trace_val.jsonl" \
    --test-trace "${INTRINSIC_ROOT}/test/${cell}/selection_trace_test.jsonl" \
    --train-raw "$(raw_path train)" \
    --val-raw "$(raw_path val)" \
    --test-raw "$(raw_path test)" \
    --dataset liar_raw \
    --label-schema liar6 \
    --output-dir "${DIAGNOSTIC_BUILD_ROOT}/${cell}" \
    --selection-mode trace \
    --trace-prompt-style mrec_min \
    --evidence-text-mode full \
    --top-k "$K" \
    --prompt-evidence-policy selected_set \
    --prompt-evidence-min-count 0 \
    --prompt-evidence-max-count "$K" \
    --prompt-evidence-max-length-guard warn \
    --expected-chunk-mmr-fingerprint "" \
    --prompt-model-name-or-path "$MODEL_PATH" \
    --train-model-name-or-path "$MODEL_PATH" \
    --no-progress
done

status "phase 3/6: projecting exact verifier-visible prefixes"
for split in "${EXPECTED_SPLITS[@]}"; do
  projection_args=(
    "$PYTHON_BIN" scripts/phase5_selectors/build/project_role_rescue_prompt_feasible.py
    --role-dir "${INTRINSIC_ROOT}/${split}"
    --diagnostic-build-root "$DIAGNOSTIC_BUILD_ROOT"
    --output-dir "${PROMPT_FEASIBLE_ROOT}/${split}"
    --split "$split"
    --overwrite
  )
  for cell in "${CELLS[@]}"; do
    projection_args+=(--cell "$cell")
  done
  for cell in learned_fixed5 cor opp ctx retr random full; do
    projection_args+=(--shared-count-cell "$cell")
  done
  if [[ "$split" == "val" ]]; then
    projection_args+=(--external-cell native_gate_anchor)
  fi
  "${projection_args[@]}"
done

status "phase 4/6: rebuilding strict lossless verifier inputs"
for cell in "${CELLS[@]}"; do
  "$PYTHON_BIN" scripts/phase5_selectors/build/build_trace_verifier_data.py \
    --config "$CONFIG_PATH" \
    --train-trace "${PROMPT_FEASIBLE_ROOT}/train/${cell}/selection_trace_train.jsonl" \
    --val-trace "${PROMPT_FEASIBLE_ROOT}/val/${cell}/selection_trace_val.jsonl" \
    --test-trace "${PROMPT_FEASIBLE_ROOT}/test/${cell}/selection_trace_test.jsonl" \
    --train-raw "$(raw_path train)" \
    --val-raw "$(raw_path val)" \
    --test-raw "$(raw_path test)" \
    --dataset liar_raw \
    --label-schema liar6 \
    --output-dir "${FORMAL_BUILD_ROOT}/${cell}" \
    --selection-mode trace \
    --trace-prompt-style mrec_min \
    --evidence-text-mode full \
    --top-k "$K" \
    --prompt-evidence-policy selected_set \
    --prompt-evidence-min-count 0 \
    --prompt-evidence-max-count "$K" \
    --prompt-evidence-max-length-guard warn \
    --expected-chunk-mmr-fingerprint "" \
    --prompt-model-name-or-path "$MODEL_PATH" \
    --train-model-name-or-path "$MODEL_PATH" \
    --forbid-prompt-truncation \
    --no-progress
done

status "phase 5/6: installing the frozen native-equivalence gate cell"
mkdir -p "${FORMAL_BUILD_ROOT}/native_gate_anchor/build"
cp -f "$NATIVE_GATE_BUILD" "${FORMAL_BUILD_ROOT}/native_gate_anchor/build/build_val.jsonl"

status "phase 6/6: running row, surface, split, and role-realization gates"
"$PYTHON_BIN" - \
  "$INTRINSIC_ROOT" \
  "$PROMPT_FEASIBLE_ROOT" \
  "$DIAGNOSTIC_BUILD_ROOT" \
  "$FORMAL_BUILD_ROOT" \
  "$NATIVE_GATE_BUILD" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

intrinsic_root, trace_root, diagnostic_root, build_root, native_gate = map(
    Path, sys.argv[1:]
)
cells = ("r_only", "learned_fixed5", "cor", "opp", "ctx", "retr", "random", "full")
expected_rows = {"train": 10065, "val": 1274, "test": 1251}


def iter_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def visible_uids(row):
    count = int(row.get("evidence_count", 0))
    return [str(item.get("candidate_uid") or "") for item in row["candidates"][:count]]


def event_sequence(path):
    return [str(row.get("event_id") or "") for _, row in iter_jsonl(path)]


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for cell in cells:
    cfg_path = build_root / cell / "train.resolved.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    configured_paths = []
    split_event_sets = {}
    for split, expected in expected_rows.items():
        intrinsic_path = intrinsic_root / split / cell / f"selection_trace_{split}.jsonl"
        trace_path = trace_root / split / cell / f"selection_trace_{split}.jsonl"
        diagnostic_path = diagnostic_root / cell / "build" / f"build_{split}.jsonl"
        build_path = build_root / cell / "build" / f"build_{split}.jsonl"
        configured = Path(str(cfg["data"][f"{split}_candidates"]))
        if configured.resolve() != build_path.resolve():
            raise SystemExit(f"{cell}:{split}: train config path mismatch")
        configured_paths.append(configured.resolve())

        intrinsic_rows = iter_jsonl(intrinsic_path)
        trace_rows = iter_jsonl(trace_path)
        diagnostic_rows = iter_jsonl(diagnostic_path)
        build_rows = iter_jsonl(build_path)
        event_ids = []
        count = 0
        while True:
            batch = [next(rows, None) for rows in (intrinsic_rows, trace_rows, diagnostic_rows, build_rows)]
            if all(item is None for item in batch):
                break
            if any(item is None for item in batch):
                raise SystemExit(f"{cell}:{split}: artifact row counts differ")
            intrinsic = batch[0][1]
            trace = batch[1][1]
            diagnostic = batch[2][1]
            build = batch[3][1]
            event_id = str(trace.get("event_id") or "")
            if any(str(row.get("event_id") or "") != event_id for row in (intrinsic, diagnostic, build)):
                raise SystemExit(f"{cell}:{split}: event order mismatch near row {count + 1}")
            selected_uids = [str(value) for value in trace.get("selected_candidate_uids") or []]
            if selected_uids != visible_uids(build):
                raise SystemExit(f"{cell}:{split}:{event_id}: formal visible UID mismatch")
            if build.get("was_truncated") is not False or build.get("evidence_text_truncated") is not False:
                raise SystemExit(f"{cell}:{split}:{event_id}: formal build is truncated")
            if diagnostic.get("evidence_text_truncated") is not False:
                raise SystemExit(f"{cell}:{split}:{event_id}: diagnostic evidence text was truncated")
            realization = trace.get("realization_metadata") or {}
            if realization.get("diagnostic_surface_matches_projection") is True:
                for field in ("prompt", "prompt_input_ids", "target", "evidence_count"):
                    if diagnostic.get(field) != build.get(field):
                        raise SystemExit(f"{cell}:{split}:{event_id}: diagnostic/formal {field} drift")
            role_meta = trace.get("role_rescue_metadata") or {}
            if role_meta.get("selection_uses_gold_label") is not False:
                raise SystemExit(f"{cell}:{split}:{event_id}: gold leakage contract missing")
            if role_meta.get("selection_uses_verifier_output") is not False:
                raise SystemExit(f"{cell}:{split}:{event_id}: verifier leakage contract missing")
            event_ids.append(event_id)
            count += 1
        if count != expected or len(set(event_ids)) != expected:
            raise SystemExit(f"{cell}:{split}: rows/unique={count}/{len(set(event_ids))}, expected={expected}")
        split_event_sets[split] = set(event_ids)
    if len(set(configured_paths)) != 3:
        raise SystemExit(f"{cell}: configured split paths are not distinct")
    if (
        split_event_sets["train"] & split_event_sets["val"]
        or split_event_sets["train"] & split_event_sets["test"]
        or split_event_sets["val"] & split_event_sets["test"]
    ):
        raise SystemExit(f"{cell}: event IDs overlap across splits")
    report = json.loads((build_root / cell / "build" / "build_report.json").read_text(encoding="utf-8"))
    for split in expected_rows:
        split_report = report["splits"][split]
        if int(split_report.get("skipped_total", -1)) != 0:
            raise SystemExit(f"{cell}:{split}: build skipped rows")
        if float(split_report.get("prompt_truncation_rate", -1)) != 0.0:
            raise SystemExit(f"{cell}:{split}: non-zero formal prompt truncation")
    print(f"{cell}: strict all-split materialization gate PASS")

for split in expected_rows:
    matched = ("learned_fixed5", "cor", "opp", "ctx", "retr", "random", "full")
    iterators = {
        cell: iter_jsonl(build_root / cell / "build" / f"build_{split}.jsonl")
        for cell in matched
    }
    for row_index in range(expected_rows[split]):
        rows = {cell: next(iterator)[1] for cell, iterator in iterators.items()}
        event_ids = {str(row.get("event_id") or "") for row in rows.values()}
        evidence_counts = {int(row.get("evidence_count", -1)) for row in rows.values()}
        if len(event_ids) != 1 or len(evidence_counts) != 1:
            raise SystemExit(
                f"{split}: matched cells differ at row {row_index + 1}: "
                f"event_ids={event_ids}, evidence_counts={evidence_counts}"
            )
    print(f"{split}: seven fixed-capacity cells share event-wise visible counts")

copied_gate = build_root / "native_gate_anchor" / "build" / "build_val.jsonl"
if file_sha(copied_gate) != file_sha(native_gate):
    raise SystemExit("native gate copy differs from reference contract artifact")
role_val = event_sequence(build_root / "full" / "build" / "build_val.jsonl")
gate_val = event_sequence(copied_gate)
if role_val != gate_val or len(role_val) != expected_rows["val"]:
    raise SystemExit("native gate event sequence differs from role-rescue validation cells")

manifest_path = trace_root / "val" / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("all_ready") is not True:
    raise SystemExit("validation matrix manifest is not ready")
manifest_cells = [str(item.get("cell_id") or "") for item in manifest.get("cells") or []]
expected_cells = [*cells, "native_gate_anchor"]
if manifest_cells != expected_cells or int(manifest.get("cell_count", -1)) != len(expected_cells):
    raise SystemExit(f"validation matrix cells differ: {manifest_cells}")
if int(manifest.get("event_count", -1)) != expected_rows["val"]:
    raise SystemExit("validation matrix event count differs")
print("native equivalence gate and validation matrix manifest PASS")
PY

status "typed role-rescue verifier data ready at ${FORMAL_BUILD_ROOT}"
