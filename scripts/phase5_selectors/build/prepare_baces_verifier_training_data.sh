#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/models/Ministral-3-8B-Instruct-2512}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_minmax5_10.yaml}"
ABC_ROOT="${ABC_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1}"
FEATURE_ROOT="${FEATURE_ROOT:-${ABC_ROOT}/04_evidence_map}"
LEARNED_TRACE_ROOT="${LEARNED_TRACE_ROOT:-${ABC_ROOT}/05_mrec_v0_2_learned_marginal_proxy_fullpool}"
REFERENCE_BUILD_ROOT="${REFERENCE_BUILD_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${ABC_ROOT}/10_baces_verifier_training_v0_1}"
INTRINSIC_ROOT="${INTRINSIC_ROOT:-${ARTIFACT_ROOT}/factorial}"
PROMPT_FEASIBLE_ROOT="${PROMPT_FEASIBLE_ROOT:-${ARTIFACT_ROOT}/prompt_feasible}"
DIAGNOSTIC_BUILD_ROOT="${DIAGNOSTIC_BUILD_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__baces_verifier_training_diagnostic_v0_1}"
FORMAL_BUILD_ROOT="${FORMAL_BUILD_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__baces_verifier_training_v0_1}"
FROZEN_VAL_BUILD_ROOT="${FROZEN_VAL_BUILD_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__baces_factorial_prompt_feasible_v0_2__val}"

CELLS=(
  "baces_exact__ordinal_replay_minmax5_10"
  "baces_exact__matched_token_cap"
)
CONTROLLERS=(
  "ordinal_replay_minmax5_10"
  "matched_token_cap"
)

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

timestamp() {
  date '+%F %T %Z %z'
}

status() {
  printf '[%s] %s\n' "$(timestamp)" "$1"
}

raw_path() {
  printf '%s\n' "data/raw/LIAR-RAW/$1.json"
}

status "phase 1/4: materializing the targeted 1x2 intrinsic factorial"
for split in train val test; do
  "$PYTHON_BIN" scripts/phase5_selectors/build/build_baces_factorial_traces.py \
    --features "${FEATURE_ROOT}/candidate_evidence_map_features_${split}.jsonl" \
    --learned-trace "${LEARNED_TRACE_ROOT}/selection_trace_${split}.jsonl" \
    --reference-build "${REFERENCE_BUILD_ROOT}/build/build_${split}.jsonl" \
    --split "$split" \
    --output-dir "${INTRINSIC_ROOT}/${split}" \
    --selector baces_exact \
    --controller ordinal_replay_minmax5_10 \
    --controller matched_token_cap
done

status "phase 2/4: realizing diagnostic prompts for the two controller slates"
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
    --top-k 10 \
    --prompt-evidence-policy selected_set \
    --prompt-evidence-min-count 0 \
    --prompt-evidence-max-count 10 \
    --prompt-evidence-max-length-guard warn \
    --expected-chunk-mmr-fingerprint "" \
    --prompt-model-name-or-path "$MODEL_PATH" \
    --train-model-name-or-path "$MODEL_PATH" \
    --no-progress
done

status "phase 3/4: projecting the verifier-visible prompt-feasible slates"
for split in train val test; do
  "$PYTHON_BIN" scripts/phase5_selectors/build/project_baces_factorial_prompt_feasible.py \
    --factorial-dir "${INTRINSIC_ROOT}/${split}" \
    --build-root "$DIAGNOSTIC_BUILD_ROOT" \
    --output-dir "${PROMPT_FEASIBLE_ROOT}/${split}" \
    --split "$split" \
    --cell "${CELLS[0]}" \
    --cell "${CELLS[1]}" \
    --overwrite
done

status "phase 4/4: rebuilding strict all-split verifier inputs"
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
    --top-k 10 \
    --prompt-evidence-policy selected_set \
    --prompt-evidence-min-count 0 \
    --prompt-evidence-max-count 10 \
    --prompt-evidence-max-length-guard warn \
    --expected-chunk-mmr-fingerprint "" \
    --prompt-model-name-or-path "$MODEL_PATH" \
    --train-model-name-or-path "$MODEL_PATH" \
    --forbid-prompt-truncation \
    --no-progress
done

status "running cross-split, trace/build, and frozen-validation equivalence gates"
"$PYTHON_BIN" - "$PROMPT_FEASIBLE_ROOT" "$FORMAL_BUILD_ROOT" "$FROZEN_VAL_BUILD_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

trace_root, build_root, frozen_val_root = map(Path, sys.argv[1:])
cells = (
    "baces_exact__ordinal_replay_minmax5_10",
    "baces_exact__matched_token_cap",
)
expected_rows = {"train": 10065, "val": 1274, "test": 1251}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def visible_uids(row):
    count = int(row.get("evidence_count", 0))
    return [str(item.get("candidate_uid") or "") for item in row["candidates"][:count]]


def frozen_surface_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for _, row in iter_jsonl(path):
        payload = {
            "event_id": row.get("event_id"),
            "prompt": row.get("prompt"),
            "target": row.get("target"),
            "prompt_input_ids": row.get("prompt_input_ids"),
        }
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


for cell in cells:
    split_ids = {}
    config_path = build_root / cell / "train.resolved.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidate_paths = []
    for split, expected in expected_rows.items():
        trace_path = trace_root / split / cell / f"selection_trace_{split}.jsonl"
        build_path = build_root / cell / "build" / f"build_{split}.jsonl"
        configured = Path(str(cfg["data"][f"{split}_candidates"]))
        if configured.resolve() != build_path.resolve():
            raise SystemExit(f"{cell}:{split}: config path mismatch")
        candidate_paths.append(configured.resolve())
        event_ids = []
        trace_rows = iter_jsonl(trace_path)
        build_rows = iter_jsonl(build_path)
        count = 0
        while True:
            trace_item = next(trace_rows, None)
            build_item = next(build_rows, None)
            if trace_item is None or build_item is None:
                if trace_item is not None or build_item is not None:
                    raise SystemExit(f"{cell}:{split}: trace/build row count mismatch")
                break
            trace_line, trace = trace_item
            build_line, build = build_item
            event_id = str(trace.get("event_id") or "")
            if event_id != str(build.get("event_id") or ""):
                raise SystemExit(f"{cell}:{split}:{trace_line}: event mismatch")
            selected_uids = [str(value) for value in trace.get("selected_candidate_uids") or []]
            if selected_uids != visible_uids(build):
                raise SystemExit(f"{cell}:{split}:{event_id}: visible UID mismatch")
            if build.get("was_truncated") is not False:
                raise SystemExit(f"{cell}:{split}:{event_id}: prompt was truncated")
            if build.get("evidence_text_truncated") is not False:
                raise SystemExit(f"{cell}:{split}:{event_id}: evidence text was truncated")
            ids = build.get("prompt_input_ids")
            if not isinstance(ids, list) or not ids:
                raise SystemExit(f"{cell}:{split}:{event_id}: missing prompt_input_ids")
            event_ids.append(event_id)
            count += 1
        if count != expected or len(set(event_ids)) != expected:
            raise SystemExit(
                f"{cell}:{split}: rows/unique={count}/{len(set(event_ids))}, expected={expected}"
            )
        split_ids[split] = set(event_ids)
    if len(set(candidate_paths)) != 3:
        raise SystemExit(f"{cell}: configured split paths are not distinct")
    if split_ids["train"] & split_ids["val"] or split_ids["train"] & split_ids["test"] or split_ids["val"] & split_ids["test"]:
        raise SystemExit(f"{cell}: event IDs overlap across dataset splits")
    new_val = build_root / cell / "build" / "build_val.jsonl"
    old_val = frozen_val_root / cell / "build" / "build_val.jsonl"
    if frozen_surface_sha(new_val) != frozen_surface_sha(old_val):
        raise SystemExit(f"{cell}: rebuilt validation prompt surface differs from frozen v0.2")
    report = json.loads(
        (build_root / cell / "build" / "build_report.json").read_text(encoding="utf-8")
    )
    for split in expected_rows:
        split_report = report["splits"][split]
        if int(split_report.get("skipped_total", -1)) != 0:
            raise SystemExit(f"{cell}:{split}: build skipped rows")
        if float(split_report.get("prompt_truncation_rate", -1)) != 0.0:
            raise SystemExit(f"{cell}:{split}: non-zero prompt truncation rate")
    print(f"{cell}: all materialization gates PASS")
PY

status "BACES verifier training data are ready at ${FORMAL_BUILD_ROOT}"

