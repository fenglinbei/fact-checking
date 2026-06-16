#!/usr/bin/env python3
"""Stage and audit sentence-trace selector sources for clean experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any


CLEAN_SELECTOR_NAME = "sentence_rule_step_adaptive5_10"
CLEAN_GRAPH_VERSION = "sentence_evidence_chain_graph"
CLEAN_ADAPTIVE_POLICY = "sentence_rule_step"

LIAR_RAW_TRACE_ROOT = Path("outputs/selectors/evidence_chain_graph")
LIAR_RAW_DEFAULT_SOURCES = {
    "train": LIAR_RAW_TRACE_ROOT / "v0_6c_adaptive5_10_train" / "selection_trace_train.jsonl",
    "val": LIAR_RAW_TRACE_ROOT / "v0_6c_adaptive5_10_val" / "selection_trace_val.jsonl",
    "test": LIAR_RAW_TRACE_ROOT / "v0_6c_adaptive5_10_test" / "selection_trace_test.jsonl",
}

DATASET_SPECS = {
    "liar_raw": {
        "expected_fingerprint": "432dfc970e75",
        "forbidden_fingerprints": set(),
        "default_sources": LIAR_RAW_DEFAULT_SOURCES,
    },
    "rawfc": {
        "expected_fingerprint": None,
        "forbidden_fingerprints": {"3b94476fd08e"},
        "default_sources": {},
    },
}


def normalize_dataset(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "liar": "liar_raw",
        "liarraw": "liar_raw",
        "liar_raw": "liar_raw",
        "rawfc": "rawfc",
        "raw_fc": "rawfc",
    }
    if normalized not in aliases:
        raise SystemExit(f"Unsupported dataset: {value}")
    return aliases[normalized]


def parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    if not splits:
        raise SystemExit("At least one split is required.")
    unknown = sorted(set(splits) - {"train", "val", "test"})
    if unknown:
        raise SystemExit(f"Unsupported split(s): {', '.join(unknown)}")
    return splits


def resolve_source_path(dataset: str, split: str, source_root: Path | None) -> Path:
    if source_root is not None:
        candidates = [
            Path(f"{source_root}_{split}") / f"selection_trace_{split}.jsonl",
            source_root / split / f"selection_trace_{split}.jsonl",
            source_root / f"selection_trace_{split}.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        joined = "\n  ".join(str(path) for path in candidates)
        raise SystemExit(f"No source trace found for {dataset}/{split}. Tried:\n  {joined}")

    default_sources: dict[str, Path] = DATASET_SPECS[dataset]["default_sources"]  # type: ignore[assignment]
    if split not in default_sources:
        raise SystemExit(
            f"No default source trace is configured for {dataset}/{split}. "
            "Pass --source-root or rebuild clean sentence sources first."
        )
    path = default_sources[split]
    if not path.exists():
        raise SystemExit(f"Default source trace does not exist: {path}")
    return path


def row_fingerprint(row: dict[str, Any]) -> str | None:
    values = [
        row.get("chunk_mmr_fingerprint"),
        row.get("fingerprint"),
        (row.get("candidate_pool_metadata") or {}).get("chunk_mmr_fingerprint"),
        (row.get("candidate_pool_metadata") or {}).get("fingerprint"),
    ]
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def iter_candidate_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("candidate_pool")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    return []


def validate_sentence_candidates(
    row: dict[str, Any],
    split: str,
    line_no: int,
    *,
    allow_multi_sentence_candidates: bool = False,
) -> tuple[int, int, int]:
    checked = 0
    bad = 0
    multi_sentence = 0
    for candidate in iter_candidate_pool(row):
        indices = candidate.get("chunk_sent_indices")
        checked += 1
        if not isinstance(indices, list) or not indices:
            bad += 1
            if bad <= 3:
                claim_id = row.get("claim_id") or row.get("id") or "<unknown>"
                raise ValueError(
                    f"{split}:{line_no} claim={claim_id} has non-sentence candidate "
                    f"chunk_sent_indices={indices!r}"
                )
            continue
        if len(indices) != 1:
            multi_sentence += 1
            if not allow_multi_sentence_candidates:
                bad += 1
                if bad <= 3:
                    claim_id = row.get("claim_id") or row.get("id") or "<unknown>"
                    raise ValueError(
                        f"{split}:{line_no} claim={claim_id} has non-sentence candidate "
                        f"chunk_sent_indices={indices!r}"
                    )
    if checked == 0:
        claim_id = row.get("claim_id") or row.get("id") or "<unknown>"
        raise ValueError(f"{split}:{line_no} claim={claim_id} has no auditable candidate_pool.")
    return checked, bad, multi_sentence


def clean_row(
    row: dict[str, Any],
    *,
    selector_name: str = CLEAN_SELECTOR_NAME,
    graph_version: str = CLEAN_GRAPH_VERSION,
    adaptive_policy: str = CLEAN_ADAPTIVE_POLICY,
) -> dict[str, Any]:
    cleaned = dict(row)
    cleaned["selector_name"] = selector_name
    cleaned["graph_version"] = graph_version
    cleaned["adaptive_policy"] = adaptive_policy

    metadata = dict(cleaned.get("candidate_pool_metadata") or {})
    metadata["selector_name"] = selector_name
    metadata["graph_version"] = graph_version
    metadata["adaptive_policy"] = adaptive_policy
    cleaned["candidate_pool_metadata"] = metadata
    return cleaned


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_split(
    *,
    dataset: str,
    split: str,
    source_path: Path,
    target_path: Path,
    sample_limit: int,
    force: bool,
    selector_name: str = CLEAN_SELECTOR_NAME,
    graph_version: str = CLEAN_GRAPH_VERSION,
    adaptive_policy: str = CLEAN_ADAPTIVE_POLICY,
    expected_fingerprint: str | None = None,
    forbidden_fingerprints: set[str] | None = None,
    allow_multi_sentence_candidates: bool = False,
) -> dict[str, Any]:
    if target_path.exists() and not force:
        return audit_existing(
            dataset=dataset,
            split=split,
            target_path=target_path,
            selector_name=selector_name,
            expected_fingerprint=expected_fingerprint,
            forbidden_fingerprints=forbidden_fingerprints,
            allow_multi_sentence_candidates=allow_multi_sentence_candidates,
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if expected_fingerprint is None:
        expected_fingerprint = DATASET_SPECS[dataset]["expected_fingerprint"]  # type: ignore[assignment]
    if forbidden_fingerprints is None:
        forbidden_fingerprints = DATASET_SPECS[dataset]["forbidden_fingerprints"]  # type: ignore[assignment]

    rows = 0
    checked_candidates = 0
    multi_sentence_candidates = 0
    fingerprints: set[str] = set()
    with source_path.open("r", encoding="utf-8") as src, target_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if sample_limit > 0 and rows >= sample_limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            fingerprint = row_fingerprint(row)
            if fingerprint:
                fingerprints.add(fingerprint)
            checked, _, multi_sentence = validate_sentence_candidates(
                row,
                split,
                line_no,
                allow_multi_sentence_candidates=allow_multi_sentence_candidates,
            )
            checked_candidates += checked
            multi_sentence_candidates += multi_sentence
            dst.write(
                json.dumps(
                    clean_row(
                        row,
                        selector_name=selector_name,
                        graph_version=graph_version,
                        adaptive_policy=adaptive_policy,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            rows += 1

    if rows == 0:
        raise SystemExit(f"No rows staged from {source_path}")
    if len(fingerprints) != 1:
        raise SystemExit(f"{source_path} has inconsistent fingerprints: {sorted(fingerprints)}")
    fingerprint = next(iter(fingerprints))
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise SystemExit(
            f"{dataset}/{split} fingerprint mismatch: expected {expected_fingerprint}, got {fingerprint}"
        )
    if fingerprint in forbidden_fingerprints:
        raise SystemExit(f"{dataset}/{split} fingerprint {fingerprint} is forbidden for sentence staging.")

    manifest = {
        "dataset": dataset,
        "split": split,
        "selector_name": selector_name,
        "graph_version": graph_version,
        "adaptive_policy": adaptive_policy,
        "chunk_mmr_fingerprint": fingerprint,
        "rows": rows,
        "sample_limit": sample_limit,
        "sentence_chunk_audit": {
            "candidate_pool_rows": rows,
            "checked_candidates": checked_candidates,
            "multi_sentence_candidates": multi_sentence_candidates,
            "allow_multi_sentence_candidates": allow_multi_sentence_candidates,
            "rule": (
                "candidate_pool.chunk_sent_indices must be non-empty; multi-sentence candidates are allowed"
                if allow_multi_sentence_candidates
                else "every candidate_pool.chunk_sent_indices must contain exactly one sentence index"
            ),
        },
        "source_file_name": source_path.name,
        "source_sha256": source_sha256(source_path),
        "staged_trace": str(target_path),
    }
    manifest_path = target_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def audit_existing(
    *,
    dataset: str,
    split: str,
    target_path: Path,
    selector_name: str = CLEAN_SELECTOR_NAME,
    expected_fingerprint: str | None = None,
    forbidden_fingerprints: set[str] | None = None,
    allow_multi_sentence_candidates: bool = False,
) -> dict[str, Any]:
    rows = 0
    checked_candidates = 0
    multi_sentence_candidates = 0
    fingerprints: set[str] = set()
    with target_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("selector_name") != selector_name:
                raise SystemExit(f"{target_path}:{line_no} has selector_name={row.get('selector_name')!r}")
            fingerprint = row_fingerprint(row)
            if fingerprint:
                fingerprints.add(fingerprint)
            checked, _, multi_sentence = validate_sentence_candidates(
                row,
                split,
                line_no,
                allow_multi_sentence_candidates=allow_multi_sentence_candidates,
            )
            checked_candidates += checked
            multi_sentence_candidates += multi_sentence
            rows += 1

    if rows == 0:
        raise SystemExit(f"Existing staged trace is empty: {target_path}")
    if len(fingerprints) != 1:
        raise SystemExit(f"Existing staged trace has inconsistent fingerprints: {sorted(fingerprints)}")
    fingerprint = next(iter(fingerprints))
    if expected_fingerprint is None:
        expected_fingerprint = DATASET_SPECS[dataset]["expected_fingerprint"]  # type: ignore[assignment]
    if forbidden_fingerprints is None:
        forbidden_fingerprints = DATASET_SPECS[dataset]["forbidden_fingerprints"]  # type: ignore[assignment]
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise SystemExit(f"{dataset}/{split} fingerprint mismatch: expected {expected_fingerprint}, got {fingerprint}")
    if fingerprint in forbidden_fingerprints:
        raise SystemExit(f"{dataset}/{split} fingerprint {fingerprint} is forbidden for sentence staging.")
    return {
        "dataset": dataset,
        "split": split,
        "selector_name": selector_name,
        "chunk_mmr_fingerprint": fingerprint,
        "rows": rows,
        "sentence_chunk_audit": {
            "candidate_pool_rows": rows,
            "checked_candidates": checked_candidates,
            "multi_sentence_candidates": multi_sentence_candidates,
            "allow_multi_sentence_candidates": allow_multi_sentence_candidates,
        },
        "staged_trace": str(target_path),
        "reused_existing_staged_trace": True,
    }


def write_env_file(path: Path, traces: dict[str, Path], fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"EXPECTED_CHUNK_MMR_FINGERPRINT={shlex.quote(fingerprint)}"]
    for split, trace_path in traces.items():
        lines.append(f"{split.upper()}_TRACE={shlex.quote(str(trace_path))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="liar_raw or rawfc")
    parser.add_argument("--output-root", default="outputs/sentence_trace_method")
    parser.add_argument("--source-root", default=None, help="Optional clean source root; supports <root>_<split> layout.")
    parser.add_argument("--selector-name", default=CLEAN_SELECTOR_NAME)
    parser.add_argument("--graph-version", default=CLEAN_GRAPH_VERSION)
    parser.add_argument("--adaptive-policy", default=CLEAN_ADAPTIVE_POLICY)
    parser.add_argument("--expected-fingerprint", default=None)
    parser.add_argument(
        "--forbidden-fingerprint",
        action="append",
        default=[],
        help="Fingerprint that must not appear; may be repeated.",
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument(
        "--allow-multi-sentence-candidates",
        action="store_true",
        help=(
            "Allow candidate_pool.chunk_sent_indices with more than one sentence index. "
            "Malformed or empty indices still fail."
        ),
    )
    args = parser.parse_args()

    dataset = normalize_dataset(args.dataset)
    splits = parse_splits(args.splits)
    output_root = Path(args.output_root)
    source_root = Path(args.source_root) if args.source_root else None
    sample_suffix = f"_sample{args.sample_limit}" if args.sample_limit > 0 else ""
    staged_root = output_root / "_sources" / dataset / f"{args.selector_name}{sample_suffix}"
    forbidden_fingerprints = set(str(item) for item in (args.forbidden_fingerprint or []))

    manifests = []
    trace_paths: dict[str, Path] = {}
    fingerprints: set[str] = set()
    for split in splits:
        target_path = staged_root / split / f"selection_trace_{split}.jsonl"
        if target_path.exists() and not args.force:
            manifest = audit_existing(
                dataset=dataset,
                split=split,
                target_path=target_path,
                selector_name=str(args.selector_name),
                expected_fingerprint=args.expected_fingerprint,
                forbidden_fingerprints=forbidden_fingerprints or None,
                allow_multi_sentence_candidates=args.allow_multi_sentence_candidates,
            )
            manifests.append(manifest)
            trace_paths[split] = target_path
            fingerprints.add(manifest["chunk_mmr_fingerprint"])
            continue
        source_path = resolve_source_path(dataset, split, source_root)
        manifest = stage_split(
            dataset=dataset,
            split=split,
            source_path=source_path,
            target_path=target_path,
            sample_limit=args.sample_limit,
            force=args.force,
            selector_name=str(args.selector_name),
            graph_version=str(args.graph_version),
            adaptive_policy=str(args.adaptive_policy),
            expected_fingerprint=args.expected_fingerprint,
            forbidden_fingerprints=forbidden_fingerprints or None,
            allow_multi_sentence_candidates=args.allow_multi_sentence_candidates,
        )
        manifests.append(manifest)
        trace_paths[split] = target_path
        fingerprints.add(manifest["chunk_mmr_fingerprint"])

    if len(fingerprints) != 1:
        raise SystemExit(f"Staged splits disagree on chunk fingerprint: {sorted(fingerprints)}")
    fingerprint = next(iter(fingerprints))

    summary = {
        "dataset": dataset,
        "source_set": f"{args.selector_name}{sample_suffix}",
        "output_root": str(staged_root),
        "chunk_mmr_fingerprint": fingerprint,
        "traces": {split: str(path) for split, path in trace_paths.items()},
        "splits": manifests,
    }
    (staged_root / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.env_file:
        write_env_file(Path(args.env_file), trace_paths, fingerprint)
    if args.print_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
