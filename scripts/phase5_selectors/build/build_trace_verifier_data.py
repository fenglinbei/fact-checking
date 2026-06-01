#!/usr/bin/env python3
"""Build verifier-ready JSONL from selector/control trace files.

The input trace format is the one emitted by selector eval scripts:
``candidate_pool`` plus ``selector_ordered_indices`` in candidate-pool
coordinates.  This lets selection-only controls be promoted to an inference
dataset without rerunning retrieval or changing the verifier pipeline.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from fact_checking.build.candidates import _build_training_row, _load_prompt_tokenizer

load_prompt_tokenizer = _load_prompt_tokenizer
build_training_row = _build_training_row
from fact_checking.config import save_yaml
from fact_checking.data.io import load_split
from fact_checking.selectors.metrics import ordered_selection_metrics, summarize_ordered_selection
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    write_json,
    write_jsonl,
)


SELECTION_MODES = (
    "trace",
    "hybrid_score_topk",
    "candidate_pool_topk",
    "same_set_hybrid_order",
    "same_set_candidate_pool_order",
    "same_set_random_order",
)
TRACE_PROMPT_STYLES = ("plain", "trace_lite")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build verifier data from selector/control traces.")
    p.add_argument("--config", default="configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml")
    p.add_argument("--train-trace", default=None)
    p.add_argument("--val-trace", default=None)
    p.add_argument("--test-trace", default=None)
    p.add_argument("--train-oracle-results", default=None)
    p.add_argument("--val-oracle-results", default=None)
    p.add_argument("--test-oracle-results", default=None)
    p.add_argument("--train-raw", default="data/raw/LIAR-RAW/train.json")
    p.add_argument("--val-raw", default="data/raw/LIAR-RAW/val.json")
    p.add_argument("--test-raw", default="data/raw/LIAR-RAW/test.json")
    p.add_argument("--dataset", default=None, help="Raw split format: liar_raw or rawfc.")
    p.add_argument("--label-schema", default=None, help="Label schema: liar6 or rawfc3.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--selection-mode", default="trace", choices=SELECTION_MODES)
    p.add_argument("--trace-prompt-style", default="plain", choices=TRACE_PROMPT_STYLES)
    p.add_argument("--expected-selector-name", default="")
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--prompt-model-name-or-path", default=None)
    p.add_argument("--train-model-name-or-path", default=None)
    p.add_argument("--model-base-path", default=None)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--val-only", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_experiment_config(args.config)
    prompt_cfg = dict((cfg.get("build", {}) or {}).get("prompt", {}) or {})
    label_schema = str(
        args.label_schema
        or prompt_cfg.get("label_schema")
        or ((cfg.get("build", {}) or {}).get("data", {}) or {}).get("label_schema")
        or cfg.get("label_schema")
        or "liar6"
    )
    prompt_cfg["label_schema"] = label_schema
    if args.prompt_model_name_or_path:
        prompt_cfg["model_name_or_path"] = args.prompt_model_name_or_path
    if args.model_base_path and prompt_cfg.get("model_name_or_path"):
        prompt_cfg["model_name_or_path"] = _resolve_model_path(
            str(prompt_cfg["model_name_or_path"]),
            args.model_base_path,
        )
    tokenizer = load_prompt_tokenizer(str(prompt_cfg["model_name_or_path"]))

    split_specs = []
    if not args.val_only:
        train_source = _resolve_split_source("train", args.train_trace, args.train_oracle_results)
        split_specs.append(("train", train_source[0], train_source[1], args.train_raw))
    val_source = _resolve_split_source("val", args.val_trace, args.val_oracle_results)
    split_specs.append(("val", val_source[0], val_source[1], args.val_raw))
    test_source = _resolve_optional_split_source(args.test_trace, args.test_oracle_results)
    if test_source is not None:
        split_specs.append(("test", test_source[0], test_source[1], args.test_raw))

    split_paths: dict[str, str] = {}
    reports: dict[str, Any] = {}
    for split, source_type, source_path, raw_path in split_specs:
        rows, report = _build_split(
            split=split,
            source_type=source_type,
            source_path=Path(source_path),
            raw_path=Path(raw_path),
            dataset=args.dataset,
            label_schema=label_schema,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            selection_mode=str(args.selection_mode),
            trace_prompt_style=str(args.trace_prompt_style),
            expected_selector_name=str(args.expected_selector_name or ""),
            top_k=int(args.top_k),
            random_seed=int(args.random_seed),
            expected_chunk_mmr_fingerprint=str(args.expected_chunk_mmr_fingerprint or ""),
            sample_limit=args.sample_limit,
            show_progress=not args.no_progress,
        )
        out_path = output_dir / f"build_{split}.jsonl"
        write_jsonl(out_path, rows)
        split_paths[split] = str(out_path)
        reports[split] = report

    if "train" not in split_paths:
        split_paths["train"] = split_paths["val"]
    if "test" not in split_paths:
        split_paths["test"] = split_paths["val"]

    train_config = _build_train_config(
        cfg=cfg,
        output_dir=output_dir,
        split_paths=split_paths,
        label_schema=label_schema,
        model_base_path=args.model_base_path,
        train_model_name_or_path=args.train_model_name_or_path,
    )
    train_config_path = output_dir / "train.resolved.yaml"
    save_yaml(train_config, train_config_path)

    report = {
        "config": args.config,
        "output_dir": str(output_dir),
        "selection_mode": args.selection_mode,
        "trace_prompt_style": args.trace_prompt_style,
        "expected_selector_name": args.expected_selector_name,
        "top_k": int(args.top_k),
        "random_seed": int(args.random_seed),
        "expected_chunk_mmr_fingerprint": args.expected_chunk_mmr_fingerprint,
        "val_only": bool(args.val_only),
        "prompt_model_name_or_path": str(prompt_cfg["model_name_or_path"]),
        "label_schema": label_schema,
        "split_paths": split_paths,
        "train_config": str(train_config_path),
        "splits": reports,
        "notes": [
            "Rows are derived from selector/control trace candidate_pool coordinates.",
            "No retrieval, selector scoring, verifier training, or oracle search is run here.",
            "Use selection_mode=same_set_random_order with multiple wrapper seeds to estimate random-order means.",
        ],
    }
    write_json(output_dir / "build_report.json", report)

    print(f"Wrote selector trace verifier data to {output_dir}")
    for split, split_report in reports.items():
        print(
            "{split}: rows={rows} skipped={skipped} trunc={trunc:.4f} "
            "mean_evidence={mean_ev:.3f}".format(
                split=split,
                rows=split_report["n_rows"],
                skipped=split_report["skipped_total"],
                trunc=split_report["prompt_truncation_rate"],
                mean_ev=float(split_report["evidence_count"].get("mean", 0.0)),
            )
        )
    print(f"Train config: {train_config_path}")


def _build_split(
    *,
    split: str,
    source_type: str,
    source_path: Path,
    raw_path: Path,
    dataset: str | None,
    label_schema: str,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    selection_mode: str,
    trace_prompt_style: str,
    expected_selector_name: str,
    top_k: int,
    random_seed: int,
    expected_chunk_mmr_fingerprint: str,
    sample_limit: int | None,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_by_event = {
        sample.event_id: sample
        for sample in load_split(raw_path, dataset=dataset, label_schema=label_schema)
    }
    source_rows = _read_jsonl(source_path)
    if sample_limit is not None:
        source_rows = source_rows[: int(sample_limit)]

    out_rows: list[dict[str, Any]] = []
    metric_traces: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    selector_names: Counter[str] = Counter()
    fp_counter: Counter[str] = Counter()
    selected_len_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    prompt_tokens: list[int] = []
    evidence_counts: list[int] = []
    evidence_counts_before: list[int] = []

    iterator = tqdm(
        source_rows,
        desc=f"trace-verifier [{split}]",
        unit="claim",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for source_row in iterator:
        trace = _normalize_source_row(source_row, source_type=source_type)
        event_id = str(trace.get("event_id") or "")
        if not event_id:
            skipped["missing_event_id"] += 1
            continue

        selector_name = str(trace.get("selector_name") or "")
        selector_names[selector_name] += 1
        if expected_selector_name and selector_name != expected_selector_name:
            raise ValueError(
                f"{split}:{event_id} selector_name mismatch: "
                f"expected {expected_selector_name!r}, got {selector_name!r}."
            )

        sample = raw_by_event.get(event_id)
        if sample is None:
            skipped["missing_raw_sample"] += 1
            continue

        fingerprint = _trace_fingerprint(trace)
        fp_counter[fingerprint] += 1
        if expected_chunk_mmr_fingerprint and fingerprint != expected_chunk_mmr_fingerprint:
            raise ValueError(
                f"{split}:{event_id} chunk_mmr_fingerprint mismatch: "
                f"expected {expected_chunk_mmr_fingerprint}, got {fingerprint}."
            )

        try:
            selected_indices = _select_indices(
                trace,
                mode=selection_mode,
                top_k=top_k,
                random_seed=random_seed,
            )
            candidates = _selected_candidates(trace, selected_indices, selection_mode=selection_mode)
        except ValueError as exc:
            raise ValueError(f"{split}:{event_id}: {exc}") from exc
        if not candidates:
            skipped["no_selected_evidence"] += 1
            continue

        if trace_prompt_style == "trace_lite":
            claim, candidates = _apply_trace_lite_prompt_fields(
                claim=sample.claim,
                candidates=candidates,
                claim_atoms=trace.get("claim_atoms") or [],
            )
        else:
            claim = sample.claim

        retrieval_row = {
            "event_id": sample.event_id,
            "claim": claim,
            "label": sample.label,
            "label_schema": label_schema,
            "explain": sample.explain,
            "candidates": candidates,
        }
        training_row = build_training_row(retrieval_row, tokenizer, prompt_cfg)
        training_row["trace_prompt_style"] = trace_prompt_style
        training_row["selector_trace"] = {
            "source_type": source_type,
            "source_path": str(source_path),
            "selector_name": selector_name,
            "selection_mode": selection_mode,
            "top_k": int(top_k),
            "random_seed": int(random_seed),
            "chunk_mmr_fingerprint": fingerprint,
            "oracle_ordered_indices": [int(x) for x in (trace.get("oracle_ordered_indices") or [])],
            "selected_indices": [int(x) for x in selected_indices],
        }
        out_rows.append(training_row)

        metrics = ordered_selection_metrics(
            [int(x) for x in (trace.get("oracle_ordered_indices") or [])],
            selected_indices,
            top_k=top_k,
        )
        metric_trace = {
            "event_id": event_id,
            "gold_label": training_row.get("gold_label", ""),
            "selector_name": selector_name,
        }
        metric_trace.update(metrics)
        metric_traces.append(metric_trace)

        selected_len_counter[str(len(selected_indices))] += 1
        label_counter[str(training_row.get("gold_label", ""))] += 1
        prompt_tokens.append(int(training_row.get("prompt_token_count", 0)))
        evidence_counts.append(int(training_row.get("evidence_count", 0)))
        evidence_counts_before.append(
            int(training_row.get("evidence_count_before", training_row.get("evidence_count", 0)))
        )

    report = {
        "split": split,
        "source_type": source_type,
        "source_path": str(source_path),
        "raw_path": str(raw_path),
        "selection_mode": selection_mode,
        "trace_prompt_style": trace_prompt_style,
        "top_k": int(top_k),
        "random_seed": int(random_seed),
        "n_source_rows": len(source_rows),
        "n_rows": len(out_rows),
        "skipped": dict(skipped),
        "skipped_total": int(sum(skipped.values())),
        "selector_names": dict(selector_names),
        "labels": dict(label_counter),
        "chunk_mmr_fingerprints": dict(fp_counter),
        "selected_index_lengths": dict(selected_len_counter),
        "selection_metrics": summarize_ordered_selection(metric_traces),
        "prompt_truncation_rate": float(
            sum(1 for row in out_rows if bool(row.get("was_truncated"))) / max(len(out_rows), 1)
        ),
        "prompt_token_count": _summary(prompt_tokens),
        "evidence_count": _summary(evidence_counts),
        "evidence_count_before": _summary(evidence_counts_before),
    }
    return out_rows, report


def _apply_trace_lite_prompt_fields(
    *,
    claim: str,
    candidates: list[dict[str, Any]],
    claim_atoms: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    atom_lines: list[str] = []
    for atom in claim_atoms:
        if not isinstance(atom, dict):
            continue
        atom_id = _compact_whitespace(atom.get("atom_id") or atom.get("node_id") or "")
        atom_text = _compact_whitespace(atom.get("text") or "")
        if atom_id and atom_text:
            atom_lines.append(f"{atom_id}: {atom_text}")

    rendered_claim = str(claim)
    if atom_lines:
        rendered_claim = f"{rendered_claim.rstrip()}\n\nClaim atoms:\n" + "\n".join(atom_lines)

    rendered_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        copied = dict(candidate)
        covers = _render_covered_atom_ids(copied.get("covered_atom_ids"))
        relation = _compact_whitespace(copied.get("map_relation") or "") or "unknown"
        directness = _compact_whitespace(copied.get("map_directness") or "") or "unknown"
        text = str(copied.get("text", "")).strip()
        copied["text"] = f"[covers={covers}; relation={relation}; directness={directness}]\n{text}"
        rendered_candidates.append(copied)
    return rendered_claim, rendered_candidates


def _render_covered_atom_ids(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = list(value) if isinstance(value, tuple) else []
    rendered = [_compact_whitespace(item) for item in items]
    rendered = [item for item in rendered if item]
    if not rendered:
        return "none"
    return ",".join(rendered)


def _compact_whitespace(value: Any) -> str:
    return " ".join(str(value).split())


def _resolve_split_source(split: str, trace_path: str | None, oracle_path: str | None) -> tuple[str, str]:
    if trace_path and oracle_path:
        raise ValueError(f"Use only one of --{split}-trace or --{split}-oracle-results.")
    if trace_path:
        return "trace", trace_path
    if oracle_path:
        return "oracle_results", oracle_path
    raise ValueError(f"--{split}-trace or --{split}-oracle-results is required.")


def _resolve_optional_split_source(
    trace_path: str | None,
    oracle_path: str | None,
) -> tuple[str, str] | None:
    if trace_path and oracle_path:
        raise ValueError("Use only one of --test-trace or --test-oracle-results.")
    if trace_path:
        return "trace", trace_path
    if oracle_path:
        return "oracle_results", oracle_path
    return None


def _normalize_source_row(row: dict[str, Any], *, source_type: str) -> dict[str, Any]:
    if source_type == "trace":
        return row
    if source_type != "oracle_results":
        raise ValueError(f"unknown source_type={source_type!r}")
    metadata = dict(row.get("candidate_pool_metadata") or {})
    return {
        "event_id": row.get("event_id", ""),
        "claim": row.get("claim", ""),
        "gold_label": row.get("gold_label", ""),
        "candidate_pool": row.get("candidate_pool") or [],
        "candidate_scores": row.get("candidate_scores") or [],
        "oracle_ordered_indices": [int(x) for x in (row.get("selected_indices") or [])],
        "selector_ordered_indices": [int(x) for x in (row.get("selected_indices") or [])],
        "selector_name": "oracle_results",
        "fingerprint": str(metadata.get("chunk_mmr_fingerprint") or ""),
        "candidate_pool_metadata": metadata,
    }


def _select_indices(
    trace: dict[str, Any],
    *,
    mode: str,
    top_k: int,
    random_seed: int,
) -> list[int]:
    pool = trace.get("candidate_pool") or []
    if not isinstance(pool, list) or not pool:
        raise ValueError("trace has no candidate_pool")
    n = len(pool)
    if mode == "trace":
        selected = _ordered_trace_indices(trace)
    elif mode == "hybrid_score_topk":
        selected = sorted(range(n), key=lambda idx: _hybrid_score(trace, idx), reverse=True)
    elif mode == "candidate_pool_topk":
        selected = list(range(n))
    elif mode in {"same_set_hybrid_order", "same_set_candidate_pool_order", "same_set_random_order"}:
        selected = _ordered_trace_indices(trace)
        selected = _dedupe_in_range(selected, n)[:top_k]
        selected_set = set(selected)
        if mode == "same_set_hybrid_order":
            selected = [
                idx
                for idx in sorted(range(n), key=lambda item: _hybrid_score(trace, item), reverse=True)
                if idx in selected_set
            ]
        elif mode == "same_set_candidate_pool_order":
            selected = [idx for idx in range(n) if idx in selected_set]
        else:
            rng = np.random.default_rng(int(random_seed))
            selected = list(selected)
            rng.shuffle(selected)
    else:
        raise ValueError(f"unknown selection mode: {mode}")

    selected = _dedupe_in_range(selected, n)[:top_k]
    if not selected:
        raise ValueError("selection produced no indices")
    return selected


def _selected_candidates(
    trace: dict[str, Any],
    selected_indices: list[int],
    *,
    selection_mode: str,
) -> list[dict[str, Any]]:
    pool = trace.get("candidate_pool") or []
    scores_by_idx = _candidate_scores_by_idx(trace)
    selected: list[dict[str, Any]] = []
    for rank, idx in enumerate(selected_indices):
        candidate = dict(pool[idx])
        score = dict(scores_by_idx.get(idx, {}))
        candidate.update(
            {
                "selector_trace_rank": int(rank),
                "selector_candidate_idx": int(idx),
                "selector_selection_mode": selection_mode,
                "candidate_idx": int(candidate.get("candidate_idx", idx)),
                "candidate_uid": str(candidate.get("candidate_uid") or score.get("candidate_uid") or ""),
                "hybrid_rank": int(score.get("hybrid_rank", idx)),
                "dense_score": _float_or_default(score.get("dense_score", candidate.get("dense_score")), 0.0),
                "lexical_score": _float_or_default(score.get("lexical_score", candidate.get("lexical_score")), 0.0),
                "bm25_score": _float_or_default(score.get("bm25_score", candidate.get("bm25_score")), 0.0),
                "hybrid_score": _float_or_default(score.get("hybrid_score", candidate.get("hybrid_score")), 0.0),
            }
        )
        if "selector_score" in score:
            candidate["selector_score"] = _float_or_default(score.get("selector_score"), 0.0)
        if "sequential_selected_step" in score:
            candidate["sequential_selected_step"] = int(score["sequential_selected_step"])
        selected.append(candidate)
    return selected


def _ordered_trace_indices(trace: dict[str, Any]) -> list[int]:
    raw = trace.get("selector_ordered_indices") or []
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _dedupe_in_range(indices: list[int], n: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= n or idx in seen:
            continue
        seen.add(idx)
        out.append(int(idx))
    return out


def _hybrid_score(trace: dict[str, Any], idx: int) -> float:
    scores_by_idx = _candidate_scores_by_idx(trace)
    score = scores_by_idx.get(idx, {})
    pool = trace.get("candidate_pool") or []
    candidate = pool[idx] if idx < len(pool) and isinstance(pool[idx], dict) else {}
    return _float_or_default(score.get("hybrid_score", candidate.get("hybrid_score")), 0.0)


def _candidate_scores_by_idx(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for fallback_idx, item in enumerate(trace.get("candidate_scores") or []):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("candidate_idx", fallback_idx))
        except (TypeError, ValueError):
            idx = fallback_idx
        out[idx] = item
    return out


def _trace_fingerprint(trace: dict[str, Any]) -> str:
    if trace.get("fingerprint"):
        return str(trace.get("fingerprint"))
    metadata = trace.get("candidate_pool_metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("chunk_mmr_fingerprint") or "")
    return ""


def _load_experiment_config(config_path: str) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    project_root = Path(__file__).resolve().parents[3]
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    experiment_dir = project_root / "configs" / "experiment"
    try:
        rel = path.resolve().relative_to(experiment_dir.resolve())
    except ValueError:
        cfg = OmegaConf.load(path)
        return dict(OmegaConf.to_container(cfg, resolve=True) or {})
    if len(rel.parts) != 1:
        cfg = OmegaConf.load(path)
        return dict(OmegaConf.to_container(cfg, resolve=True) or {})
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(project_root / "configs")):
        cfg = compose(config_name="pipeline/default", overrides=[f"experiment={rel.stem}"])
    return dict(OmegaConf.to_container(cfg, resolve=True) or {})


def _build_train_config(
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    split_paths: dict[str, str],
    label_schema: str | None,
    model_base_path: str | None,
    train_model_name_or_path: str | None,
) -> dict[str, Any]:
    train_model = train_model_name_or_path or str((cfg.get("train", {}) or {}).get("model_name_or_path", ""))
    if model_base_path and train_model:
        train_model = _resolve_model_path(train_model, model_base_path)
    sft_train = dict(cfg.get("sft_train", {}) or {})
    resolved_label_schema = str(
        label_schema
        or sft_train.get("label_schema")
        or cfg.get("label_schema")
        or ((cfg.get("build", {}) or {}).get("prompt", {}) or {}).get("label_schema")
        or ((cfg.get("build", {}) or {}).get("data", {}) or {}).get("label_schema")
        or "liar6"
    )
    sft_train["label_schema"] = resolved_label_schema
    sft_train["resolved_output_dir"] = True
    train_cfg = {
        "label_schema": resolved_label_schema,
        "output_dir": str(output_dir / "train"),
        "data": {
            "train_candidates": split_paths["train"],
            "val_candidates": split_paths["val"],
            "test_candidates": split_paths["test"],
        },
        "model_name_or_path": train_model,
        "baseline": dict(cfg.get("baseline", {}) or {}),
        "sft_train": sft_train,
    }
    train_cfg["baseline"]["label_schema"] = resolved_label_schema
    for key in ("tracking", "wandb", "swanlab"):
        if key in cfg:
            train_cfg[key] = cfg[key]
    if isinstance(train_cfg.get("swanlab"), dict):
        swanlab = dict(train_cfg["swanlab"])
        swanlab["experiment_name"] = str(swanlab.get("experiment_name") or "selector_trace_verifier")
        train_cfg["swanlab"] = swanlab
    return train_cfg


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_model_path(raw: str, base_path: str) -> str:
    if raw.startswith("/data/models/"):
        return raw.replace("/data/models/", base_path.rstrip("/") + "/", 1)
    return raw


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


if __name__ == "__main__":
    main()
