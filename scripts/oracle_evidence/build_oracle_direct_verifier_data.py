#!/usr/bin/env python3
"""Build verifier-ready JSONL from oracle-selected evidence.

This script bypasses learned selectors entirely.  For each claim, it takes the
oracle result's selected evidence set, renders the same prompt format used by
the normal build pipeline, and writes prebuilt ``build_<split>.jsonl`` files
that can be consumed by ``sft.label_token_trainer``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from fact_checking.build.candidates import _build_training_row, _load_prompt_tokenizer
from fact_checking.config import save_yaml
from fact_checking.data.io import load_split
from fact_checking.oracle_pointwise import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build direct verifier data from oracle evidence sets.")
    p.add_argument("--config", default="configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml")
    p.add_argument("--train-oracle-results", required=True)
    p.add_argument("--val-oracle-results", required=True)
    p.add_argument("--test-oracle-results", default=None)
    p.add_argument("--train-raw", default="data/raw/LIAR-RAW/train.json")
    p.add_argument("--val-raw", default="data/raw/LIAR-RAW/val.json")
    p.add_argument("--test-raw", default="data/raw/LIAR-RAW/test.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--expected-chunk-mmr-fingerprint", default="432dfc970e75")
    p.add_argument("--prompt-model-name-or-path", default=None)
    p.add_argument("--train-model-name-or-path", default=None)
    p.add_argument("--model-base-path", default=None)
    p.add_argument("--order", default="oracle", choices=["oracle", "hybrid", "candidate_pool"])
    p.add_argument("--filter", default="all", choices=["all", "oracle_correct", "margin_positive"])
    p.add_argument("--allow-selected-texts-fallback", action="store_true")
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_experiment_config(args.config)
    prompt_cfg = dict((cfg.get("build", {}) or {}).get("prompt", {}) or {})
    if args.prompt_model_name_or_path:
        prompt_cfg["model_name_or_path"] = args.prompt_model_name_or_path
    if args.model_base_path and prompt_cfg.get("model_name_or_path"):
        prompt_cfg["model_name_or_path"] = _resolve_model_path(
            str(prompt_cfg["model_name_or_path"]),
            args.model_base_path,
        )
    tokenizer = _load_prompt_tokenizer(str(prompt_cfg["model_name_or_path"]))

    split_specs = [
        ("train", args.train_oracle_results, args.train_raw),
        ("val", args.val_oracle_results, args.val_raw),
    ]
    if args.test_oracle_results:
        split_specs.append(("test", args.test_oracle_results, args.test_raw))

    split_paths: dict[str, str] = {}
    reports: dict[str, Any] = {}
    for split, oracle_path, raw_path in split_specs:
        rows, report = _build_split(
            split=split,
            oracle_path=Path(oracle_path),
            raw_path=Path(raw_path),
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            expected_chunk_mmr_fingerprint=args.expected_chunk_mmr_fingerprint,
            order=args.order,
            row_filter=args.filter,
            allow_selected_texts_fallback=args.allow_selected_texts_fallback,
            sample_limit=args.sample_limit,
            show_progress=not args.no_progress,
        )
        out_path = output_dir / f"build_{split}.jsonl"
        write_jsonl(out_path, rows)
        split_paths[split] = str(out_path)
        reports[split] = report

    if "test" not in split_paths:
        split_paths["test"] = split_paths["val"]

    train_config = _build_train_config(
        cfg=cfg,
        output_dir=output_dir,
        split_paths=split_paths,
        model_base_path=args.model_base_path,
        train_model_name_or_path=args.train_model_name_or_path,
    )
    train_config_path = output_dir / "train.resolved.yaml"
    save_yaml(train_config, train_config_path)

    report = {
        "config": args.config,
        "output_dir": str(output_dir),
        "expected_chunk_mmr_fingerprint": args.expected_chunk_mmr_fingerprint,
        "order": args.order,
        "filter": args.filter,
        "prompt_model_name_or_path": str(prompt_cfg["model_name_or_path"]),
        "split_paths": split_paths,
        "train_config": str(train_config_path),
        "splits": reports,
        "notes": [
            "Rows use oracle-selected evidence directly; no selector is trained or applied.",
            "selected_indices are interpreted as coordinates into each oracle row's candidate_pool.",
            "This is a gold-conditioned upper-bound diagnostic and must not be used as a deployable test protocol.",
        ],
    }
    write_json(output_dir / "build_report.json", report)

    print(f"Wrote oracle direct verifier data to {output_dir}")
    for split, split_report in reports.items():
        print(
            "{split}: rows={rows} skipped={skipped} oracle_acc={acc:.4f} trunc={trunc:.4f}".format(
                split=split,
                rows=split_report["n_rows"],
                skipped=split_report["skipped_total"],
                acc=split_report["oracle_correct_rate"],
                trunc=split_report["prompt_truncation_rate"],
            )
        )
    print(f"Train config: {train_config_path}")


def _load_experiment_config(config_path: str) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
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
        cfg = compose(
            config_name="pipeline/default",
            overrides=[f"experiment={rel.stem}"],
        )
    return dict(OmegaConf.to_container(cfg, resolve=True) or {})


def _build_split(
    *,
    split: str,
    oracle_path: Path,
    raw_path: Path,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    expected_chunk_mmr_fingerprint: str,
    order: str,
    row_filter: str,
    allow_selected_texts_fallback: bool,
    sample_limit: int | None,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_by_event = {sample.event_id: sample for sample in load_split(raw_path)}
    oracle_rows = _read_jsonl(oracle_path)
    if sample_limit is not None:
        oracle_rows = oracle_rows[: int(sample_limit)]

    out_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    fp_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    prompt_tokens: list[int] = []
    evidence_counts: list[int] = []
    oracle_correct = 0

    iterator = tqdm(
        oracle_rows,
        desc=f"oracle-direct [{split}]",
        unit="claim",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for rec in iterator:
        if not _filter_passes(rec, row_filter):
            skipped["filter"] += 1
            continue
        event_id = str(rec.get("event_id", ""))
        sample = raw_by_event.get(event_id)
        if sample is None:
            skipped["missing_raw_sample"] += 1
            continue
        metadata = dict(rec.get("candidate_pool_metadata") or {})
        fp = str(metadata.get("chunk_mmr_fingerprint") or "")
        fp_counter[fp] += 1
        if expected_chunk_mmr_fingerprint and fp != expected_chunk_mmr_fingerprint:
            raise ValueError(
                f"{split}:{event_id} chunk_mmr_fingerprint mismatch: "
                f"expected {expected_chunk_mmr_fingerprint}, got {fp}."
            )

        try:
            candidates = _oracle_selected_candidates(
                rec,
                order=order,
                allow_selected_texts_fallback=allow_selected_texts_fallback,
            )
        except ValueError as exc:
            raise ValueError(f"{split}:{event_id}: {exc}") from exc
        if not candidates:
            skipped["no_selected_evidence"] += 1
            continue

        retrieval_row = {
            "event_id": sample.event_id,
            "claim": sample.claim,
            "label": sample.label,
            "explain": sample.explain,
            "candidates": candidates,
        }
        training_row = _build_training_row(retrieval_row, tokenizer, prompt_cfg)
        training_row["oracle_direct"] = {
            "source_oracle_results": str(oracle_path),
            "candidate_pool_fingerprint": str(rec.get("candidate_pool_fingerprint") or ""),
            "chunk_mmr_fingerprint": fp,
            "search_objective": str(rec.get("search_objective") or ""),
            "oracle_is_correct": bool(rec.get("is_correct")),
            "oracle_pred_label": str(rec.get("pred_label") or ""),
            "oracle_margin": _float_or_none(rec.get("margin")),
            "oracle_selected_indices": [int(x) for x in (rec.get("selected_indices") or [])],
            "order": order,
        }
        out_rows.append(training_row)
        label_counter[str(training_row.get("gold_label", ""))] += 1
        prompt_tokens.append(int(training_row.get("prompt_token_count", 0)))
        evidence_counts.append(int(training_row.get("evidence_count", 0)))
        oracle_correct += int(bool(rec.get("is_correct")))

    report = {
        "split": split,
        "oracle_results": str(oracle_path),
        "raw_path": str(raw_path),
        "n_oracle_rows": len(oracle_rows),
        "n_rows": len(out_rows),
        "skipped": dict(skipped),
        "skipped_total": int(sum(skipped.values())),
        "labels": dict(label_counter),
        "chunk_mmr_fingerprints": dict(fp_counter),
        "oracle_correct_rate": float(oracle_correct / max(len(out_rows), 1)),
        "prompt_truncation_rate": float(
            sum(1 for row in out_rows if bool(row.get("was_truncated"))) / max(len(out_rows), 1)
        ),
        "prompt_token_count": _summary(prompt_tokens),
        "evidence_count": _summary(evidence_counts),
    }
    return out_rows, report


def _oracle_selected_candidates(
    rec: dict[str, Any],
    *,
    order: str,
    allow_selected_texts_fallback: bool,
) -> list[dict[str, Any]]:
    pool = rec.get("candidate_pool") or []
    selected_indices = [int(x) for x in (rec.get("selected_indices") or [])]
    scores_by_idx = {
        int(item.get("candidate_idx", i)): item
        for i, item in enumerate(rec.get("candidate_scores") or [])
    }
    if pool and selected_indices:
        max_idx = len(pool) - 1
        bad = [idx for idx in selected_indices if idx < 0 or idx > max_idx]
        if bad:
            raise ValueError(f"selected_indices outside candidate_pool: {bad}")
        selected: list[dict[str, Any]] = []
        for rank, idx in enumerate(selected_indices):
            candidate = dict(pool[idx])
            score = dict(scores_by_idx.get(idx, {}))
            candidate.update({
                "oracle_selected_rank": int(rank),
                "oracle_candidate_idx": int(idx),
                "oracle_hybrid_rank": int(score.get("hybrid_rank", idx)),
                "dense_score": float(score.get("dense_score", candidate.get("dense_score", 0.0))),
                "lexical_score": float(score.get("lexical_score", candidate.get("lexical_score", 0.0))),
                "bm25_score": float(score.get("bm25_score", candidate.get("bm25_score", 0.0))),
                "hybrid_score": float(score.get("hybrid_score", candidate.get("hybrid_score", 0.0))),
            })
            selected.append(candidate)
    elif allow_selected_texts_fallback:
        selected = [
            {
                "text": str(text),
                "oracle_selected_rank": int(rank),
                "oracle_candidate_idx": int(rank),
                "hybrid_score": 0.0,
            }
            for rank, text in enumerate(rec.get("selected_texts") or [])
            if str(text).strip()
        ]
    else:
        raise ValueError("missing candidate_pool/selected_indices; use --allow-selected-texts-fallback for legacy rows")

    if order == "hybrid":
        selected.sort(key=lambda item: float(item.get("hybrid_score", 0.0)), reverse=True)
    elif order == "candidate_pool":
        selected.sort(key=lambda item: int(item.get("oracle_candidate_idx", 0)))
    return selected


def _filter_passes(rec: dict[str, Any], row_filter: str) -> bool:
    if row_filter == "all":
        return True
    if row_filter == "oracle_correct":
        return bool(rec.get("is_correct"))
    if row_filter == "margin_positive":
        return float(rec.get("margin", 0.0)) > 0.0
    raise ValueError(f"Unknown filter: {row_filter}")


def _build_train_config(
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    split_paths: dict[str, str],
    model_base_path: str | None,
    train_model_name_or_path: str | None,
) -> dict[str, Any]:
    train_model = train_model_name_or_path or str((cfg.get("train", {}) or {}).get("model_name_or_path", ""))
    if model_base_path and train_model:
        train_model = _resolve_model_path(train_model, model_base_path)
    train_cfg = {
        "output_dir": str(output_dir / "train"),
        "data": {
            "train_candidates": split_paths["train"],
            "val_candidates": split_paths["val"],
            "test_candidates": split_paths["test"],
        },
        "model_name_or_path": train_model,
        "baseline": dict(cfg.get("baseline", {}) or {}),
        "sft_train": dict(cfg.get("sft_train", {}) or {}),
    }
    for key in ("tracking", "wandb", "swanlab"):
        if key in cfg:
            train_cfg[key] = cfg[key]
    if isinstance(train_cfg.get("swanlab"), dict):
        swanlab = dict(train_cfg["swanlab"])
        swanlab["experiment_name"] = str(swanlab.get("experiment_name") or "oracle_sentence_direct_verifier")
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


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
