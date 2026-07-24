#!/usr/bin/env python3
"""Fail-closed artifact and process checks for the RAWFC seed43 tail run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


SPLITS = ("train", "val", "test")


def _emit(payload: dict[str, Any], exit_code: int = 0) -> int:
    print(json.dumps(payload, sort_keys=True))
    return exit_code


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_contract(
    config: Path,
    *,
    expected_seed: int,
    allow_missing_seed: bool,
) -> tuple[bool, dict[str, Any]]:
    payload = _load_yaml(config)
    train = payload.get("sft_train") or {}
    if not isinstance(train, dict):
        return False, {"path": str(config), "reason": "sft_train_not_mapping"}
    recorded = train.get("seed")
    if recorded is None and allow_missing_seed:
        effective = 42
        source = "trainer_default"
    else:
        effective = recorded
        source = "resolved_config"
    valid = type(effective) is int and effective == expected_seed
    return valid, {
        "path": str(config),
        "recorded": recorded,
        "effective": effective,
        "expected": expected_seed,
        "source": source,
    }


def _build_contract(candidate: Path, canonical: Path | None) -> tuple[bool, dict[str, Any]]:
    result: dict[str, Any] = {
        "candidate_root": str(candidate),
        "canonical_root": None if canonical is None else str(canonical),
        "splits": {},
    }
    valid = True
    for split in SPLITS:
        candidate_path = candidate / "build" / f"build_{split}.jsonl"
        item: dict[str, Any] = {"candidate_path": str(candidate_path)}
        if not candidate_path.is_file() or candidate_path.stat().st_size <= 0:
            item.update(status="missing_candidate")
            valid = False
            result["splits"][split] = item
            continue
        candidate_sha = _sha256(candidate_path)
        item["candidate_sha256"] = candidate_sha
        if canonical is None:
            item["status"] = "ready"
        else:
            canonical_path = canonical / "build" / f"build_{split}.jsonl"
            item["canonical_path"] = str(canonical_path)
            if not canonical_path.is_file() or canonical_path.stat().st_size <= 0:
                item["status"] = "missing_canonical"
                valid = False
            else:
                canonical_sha = _sha256(canonical_path)
                item["canonical_sha256"] = canonical_sha
                item["status"] = "matched" if candidate_sha == canonical_sha else "mismatch"
                valid = valid and candidate_sha == canonical_sha
        result["splits"][split] = item
    result["status"] = "ready" if valid else "invalid"
    return valid, result


def _metric_contract(path: Path, expected_samples: int) -> tuple[bool, dict[str, Any]]:
    payload = _load_json(path)
    fields = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    numeric = all(
        isinstance(payload.get(field), (int, float)) and not isinstance(payload.get(field), bool)
        for field in fields
    )
    valid = payload.get("num_samples") == expected_samples and numeric
    return valid, {
        "path": str(path),
        "num_samples": payload.get("num_samples"),
        **{field: payload.get(field) for field in fields},
    }


def _prediction_contract(path: Path, expected_samples: int) -> tuple[bool, dict[str, Any]]:
    count = 0
    invalid_lines: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                invalid_lines.append(line_no)
                continue
            count += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines.append(line_no)
                continue
            if not isinstance(payload, dict):
                invalid_lines.append(line_no)
    valid = count == expected_samples and not invalid_lines
    return valid, {
        "path": str(path),
        "num_predictions": count,
        "expected_predictions": expected_samples,
        "invalid_lines": invalid_lines[:20],
        "sha256": _sha256(path),
    }


def command_config(args: argparse.Namespace) -> int:
    try:
        valid, detail = _seed_contract(
            args.config,
            expected_seed=args.expected_seed,
            allow_missing_seed=args.allow_missing_seed,
        )
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        return _emit({"status": "invalid", "error": str(exc)}, 3)
    return _emit({"status": "ready" if valid else "invalid", "seed": detail}, 0 if valid else 3)


def command_builds(args: argparse.Namespace) -> int:
    try:
        valid, detail = _build_contract(args.candidate_root, args.canonical_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _emit({"status": "invalid", "error": str(exc)}, 4)
    return _emit(detail, 0 if valid else 4)


def command_full(args: argparse.Namespace) -> int:
    root = args.root
    marker = root / "train" / "training_complete.json"
    resolved = root / "train.resolved.yaml"
    try:
        complete = _load_json(marker)
        marker_valid = (
            complete.get("completed") is True
            and isinstance(complete.get("global_step"), (int, float))
            and not isinstance(complete.get("global_step"), bool)
            and complete["global_step"] > 0
        )
        seed_valid, seed_detail = _seed_contract(
            resolved,
            expected_seed=args.expected_seed,
            allow_missing_seed=args.allow_missing_seed,
        )
        metrics: dict[str, Any] = {}
        metrics_valid = True
        for split in ("val", "test"):
            valid, detail = _metric_contract(
                root / "eval" / split / "best" / "label_token" / "metrics.json",
                args.expected_num_samples,
            )
            metrics[split] = detail
            metrics_valid = metrics_valid and valid
        builds_valid, builds = _build_contract(root, args.canonical_build_root)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return _emit(
            {"status": "incomplete", "root": str(root), "error": str(exc)},
            3,
        )
    valid = marker_valid and seed_valid and metrics_valid and builds_valid
    return _emit(
        {
            "status": "complete" if valid else "incomplete",
            "root": str(root),
            "training_complete": {
                "path": str(marker),
                "completed": complete.get("completed"),
                "global_step": complete.get("global_step"),
            },
            "seed": seed_detail,
            "metrics": metrics,
            "builds": builds,
        },
        0 if valid else 3,
    )


def command_best_adapter(args: argparse.Namespace) -> int:
    root = args.root
    resolved = root / "train.resolved.yaml"
    marker = root / "train" / "training_complete.json"
    adapter = root / "train" / "best" / "adapter_model.safetensors"
    try:
        seed_valid, seed_detail = _seed_contract(
            resolved,
            expected_seed=args.expected_seed,
            allow_missing_seed=False,
        )
        marker_missing = not marker.exists()
        adapter_size = adapter.stat().st_size
        adapter_sha = _sha256(adapter)
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        return _emit(
            {"status": "invalid", "root": str(root), "error": str(exc)},
            3,
        )
    adapter_valid = adapter_size > 0 and adapter_sha == args.expected_adapter_sha
    valid = seed_valid and marker_missing and adapter_valid
    return _emit(
        {
            "status": "ready" if valid else "invalid",
            "root": str(root),
            "seed": seed_detail,
            "training_complete": {
                "path": str(marker),
                "missing": marker_missing,
            },
            "adapter": {
                "path": str(adapter),
                "size_bytes": adapter_size,
                "sha256": adapter_sha,
                "expected_sha256": args.expected_adapter_sha,
                "matched": adapter_valid,
            },
        },
        0 if valid else 3,
    )


def command_export(args: argparse.Namespace) -> int:
    root = args.root
    output = root / "eval" / args.split / "best" / "label_token"
    metrics_path = output / "metrics.json"
    predictions_path = output / f"{args.split}_predictions.jsonl"
    try:
        metric_valid, metric_detail = _metric_contract(
            metrics_path, args.expected_num_samples
        )
        metrics = _load_json(metrics_path)
        parse_error_rate = metrics.get("parse_error_rate")
        parse_valid = (
            isinstance(parse_error_rate, (int, float))
            and not isinstance(parse_error_rate, bool)
            and parse_error_rate == 0
        )
        prediction_valid, prediction_detail = _prediction_contract(
            predictions_path, args.expected_num_samples
        )
        metrics_sha = _sha256(metrics_path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        return _emit(
            {
                "status": "incomplete",
                "root": str(root),
                "split": args.split,
                "error": str(exc),
            },
            4,
        )
    valid = metric_valid and parse_valid and prediction_valid
    metric_detail.update(
        {
            "parse_error_rate": parse_error_rate,
            "parse_failures": 0 if parse_valid else None,
            "sha256": metrics_sha,
        }
    )
    return _emit(
        {
            "status": "complete" if valid else "incomplete",
            "root": str(root),
            "split": args.split,
            "metrics": metric_detail,
            "predictions": prediction_detail,
        },
        0 if valid else 4,
    )


def _read_argv(path: Path) -> tuple[str, ...]:
    return tuple(
        token.decode("utf-8", errors="replace")
        for token in path.read_bytes().split(b"\0")
        if token
    )


def _is_accelerate_launch(argv: tuple[str, ...]) -> bool:
    for index, token in enumerate(argv):
        if Path(token).name == "accelerate" and index + 1 < len(argv):
            if argv[index + 1] == "launch":
                return True
    return False


def command_accelerate(args: argparse.Namespace) -> int:
    matches: list[dict[str, Any]] = []
    try:
        entries = list(args.proc_root.iterdir())
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return _emit({"status": "invalid", "error": str(exc)}, 5)
    for entry in entries:
        if not entry.name.isdigit() or not entry.is_dir():
            continue
        try:
            argv = _read_argv(entry / "cmdline")
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if _is_accelerate_launch(argv):
            matches.append({"pid": int(entry.name), "argv": list(argv)})
    matches.sort(key=lambda item: int(item["pid"]))
    return _emit(
        {"status": "ready", "count": len(matches), "processes": matches},
        0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    config = subparsers.add_parser("config")
    config.add_argument("--config", type=Path, required=True)
    config.add_argument("--expected-seed", type=int, required=True)
    config.add_argument("--allow-missing-seed", action="store_true")
    config.set_defaults(func=command_config)

    builds = subparsers.add_parser("builds")
    builds.add_argument("--candidate-root", type=Path, required=True)
    builds.add_argument("--canonical-root", type=Path)
    builds.set_defaults(func=command_builds)

    full = subparsers.add_parser("full")
    full.add_argument("--root", type=Path, required=True)
    full.add_argument("--expected-seed", type=int, required=True)
    full.add_argument("--allow-missing-seed", action="store_true")
    full.add_argument("--expected-num-samples", type=int, default=200)
    full.add_argument("--canonical-build-root", type=Path)
    full.set_defaults(func=command_full)

    best_adapter = subparsers.add_parser("best-adapter")
    best_adapter.add_argument("--root", type=Path, required=True)
    best_adapter.add_argument("--expected-seed", type=int, required=True)
    best_adapter.add_argument("--expected-adapter-sha", required=True)
    best_adapter.set_defaults(func=command_best_adapter)

    export = subparsers.add_parser("export")
    export.add_argument("--root", type=Path, required=True)
    export.add_argument("--split", choices=("val", "test"), required=True)
    export.add_argument("--expected-num-samples", type=int, default=200)
    export.set_defaults(func=command_export)

    accelerate = subparsers.add_parser("accelerate")
    accelerate.add_argument("--proc-root", type=Path, default=Path("/proc"))
    accelerate.set_defaults(func=command_accelerate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
