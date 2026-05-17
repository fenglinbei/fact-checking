from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _stringify_mapping_keys(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _stringify_mapping_keys(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_stringify_mapping_keys(value) for value in payload]
    if isinstance(payload, tuple):
        return [_stringify_mapping_keys(value) for value in payload]
    return payload


def stable_json(payload: Any) -> str:
    payload = _stringify_mapping_keys(payload)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(payload: Any, *, length: int = 12) -> str:
    digest = hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()
    return digest[:length]


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_split_paths(output_dir: Path) -> dict[str, Path]:
    return {split: output_dir / f"build_{split}.jsonl" for split in ("train", "val", "test")}


def paths_exist(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def phase_done(manifest: dict[str, Any], phase: str) -> bool:
    return manifest.get("phases", {}).get(phase, {}).get("status") == "completed"


def mark_phase(manifest: dict[str, Any], phase: str, payload: dict[str, Any]) -> dict[str, Any]:
    phases = manifest.setdefault("phases", {})
    phases[phase] = {
        "status": payload.pop("status", "completed"),
        "updated_at": now_string(),
        **payload,
    }
    return manifest
