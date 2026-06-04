#!/usr/bin/env python
"""Prepare a resolved RAWFC v0.6c config for a backbone migration run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


BASE_EXPERIMENTS = {
    "lora": "v0_6c_rawfc3_rule_step_adaptive5_10_eval25",
    "fullft": "v0_6c_rawfc3_rule_step_adaptive5_10_eval25_fullft",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping YAML at {path}, got {type(payload).__name__}")
    return payload


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_to_path(
    *,
    project_root: Path,
    current_path: Path,
    entry: Any,
    experiment_override: str | None,
) -> Path | None:
    if isinstance(entry, str):
        if entry == "_self_":
            return None
        if entry.startswith("/"):
            group_value = entry.lstrip("/")
            if ":" not in group_value:
                raise ValueError(f"Unsupported defaults entry {entry!r} in {current_path}")
            group, value = [part.strip() for part in group_value.split(":", 1)]
            return project_root / "configs" / group / f"{value}.yaml"
        return current_path.parent / f"{entry}.yaml"

    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ValueError(f"Unsupported defaults mapping {entry!r} in {current_path}")
        raw_group, raw_value = next(iter(entry.items()))
        group = str(raw_group).lstrip("/")
        value = str(raw_value)
        if group == "experiment" and experiment_override:
            value = experiment_override
        return project_root / "configs" / group / f"{value}.yaml"

    raise ValueError(f"Unsupported defaults entry {entry!r} in {current_path}")


def _load_with_defaults(
    *,
    project_root: Path,
    path: Path,
    experiment_override: str | None = None,
    stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Config defaults cycle detected: {cycle}")

    payload = _load_yaml(path)
    defaults = list(payload.get("defaults", []) or [])
    self_payload = {key: value for key, value in payload.items() if key != "defaults"}
    result: dict[str, Any] = {}
    saw_self = False

    for entry in defaults:
        if entry == "_self_":
            result = _merge_dicts(result, self_payload)
            saw_self = True
            continue
        default_path = _default_to_path(
            project_root=project_root,
            current_path=path,
            entry=entry,
            experiment_override=experiment_override,
        )
        if default_path is None:
            continue
        result = _merge_dicts(
            result,
            _load_with_defaults(
                project_root=project_root,
                path=default_path,
                experiment_override=None,
                stack=(*stack, path),
            ),
        )

    if not saw_self:
        result = _merge_dicts(result, self_payload)
    return result


def _compose_experiment(project_root: Path, experiment: str) -> dict[str, Any]:
    return _load_with_defaults(
        project_root=project_root,
        path=project_root / "configs" / "pipeline" / "default.yaml",
        experiment_override=experiment,
    )


def _set_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    node: dict[str, Any] = payload
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _chat_template_for_backbone(backbone: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "mode": "tokenizer_default",
        "add_generation_prompt": True,
        "template_kwargs": {},
    }
    if backbone in {"qwen3_17b", "qwen3_8b"}:
        cfg["template_kwargs"] = {"enable_thinking": False}
        cfg["thinking_control"] = "hard_enable_thinking_false"
    elif backbone == "qwen3_4b_2507":
        cfg["thinking_control"] = "native_non_thinking"
    return cfg


def _prepare_config(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    base_experiment = str(args.base_experiment or BASE_EXPERIMENTS[args.finetune])
    payload = _compose_experiment(project_root, base_experiment)
    model_path = str(args.model_path)
    case_name = str(args.case_name)

    _set_path(payload, "model_name_or_path", model_path)
    _set_path(payload, "build.prompt.model_name_or_path", model_path)
    _set_path(payload, "build.prompt.chat_template", _chat_template_for_backbone(str(args.backbone)))
    _set_path(payload, "train.model_name_or_path", model_path)
    _set_path(payload, "experiment.name", case_name)
    _set_path(payload, "baseline.variant", case_name)
    _set_path(payload, "swanlab.experiment_name", case_name)

    if args.finetune == "fullft":
        _set_path(payload, "sft_train.lora.enabled", False)
        _set_path(payload, "infer.merge_lora_cache.enabled", False)
        if float(args.size_b) >= 7.0:
            _set_path(payload, "sft_train.per_device_train_batch_size", 1)
            _set_path(payload, "sft_train.gradient_accumulation_steps", 8)

    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--size-b", type=float, required=True)
    p.add_argument("--finetune", choices=sorted(BASE_EXPERIMENTS), required=True)
    p.add_argument("--case-name", required=True)
    p.add_argument("--base-experiment", default=None)
    p.add_argument(
        "--output-root",
        default="outputs/cache/backbone_migration/configs",
        help="Directory for generated resolved configs.",
    )
    p.add_argument("--print-path", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = _project_root()
    payload = _prepare_config(args, project_root)
    output_root = Path(str(args.output_root))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_dir = output_root / str(args.finetune)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.case_name}.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    if args.print_path:
        print(output_path)
    else:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
