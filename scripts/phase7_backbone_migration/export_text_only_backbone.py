#!/usr/bin/env python
"""Export the language tower from a multimodal backbone as a CausalLM checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig


@dataclass(frozen=True)
class ExportSpec:
    family: str
    source_prefix: str
    architecture: str
    copy_embed_to_lm_head: bool = False
    keep_quantization_extras: bool = False


EXPORT_SPECS = {
    "gemma4": ExportSpec(
        family="gemma4",
        source_prefix="model.language_model.",
        architecture="Gemma4ForCausalLM",
        copy_embed_to_lm_head=True,
    ),
    "mistral3": ExportSpec(
        family="mistral3",
        source_prefix="language_model.",
        architecture="Ministral3ForCausalLM",
        keep_quantization_extras=True,
    ),
}

TOKENIZER_FILES = [
    "chat_template.jinja",
    "generation_config.json",
    "params.json",
    "processor_config.json",
    "tekken.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping JSON at {path}")
    return payload


def _detect_family(config_payload: dict[str, Any]) -> str:
    model_type = str(config_payload.get("model_type", "")).lower()
    if model_type in EXPORT_SPECS:
        return model_type
    raise ValueError(f"Unsupported multimodal model_type={model_type!r}; supported={sorted(EXPORT_SPECS)}")


def _text_config_payload(source: Path, spec: ExportSpec) -> dict[str, Any]:
    root_payload = _read_json(source / "config.json")
    text_config = root_payload.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"{source}/config.json does not contain a text_config mapping")
    payload = dict(text_config)
    payload["architectures"] = [spec.architecture]
    payload.setdefault("dtype", root_payload.get("dtype", "bfloat16"))
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in root_payload and key not in payload:
            payload[key] = root_payload[key]
    if spec.keep_quantization_extras and isinstance(root_payload.get("quantization_config"), dict):
        payload["quantization_config"] = root_payload["quantization_config"]
    return payload


def _source_weight_files(source: Path) -> list[Path]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index_payload = _read_json(index_path)
        weight_map = index_payload.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight_map in {index_path}")
        return sorted({source / str(filename) for filename in weight_map.values()})
    single = source / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No HF safetensors weights found under {source}")


def _mapped_key(source_key: str, spec: ExportSpec) -> str | None:
    if not source_key.startswith(spec.source_prefix):
        return None
    suffix = source_key[len(spec.source_prefix) :]
    if spec.family == "gemma4":
        return "model." + suffix
    return suffix


def _quantization_extra_key(key: str) -> bool:
    return key.endswith(".activation_scale") or key.endswith(".weight_scale_inv")


def _infer_expected_state_keys(config_dir: Path) -> set[str]:
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    return set(model.state_dict().keys())


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _flush_shard(
    *,
    output_dir: Path,
    shards: list[tuple[Path, list[str]]],
    tensors: dict[str, torch.Tensor],
) -> None:
    if not tensors:
        return
    shard_path = output_dir / f"model-{len(shards) + 1:05d}.safetensors"
    save_file(tensors, str(shard_path), metadata={"format": "pt"})
    shards.append((shard_path, sorted(tensors)))
    tensors.clear()


def _iter_mapped_tensors(
    *,
    source: Path,
    spec: ExportSpec,
    expected_keys: set[str],
) -> Iterable[tuple[str, torch.Tensor, bool]]:
    emitted_lm_head = False
    for weight_file in _source_weight_files(source):
        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            for source_key in handle.keys():
                target_key = _mapped_key(source_key, spec)
                if target_key is None:
                    continue
                if target_key not in expected_keys and not (
                    spec.keep_quantization_extras and _quantization_extra_key(target_key)
                ):
                    continue
                tensor = handle.get_tensor(source_key)
                yield target_key, tensor, False
                if (
                    spec.copy_embed_to_lm_head
                    and source_key == f"{spec.source_prefix}embed_tokens.weight"
                    and "lm_head.weight" in expected_keys
                ):
                    emitted_lm_head = True
                    yield "lm_head.weight", tensor, True
    if spec.copy_embed_to_lm_head and "lm_head.weight" in expected_keys and not emitted_lm_head:
        raise RuntimeError("Could not synthesize lm_head.weight from embed_tokens.weight")


def export_text_only_checkpoint(
    *,
    source: Path,
    output: Path,
    family: str | None,
    shard_size_gb: float,
    force: bool,
    dry_run: bool = False,
) -> Path:
    source = source.resolve()
    output = output.resolve()
    root_payload = _read_json(source / "config.json")
    detected_family = family or _detect_family(root_payload)
    spec = EXPORT_SPECS[detected_family]

    if output.exists() and not dry_run:
        if not force:
            raise FileExistsError(f"Output already exists: {output}. Use --force to rebuild.")
        shutil.rmtree(output)
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)

    config_payload = _text_config_payload(source, spec)
    if dry_run:
        import tempfile

        tmp_dir_ctx = tempfile.TemporaryDirectory()
        config_dir = Path(tmp_dir_ctx.name)
        (config_dir / "config.json").write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        tmp_dir_ctx = None
        config_dir = output
        (output / "config.json").write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for filename in TOKENIZER_FILES:
            src_file = source / filename
            if src_file.exists():
                shutil.copy2(src_file, output / filename)

    expected_keys = _infer_expected_state_keys(config_dir)
    if dry_run:
        mapped: set[str] = set()
        for weight_file in _source_weight_files(source):
            with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
                for source_key in handle.keys():
                    target_key = _mapped_key(source_key, spec)
                    if target_key is None:
                        continue
                    if target_key in expected_keys or (
                        spec.keep_quantization_extras and _quantization_extra_key(target_key)
                    ):
                        mapped.add(target_key)
                    if (
                        spec.copy_embed_to_lm_head
                        and source_key == f"{spec.source_prefix}embed_tokens.weight"
                        and "lm_head.weight" in expected_keys
                    ):
                        mapped.add("lm_head.weight")
        missing = sorted(expected_keys - mapped)
        summary = {
            "source": str(source),
            "family": spec.family,
            "architecture": spec.architecture,
            "expected_keys": len(expected_keys),
            "mapped_keys": len(mapped),
            "missing_keys": missing[:20],
            "dry_run": True,
        }
        print(json.dumps(summary, indent=2))
        if tmp_dir_ctx is not None:
            tmp_dir_ctx.cleanup()
        if missing:
            raise RuntimeError(f"Dry-run missing {len(missing)} expected text weights")
        return output

    shard_limit = max(int(shard_size_gb * 1024**3), 1)
    current: dict[str, torch.Tensor] = {}
    current_size = 0
    shards: list[tuple[Path, list[str]]] = []
    written: set[str] = set()

    for target_key, tensor, separate_shard in _iter_mapped_tensors(
        source=source,
        spec=spec,
        expected_keys=expected_keys,
    ):
        tensor_size = _tensor_nbytes(tensor)
        if separate_shard:
            _flush_shard(output_dir=output, shards=shards, tensors=current)
            current_size = 0
            current[target_key] = tensor
            written.add(target_key)
            _flush_shard(output_dir=output, shards=shards, tensors=current)
            continue
        if current and current_size + tensor_size > shard_limit:
            _flush_shard(output_dir=output, shards=shards, tensors=current)
            current_size = 0
        current[target_key] = tensor
        current_size += tensor_size
        written.add(target_key)
    _flush_shard(output_dir=output, shards=shards, tensors=current)

    missing = sorted(expected_keys - written)
    if missing:
        raise RuntimeError(f"Export missing {len(missing)} expected text weights: {missing[:20]}")

    if len(shards) == 1:
        only_path, keys = shards[0]
        final_path = output / "model.safetensors"
        only_path.rename(final_path)
        shards = [(final_path, keys)]
    else:
        total = len(shards)
        weight_map: dict[str, str] = {}
        renamed: list[tuple[Path, list[str]]] = []
        for idx, (path, keys) in enumerate(shards, start=1):
            final_name = f"model-{idx:05d}-of-{total:05d}.safetensors"
            final_path = output / final_name
            path.rename(final_path)
            renamed.append((final_path, keys))
            for key in keys:
                weight_map[key] = final_name
        total_size = sum(path.stat().st_size for path, _ in renamed)
        index_payload = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (output / "model.safetensors.index.json").write_text(
            json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    summary = {
        "source": str(source),
        "family": spec.family,
        "architecture": spec.architecture,
        "written_expected_keys": len(expected_keys),
        "num_shards": len(shards),
        "shard_size_gb": shard_size_gb,
    }
    (output / "text_only_export.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", choices=sorted(EXPORT_SPECS), default=None)
    parser.add_argument("--shard-size-gb", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_text_only_checkpoint(
        source=args.source,
        output=args.output,
        family=args.family,
        shard_size_gb=float(args.shard_size_gb),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
