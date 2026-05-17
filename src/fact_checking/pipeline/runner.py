from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from fact_checking.build.candidates import BUILD_LOGIC_VERSION, run_build
from fact_checking.config import save_yaml
from fact_checking.pipeline.artifacts import (
    build_split_paths,
    fingerprint,
    mark_phase,
    paths_exist,
    read_json,
    phase_done,
    write_json,
)


def _run_subprocess_tee(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    """Run a subprocess and tee its combined stdout/stderr to both the terminal and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        header = "$ " + " ".join(command) + "\n\n"
        log_file.write(header)
        log_file.flush()
        sys.stdout.write(header)
        sys.stdout.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _sync_prompt_stats(build_dir: Path, run_dir: Path) -> None:
    """Copy prompt_stats from build cache to run directory if available."""
    prompt_stats_src = build_dir / "prompt_stats"
    if not prompt_stats_src.is_dir():
        return
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    target_dir = train_dir / "prompt_stats"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(prompt_stats_src, target_dir)


@dataclass(frozen=True)
class PipelineState:
    run_dir: Path
    manifest_path: Path
    run_id: str
    build_id: str
    build_dir: Path


class PipelineRunner:
    def __init__(self, cfg: Any, *, project_root: str | Path | None = None) -> None:
        self.cfg = OmegaConf.to_container(cfg, resolve=True)
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.state = self._build_state()
        self.state.run_dir.mkdir(parents=True, exist_ok=True)
        (self.state.run_dir / "configs").mkdir(parents=True, exist_ok=True)
        (self.state.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        steps = self._resolve_steps()
        print(f"[pipeline] run_dir={self.state.run_dir}", flush=True)
        print(f"[pipeline] steps={steps}", flush=True)
        build_paths: dict[str, Path] | None = None
        infer_config_path: Path | None = None

        if "build" in steps or "train" in steps:
            build_paths = self._run_build(manifest)
            if "infer" in steps and "train" not in steps:
                infer_config_path = self._write_train_config(build_paths, self._train_dir())
        elif "infer" in steps:
            build_paths = build_split_paths(self.state.build_dir)

        if "train" in steps:
            assert build_paths is not None
            self._run_train(manifest, build_paths)

        if "infer" in steps:
            self._run_infer(manifest, config_path=infer_config_path)

        self._save_manifest(manifest)
        print(f"[pipeline] DONE run_dir={self.state.run_dir}", flush=True)
        return manifest

    def _compute_sweep_slug(self) -> str:
        try:
            from hydra.core.hydra_config import HydraConfig

            overrides = list(HydraConfig.get().overrides.task)
        except Exception:
            return ""
        keep: list[str] = []
        for ov in overrides:
            if "=" not in ov:
                continue
            key = ov.split("=", 1)[0].lstrip("+~")
            if key in ("experiment", "pipeline.mode", "pipeline.steps", "pipeline.resume", "pipeline.force.build", "pipeline.force.train", "pipeline.force.infer"):
                continue
            if key.startswith("infer.") or key.startswith("pipeline."):
                continue
            keep.append(ov.replace("=", "-"))
        return ",".join(keep)

    def _configured_output_subdir(self) -> str:
        pipeline_cfg = self.cfg.get("pipeline", {})
        output_subdir = str(pipeline_cfg.get("output_subdir", "") or "").strip()
        if not output_subdir:
            return ""
        path = Path(output_subdir)
        if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in (".", ".."):
            raise ValueError(
                "pipeline.output_subdir must be a single directory name. "
                "Use pipeline.run_dir for full or nested paths."
            )
        return output_subdir

    def _build_state(self) -> PipelineState:
        build_cfg = self.cfg["build"]
        pipeline_cfg = self.cfg.get("pipeline", {})
        experiment_name = str(self.cfg.get("experiment", {}).get("name", "default"))
        build_id = fingerprint({"build": build_cfg, "build_logic_version": BUILD_LOGIC_VERSION})
        run_id = fingerprint(
            {
                "experiment": self.cfg.get("experiment", {}),
                "build_id": build_id,
                "baseline": self.cfg.get("baseline", {}),
                "sft_train": self.cfg.get("sft_train", {}),
                "train": self.cfg.get("train", {}),
            }
        )
        run_dir_cfg = str(pipeline_cfg.get("run_dir", "") or "").strip()
        if run_dir_cfg:
            run_dir = Path(run_dir_cfg)
            if not run_dir.is_absolute():
                run_dir = self.project_root / run_dir
        else:
            output_root = Path(str(pipeline_cfg.get("output_root", "outputs/runs")))
            if not output_root.is_absolute():
                output_root = self.project_root / output_root
            output_subdir = self._configured_output_subdir()
            if output_subdir:
                leaf = f"{output_subdir}__{run_id[:8]}"
            else:
                sweep_slug = self._compute_sweep_slug()
                leaf = f"{sweep_slug}__{run_id[:8]}" if sweep_slug else run_id
            run_dir = output_root / experiment_name / leaf

        cache_root = Path(str(pipeline_cfg.get("cache_root", "outputs/cache")))
        if not cache_root.is_absolute():
            cache_root = self.project_root / cache_root
        build_dir = cache_root / "build" / build_id
        return PipelineState(
            run_dir=run_dir,
            manifest_path=run_dir / "manifest.json",
            run_id=run_id,
            build_id=build_id,
            build_dir=build_dir,
        )

    def _resolve_steps(self) -> list[str]:
        pipeline_cfg = self.cfg.get("pipeline", {})
        explicit_steps = list(pipeline_cfg.get("steps", []) or [])
        mode = str(pipeline_cfg.get("mode", "full")).strip().lower()
        if explicit_steps:
            steps = explicit_steps
        elif mode == "full":
            steps = ["build", "train", "infer"]
        elif mode == "build":
            steps = ["build"]
        elif mode == "train":
            steps = ["build", "train"]
        elif mode == "infer":
            steps = ["infer"]
        else:
            raise ValueError(f"Unsupported pipeline.mode={mode!r}. Use full, build, train, or infer.")

        allowed = {"build", "train", "infer"}
        unknown = [step for step in steps if step not in allowed]
        if unknown:
            raise ValueError(f"Unsupported pipeline step(s): {unknown}. Use build, train, infer.")
        return steps

    def _load_manifest(self) -> dict[str, Any]:
        manifest = read_json(self.state.manifest_path)
        manifest.setdefault("run_id", self.state.run_id)
        manifest.setdefault("run_dir", str(self.state.run_dir))
        manifest.setdefault("build_id", self.state.build_id)
        manifest.setdefault("phases", {})
        manifest["config"] = {
            "experiment": self.cfg.get("experiment", {}),
            "baseline": self.cfg.get("baseline", {}),
        }
        return manifest

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        write_json(self.state.manifest_path, manifest)

    def _resume_enabled(self) -> bool:
        return bool(self.cfg.get("pipeline", {}).get("resume", True))

    def _force_phase(self, phase: str) -> bool:
        return bool(self.cfg.get("pipeline", {}).get("force", {}).get(phase, False))

    def _run_build(self, manifest: dict[str, Any]) -> dict[str, Path]:
        build_paths = build_split_paths(self.state.build_dir)
        build_manifest_path = self.state.build_dir / "manifest.json"
        build_manifest = read_json(build_manifest_path)
        can_reuse = (
            self._resume_enabled()
            and not self._force_phase("build")
            and build_manifest.get("status") == "completed"
            and build_manifest.get("build_id") == self.state.build_id
            and paths_exist(build_paths)
        )
        if can_reuse:
            mark_phase(
                manifest,
                "build",
                {
                    "build_id": self.state.build_id,
                    "output_dir": str(self.state.build_dir),
                    "outputs": {key: str(value) for key, value in build_paths.items()},
                    "reused": True,
                },
            )
            self._save_manifest(manifest)
            _sync_prompt_stats(self.state.build_dir, self.state.run_dir)
            return build_paths

        result = run_build(self.cfg["build"], output_dir=self.state.build_dir)
        build_paths = {split: path.resolve() for split, path in result.split_paths.items()}
        write_json(
            build_manifest_path,
            {
                "status": "completed",
                "build_id": self.state.build_id,
                "output_dir": str(self.state.build_dir),
                "outputs": {key: str(value) for key, value in build_paths.items()},
            },
        )
        mark_phase(
            manifest,
            "build",
            {
                "build_id": self.state.build_id,
                "output_dir": str(self.state.build_dir),
                "outputs": {key: str(value) for key, value in build_paths.items()},
                "reused": False,
            },
        )
        self._save_manifest(manifest)
        _sync_prompt_stats(self.state.build_dir, self.state.run_dir)
        return build_paths

    def _run_train(self, manifest: dict[str, Any], build_paths: dict[str, Path]) -> None:
        train_dir = self._train_dir()
        best_dir = train_dir / str(self.cfg.get("train", {}).get("checkpoint_for_infer", "best"))
        can_reuse = (
            self._resume_enabled()
            and not self._force_phase("train")
            and phase_done(manifest, "train")
            and best_dir.exists()
        )
        train_config_path = self._write_train_config(build_paths, train_dir)
        if can_reuse:
            print(f"[pipeline] train: reuse checkpoint={best_dir}", flush=True)
            mark_phase(
                manifest,
                "train",
                {
                    "output_dir": str(train_dir),
                    "config_path": str(train_config_path),
                    "checkpoint": str(best_dir),
                    "reused": True,
                },
            )
            self._save_manifest(manifest)
            return

        command = self._train_command(train_config_path)
        log_path = self.state.run_dir / "logs" / "train.log"
        env = self._subprocess_env(cuda_visible_devices=str(self.cfg.get("train", {}).get("cuda_visible_devices", "")))
        print(f"[pipeline] train: launching backend={command[0]} log={log_path}", flush=True)
        _run_subprocess_tee(command, cwd=self.project_root, env=env, log_path=log_path)
        print(f"[pipeline] train: subprocess exited 0; best_dir_exists={best_dir.exists()} ({best_dir})", flush=True)

        mark_phase(
            manifest,
            "train",
            {
                "output_dir": str(train_dir),
                "config_path": str(train_config_path),
                "checkpoint": str(best_dir),
                "command": command,
                "log_path": str(log_path),
                "reused": False,
            },
        )
        self._save_manifest(manifest)

    def _run_infer(self, manifest: dict[str, Any], *, config_path: Path | None = None) -> None:
        train_dir = self._train_dir()
        infer_cfg = dict(self.cfg.get("infer", {}) or {})
        infer_kind = str(infer_cfg.get("kind", "generative")).strip().lower()
        if infer_kind == "classifier":
            from sft.classifier_infer import run_classifier_inference as run_inference
        else:
            from fact_checking.infer.api import run_api_inference as run_inference

        split = str(infer_cfg.get("split", "test"))
        checkpoint = str(infer_cfg.get("checkpoint", self.cfg.get("train", {}).get("checkpoint_for_infer", "best")))
        infer_id = fingerprint({"infer": infer_cfg})
        infer_dir = self.state.run_dir / "infer" / split / checkpoint / infer_id
        metrics_path = infer_dir / "api" / "metrics.json"
        can_reuse = (
            self._resume_enabled()
            and not self._force_phase("infer")
            and phase_done(manifest, "infer")
            and metrics_path.exists()
        )
        if can_reuse:
            print(f"[pipeline] infer: reuse metrics={metrics_path}", flush=True)
            mark_phase(
                manifest,
                "infer",
                {
                    "output_dir": str(infer_dir),
                    "infer_id": infer_id,
                    "metrics_path": str(metrics_path),
                    "reused": True,
                },
            )
            self._save_manifest(manifest)
            return

        checkpoint_dir = train_dir / checkpoint
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"[pipeline] infer: checkpoint dir not found: {checkpoint_dir}. "
                f"Train may not have saved '{checkpoint}'. Inspect logs at {self.state.run_dir / 'logs' / 'train.log'}."
            )
        print(
            f"[pipeline] infer: starting split={split} checkpoint={checkpoint_dir} infer_dir={infer_dir}",
            flush=True,
        )

        infer_config_override = str(infer_cfg.get("config_path", "") or "").strip()
        if infer_config_override:
            train_config_path = Path(infer_config_override)
            if not train_config_path.is_absolute():
                train_config_path = self.project_root / train_config_path
        elif config_path is not None:
            train_config_path = config_path
        else:
            train_config_path = train_dir.parent / "configs" / "train.resolved.yaml"
        artifacts = run_inference(
            run_dir=train_dir,
            checkpoint=checkpoint,
            split=split,
            config_path=train_config_path,
            infer_cfg=infer_cfg,
            eval_dir=infer_dir / "api",
            log_dir=self.state.run_dir / "logs",
        )
        mark_phase(
            manifest,
            "infer",
            {
                "output_dir": str(infer_dir),
                "infer_id": infer_id,
                "artifacts": artifacts,
                "reused": False,
            },
        )
        self._save_manifest(manifest)

    def _train_dir(self) -> Path:
        configured = str(self.cfg.get("train", {}).get("run_dir", "") or "").strip()
        if configured:
            path = Path(configured)
            return path if path.is_absolute() else self.project_root / path
        return self.state.run_dir / "train"

    def _write_train_config(self, build_paths: dict[str, Path], train_dir: Path) -> Path:
        train_dir.mkdir(parents=True, exist_ok=True)
        sft_train = dict(self.cfg.get("sft_train", {}) or {})
        sft_train["resolved_output_dir"] = True
        train_cfg = {
            "output_dir": str(train_dir),
            "data": {
                "train_candidates": str(build_paths["train"]),
                "val_candidates": str(build_paths["val"]),
                "test_candidates": str(build_paths["test"]),
            },
            "model_name_or_path": str(self.cfg.get("train", {}).get("model_name_or_path", "")),
            "baseline": dict(self.cfg.get("baseline", {}) or {}),
            "sft_train": sft_train,
        }
        for key in ("tracking", "wandb", "swanlab"):
            if key in self.cfg:
                train_cfg[key] = self.cfg[key]

        run_label = self._configured_output_subdir() or self._compute_sweep_slug()
        exp_name = str(self.cfg.get("experiment", {}).get("name", "default"))
        descriptive_name = f"{exp_name}__{run_label}" if run_label else exp_name
        if isinstance(train_cfg.get("swanlab"), dict):
            sl = dict(train_cfg["swanlab"])
            sl["experiment_name"] = descriptive_name
            train_cfg["swanlab"] = sl
        if isinstance(train_cfg.get("wandb"), dict):
            wb = dict(train_cfg["wandb"])
            wb["run_name"] = descriptive_name
            train_cfg["wandb"] = wb

        path = self.state.run_dir / "configs" / "train.resolved.yaml"
        save_yaml(train_cfg, path)
        return path

    def _train_command(self, train_config_path: Path) -> list[str]:
        train_cfg = self.cfg.get("train", {})
        kind = str(train_cfg.get("kind", "generative")).strip().lower()
        backend = str(train_cfg.get("backend", "accelerate_deepspeed")).strip().lower()
        if kind == "classifier":
            return [sys.executable, "-m", "sft.classifier_trainer", "--config", str(train_config_path)]
        if backend == "single":
            return [sys.executable, "-m", "sft.trainer", "--config", str(train_config_path)]
        if backend != "accelerate_deepspeed":
            raise ValueError("train.backend supports 'accelerate_deepspeed' or 'single'.")

        deepspeed_config = Path(str(train_cfg.get("deepspeed_config", "configs/deepspeed_zero2.json")))
        if not deepspeed_config.is_absolute():
            deepspeed_config = self.project_root / deepspeed_config
        mixed_precision = str(train_cfg.get("mixed_precision", "bf16"))
        return [
            "accelerate",
            "launch",
            f"--num_processes={int(train_cfg.get('nproc_per_node', 1))}",
            f"--num_machines={int(train_cfg.get('num_machines', 1))}",
            f"--mixed_precision={mixed_precision}",
            "--use_deepspeed",
            "--deepspeed_config_file",
            str(deepspeed_config),
            "-m",
            "sft.trainer",
            "--config",
            str(train_config_path),
        ]

    def _subprocess_env(self, *, cuda_visible_devices: str = "") -> dict[str, str]:
        env = os.environ.copy()
        src_path = str(self.project_root / "src")
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("PYTHONNOUSERSITE", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        if cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        for key, value in (self.cfg.get("runtime", {}) or {}).get("env", {}).items():
            env[str(key)] = str(value)
        return env
