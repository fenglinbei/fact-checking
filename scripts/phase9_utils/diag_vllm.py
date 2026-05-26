#!/usr/bin/env python
"""诊断 vLLM 推理启动问题。用法: PYTHONPATH=src python scripts/phase9_utils/diag_vllm.py"""

import json
import os
import sys
import traceback
from pathlib import Path

# 在项目根目录运行
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / "src"))

# ---- 配置 ----
RUN_DIR = Path(
    "outputs/runs/mmr_lambda_sweep_1024/build.retrieval.mmr_lambda-0.0__36a69f69"
)
TRAIN_DIR = RUN_DIR / "train"
LOG_DIR = RUN_DIR / "logs"
CONFIG_PATH = RUN_DIR / "configs" / "train.resolved.yaml"


def step1_check_logit_adjust():
    print("=" * 60)
    print("1. 检查 logit_adjust.json")
    logit_path = TRAIN_DIR / "logit_adjust.json"
    print(f"   path={logit_path}")
    print(f"   exists={logit_path.exists()}")
    if not logit_path.exists():
        return
    try:
        with open(logit_path) as f:
            cfg = json.load(f)
        print(f"   keys: {list(cfg.keys())}")
        print(f"   tau={cfg.get('tau')}")
        print(f"   letter_token_ids={cfg.get('letter_token_ids')}")
        print(f"   log_priors={cfg.get('log_priors')}")
    except Exception:
        print("   ERROR:")
        traceback.print_exc()


def step2_build_logit_bias():
    print()
    print("=" * 60)
    print("2. 测试 build_logit_bias")
    from sft.logit_adjust import build_logit_bias, load_logit_adjust_cfg

    try:
        cfg = load_logit_adjust_cfg(TRAIN_DIR)
        print(f"   cfg is None: {cfg is None}")
        if cfg is not None:
            bias = build_logit_bias(cfg)
            print(f"   logit_bias: {bias}")
    except Exception:
        print("   ERROR:")
        traceback.print_exc()


def step3_build_vllm_command():
    print()
    print("=" * 60)
    print("3. 测试 _build_vllm_command")
    from fact_checking.infer.api import _build_vllm_command
    from sft.infer_common import build_inference_context

    ctx = build_inference_context(
        run_dir=str(TRAIN_DIR),
        checkpoint="best",
        split="test",
        config_path=str(CONFIG_PATH),
    )
    print(f"   model_name_or_path={ctx.model_name_or_path}")
    print(f"   checkpoint_dir={ctx.checkpoint_dir}")
    print(f"   is_peft={ctx.is_peft_adapter}")
    print(f"   max_length={ctx.max_length}")

    infer_cfg = {
        "provider": "vllm_openai",
        "host": "127.0.0.1",
        "port": 35001,
        "served_model_name": "fact-checking-sft",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
        "dtype": "auto",
        "cuda_visible_devices": "0,1,2,3",
        "server": {"manage": True, "stop_after_infer": True, "extra_args": []},
    }
    try:
        cmd = _build_vllm_command(context=ctx, infer_cfg=infer_cfg)
        print("   command:")
        for i, arg in enumerate(cmd):
            print(f"     [{i}] {arg}")
    except Exception:
        print("   ERROR:")
        traceback.print_exc()

    return ctx


def step4_ensure_vllm_server(ctx):
    print()
    print("=" * 60)
    print("4. 测试 _ensure_vllm_server（实际启动 vLLM）")
    from fact_checking.infer.api import _ensure_vllm_server

    infer_cfg = {
        "provider": "vllm_openai",
        "host": "127.0.0.1",
        "port": 35001,
        "wait_seconds": 300,
        "served_model_name": "fact-checking-sft",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
        "dtype": "auto",
        "cuda_visible_devices": "0,1,2,3",
        "server": {"manage": True, "stop_after_infer": True, "extra_args": []},
    }
    log_path = LOG_DIR / "vllm_diag.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"   log_path={log_path}")

    proc = None
    try:
        proc = _ensure_vllm_server(
            context=ctx, infer_cfg=infer_cfg, log_path=log_path
        )
        print(f"   SUCCESS: vLLM server started, pid={proc.pid}")
    except Exception as exc:
        print(f"   ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print()
        if log_path.exists():
            size = log_path.stat().st_size
            print(f"   vllm log exists (size={size}):")
            print(log_path.read_text()[:3000])
        else:
            print("   vllm log does NOT exist!")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()


def main():
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    step1_check_logit_adjust()
    step2_build_logit_bias()
    ctx = step3_build_vllm_command()
    step4_ensure_vllm_server(ctx)
    print()
    print("=" * 60)
    print("诊断完成")


if __name__ == "__main__":
    main()
