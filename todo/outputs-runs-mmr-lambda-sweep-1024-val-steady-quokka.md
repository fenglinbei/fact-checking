# MMR-λ sweep（max_length=1024）：val 先升后降 & val/test 不一致 诊断与方案

## Context

`scripts/pipeline/run_mmr_lambda_sweep_1024.sh` 当前跑了 3 个 λ（0.0/0.1/0.2）。
观察到两个症状：

1. **训练中 val Macro-F1 先升后降**（每个 λ 的曲线都呈倒 U 型，峰值在 epoch≈1 之后开始回落）
2. **训练结束后用 vLLM-API 推理 test，结果显著低于训练时观测到的 val 指标**

需要先把这两个症状的根因弄清楚（而不是急着调参或重训），再决定整改方案。
本计划只做**诊断与改造方案设计**，不立即执行训练或代码修改。

---

## 1. 关键事实（从代码与产物中实测）

### 1.1 训练超参（来自 `run_mmr_lambda_sweep_1024.sh` + `configs/experiment/` 继承）
- `sft_train.max_length=1024`, `num_epochs=2`, `eval_steps=save_steps=100`
- batch: `per_device_train=1`, `grad_accum=8`，bf16，LR=1e-5
- LoRA: r=16, alpha=32, dropout=0.05
- 总 600 steps ≈ 2 epoch（每 epoch ~300 step）

### 1.2 val 曲线（已读出每个 step 的 `metrics.json`）

| step | λ=0.0 macro_f1 | λ=0.1 macro_f1 | λ=0.2 macro_f1 |
|-----|-----:|-----:|-----:|
| 100 | 0.198 | 0.176 | 0.205 |
| 200 | 0.250 | 0.249 | 0.239 |
| **300** | **0.253** | **0.277 ★** | 0.265 |
| 400 | 0.245 | 0.259 | 0.263 |
| 500 | 0.253 | 0.252 | **0.264 ★** |
| 600 | 0.221 | 0.228 | 0.219 |

★ = 该 λ 的 best step。三组都在 epoch=1 末（step-300）附近达峰，第 2 个 epoch 全面回落 ~5pp。

### 1.3 train val vs vLLM-API test（best 检查点）

| λ | best val step | **best** val F1 | last (step-600) val F1 | API test F1 | API test parse_error_rate |
|---|---|---:|---:|---:|---:|
| 0.0 | 300 | 0.253 | 0.221 | **0.154** | **0.226** |
| 0.1 | 300 | 0.277 | 0.228 | **0.150** | **0.226** |
| 0.2 | 500 | 0.264 | 0.219 | **0.160** | **0.195** |

- 与 best val 比，test 掉了 ~0.10–0.13 absolute（**40%↓**）
- 训练 val 的 `parse_error_rate=0.000`，但 test 推理 `parse_error_rate≈22.6%`——这是**强信号**

### 1.4 训练 vs 推理：两条 vLLM 路径**不同**

| 维度 | 训练时的 online val<br>(`src/sft/vllm_online_eval.py`) | 训练后 test 推理<br>(`src/fact_checking/infer/api.py`) |
|---|---|---|
| 入口 | 进程内 `vllm.LLM(...).generate(...)` | 子进程 `vllm.entrypoints.openai.api_server` |
| LoRA | **`model.merge_adapter()`** 后把整体 state_dict `load_weights` 进 vLLM 模型（=dense） | **`--enable-lora --lora-modules <name>=<adapter_dir>`**（vLLM 原生动态 LoRA） |
| 端点 | Python `LLM.generate` | HTTP POST `/v1/completions` |
| logit 调整 | `SamplingParams(logits_processors=[create_logit_adjust_processor(cfg)])` | OpenAI API 的 `logit_bias`（仅 token-level bias） |
| max_tokens / temperature | `train_cfg.max_new_tokens / temperature` | `infer.max_new_tokens / temperature`（fallback 到 `baseline`） |

→ **同一 adapter，"merge 后 load" 与 "vLLM 动态 LoRA" 在 vLLM 实现里并不必然数值等价**，差异可能来自：

- vLLM LoRA 实现可能不支持训练时 `modules_to_save`（例如 `lm_head`/`embed_tokens` 的 trainable 部分被忽略）
- vLLM LoRA 的 `target_modules` 覆盖与 PEFT 配置可能不完全一致
- `--max-lora-rank` 与训练 `r` 是否对得上
- 若启用 logit_adjust：`logits_processors` ≠ `logit_bias`（前者可做归一化/先验校正，后者只是常数加在 logits 上）

### 1.5 best checkpoint 保存逻辑（`src/sft/trainer.py:621-623`）
```python
if macro_f1 > best_macro_f1:
    best_macro_f1 = macro_f1
    save_model(accelerator, model, tokenizer, output_dir / "best")
```
仅保存"截止当前 step 的 val F1 历史最高"——逻辑正确。

### 1.6 实际产物（完整目录结构）
每个 `build.retrieval.mmr_lambda-X__hash/train/` 下都持久化了：
- `best/` — LoRA adapter（`adapter_config.json` + `adapter_model.safetensors` + tokenizer 全套）
- `checkpoint-{100..600}/` — 每 100 step 一个 adapter 快照
- `final/` — step-600 的快照（与 last eval 对应）
- `eval/step-{100..600}/` — 在线 val 的 metrics + 预测
- `logit_adjust.json` — 训练时持久化的先验校正配置（推理时由 `load_logit_adjust_cfg(run_dir)` 读取）
- `tokenized_cache/` — 预 tokenize 缓存
- `config.resolved.yaml` — 推理脚本读取此处恢复 train_cfg/baseline_cfg

另外还有 `build.retrieval.mmr_lambda-0.3__f059f548/`，目前训练到 step-300 还在跑（已写出 `best/` 但 step-400+ 未到、`infer/` 未跑）。

→ **关键：B1 复现实验不需要重训**，直接对现有 `best/` 跑 `infer.split=val` 即可对照。

---

## 2. 诊断结论

### 问题 1：val 先升后降 = 经典过拟合

- 数据集小（liar-raw 约 10k train），LoRA r=16 容量足以记住
- 2 epoch 已经过头，最佳点都在 epoch 1 末
- 目前**已经**有 best-checkpoint 持久化，所以即便继续训练到 step 600 也不会丢最佳点。
- 真正的浪费：训练时间多了一倍；且**没有 early stopping with patience**——若曲线在末段刚好刷新一次"虚高"，会把 best 覆盖成噪声点
- **结论：不必"中途停训"，但应该：① 降到 1 epoch 或加 EarlyStopping callback，② 验证 `best/` 真的是峰值 step**

### 问题 2：val vs test 差距大 ≠ 单纯数据分布差异

证据：
- val/test 都是 1251 样本、同一 build 阶段、同一 prompt 模板（已在 `infer_common._load_prebuilt_samples` 与 `build_inference_context` 共用同一段代码确认）
- val 在训练时 `parse_error_rate≡0`，但同一 best ckpt 走 API 推理时**22.6%**
- 22.6% 的解析失败说明模型输出格式漂移（看到 `raw_output="A"` 而不是 `"Label: A"`）→ **LoRA 在 API 路径下没完整生效或丢失了 modules_to_save 部分**
- 该差距远超"数据集分布"能解释的范围（liar-raw 的 val/test 同源、同分布）

**根因**：训练时的 val 与推理时的 test 走了两条不同的 vLLM 路径（merge_adapter vs 原生 LoRA）。这是**"训练-推理不一致"**，不是单纯过拟合。

---

## 3. 推荐方案（按确认的优先级排序）

### 阶段 1（优先）：Q2 — 消除 val/test 引擎不一致

#### 1.1 B1 复现实验（零代码改动，~5 min）
现有 `train/best/` adapter 完好，直接把 **val** 集喂给 API 推理路径再算一次 F1。

```bash
python -m fact_checking.pipeline.run \
    experiment=mmr_lambda_sweep_1024 \
    pipeline.mode=infer \
    "build.retrieval.mmr_lambda=0.1" \
    infer.split=val
```
> 注：该环节需要先构建一个脚本，我去实际服务器运行后拿到结果在pull下来

附加诊断（一并采集）：
- 读 `train/logit_adjust.json`，确认 `enabled` 和 type；若不是纯 token-level bias，OpenAI `logit_bias` 无法等价表达，需独立改造
- `sha256sum train/best/adapter_model.safetensors train/checkpoint-300/adapter_model.safetensors`，验证 best/ 即 step-300

判读：
- **API-val F1 ≈ online-val（≈0.27, parse_error≈0）** → 引擎一致，test 低是数据分布问题，停在此处汇报 val/test gap
- **API-val F1 ≈ API-test（≈0.15, parse_error≈22%）** → 引擎差异，进入 1.2

#### 1.2 B2a 修复：推理路径合并 adapter 后再起 server

集中改动 `src/fact_checking/infer/api.py:289-375`（`_ensure_vllm_server` / `_build_vllm_command`）：

```python
# 现状（_build_vllm_command 末段）
if checkpoint_has_peft_adapter(checkpoint_dir):
    command.extend(["--enable-lora", "--max-lora-rank", str(r),
                    "--lora-modules", f"{served_model_name}={checkpoint_dir}"])

# 改造后：起 server 前先 merge
if checkpoint_has_peft_adapter(checkpoint_dir):
    merged_dir = _merge_lora_to_tmp(
        base_model=context.model_name_or_path,
        adapter_dir=checkpoint_dir,
        tokenizer_dir=checkpoint_dir,
        dtype=infer_cfg.get("dtype", "bfloat16"),
    )
    command[command.index("--model") + 1] = str(merged_dir)
    command[command.index("--tokenizer") + 1] = str(merged_dir)
    # 不再加 --enable-lora / --lora-modules / --max-lora-rank
```

新增 `_merge_lora_to_tmp(...)`：
- 用 `transformers.AutoModelForCausalLM.from_pretrained(base, torch_dtype=...)` 加载 base
- `PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()` 得到 dense
- `.save_pretrained(tempdir)`，并把 `adapter_dir` 的 tokenizer 文件 cp 进去（直接调 `tokenizer.save_pretrained(tempdir)` 更稳）
- 把临时目录路径返回；推理完成后由 `run_api_inference` 的 `finally` 清理

复用：`src/sft/runtime/adapters.py:is_peft_model` / `checkpoint_has_peft_adapter`，PEFT 标准 `merge_and_unload()` API。

附加：若 `logit_adjust.json.enabled=true` 且实现非线性，把 `logit_adjust` 强制走 logits_processors（OpenAI completions 端点接受 `extra_body={"logits_processors": [...]}`，需服务端支持）；否则保留现有的 `build_logit_bias` 即可。

### 阶段 2：Q1 — patience-based early stopping

集中改动 `src/sft/trainer.py:472-625`：

```python
best_macro_f1 = float("-inf")
patience = int(train_cfg.get("early_stopping_patience", 3))   # 新 cfg
no_improve_count = 0
should_stop = False

# 在 eval 分支末尾（save_model("best") 之后）：
if macro_f1 > best_macro_f1:
    best_macro_f1 = macro_f1
    save_model(accelerator, model, tokenizer, output_dir / "best")
    no_improve_count = 0
else:
    no_improve_count += 1
    if no_improve_count >= patience:
        if accelerator.is_main_process:
            logger.info(f"[early-stop] no val improvement for {patience} evals, stopping at step={global_step}")
        should_stop = True

# 内/外层循环检测 should_stop 并 break；保持 final/ 保存逻辑不变
```

新 config（在 `configs/experiment/mmr_lambda_sweep_1024.yaml` 或父 `b0.yaml`）：
```yaml
sft_train:
  early_stopping_patience: 3    # 3 次 eval 不刷新就停（当前 eval_steps=100，等价 patience=300 step）
```

不动 `num_epochs`（保持 2），但实际训练会在 step ~600 之前自动停。若以后 dataset/eval_steps 改变，patience 仍生效。

### 阶段 3（可选，长远）
若 B2a 验证收益后想彻底消除两条 vLLM 路径，再考虑 B2c：训练时 val 也起 sidecar OpenAI server。工程量大，不在本计划范围。

---

## 4. 关键文件清单（实施时会动到）

| 文件 | 用途 | 阶段 |
|---|---|---|
| `src/fact_checking/infer/api.py` (L289-375) | 起 server 前 `merge_and_unload()` 出临时 dense 目录；删 `--enable-lora` 分支；推理完成后清理临时目录 | 阶段 1（B2a） |
| `src/sft/trainer.py` (L460-625) | 在 eval 分支后加 `no_improve_count` / `patience` / `should_stop`；外层 epoch 循环检测 break | 阶段 2 |
| `configs/experiment/mmr_lambda_sweep_1024.yaml`（或父 `b0.yaml`） | 新增 `sft_train.early_stopping_patience: 3` | 阶段 2 |

直接复用、不需新写：
- `src/sft/runtime/adapters.py::checkpoint_has_peft_adapter / is_peft_model` —— adapter 检测
- PEFT 标准 `PeftModel.from_pretrained(...).merge_and_unload()` —— merge 实现
- `src/sft/logit_adjust.py::build_logit_bias / create_logit_adjust_processor / load_logit_adjust_cfg` —— logit_adjust 已经在两条路径都接好

---

## 5. 验证（端到端）

### Step 1 — 阶段 1.1 B1 复现实验（先跑，再决定是否做 1.2）

```bash
python -m fact_checking.pipeline.run \
    experiment=mmr_lambda_sweep_1024 \
    pipeline.mode=infer \
    "build.retrieval.mmr_lambda=0.1" \
    infer.split=val
```

对照三组数字：
| 来源 | 文件 | 已知/期望 |
|---|---|---|
| 训练时 online val | `train/eval/step-300/metrics.json` | macro_f1=0.277, parse_error=0.000 |
| API val（**本步新跑**） | `infer/val/best/.../api/metrics.json` | 待测 |
| API test（已有） | `infer/test/best/.../api/metrics.json` | macro_f1=0.150, parse_error=0.226 |

### Step 2 — 阶段 1.2 实施 B2a（如 Step 1 证明是引擎差异）

实施完后用 λ=0.0/0.1/0.2 三套 best/ 重跑 test，确认：
- API-test macro_f1 ≥ 0.22（达到训练时 last-step val 水平）
- API-test parse_error_rate < 5%
- API-val 与 online-val 差距 < 0.02（一致性回升）

### Step 3 — 阶段 2 实施 early stopping，重跑 λ=0.0/0.1/0.2 一组对照

```bash
PIPELINE_MODE=full MMR_LAMBDAS=0.1 \
    bash scripts/pipeline/run_mmr_lambda_sweep_1024.sh \
    sft_train.early_stopping_patience=3
```

接受标准：
- 训练在 step ~400 左右自动停（peak at step-300 + patience=3 × eval_steps=100 = step-600，但 step-400 后多半已经触发）
- best val F1 ≥ 当前 0.27（不损失峰值）
- 时长比 2 epoch 缩短 ≥ 30%

### Step 4 — 拓展到全 sweep（λ=0.3 ~ 1.0）
确认 Step 2、3 都满足接受标准后，再把剩下 8 个 λ 跑完，得到完整 sweep 曲线。
