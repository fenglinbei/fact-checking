# v0.6d Trace-Lite Verifier Prompt Ablation Plan

日期：2026-06-01

状态：implementation plan

## 1. 目标

在不改变 v0.6d selector、候选池、evidence 数量和 evidence 顺序的前提下，新增一个 verifier prompt ablation：

```text
plain verifier prompt
vs
trace-lite verifier prompt
```

核心问题是：

```text
selected evidence 固定时，轻量 evidence-map 结构标签是否能帮助 verifier 更好利用证据？
```

这里的 trace-lite 只使用 downstream 已有、oracle-free 的 map 字段，不生成 rationale，不暴露 selector rule，不加入 scores / oracle / gold 信息。

## 2. 背景与约束

v0.5c 的 prompt diagnostic 已经证明 `map_full` 对旧 verifier 是明显 OOD：同一 oracle evidence 下，`map_full` 相比 `plain_original` macro-F1 下降约 0.35。因此 v0.6d 不应复用 full map prompt。

本轮 trace-lite 的设计原则是：

1. 只给 verifier 很短的结构提示。
2. 原始 evidence text 仍是主体。
3. 不让模型读取 selector policy trace。
4. 不改变 selection trace schema。
5. 不重新跑 evidence-map API。
6. full pipeline 仍使用 `SOURCE_TYPE=trace`、`TRACE_SELECTION_MODE=trace`。

## 3. Prompt 设计

### 3.1 Baseline plain prompt

当前 full pipeline 通过 `build_trace_verifier_data.py` 选出 trace evidence 后，调用通用 `build_training_row()`，最终 prompt 形态近似：

```text
Claim:
<claim>

Evidence:
[1] <raw evidence text>
[2] <raw evidence text>
...
```

### 3.2 新增 trace-lite prompt

trace-lite 仍复用现有 label instruction、label definitions、chat template 和 auto truncation 逻辑，但将 claim 与 candidate text 做轻量增强。

Claim 增强：

```text
<original claim>

Claim atoms:
A1: <atom text>
A2: <atom text>
...
```

Evidence text 增强：

```text
[covers=A1,A2; relation=support; directness=direct]
<original evidence text>
```

若 evidence 没覆盖 atom：

```text
[covers=none; relation=background; directness=context]
<original evidence text>
```

### 3.3 允许字段

trace-lite 只使用：

- `trace.claim_atoms[].atom_id`
- `trace.claim_atoms[].text`
- selected candidate `covered_atom_ids`
- selected candidate `map_relation`
- selected candidate `map_directness`
- selected candidate `text`

### 3.4 禁止字段

不得渲染进 prompt：

- `gold_label`
- `oracle_ordered_indices`
- `oracle_ordered_keys`
- oracle metrics，例如 `jaccard@5` / `recall@5`
- `selector_score`
- `selection_steps`
- `adaptive_stop_reason`
- `sufficiency_state`
- `P1/P2/P3` rule name
- `candidate_uid` / `candidate_key`
- retrieval / fusion / map scores

## 4. 实现计划

### 4.1 `build_trace_verifier_data.py`

新增 CLI 参数：

```bash
--trace-prompt-style plain|trace_lite
```

默认值：

```text
plain
```

保持所有现有 pipeline 行为不变。

在 `_build_split()` 中，当前流程是：

```python
candidates = _selected_candidates(trace, selected_indices, selection_mode=selection_mode)
retrieval_row = {
    "event_id": sample.event_id,
    "claim": sample.claim,
    "label": sample.label,
    "explain": sample.explain,
    "candidates": candidates,
}
training_row = build_training_row(retrieval_row, tokenizer, prompt_cfg)
```

改为：

```python
candidates = _selected_candidates(trace, selected_indices, selection_mode=selection_mode)
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
    "explain": sample.explain,
    "candidates": candidates,
}
training_row = build_training_row(retrieval_row, tokenizer, prompt_cfg)
training_row["trace_prompt_style"] = trace_prompt_style
```

新增 helper：

```python
def _apply_trace_lite_prompt_fields(
    *,
    claim: str,
    candidates: list[dict[str, Any]],
    claim_atoms: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    ...
```

helper 行为：

- claim atoms 按 trace 中顺序渲染。
- atom text 去掉换行并压缩 whitespace。
- candidate 复制后只改 `text` 字段，不改原始 candidate metadata。
- header 格式固定为：

```text
[covers=<ids>; relation=<relation>; directness=<directness>]
<text>
```

- `covered_atom_ids` 为空时用 `none`。
- relation / directness 为空时用 `unknown`。

`build_report.json` 增加：

```json
"trace_prompt_style": "plain|trace_lite"
```

split report 增加：

```json
"trace_prompt_style": "plain|trace_lite"
```

### 4.2 `run_selector_trace_full_pipeline.sh`

新增环境变量：

```bash
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-plain}"
```

build-data args 追加：

```bash
--trace-prompt-style "${TRACE_PROMPT_STYLE}"
```

打印日志中增加：

```text
[selector-trace-full] trace_prompt_style=<style>
```

### 4.3 v0.6d ablation runner

不改变 `run_v0_6d_sufficiency_contradiction_all_pipelines.sh` 默认行为。默认仍为 `plain`。

新增一个薄 wrapper：

```text
scripts/phase5_selectors/run/run_v0_6d_trace_lite_prompt_ablation_all_pipelines.sh
```

该 wrapper 固定：

```bash
TRACE_PROMPT_STYLE=trace_lite
LORA_CASE_NAME=v0_6d_sufficiency_contradiction_trace_lite
FULLFT_CASE_NAME=v0_6d_sufficiency_contradiction_trace_lite_fullft
```

其余参数沿用 v0.6d：

```bash
SOURCE_TYPE=trace
TRACE_SELECTION_MODE=trace
TOP_K=10
EXPECTED_SELECTOR_NAME=v0_6d_sufficiency_contradiction_adaptive5_10
TRAIN_TRACE=outputs/selectors/evidence_chain_graph/v0_6d_sufficiency_contradiction_train/selection_trace_train.jsonl
VAL_TRACE=outputs/selectors/evidence_chain_graph/v0_6d_sufficiency_contradiction_val/selection_trace_val.jsonl
```

## 5. 实验矩阵

主对照：

| selector | prompt style | train mode | case name |
| --- | --- | --- | --- |
| v0.6d | plain | LoRA | `v0_6d_sufficiency_contradiction_adaptive5_10` |
| v0.6d | trace_lite | LoRA | `v0_6d_sufficiency_contradiction_trace_lite` |
| v0.6d | plain | FullFT | `v0_6d_sufficiency_contradiction_adaptive5_10_fullft` |
| v0.6d | trace_lite | FullFT | `v0_6d_sufficiency_contradiction_trace_lite_fullft` |

可选 follow-up：

```text
trace_lite_no_atoms
```

只在 trace-lite 提升但 truncation 明显升高时再做。该版本只给 evidence header，不在 claim 后追加 `Claim atoms`。

## 6. 评估指标

主指标：

- macro-F1
- accuracy
- half-true F1
- mostly-true F1
- true-side F1 / macro-F1-plus-true-side

健康指标：

- prompt truncation rate
- evidence_count mean / p50 / p95
- evidence_count_before mean
- prompt_token_count mean / p95 / max
- selected_index_lengths

对照解释：

- 若 trace-lite 指标提升且 truncation 持平：说明轻量 map structure 帮助 verifier evidence use。
- 若 trace-lite 指标下降但 truncation 持平：说明即便轻量结构也存在 prompt OOD。
- 若 trace-lite 指标下降且 truncation 上升：需要跑 `trace_lite_no_atoms` 判断是否为 token overhead。
- 若 LoRA 提升但 FullFT 不提升：可能是 LoRA 更依赖提示格式，FullFT 已能从 raw evidence 学到结构。

## 7. 测试计划

Unit tests：

- `plain` 默认输出与当前 prompt 完全一致。
- `trace_lite` 会在 claim 中追加 `Claim atoms`。
- `trace_lite` 会在 selected evidence text 前追加 `covers/relation/directness` header。
- 缺失 atom / relation / directness 时 fallback 为 `none` / `unknown`，不报错。
- forbidden fields 不出现在 rendered prompt 中。
- `build_report.json` 与 split report 包含 `trace_prompt_style`。

Smoke checks：

```bash
SAMPLE_LIMIT=20 RUN_GRAPH_BUILD=false RUN_TRAIN=false RUN_INFER=false \
TRACE_PROMPT_STYLE=trace_lite \
OUTPUT_ROOT=outputs/selector_trace_verifier/smoke_v0_6d_trace_lite \
bash scripts/phase5_selectors/run/run_v0_6d_sufficiency_contradiction_all_pipelines.sh
```

检查：

- LoRA / FullFT build-data 均成功。
- `build_report.json` 中 `trace_prompt_style=trace_lite`。
- prompt snapshot 中包含 `Claim atoms:` 和 evidence header。
- `prompt_truncation_rate` 不异常升高。

Full acceptance：

- trace-lite LoRA / FullFT 都产出 `build_report.json`、`train.resolved.yaml`、checkpoint、val inference metrics。
- plain v0.6d 目录不被覆盖。
- selector trace 不需要重建。

## 8. 运行方式

默认 plain v0.6d：

```bash
bash scripts/phase5_selectors/run/run_v0_6d_sufficiency_contradiction_all_pipelines.sh
```

trace-lite ablation：

```bash
bash scripts/phase5_selectors/run/run_v0_6d_trace_lite_prompt_ablation_all_pipelines.sh
```

或手动指定：

```bash
TRACE_PROMPT_STYLE=trace_lite \
LORA_CASE_NAME=v0_6d_sufficiency_contradiction_trace_lite \
FULLFT_CASE_NAME=v0_6d_sufficiency_contradiction_trace_lite_fullft \
bash scripts/phase5_selectors/run/run_v0_6d_sufficiency_contradiction_all_pipelines.sh
```

## 9. Assumptions

- v0.6d selector trace 已存在，并且 selector name 为 `v0_6d_sufficiency_contradiction_adaptive5_10`。
- evidence map 字段来自 v0.6b compact map，不重新跑 API。
- 本 ablation 只比较 verifier prompt rendering，不改变 selected set。
- trace-lite 不作为默认主方法，先作为 ablation；若显著提升且 truncation 健康，再考虑升级为主 pipeline 默认。
