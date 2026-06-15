# AA-QEC 三阶段实验实现设计

## 0. 设计目标

本文把 `aa_qec_roadmap_experiment_plan.md` 中的 AA-QEC 路线落成一组可实现、可审计、可复用的三阶段实验实现方案。

当前已知实验信号是：

```text
RAWFC: QEC-MIN > plain
RAWFC: QEC-MAP ≈ plain
```

因此第一版升级不继续增加 verifier prompt 字段，也不把 `relation`、`directness`、`confidence` 等 map 信息默认暴露给 verifier。主线保持：

```text
prompt = qec_min
每条 evidence 只渲染:
Check: <cue_text>
<evidence_text>
```

升级重点从 prompt 转移到 evidence chain construction：

```text
claim atoms 决定验证骨架；
QD questions 决定自然语言 cue；
evidence sentence 本身作为 answer；
map relation/directness/confidence 只用于内部选链和 diagnostics。
```

第一版工程目标不是立即完成所有论文实验，而是让以下三阶段都能稳定产出标准 trace/source，并复用现有 `sentence_trace_method` LoRA 训练入口：

```text
Stage 1: AA-QEC-View
Stage 2: AA-QEC-Constrained
Stage 3: AA-QEC-Full
```

---

## 1. 范围与非目标

### 1.1 第一版纳入范围

第一版实现只覆盖以下内容：

```text
1. 新增 AA-QEC trace/source 构造逻辑；
2. 让 AA-QEC 输出与现有 selection_trace pipeline 兼容；
3. 让 qec_min/qec_map prompt builder 优先使用 trace.chain_steps 中的 cue_text；
4. 提供 RAWFC 三阶段单 seed 运行 wrapper；
5. 提供 build sanity 和 dry-run/test 验证；
6. 输出足够的 chain diagnostics，支持后续实验报告。
```

### 1.2 第一版不纳入范围

以下内容后置，不混入第一版三阶段实现：

```text
1. LIAR-RAW 全量迁移；
2. 多 seed confirmation；
3. cue policy ablation 全矩阵；
4. qec_map/no-relation/shuffled-relation 系列诊断；
5. case study HTML 可视化；
6. 改动 LoRA trainer 或 label-token infer 主逻辑；
7. 把 AA-QEC 直接并入旧 v0.7 budgeted marginal objective。
```

这些后置项依赖三阶段 trace 稳定后再展开。

---

## 2. 推荐实现方案

采用 **trace-first incremental** 方案：

```text
新增 AA-QEC selector/source 生成层；
输出标准 selection_trace_*.jsonl；
训练继续复用 run_lora_matrix.sh / run_one.sh；
prompt 继续复用 build_trace_verifier_data.py 的 qec_min/qec_map。
```

不推荐把 AA-QEC 直接塞进 `evidence_chain_graph.py`，原因是该文件已经承载 v0.6/v0.7 多套 selector 逻辑，继续堆叠会让 primary/secondary/fallback 规则和原有 budgeted marginal objective 纠缠在一起。

不推荐只在 `build_trace_verifier_data.py` 中做 AA-QEC order/filter，原因是这样 AA-QEC 不是一个可独立审计的 selector/source，后续 diagnostics、case study、source staging、实验索引都会变得不干净。

---

## 3. 文件边界

### 3.1 新增核心 selector 文件

新增：

```text
src/fact_checking/selectors/atom_anchored_qec.py
```

职责：

```text
1. 定义 AA-QEC 参数 dataclass；
2. 从 evidence-map candidate row 构造 atom-aware chain；
3. 支持 Stage 1/2/3 三种 candidate scope / selection policy；
4. 输出 graph row 与 selection_trace；
5. 计算 chain diagnostics；
6. 不做文件 IO，不调用训练逻辑。
```

建议核心 API：

```python
@dataclass(frozen=True)
class AtomAnchoredQECParams:
    candidate_top_n: int = 20
    min_chain_steps: int = 5
    max_chain_steps: int = 10
    max_secondary_per_atom: int = 1
    min_secondary_confidence: float = 0.4
    cue_policy: str = "qd_prefer"
    candidate_scope: str = "selected"  # selected | top20
    selection_policy: str = "keep_all_reorder"
    source_selector_name: str = "v0_7_budgeted_marginal_chain_adaptive5_10"


def atom_anchored_qec_selector_name(params: AtomAnchoredQECParams) -> str:
    ...


def build_atom_anchored_qec_row(row: dict[str, Any], *, params: AtomAnchoredQECParams) -> dict[str, Any]:
    ...
```

### 3.2 新增 build 脚本

新增：

```text
scripts/phase5_selectors/build/build_atom_anchored_qec.py
```

职责：

```text
1. 读取 v0.7 staged trace 或 evidence-map candidate source；
2. 调用 atom_anchored_qec.py；
3. 写出 chain_graph_{split}.jsonl；
4. 写出 selection_trace_{split}.jsonl；
5. 写出 graph_diagnostics.json；
6. 写出 manifest.json。
```

### 3.3 新增 run wrapper

新增：

```text
scripts/phase5_selectors/run/run_atom_anchored_qec.sh
scripts/sentence_trace_method/prepare_aa_qec_sources.sh
scripts/sentence_trace_method/run_aa_qec_three_stage_ministral3.sh
```

职责：

```text
run_atom_anchored_qec.sh:
    单 split 构造 AA-QEC graph/trace。

prepare_aa_qec_sources.sh:
    对 train/val/test 构造 AA-QEC trace；
    调用 stage_sources.py 进入 outputs/sentence_trace_method/_sources；
    保持 selector_name / graph_version / adaptive_policy 可审计。

run_aa_qec_three_stage_ministral3.sh:
    展开 RAWFC Stage 1/2/3 实验矩阵；
    统一设置 Ministral prompt_input_ids、LoRA、EBS、LR、epoch、eval cadence；
    支持 MODE=build|train|eval|full 和 DRY_RUN=true。
```

### 3.4 修改现有 prompt builder

修改：

```text
scripts/phase5_selectors/build/build_trace_verifier_data.py
```

目标：

```text
qec_min/qec_map 优先使用 trace.chain_steps；
如果 trace 没有 chain_steps，则保留当前 selected_candidates + QD route fallback 逻辑。
```

这样 AA-QEC 和现有 v0.7 QEC-MIN 能共用 prompt style。

---

## 4. Trace schema

AA-QEC trace 必须继续提供现有 verifier builder 依赖的字段：

```json
{
  "event_id": "...",
  "claim": "...",
  "gold_label": "...",
  "selector_name": "aa_qec_view_keep_all_qd_prefer_selected_min5_10",
  "graph_version": "atom_anchored_qec_v1",
  "fingerprint": "432dfc970e75",
  "candidate_pool_metadata": {
    "chunk_mmr_fingerprint": "432dfc970e75",
    "selector_name": "aa_qec_view_keep_all_qd_prefer_selected_min5_10",
    "graph_version": "atom_anchored_qec_v1",
    "adaptive_policy": "aa_qec_view"
  },
  "candidate_pool": [],
  "candidate_scores": [],
  "selector_ordered_indices": [0, 3, 5],
  "selected_indices": [0, 3, 5],
  "oracle_ordered_indices": [],
  "selected_candidates": [],
  "claim_atoms": [],
  "chain_steps": [],
  "chain_diagnostics": {}
}
```

每个 `chain_steps` 元素保存完整信息，但 prompt 只读取 `cue_text` 和 evidence text：

```json
{
  "step": 1,
  "atom_id": "A1",
  "atom_text": "...",
  "cue_text": "...",
  "cue_source": "qd_question",
  "candidate_idx": 3,
  "evidence_id": "E03",
  "evidence_text": "...",
  "role": "primary",
  "relation": "support",
  "directness": "direct",
  "map_confidence": 0.82,
  "evidence_map_quality_score": 0.74,
  "from_qd": true,
  "qd_question_id": "q2",
  "qd_question_rank": 1,
  "qd_question_hybrid_score": 0.71,
  "covered_by_previous_step": false,
  "anchor_step": 0
}
```

Prompt 渲染只使用：

```text
Check: {cue_text}
{evidence_text}
```

不渲染：

```text
role
relation
directness
confidence
score
source type
route rank
```

---

## 5. AA-QEC 规则设计

### 5.1 Atom 顺序

Atom 顺序使用 `claim_atoms` 的原始顺序。若 atom 带有 `importance`，只用于同 atom 内候选 tie-break，不改变主遍历顺序。

这样可以避免把 chain 顺序变成另一个隐式加权 objective。

### 5.2 Cue policy

第一版默认：

```text
cue_policy = qd_prefer
```

选择规则：

```text
1. 如果 candidate 有可用 qd_question_routes，选择最佳 question；
2. 如果没有可用 question，使用当前 atom text；
3. 如果没有 covered atom，fallback 到 Verify the main factual claim.
```

可用 question 的过滤规则：

```text
1. question 非空；
2. question 与 claim 不完全相同；
3. question 至少包含 4 个英文词；
4. 排除 "Is this claim true?" 这类过泛化问题。
```

最佳 question 排序：

```text
rank 小优先；
rank 相同 hybrid_score 高优先；
仍相同 question_id 字典序优先。
```

### 5.3 Primary evidence

对每个 atom，候选集合为：

```text
candidate.covered_atom_ids contains atom_id
```

Primary 排序采用 lexicographic priority：

```text
direct support/refute
> partial support/refute
> direct qualify/mixed
> partial qualify/mixed
> context support/refute
> context qualify/mixed
> background/context fallback
```

同级 tie-break：

```text
map_confidence 高优先
> evidence_map_quality_score 高优先
> qd_max_question_hybrid 高优先
> base_score 高优先
> 原 candidate rank 靠前优先
```

### 5.4 Secondary evidence

每个 atom 最多加入一条 secondary evidence。

若 primary 是：

```text
support -> secondary 可选 refute / qualify / mixed
refute -> secondary 可选 support / qualify / mixed
qualify/mixed -> secondary 可选 support / refute
```

过滤条件：

```text
1. 不是 primary 本身；
2. 不是 duplicate；
3. directness 至少为 partial，除非没有其他候选；
4. map_confidence >= 0.4；
5. 如果超过 max_chain_steps，secondary 优先被丢弃。
```

### 5.5 Multi-atom 去重

同一 evidence 只进入 prompt 一次。

如果某条 evidence 覆盖多个 atoms：

```text
1. 第一次被选中时进入 chain；
2. 它覆盖的 atom 全部标记为 covered；
3. 后续 atom 若最佳 evidence 已使用，则尝试找新的补充 evidence；
4. 若没有新的有效 evidence，则记录 covered_by_previous_step。
```

### 5.6 Budget

第一版默认：

```text
candidate_top_n = 20
min_chain_steps = 5
max_chain_steps = 10
max_secondary_per_atom = 1
min_secondary_confidence = 0.4
```

构造流程：

```text
1. 按 atom 顺序选择 primary evidence；
2. 对有冲突/限定价值的 atom 加 secondary evidence；
3. 去重；
4. 若 chain < min_chain_steps，用 fallback evidence 补足；
5. 若 chain > max_chain_steps，裁剪到 max_chain_steps。
```

Fallback 补足优先级：

```text
1. v0.7 order 中尚未使用的 evidence；
2. 全局高 map_quality / directness / base_score evidence；
3. 避免 duplicate 和 irrelevant/background evidence。
```

裁剪优先级：

```text
保留 primary；
保留 direct support/refute；
保留覆盖未解决 atom 的 evidence；
保留 qualifier/counter；
丢弃 fallback；
丢弃 background/context；
丢弃重复 source / duplicate。
```

---

## 6. 三阶段实验设计

### 6.1 Stage 1: AA-QEC-View

目标：

```text
在同一批 v0.7 selected evidence 上，只改变 order/cue assignment，验证 atom anchoring 是否有价值。
```

实现配置：

```text
candidate_scope = selected
selection_policy = keep_all_reorder
cue_policy = qd_prefer
min_chain_steps = inherited
max_chain_steps = inherited
prompt = qec_min
```

RAWFC 单 seed 第一轮：

| ID | Source | Evidence set | Order | Prompt | 说明 |
|---|---|---|---|---|---|
| B0 | v0.7 adaptive5_10 | v0.7 selected | v0.7 greedy | qec_min | 当前 QEC-MIN baseline |
| O1 | AA-QEC-View | v0.7 selected | atom order | qec_min | 只测 atom order |
| O2 | AA-QEC-View | v0.7 selected | atom order + primary-before-secondary | qec_min | 更强 chain view |
| O3 | AA-QEC-View | v0.7 selected | shuffled | qec_min | 负对照 |

进入 Stage 2 条件：

```text
O1/O2 >= B0 + 0.005 selection 或 macro_f1；
或 O1/O2 与 B0 持平但 O3 明显下降。
```

### 6.2 Stage 2: AA-QEC-Constrained

目标：

```text
候选仍限制在 v0.7 selected evidence 内，但用 atom primary / secondary / fallback 规则重构 chain。
```

实现配置：

```text
candidate_scope = selected
min_chain_steps = 5
max_chain_steps = 10
cue_policy = qd_prefer
prompt = qec_min
```

RAWFC 单 seed第一轮：

| ID | Selection policy | Budget | Prompt | 说明 |
|---|---|---|---|---|
| C0 | keep_all | 原 v0.7 | qec_min | baseline |
| C1 | primary_only | max10 | qec_min | 去冗余 primary chain |
| C2 | primary_secondary | max10 | qec_min | qualifier/counter step |
| C3 | primary_secondary_fallback_min5 | min5/max10 | qec_min | 主候选 |
| C4 | primary_fallback_min5_no_secondary | min5/max10 | qec_min | 隔离 secondary 贡献 |

进入 Stage 3 条件：

```text
C2 或 C3 >= C0。
```

若 C 系列全部下降，则暂缓 Stage 3 全量训练，但仍可做 build sanity 以定位原因。

### 6.3 Stage 3: AA-QEC-Full

目标：

```text
从 top20 candidate pool 直接构造 AA-QEC，检查是否能替代 v0.7 adaptive5_10。
```

实现配置：

```text
candidate_scope = top20
cue_policy = qd_prefer
prompt = qec_min
```

RAWFC 单 seed 第一轮：

| ID | Selection policy | Budget | Prompt | 说明 |
|---|---|---|---|---|
| F0 | v0.7 adaptive5_10 | min5/max10 | qec_min | 当前主线 baseline |
| F1 | primary_only_fallback_min5 | min5/max10 | qec_min | 纯 atom primary |
| F2 | primary_secondary_fallback_min5 | min5/max10 | qec_min | 完整 AA-QEC |
| F3 | primary_secondary_dynamic | no min5/max10 | qec_min | 动态链长 |

后置 ablation：

| ID | Selection policy | Budget | Prompt | 说明 |
|---|---|---|---|---|
| F4 | primary_secondary_fallback_min5 | min5/max10 | plain | 区分 selector 与 QEC prompt 收益 |
| F5 | primary_secondary_fallback_min5 | min5/max10 | qec_map | 检查 map-visible 是否在 AA-QEC 下有效 |

AA-QEC 成为主方法候选的条件：

```text
F2 >= F0 + 0.005 selection 或 macro_f1；
并且后续多 seed 均值仍优于 F0。
```

如果 F2 与 F0 持平，但 chain diagnostics 更好、prompt 更短、case study 更清楚，则 AA-QEC 可作为 interpretability-oriented 主方法候选，性能表述需要保守。

---

## 7. 运行口径

### 7.1 第一版 RAWFC 训练配置

RAWFC 三阶段第一轮沿用当前 QEC-MIN 口径：

```text
dataset = rawfc
model = ministral3_8b
TRACE_PROMPT_STYLE = qec_min
LoRA r = 16
LoRA alpha = 32
LoRA dropout = 0.05
EBS = 16
SFT_GRADIENT_ACCUMULATION_STEPS = 4
DEEPSPEED_CONFIG = configs/deepspeed_zero2_bsz1_ga4.json
SFT_LEARNING_RATE = 1e-5
SFT_NUM_TRAIN_EPOCHS = 10
SFT_EVAL_STEPS = 50
SFT_SAVE_STEPS = 50
SFT_EARLY_STOPPING_PATIENCE = 8
REQUIRE_PROMPT_INPUT_IDS = true
```

RAWFC 主指标：

```text
eval/val/best/label_token/metrics.json
```

Test 只在 val 选出候选后做最终确认。

### 7.2 后续 LIAR-RAW 迁移口径

LIAR-RAW 后续迁移时沿用现有 Ministral transfer 口径：

```text
dataset = liar_raw
model = ministral3_8b
EBS = 16
SFT_GRADIENT_ACCUMULATION_STEPS = 4
SFT_LEARNING_RATE = 2e-5
SFT_NUM_TRAIN_EPOCHS = 12
SFT_EVAL_STEPS = 100
SFT_SAVE_STEPS = 100
SFT_EARLY_STOPPING_PATIENCE = 8
REQUIRE_PROMPT_INPUT_IDS = true
LIAR_CLASS_WEIGHTS = current Ministral transfer defaults
```

LIAR-RAW 主指标：

```text
eval/val/best/label_token_logit_adjust_tau0p75/metrics.json
```

---

## 8. Build sanity 指标

每个 AA-QEC source build 必须在 `graph_diagnostics.json` 和 `build_report.json` 中能追踪以下指标：

```text
n_rows
skipped_total
prompt_token_count.mean / p50 / p90 / p95 / max
prompt_truncation_rate
evidence_count.mean
evidence_count_before.mean
chain_steps.mean
atom_coverage_rate
uncovered_atom_rate
qd_cue_rate
atom_cue_rate
claim_fallback_rate
repeated_question_rate
duplicate_evidence_rate
secondary_step_rate
fallback_fill_rate
multi_atom_evidence_rate
primary_step_count.mean
secondary_step_count.mean
fallback_step_count.mean
```

训练前硬性检查：

```text
1. train/val/test row count 与 source trace 一致；
2. selector_name 与 expected-selector-name 一致；
3. fingerprint 没有漂移；
4. prompt_input_ids 存在；
5. prompt_truncation_rate 不显著高于当前 qec_min baseline；
6. qd_cue_rate / atom_cue_rate / fallback_rate 不为异常值；
7. O1/O2/C/F 系列 selected indices 均在 candidate_pool 坐标范围内。
```

---

## 9. 测试与验证

### 9.1 单元测试

新增：

```text
src/fact_checking/selectors/test_atom_anchored_qec.py
```

覆盖：

```text
1. Stage 1 keep_all_reorder 保留 selected set 但改变 order；
2. shuffled order 使用固定 random_seed 可复现；
3. primary 选择遵循 direct support/refute 优先；
4. secondary 选择捕捉 counter/qualifier；
5. duplicate evidence 不重复进入 prompt；
6. fallback 能补足 min_chain_steps；
7. max_chain_steps 裁剪优先丢弃 fallback/background；
8. top20 scope 不依赖 v0.7 selected set；
9. trace selector_ordered_indices 与 candidate_pool 坐标一致；
10. chain_steps cue_text 与 selected evidence 顺序一致。
```

扩展：

```text
scripts/phase5_selectors/build/test_build_trace_verifier_data.py
```

覆盖：

```text
1. trace.chain_steps 存在时 qec_min 使用 chain_steps.cue_text；
2. trace.chain_steps 不存在时保留当前 QD route / atom fallback 行为；
3. qec_map 仍只在 prompt 中显示 covers/relation/directness；
4. chain_steps 中 role/relation/directness/confidence 不泄漏进 qec_min prompt。
```

扩展：

```text
scripts/sentence_trace_method/test_experiment_matrix_scripts.py
```

覆盖：

```text
1. run_aa_qec_three_stage_ministral3.sh 的 DRY_RUN 展开 RAWFC Stage 1/2/3；
2. RAWFC 使用 lr1e-5/ep10/eval50/pat8/EBS16；
3. REQUIRE_PROMPT_INPUT_IDS=true；
4. CASE_SUFFIX 与 selector_name 可区分 O/C/F 各实验；
5. 默认不展开 LIAR-RAW、多 seed、F4/F5 后置实验。
```

### 9.2 命令级验证

最小验证命令：

```bash
PYTHONPATH=.:src /data/liaozijie/conda/accelerate-fc/bin/python -m pytest \
  src/fact_checking/selectors/test_atom_anchored_qec.py \
  scripts/phase5_selectors/build/test_build_trace_verifier_data.py \
  scripts/sentence_trace_method/test_experiment_matrix_scripts.py -v

bash -n scripts/phase5_selectors/run/run_atom_anchored_qec.sh
bash -n scripts/sentence_trace_method/prepare_aa_qec_sources.sh
bash -n scripts/sentence_trace_method/run_aa_qec_three_stage_ministral3.sh

DRY_RUN=true MODE=build RUN_STAGE=1 bash scripts/sentence_trace_method/run_aa_qec_three_stage_ministral3.sh
```

Build smoke：

```bash
PYTHON_BIN=/data/liaozijie/conda/accelerate-fc/bin/python \
DATASETS=rawfc \
SPLITS=val \
SAMPLE_LIMIT=5 \
STAGES=stage1 \
MODE=build \
bash scripts/sentence_trace_method/run_aa_qec_three_stage_ministral3.sh
```

---

## 10. 命名约定

建议 selector name 可读且包含关键实验轴：

```text
aa_qec_view_keep_all_qd_prefer_selected_min5_10
aa_qec_view_primary_secondary_order_qd_prefer_selected_min5_10
aa_qec_constrained_primary_only_qd_prefer_selected_max10
aa_qec_constrained_primary_secondary_fallback_qd_prefer_selected_min5_10
aa_qec_full_primary_only_fallback_qd_prefer_top20_min5_10
aa_qec_full_primary_secondary_fallback_qd_prefer_top20_min5_10
aa_qec_full_primary_secondary_dynamic_qd_prefer_top20
```

对应 `CASE_SUFFIX` 使用更短形式：

```text
__aa_qec_o1_view_atom_order
__aa_qec_o2_view_primary_secondary_order
__aa_qec_o3_view_shuffled
__aa_qec_c1_primary
__aa_qec_c2_primary_secondary
__aa_qec_c3_primary_secondary_fallback
__aa_qec_c4_primary_fallback_no_secondary
__aa_qec_f1_full_primary
__aa_qec_f2_full_primary_secondary
__aa_qec_f3_full_dynamic
```

RAWFC LoRA suffix 保持：

```text
_lora_ebs16_lr1em5_ep10_eval50_pat8_rawfc
```

---

## 11. 成功标准

### 11.1 工程成功标准

第一版实现完成的标准：

```text
1. RAWFC Stage 1/2/3 均可生成 train/val/test AA-QEC source；
2. build_trace_verifier_data.py 能从 AA-QEC source 构造 qec_min build rows；
3. build rows 包含 prompt_input_ids；
4. dry-run 能展开全部 RAWFC 单 seed 三阶段矩阵；
5. 单元测试和 bash -n 通过；
6. build sanity 报告包含 chain diagnostics；
7. 现有 qec_min/qec_map 测试不回归。
```

### 11.2 实验推进标准

Stage 1 到 Stage 2：

```text
O1/O2 >= B0 + 0.005 selection 或 macro_f1；
或 O3 明显下降，证明 order 有影响。
```

Stage 2 到 Stage 3：

```text
C2 或 C3 >= C0。
```

AA-QEC 进入多 seed：

```text
F2 >= F0 + 0.005 selection 或 macro_f1；
或 F2 与 F0 持平但 diagnostics/case study 明显更优。
```

---

## 12. 风险与应对

### 12.1 QD route 字段缺失

风险：

```text
candidate_pool 中 qd_question_routes 可能被 _candidate_trace_output 过滤掉。
```

应对：

```text
1. AA-QEC trace 输出必须保留 qd_question_routes、qd_max_question_hybrid、from_qd、qd_pool_rank；
2. 若老 trace 缺失 QD route，cue fallback 到 atom text；
3. build sanity 报告 qd_cue_rate，低于预期时先不训练。
```

### 12.2 prompt truncation 上升

风险：

```text
qec_min 添加 Check cue 后 prompt 变长，可能导致 evidence 截断。
```

应对：

```text
1. Stage 0/build sanity 先比较 prompt_token_count 和 truncation_rate；
2. 若 AA-QEC 明显高于当前 QEC-MIN baseline，先缩短 cue 或限制 repeated question；
3. 不直接启动 full training。
```

### 12.3 selector_name / source staging 混淆

风险：

```text
多个 O/C/F source 共存，run_lora_matrix.sh 可能复用旧 source。
```

应对：

```text
1. 每个 selector_name 独立 stage 到 outputs/sentence_trace_method/_sources/<dataset>/<selector_name>/；
2. wrapper 每个 case 显式设置 SELECTOR_NAME 和 EXPECTED_SELECTOR_NAME；
3. FORCE_STAGE 默认只在 source 生成 wrapper 内控制，不让训练 wrapper 静默改 source。
```

### 12.4 Stage 2 过度过滤证据

风险：

```text
primary-only 或 secondary 规则可能破坏 RAWFC 当前偏好的证据量。
```

应对：

```text
1. C3 使用 fallback to min5 作为主候选；
2. C1/C2/C4 用于解释去冗余和 secondary 贡献；
3. evidence_count.mean 和 fallback_fill_rate 必须进入报告。
```

---

## 13. 后续扩展

三阶段 RAWFC 单 seed 完成后，再按结果选择后续工作：

```text
1. 对 F2、最佳 constrained variant、B0/F0 做 3 seed confirmation；
2. 若差距 < 0.015 selection 或 macro_f1，扩到 5 seeds；
3. 迁移 LIAR-RAW best two AA-QEC variants；
4. 做 cue policy ablation: qd_prefer / atom_only / qd_only_with_claim_fallback / no_repeated_question；
5. 做 F4/F5 prompt ablation；
6. 做 shuffled cue / wrong cue 负对照；
7. 更新 case study HTML 或 Markdown 报告。
```

---

## 14. 结论

第一版 AA-QEC 实现应保持主 prompt 极简，把复杂度放在可审计的 selector trace 中：

```text
不是增加 prompt 字段；
而是让 chain construction 从 greedy evidence order 升级为 atom-anchored construction。
```

推荐的执行顺序是：

```text
1. 实现 AA-QEC-View；
2. 跑 RAWFC build sanity 和 Stage 1；
3. 若 Stage 1 有信号，实现 AA-QEC-Constrained；
4. 跑 RAWFC Stage 2；
5. 若 Stage 2 有信号，实现 AA-QEC-Full；
6. 跑 RAWFC Stage 3；
7. 只把胜出的 top variants 推入多 seed 和 LIAR-RAW 迁移。
```

