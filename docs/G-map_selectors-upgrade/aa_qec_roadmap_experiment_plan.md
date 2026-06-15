# Atom-Anchored QA Evidence Chain 路线图与实验计划

## 0. 当前结论与下一步目标

当前 RAWFC 实验现象：

```text
QEC-MIN 最优
QEC-MAP 与 plain 基本持平
```

这说明：

```text
Check cue 有帮助；
显式暴露 relation / directness / covers 等 map 标签暂时没有明显收益。
```

因此下一步不建议继续增加 prompt 字段，也不建议加入生成式 `Answer`。更合理的升级方向是：

```text
保留 QEC-MIN 的极简 prompt；
把 evidence chain 的构造逻辑从 v0.7 greedy order 升级为 atom-anchored chain construction。
```

最终目标方法可以命名为：

```text
Atom-Anchored QA Evidence Chain
AA-QEC
```

或更保守地命名为：

```text
Atom-Anchored Question-Guided Evidence Chain
```

核心思想：

```text
claim atom 决定链条要验证什么；
QD question 决定每一步如何自然语言表达；
evidence sentence 本身作为 answer；
relation/directness 等 map 信息只用于内部选链，不默认暴露给 verifier。
```

---

## 1. 方法定位

### 1.1 当前 QEC-MIN

当前 QEC-MIN 本质上是：

```text
v0.7 adaptive5_10 先选 evidence；
然后给每条 selected evidence 加一个 Check cue。
```

Prompt 形式：

```text
Check: <question_or_atom>
<evidence text>
```

它的优点是 prompt 很轻，不引入 answer，不暴露 relation/directness，能够把普通 evidence list 转成轻量 chain view。

但它仍然有一个限制：

```text
chain 顺序主要来自 v0.7 marginal-gain greedy selection；
不一定对应 claim atom 的语义结构。
```

### 1.2 AA-QEC

AA-QEC 的目标是让 chain construction 本身变成 atom-aware：

```text
先按 claim atoms 建立验证骨架；
再为每个 atom 选择 primary evidence；
必要时加入 qualifier / counter evidence；
最后用 QD question 或 atom text 作为 Check cue。
```

也就是：

```text
Atom -> Question/Cue -> Evidence
```

而不是：

```text
Evidence set -> Greedy order -> Add cue
```

AA-QEC 的主 prompt 仍然保持 QEC-MIN：

```text
Check: <cue_text>
<evidence_text>
```

不要默认显示：

```text
Answer
relation
directness
confidence
score
route rank
source type
```

这些字段进入 trace 和 diagnostics，不进入主 prompt。

---

## 2. 推荐总路线

建议按三层逐步升级，避免一次改太多变量。

### Stage 1: AA-QEC-View

固定当前 v0.7 adaptive5_10 选出的 selected evidence set，只改变 evidence order 和 cue assignment。

目标问题：

```text
在同一批 evidence 上，atom-anchored ordering 是否比 v0.7 greedy ordering 更好？
```

这一阶段不改变 selector，只改变 view/order，因此风险最低。

### Stage 2: AA-QEC-Constrained

候选范围仍限制在 v0.7 selected evidence 内，但重新按 atom primary / secondary 规则构造 chain。

目标问题：

```text
在 v0.7 selected evidence 内做 atom-aware filtering / reconstruction 是否更好？
```

这一阶段可以减少冗余 evidence，并检查 qualifier/counter step 是否有帮助。

### Stage 3: AA-QEC-Full

直接从 top20 candidate pool 或 union candidate pool 中构造 atom-anchored chain，不依赖 v0.7 selected set。

目标问题：

```text
AA-QEC 能否作为完整 selector 替代 v0.7 adaptive5_10？
```

如果 Stage 3 赢，AA-QEC 可以作为最终主方法；v0.7 adaptive5_10 作为 strong graph-based baseline 或 ablation。

---

## 3. AA-QEC chain step schema

建议 trace 中每个 step 保存完整结构，但 prompt 只渲染 cue + evidence。

```json
{
  "step": 1,
  "atom_id": "A1",
  "atom_text": "...",
  "cue_text": "...",
  "cue_source": "qd_question",
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
  "qd_question_hybrid_score": 0.71
}
```

Prompt 渲染时只使用：

```text
Check: {cue_text}
{evidence_text}
```

---

## 4. 输入字段要求

AA-QEC 需要每条 candidate 尽量包含：

```text
text
covered_atom_ids
map_relation
map_directness
map_confidence
evidence_map_quality_score
base_score
qd_question_routes
from_qd
from_baseline
baseline_rank
qd_pool_rank
qd_rrf_score
qd_question_hit_count
qd_max_question_hybrid
duplicate_group
source_group
```

第一版最小必需字段：

```text
text
covered_atom_ids
map_relation
map_directness
map_confidence
evidence_map_quality_score
qd_question_routes
```

如果 `qd_question_routes` 缺失，fallback 到 atom cue。

---

## 5. Primary evidence 选择规则

对每个 claim atom `A_i`，找到所有覆盖该 atom 的候选 evidence：

```text
candidate.covered_atom_ids contains A_i
```

然后按 lexicographic priority 选择 primary evidence。

推荐优先级：

```text
direct support/refute
> partial support/refute
> direct qualify/mixed
> partial qualify/mixed
> context support/refute
> context qualify/mixed
> background/context fallback
```

同一级别内 tie-break：

```text
map_confidence 高优先
> evidence_map_quality_score 高优先
> qd_max_question_hybrid 高优先
> base_score 高优先
> 原 candidate rank 靠前优先
```

这样避免重新引入复杂加权 objective。论文里可以解释为：

```text
For each atom, AA-QEC first seeks direct evidence that supports or refutes the atom. If direct evidence is unavailable, it falls back to partial, qualifying, or contextual evidence.
```

---

## 6. Secondary evidence 选择规则

每个 atom 最多加入一条 secondary evidence，用于 qualifier / counter。

### 6.1 何时加入 secondary

若 primary 是：

```text
support
```

secondary 可以是：

```text
refute / qualify / mixed
```

若 primary 是：

```text
refute
```

secondary 可以是：

```text
support / qualify / mixed
```

若 primary 是：

```text
qualify / mixed
```

secondary 可以是：

```text
support / refute
```

### 6.2 Secondary 过滤条件

```text
不是 primary 本身；
不是 duplicate；
directness 至少为 partial，除非没有其他候选；
map_confidence >= 0.4，第一版可作为默认阈值；
如果超过 max_chain_steps，则 secondary 优先被丢弃。
```

### 6.3 解释

Secondary evidence 不是为了增加证据数量，而是为了捕捉 RAWFC / LIAR 中常见的限定、冲突和中间标签信号。

主 prompt 仍然不显示 `role=secondary` 或 `relation`。这些只用于 trace diagnostics。

---

## 7. Multi-atom evidence 去重规则

一条 evidence 可能覆盖多个 atoms。建议：

```text
同一 evidence 只在第一次被选中时进入 prompt；
它覆盖的所有 atom 都标记为 covered；
后续 atom 如果最佳 evidence 已经被使用，则尝试选择新的补充 evidence；
如果没有新的有效 evidence，则跳过该 atom 或记录为 already_covered。
```

这样可以避免 prompt 中重复出现同一条 evidence。

Trace 中可以记录：

```text
covered_by_previous_step = true
anchor_step = <step_id>
```

但 prompt 中不显示。

---

## 8. Chain budget

由于当前 RAWFC 的最优实验来自 `v0.7 adaptive5_10 + qec_min`，第一版 AA-QEC 不建议突然变成很短链。

建议默认：

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
4. 如果 chain < min_chain_steps，则用 fallback evidence 补足；
5. 如果 chain > max_chain_steps，则裁剪到 max_chain_steps。
```

Fallback 补足规则：

```text
优先选择 v0.7 order 中尚未使用的 evidence；
否则选择全局高 map_quality / directness / base_score 的 evidence；
避免 duplicate 和 irrelevant/background evidence。
```

裁剪优先级：

```text
保留 primary evidence；
保留 direct support/refute evidence；
保留覆盖未解决 atom 的 evidence；
保留 qualifier/counter evidence；
丢弃 fallback evidence；
丢弃 background/context evidence；
丢弃重复 source / duplicate evidence。
```

---

## 9. Cue policy

AA-QEC 的 prompt cue 仍然优先使用 QD question，因为当前 QEC-MIN 的收益很可能来自自然语言 question cue。

### 9.1 默认策略：qd_prefer

```text
1. 如果 candidate 有可用 qd_question_routes，选择最佳 question；
2. 如果没有 QD route，用当前 atom text；
3. 如果没有 covered atom，用完整 claim fallback。
```

### 9.2 最佳 question route 选择

```text
过滤空 question；
过滤与 claim 完全相同的 question；
过滤过短问题，例如少于 4 个英文词；
过滤明显泛化问题，例如 "Is this claim true?"；
rank 小优先；
rank 相同则 hybrid_score 高优先；
仍相同则 question_id 靠前优先。
```

### 9.3 Cue policy 消融

必须做 cue policy 消融，因为方法名里有 QA，但主骨架是 atom。

```text
qd_prefer
atom_only
qd_only_with_claim_fallback
qd_prefer_no_repeated_question
```

如果 `qd_prefer` 最好，可以保留 AA-QEC / Atom-Anchored QA Evidence Chain 的命名。

如果 `atom_only` 接近或更好，更适合命名为：

```text
Atom-Anchored Evidence Chain
```

问题分解只作为 candidate pool augmentation，不作为 prompt 核心。

---

## 10. Prompt 渲染

主 prompt 使用 QEC-MIN。

```text
Claim:
{claim}

Evidence:
[1] Check: {cue_text}
{evidence_text}

[2] Check: {cue_text}
{evidence_text}
```

不要默认使用：

```text
QEC-MAP
Answer
Span
Relation/directness tag
Confidence/score
```

这些都放入消融。

### 10.1 QEC-MAP 的位置

当前 RAWFC 结果中 `qec_map ≈ plain`，因此 QEC-MAP 不应作为下一步主线。

保留用途：

```text
AA-QEC + qec_min
AA-QEC + qec_map
```

用于检查：

```text
atom-anchored chain 是否让 relation/directness 变得更有用？
```

若 AA-QEC 下 QEC-MAP 仍无提升，说明 map 标签应继续隐藏，只用于内部 selector。

---

## 11. 实现路线

### 11.1 新增 selector 文件

建议新增：

```text
src/fact_checking/selectors/atom_anchored_qec.py
```

核心 dataclass：

```python
@dataclass(frozen=True)
class AtomAnchoredQECParams:
    candidate_top_n: int = 20
    min_chain_steps: int = 5
    max_chain_steps: int = 10
    max_secondary_per_atom: int = 1
    min_secondary_confidence: float = 0.4
    cue_policy: str = "qd_prefer"  # qd_prefer | atom_only | qd_only
    candidate_scope: str = "top20"  # selected | top20
```

核心函数：

```python
def build_atom_anchored_qec_trace_row(row, params):
    ...

def select_primary_for_atom(atom, candidates, used_ids, params):
    ...

def select_secondary_for_atom(atom, primary, candidates, used_ids, params):
    ...

def choose_qec_cue(atom, candidate, params):
    ...

def render_chain_steps(chain_steps):
    ...
```

输出 trace 保持与现有 trace pipeline 兼容：

```json
{
  "selector_name": "atom_anchored_qec_min5_10",
  "chain_policy": "atom_anchored_qec_v1",
  "selector_ordered_indices": [...],
  "selected_candidates": [...],
  "chain_steps": [...],
  "claim_atoms": [...],
  "candidate_pool": [...]
}
```

关键是继续输出：

```text
selector_ordered_indices
selected_candidates
```

这样后续 verifier data builder 可以继续复用 trace flow。

### 11.2 修改 `build_trace_verifier_data.py`

让 `qec_min` 优先读取 `trace.chain_steps`：

```text
if trace has chain_steps:
    use chain_steps order and chain_steps.cue_text
else:
    use selected_candidates + qd route fallback
```

这样 AA-QEC 和原 v0.7 QEC-MIN 可以共用同一个 prompt style。

### 11.3 修改运行脚本

保证以下变量可配置：

```bash
TRACE_PROMPT_STYLE=qec_min
SELECTOR_NAME=atom_anchored_qec_min5_10
CANDIDATE_SCOPE=selected|top20
MIN_CHAIN_STEPS=5
MAX_CHAIN_STEPS=10
CUE_POLICY=qd_prefer|atom_only|qd_only
```

建议第一版先不要改训练脚本主体，只新增 source/trace 构建脚本，然后复用现有 LoRA 训练流程。

---

## 12. 实验计划总览

实验按四层推进：

```text
Stage 0: Build sanity
Stage 1: Order ablation / AA-QEC-View
Stage 2: Constrained reconstruction
Stage 3: Full AA-QEC selector
Stage 4: Multi-seed confirmation
```

---

## 13. Stage 0: Build sanity

目标：确认 prompt 构建、chain step、cue fallback、token/truncation 均正常。

固定：

```text
selector/source: 当前 v0.7 adaptive5_10 或 AA-QEC 构建 source
prompt: qec_min
train: no
```

| ID | Dataset | Source | Prompt | Train | 目的 |
|---|---|---|---|---:|---|
| S0 | RAWFC | v0.7 adaptive5_10 | plain | no | 当前 prompt build baseline |
| S1 | RAWFC | v0.7 adaptive5_10 | qec_min | no | 当前最优 QEC-MIN build 统计 |
| S2 | RAWFC | AA-QEC-View | qec_min | no | 检查 atom-order chain build |
| S3 | RAWFC | AA-QEC-Constrained | qec_min | no | 检查 selected-scope reconstruction |
| S4 | RAWFC | AA-QEC-Full top20 | qec_min | no | 检查 full selector build |
| S5 | LIAR-RAW | v0.7 adaptive5_10 | qec_min | no | 迁移前 build sanity |
| S6 | LIAR-RAW | AA-QEC-Full top20 | qec_min | no | 迁移前 build sanity |

Build-only 必须记录：

```text
n_rows
skipped_total
prompt_token_count.mean / p50 / p90 / max
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
```

若 `prompt_truncation_rate` 明显高于当前 QEC-MIN baseline，需要先修 prompt 长度再训练。

---

## 14. Stage 1: AA-QEC-View / order ablation

目标：同一批 v0.7 selected evidence，只改变 order，验证 atom anchoring 是否有价值。

| ID | Selector | Evidence set | Order | Prompt | 目的 |
|---|---|---|---|---|---|
| B0 | v0.7 adaptive5_10 | v0.7 selected | v0.7 greedy | qec_min | 当前 RAWFC 最优 baseline |
| O1 | v0.7 adaptive5_10 | v0.7 selected | atom order | qec_min | 只测 atom ordering |
| O2 | v0.7 adaptive5_10 | v0.7 selected | atom order + primary-before-secondary | qec_min | 测更强 atom chain view |
| O3 | v0.7 adaptive5_10 | v0.7 selected | shuffled | qec_min | 负对照，确认 order 是否重要 |

推荐先在 RAWFC 单 seed 跑。

解释规则：

```text
O1/O2 > B0:
    atom-anchored order 有价值，进入 Stage 2。

O1/O2 ≈ B0 且 O3 下降:
    order 有影响，但当前 atom order 尚未优于 v0.7 greedy。

O3 ≈ B0:
    verifier 对 order 不敏感，后续重点应转向 selection 而不是 ordering。
```

---

## 15. Stage 2: AA-QEC-Constrained

目标：候选范围仍为 v0.7 selected evidence，但用 atom primary / secondary 规则重构 chain。

| ID | Candidate scope | Selection policy | Budget | Prompt | 目的 |
|---|---|---|---|---|---|
| C0 | v0.7 selected | keep all | 原 v0.7 | qec_min | baseline |
| C1 | v0.7 selected | primary per atom | max10 | qec_min | 测去冗余 primary chain |
| C2 | v0.7 selected | primary + secondary | max10 | qec_min | 测 qualifier/counter step |
| C3 | v0.7 selected | primary + secondary + fallback to min5 | min5/max10 | qec_min | 保持 RAWFC 证据量 |
| C4 | v0.7 selected | primary + fallback to min5, no secondary | min5/max10 | qec_min | 隔离 secondary 贡献 |

预期最有希望的是：

```text
C3 = primary + secondary + fallback to min5
```

因为它既保留 RAWFC 当前偏好的证据量，也引入 atom-aware chain construction。

解释规则：

```text
C1 > C0:
    去冗余 primary chain 有效。

C2/C3 > C1:
    qualifier/counter evidence 有价值。

C3 > C2:
    RAWFC 需要 min evidence budget。

C4 ≈ C3:
    secondary 贡献不大，primary + fallback 足够。
```

---

## 16. Stage 3: AA-QEC-Full

目标：从 top20 candidate pool 直接构造 AA-QEC，完整替代 v0.7 selector。

| ID | Candidate scope | Selection policy | Budget | Prompt | 目的 |
|---|---|---|---|---|---|
| F0 | top20 | v0.7 adaptive5_10 | min5/max10 | qec_min | 当前主线 baseline |
| F1 | top20 | AA-QEC primary-only | min5/max10 | qec_min | 测纯 atom primary |
| F2 | top20 | AA-QEC primary + secondary | min5/max10 | qec_min | 测完整 AA-QEC |
| F3 | top20 | AA-QEC primary + secondary | no min5/max10 | qec_min | 测动态链长 |
| F4 | top20 | AA-QEC primary + secondary | min5/max10 | plain | 检查收益来自 selector 还是 prompt |
| F5 | top20 | AA-QEC primary + secondary | min5/max10 | qec_map | 检查 map-visible 在 AA-QEC 下是否有用 |

最重要的对比：

```text
F2 vs F0:
    AA-QEC 是否能替代 v0.7 adaptive5_10？

F2 vs F1:
    secondary / qualifier / counter 是否有贡献？

F2 vs F3:
    min5 是否必要？

F2 vs F4:
    收益来自 atom-aware selector，还是来自 QEC-MIN prompt？

F5 vs F2:
    QEC-MAP 是否在 AA-QEC 下才变得有用？
```

若资源有限，Stage 3 第一轮只跑：

```text
F0, F1, F2, F3
```

F4/F5 后置。

---

## 17. Stage 4: Cue policy ablation

目标：确认 AA-QEC 中 question cue 是否必要。

| ID | Chain selector | Cue policy | Prompt | 目的 |
|---|---|---|---|---|
| Q0 | best AA-QEC | qd_prefer | qec_min | 默认主方法 |
| Q1 | best AA-QEC | atom_only | qec_min | 检查 atom cue 是否足够 |
| Q2 | best AA-QEC | qd_only_with_claim_fallback | qec_min | 检查纯 question cue 是否足够 |
| Q3 | best AA-QEC | qd_prefer_no_repeated_question | qec_min | 检查重复 question 的干扰 |

命名决策：

```text
Q0 最优:
    保留 Atom-Anchored QA Evidence Chain。

Q1 接近或最优:
    更适合叫 Atom-Anchored Evidence Chain。

Q2 最优:
    question route 是核心，AA-QEC 中 QA 成分更强。
```

---

## 18. Stage 5: Multi-seed confirmation

单 seed 阶段只做筛选，不下最终结论。

推荐最终进入多 seed 的配置：

```text
current best baseline:
    v0.7 adaptive5_10 + qec_min

best AA-QEC variant:
    e.g. AA-QEC-F2 + qec_min

second-best AA-QEC variant:
    e.g. AA-QEC-Constrained-C3 或 AA-QEC-F3
```

RAWFC：

```text
先 3 training seeds: 13, 21, 42
如果差距 < 0.015 selection 或 macro-F1，则扩到 5 seeds: 13, 21, 42, 87, 100
```

LIAR-RAW：

```text
先迁移 best two variants；
每个 3 training seeds；
如果 qec_map 或 cue policy 现象不同，再单独做 prompt ablation。
```

注意：

```text
这里的 seed 是 verifier LoRA training seed，不是 eval seed。
Eval 尽量 deterministic。
```

---

## 19. 评价指标

### 19.1 Downstream verifier 指标

主指标：

```text
selection score
macro-F1
accuracy
```

LIAR-RAW 额外看：

```text
true-side F1
class-wise F1
pants-fire / barely-true / true confusion
```

RAWFC 额外看：

```text
false / half / true class-wise F1
false ↔ half confusion
half ↔ true confusion
```

### 19.2 Chain diagnostics

AA-QEC 必须报告 chain 质量指标，否则方法主张不够稳。

```text
atom_coverage_rate
mean_covered_atoms_per_sample
mean_chain_steps
primary_step_count
secondary_step_count
fallback_step_count
qd_cue_rate
atom_cue_rate
claim_fallback_rate
duplicate_evidence_rate
multi_atom_evidence_rate
uncovered_atom_rate
repeated_question_rate
prompt_token_mean / p90 / max
truncation_rate
```

### 19.3 统计汇报

多 seed 阶段报告：

```text
mean ± std across training seeds
每个 seed 的 raw result
val-selected tau 的 test result
```

若两个配置接近，建议做 paired bootstrap CI。

---

## 20. Tau selection 规则

所有配置统一跑：

```text
tau0
tau0p5
tau0p75
```

每个 seed 内用 val selection 选择 tau。

如果最优 tau 和次优 tau 的 val selection 差距小于：

```text
0.005
```

使用稳定性 tie-break：

```text
优先历史更稳的 tau；
优先 macro-F1 不低的 tau；
优先不会明显牺牲弱类的 tau。
```

Test 只用于最终报告，不参与选择。

---

## 21. 决策规则

### 21.1 是否进入 Stage 2

如果 Stage 1 中：

```text
O1/O2 >= B0 + 0.005 selection 或 macro-F1
```

进入 Stage 2。

如果：

```text
O1/O2 与 B0 持平，但 O3 明显下降
```

说明 order 有影响，也可以进入 Stage 2。

### 21.2 是否进入 Stage 3

如果 Stage 2 中：

```text
C3 或 C2 >= C0
```

进入 Stage 3。

如果 C 系列全部下降，说明 v0.7 selected set 内的 atom-aware filtering 可能破坏了 evidence sufficiency，Stage 3 可以暂缓。

### 21.3 是否把 AA-QEC 作为主方法

如果 Stage 3 中：

```text
F2 >= F0 + 0.005 selection 或 macro-F1
```

并且多 seed 后均值仍优于 F0，则 AA-QEC 可作为主方法。

如果：

```text
F2 与 F0 持平，但 chain diagnostics 更好、prompt 更短、case study 更清楚
```

可以把 AA-QEC 作为主方法，v0.7 作为 strong graph baseline；但需要谨慎表述性能差异。

如果：

```text
F0 明显优于 F2
```

保留 v0.7 adaptive5_10 + qec_min 作为主方法，AA-QEC 作为 interpretability ablation 或 future direction。

### 21.4 QEC-MAP 是否回到主线

只有当：

```text
AA-QEC + qec_map > AA-QEC + qec_min
```

并且：

```text
prompt_truncation_rate 没有明显上升；
no-relation / shuffled diagnostics 证明不是标签 shortcut；
多 seed 稳定。
```

才考虑把 QEC-MAP 纳入主方法。

否则继续使用：

```text
AA-QEC + qec_min
```

---

## 22. 负对照与诊断实验

建议至少做两个负对照。

### 22.1 Shuffled order

```text
同一 evidence set；
随机打乱 chain order；
prompt 仍为 qec_min。
```

作用：确认 order 是否真的有影响。

### 22.2 Wrong cue / shuffled cue

```text
同一 evidence set；
Check cue 在样本内随机错配；
evidence text 不变。
```

作用：确认 QEC-MIN 的收益是否来自 cue 与 evidence 的正确对应，而不是单纯 prompt 变长或加入了 question-like 文本。

若 wrong cue 与正确 cue 持平，说明 cue 设计可能无效或 verifier 忽略 cue。

---

## 23. Case study 计划

每个数据集至少选 5 个样本做 case study：

```text
plain
v0.7 + qec_min
AA-QEC + qec_min
```

每个样本展示：

```text
claim
gold label
predictions
chain steps
covered atoms
primary / secondary roles
是否有 qualifier/counter
错误是否来自 missing evidence、wrong cue、wrong relation、verifier misclassification
```

重点观察：

```text
AA-QEC 是否更按 claim components 组织证据；
是否减少重复 evidence；
是否更好保留 qualifier/counter evidence；
是否改善 false/half/true 边界。
```

---

## 24. 推荐执行顺序

最小可执行路线：

```text
1. 实现 AA-QEC-View：同 evidence set，只改 atom order。
2. 跑 RAWFC build sanity。
3. 跑 RAWFC Stage 1 single seed：B0/O1/O2/O3。
4. 若 O1/O2 有信号，实现 AA-QEC-Constrained。
5. 跑 RAWFC Stage 2 single seed：C0/C1/C2/C3/C4。
6. 若 C2/C3 有信号，实现 AA-QEC-Full。
7. 跑 RAWFC Stage 3 single seed：F0/F1/F2/F3。
8. 选 top-2 AA-QEC variants + baseline 跑 3 seeds。
9. 若 RAWFC 稳定，再迁移 LIAR-RAW。
10. 最后做 cue policy ablation 和 case study。
```

---

## 25. 当前最推荐的第一版实现

优先实现：

```text
AA-QEC-v1
```

配置：

```text
candidate_scope = v0.7 selected evidence
order = atom order
selection = keep all evidence, reorder only
cue_policy = qd_prefer
prompt = qec_min
min_chain_steps = inherited from v0.7 selected evidence
max_chain_steps = inherited from v0.7 selected evidence
```

这就是 Stage 1 的 O1/O2。

如果 O1/O2 有收益，再实现：

```text
AA-QEC-v2
```

配置：

```text
candidate_scope = v0.7 selected evidence
selection = primary + secondary + fallback to min5
cue_policy = qd_prefer
prompt = qec_min
min_chain_steps = 5
max_chain_steps = 10
```

如果 v2 有收益，再实现：

```text
AA-QEC-v3
```

配置：

```text
candidate_scope = top20 candidate pool
selection = primary + secondary + fallback to min5
cue_policy = qd_prefer
prompt = qec_min
min_chain_steps = 5
max_chain_steps = 10
```

v3 才是最终主方法候选。

---

## 26. 建议论文表述

若 AA-QEC + QEC-MIN 成功，方法段可以表述为：

```text
We propose Atom-Anchored QA Evidence Chain, a lightweight evidence-chain construction method that anchors verification steps to claim atoms while using decomposed questions as natural-language cues. Each chain step contains only a verification cue and its supporting evidence sentence; the evidence itself serves as the answer. This avoids generating intermediate answers and keeps the verifier input faithful to retrieved evidence.
```

中文表述：

```text
我们提出 Atom-Anchored QA Evidence Chain。该方法以 claim atom 作为验证骨架，以问题分解得到的 question 作为自然语言检查提示，并将检索到的 evidence sentence 直接作为该检查提示的回答。每一步证据链只包含一个 Check cue 和一条 evidence，从而避免生成中间 answer，同时保留证据链的可解释结构。
```

---

## 27. 一句话总结

下一步不应把 prompt 变复杂，而应把 chain construction 变得更像证据链：

```text
固定 QEC-MIN 作为主 prompt view；
用 claim atoms 组织 evidence；
用 QD questions 表达检查点；
用 evidence 本身作为 answer；
用 relation/directness 只做内部 selection 和 diagnostics。
```
