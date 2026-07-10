# Evidence Map 消融实现方案

## 一、已有实验是否覆盖

**不覆盖。** 已有 selector_mech 消融（s0–s6）与 map 消融是正交的两个维度：
- selector_mech 消融对比的是**不同的选择机制**（随机/hybrid/atom-union/启发式/learned）；
- map 消融要对比的是**同一套 learned selector 内部，map 标注不同退化程度**。

唯一相关的是 s5（map_quality_greedy），它是"有 map 无学习"，而 map 消融的 A 变体是"有学习无 map"——方向相反，不能替代。**4 个变体都需要新做。**

## 二、12 维特征对 map 标注的依赖关系

逐行核实 `extract_marginal_features`（mrec_learned_marginal.py:217-293）：

| 特征 | 依赖 map | 来源 |
|---|---|---|
| resolution_delta | ✅强 | relation+directness+confidence |
| entropy_reduction | ✅强 | 同上派生 |
| new_relation_for_atom | ✅ | relation 字段 |
| stance_tension | ✅ | relation+directness |
| corroboration_gain | ✅ | relation+directness |
| map_confidence | ✅直接 | pair.confidence |
| map_quality | ✅直接 | evidence_map_quality_score |
| new_atom_coverage | ⚠️部分 | covered_atom_ids（来自 map 标注） |
| source_novelty | ❌ | 来源 key |
| text_novelty | ❌ | 文本/duplicate_group |
| retrieval_score | ❌ | 检索分数 |
| cost_ratio | ❌ | token 预算 |

## 三、消融变体定义

| 变体 | map 退化方式 | mask 的特征 | 保留的特征 |
|---|---|---|---|
| A: no_map | map 相关全部置 0 | resolution_delta, entropy_reduction, new_relation_for_atom, stance_tension, corroboration_gain, map_confidence, map_quality, new_atom_coverage | source_novelty, text_novelty, retrieval_score, cost_ratio |
| B: no_directness | $d_{ij}$ 置常数，$\delta(d_{ij})$ 退化 | （特征维度不变，但 directness 输入退化） | 全部 12 维，但 directness 恒定 |
| C: no_confidence | $c_{ij}$ 置 1.0 | （特征维度不变，但 confidence 输入退化） | 全部 12 维，但 confidence 恒为 1.0 |
| E: no_relation | $r_{ij}$ 退化为 background | （特征维度不变，但 relation 输入退化） | 全部 12 维，但 relation 恒为 background |
| D: full_map（基准） | 无 | 无 | 全部 12 维 |

注：B/C/E 是"输入退化"而非"特征 mask"——保留特征维度但让 directness/confidence/relation 退化为常数，这样 selector 仍能学习其他维度的权重，只隔离单一信号的贡献。补 E(no_relation) 是为了与 B(no_directness)/C(no_confidence) 构成对三元组 $M(u_j,a_i)=(r_{ij},d_{ij},c_{ij})$ 的完整逐元消融。

### 证据容量对齐（关键设计）

**问题**：no_relation 退化后 atom 永远不进入 {S,R,Q,C}，resolved_rate 恒为 0，导致：
- trace 层：rho_target 永不触发，100% 跑到 max_steps
- 截断层：rho_target 不触发，退化到 min(k_max, T)，K* 失控

数据证实：主方法 fullpool trace 均值 18 步、71% 跑满 max_steps=20；截断后 K* 受 minmax5_10 的 min=5 主导（72% 样本 K*=5）。若 no_relation 让 K* 偏移到恒定 10，则性能差异无法干净归因于"缺 relation 标注"还是"证据数变了"。

**方案：本组消融全部统一用 fixed_topk k=5**

所有 map 消融变体（A/B/C/E/D）统一采用 `prompt_evidence.policy=fixed_topk, min=max=5`，强制 K*=5 恒定。这样：
- 五个变体证据容量完全相同（K*=5）
- 差异纯粹来自 map 标注质量，归因绝对干净
- D（full_map）基准用已有 `minmax5_5`（=fixed_topk k=5）主方法产物（f1=0.339），无需重跑

**no_relation 的预期行为**：relation 退化后 resolution_delta/entropy_reduction/stance_tension/corroboration_gain/new_relation_for_atom 自然归零，selector 只能靠 source_novelty/text_novelty/retrieval_score/cost_ratio/map_quality/new_atom_coverage 选证据，且 atom 不发生状态转移（trace 里 state_before=state_after=U）。这是预期现象，证明"没有 relation 标注时 selector 失去逐 atom 解析能力"。

## 四、实现方案

### 改动点 1：`extract_marginal_features` 加 `map_ablation_mode` 参数

文件：`src/fact_checking/selectors/mrec_learned_marginal.py:217`

```python
def extract_marginal_features(
    candidate: Mapping[str, Any],
    *,
    selected_steps: Sequence[Mapping[str, Any]],
    soft_state: Mapping[str, Mapping[str, float]],
    token_budget: int | None,
    pool_max_token_cost: int | None,
    map_ablation_mode: str = "full",  # 新增："full" | "no_map" | "no_directness" | "no_confidence"
) -> dict[str, float]:
```

逻辑：
- `no_map`：在计算完 features 字典后，把 8 个 map 相关特征强制置 0：
  ```python
  if map_ablation_mode == "no_map":
      for k in ("resolution_delta", "entropy_reduction", "new_relation_for_atom",
                "stance_tension", "corroboration_gain", "map_confidence",
                "map_quality", "new_atom_coverage"):
          features[k] = 0.0
  ```
- `no_directness`：在 `for pair in pairs` 循环里，把 `directness = _directness_factor(...)` 替换为常数（如 `"medium"` 对应的因子），即 `directness = _directness_factor("medium")`。
- `no_confidence`：同理，把 `confidence` 强制为 `1.0`。
- `no_relation`：同理，把 `relation = _relation_group(...)` 替换为 `"background"`（不在 `_RESOLVING_RELATION_STATES` 内，使 resolution/entropy/tension/corroboration 自然归零，但 directness/confidence/map_quality/new_atom_coverage 保留）。

### 改动点 2：训练函数透传 mode

`train_learned_marginal_proxy_weights`（:374）和 `train_learned_marginal_reward_weights`（:477）在调用 `extract_marginal_features` 时透传 `map_ablation_mode`。

### 改动点 3：推理构链透传 mode

`minimal_resolving_chain.py` 的 `_select_learned_marginal_steps`（:317）调用 `extract_marginal_features` 时透传 `map_ablation_mode`，该值从 `MRECSelectorParams` 读取。

### 改动点 4：配置与产物

- `MRECSelectorParams` 加字段 `map_ablation_mode: str = "full"`。
- 新建配置（基于 `learned_marginal_proxy_fullpool_minmax5_5.yaml`，即 fixed_topk k=5，保证证据容量对齐）：
  - `learned_marginal_proxy_fullpool_minmax5_5_map_ablation_no_map.yaml`
  - `..._no_directness.yaml`
  - `..._no_confidence.yaml`
  - `..._no_relation.yaml`
- 每个 mode 产出独立的 selector 权重文件（`learned_marginal_weights_{mode}.json`）和 trace。
- D（full_map）基准直接复用已有 `minmax5_5` 产物（=fixed_topk k=5），无需重跑。

### 改动点 5：verifier

verifier 复用主方法 checkpoint（只换 trace，不换 verifier 权重），只需重跑 build（trace 构建）+ infer。

## 五、实验流程

每个变体：
1. 训练 selector 权重（`map_ablation_mode` 控制特征退化）
2. 用该权重构建 trace（`build_mrec_traces.py`）
3. trace → verifier 数据（`build_trace_verifier_data.py`，**统一 fixed_topk k=5**）
4. verifier 推理（复用主方法 checkpoint）

共 5 个变体，其中 D（full）已有产物可复用，实际新跑 A/B/C/E 四个。

## 六、预期产物

```
outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_5_map_ablation_{no_map,no_directness,no_confidence,no_relation}/
  eval/test/best/label_token/metrics.json
```

加上已有的 full_map（minmax5_5 基准），构成 5 行消融表。

## 七、工作量估算

- 代码改动：3 个文件（mrec_learned_marginal.py / minimal_resolving_chain.py / MRECSelectorParams），约 35-45 行
- 配置：4 个 yaml（继承 minmax5_5，即 fixed_topk k=5）
- 运行：4 个变体 ×（selector 训练 + build + infer），selector 训练是 CPU 上的小模型（proxy 权重），build/infer 复用现有脚本
- D（full_map）基准复用已有 minmax5_5 产物，无需重跑
