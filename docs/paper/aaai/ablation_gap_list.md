# 组件消融与超参敏感性：待补实验清单

基于对 `configs/experiment/mrec_v0.2/`、`outputs/sentence_trace_method/` 的逐项核实，并按以下原则整理：
- **RAWFC 不做机制消融**（规模较小，易受波动影响），只保留主结果。
- **state_budget 不报告不提及**，从清单删除。
- **fixed_topk 即 minmax k_k**，已有 minmax{5_5, 7_7, 9_9} 三个实验，不单列。
- **resolve_stop 已与 minmax 结合**，不单独报告。
- **ρ_target 先改论文为 1.0**，sweep 降为低优先级。

---

## 一、组件消融（均在 LIAR-RAW 上）

### 1.1 Selector 机制消融 — ✅ 已完整，无需补

核实结论：s0–s6 全部有 test metrics。

| 变体 | acc | macro_f1 | 说明 |
|---|---|---|---|
| s0 no_evidence | 0.3070 | 0.3102 | 无证据下界 |
| s1 random_top5 | 0.3125 | 0.3168 | 随机选择 |
| s3 hybrid_mmr_top5 | 0.3181 | 0.3177 | claim 检索 + MMR |
| s6 trace_shuffle | 0.3165 | 0.3290 | 破坏顺序（证明有序性） |
| s2 hybrid_top5 | 0.3381 | 0.3451 | claim 整体检索 |
| s5 map_quality_greedy | 0.3485 | 0.3492 | 有 map 无学习 |
| s4 atom_union_source_score | 0.3501 | 0.3537 | atom-union 无学习 |
| **主方法 learned_marginal_proxy** | **0.3597** | **0.3666** | — |

梯度清晰（0.31→0.37），s6 shuffle 证明有序性。**直接可用**。

### 1.2 Chunking 消融 — ✅ LIAR 完整，无需补

LIAR：`v0_7/` 组 5 种 chunking（sentence / ctx_window / semantic / abc_claim_aware / raw）+ `chunk_*_s4_union_*` 产物均有 test。RAWFC 不做机制消融。**直接可用**。

### 1.3 Atom-Union 候选池消融 — ❌ 完全缺失（核心缺口）

当前所有实验都用完整的 Atom-Union 池（baseline + atom-route 融合），没有对照"仅 baseline 池""仅 atom-route 池""无 MMR 去冗余"。

**需要补的对照**（控制 selector/verifier 不变，只换候选池来源）：

| 变体 | 候选池构造 | 目的 |
|---|---|---|
| A: baseline_only | 仅整 claim 检索 top-k | 证明 atom-route 的增量价值 |
| B: atom_route_only | 仅 atom 各自检索 + RRF 聚合 | 证明 baseline 池的补充价值 |
| C: union_no_mmr | A+B 融合但**不做最终 MMR 去冗余** | 证明 MMR 去冗余的价值 |
| D: union_full（主方法） | A+B 融合 + MMR | — |

**补做方式**：
- 配置层：在 atom-union build 脚本里加开关 `pool_mode ∈ {baseline_only, atom_only, union_no_mmr, union_full}`，分别生成 4 套 `build_*.jsonl`。
- 复用同一 selector 权重与 verifier checkpoint，只需重跑 build + infer（不需重训 selector/verifier）。
- 数据集：LIAR-RAW。

### 1.4 Evidence Map 消融 — ❌ 完全缺失（核心缺口）

当前 selector 的 12 维特征里，`phi_res / phi_conf / phi_map / phi_new_rel / phi_tension / phi_corr` 都依赖 evidence map 标注 $M(u_j,a_i)$。没有"无 map"或"弱化 map"的对照。

**需要补的对照**（控制候选池与 verifier 不变，在 selector 特征层 mask）：

| 变体 | map 退化方式 | 目的 |
|---|---|---|
| A: no_map | map 相关 6 维特征全部置 0，selector 仅用 `phi_ent/phi_cov/phi_src_novel/phi_text_novel/phi_ret/phi_cost` | 证明 LLM 结构化标注的整体价值 |
| B: map_no_directness | 保留 relation 但 $d_{ij}$ 置常数，$\delta(d_{ij})$ 退化为常数 | 证明 directness 的价值 |
| C: map_no_confidence | 保留 relation/directness 但 $c_{ij}$ 置 1.0 | 证明置信度的价值 |
| D: full_map（主方法） | 完整 map | — |

**补做方式**：
- 在 `mrec_learned_marginal.py` 的 `extract_marginal_features` 加 `map_ablation_mode` 参数，按模式 mask 对应维度。
- 重训 selector 权重（特征变了）+ 重跑 trace 构建 + verifier 推理。
- 数据集：LIAR-RAW。

---

## 二、超参敏感性（均在 LIAR-RAW 上）

### 2.1 k_min/k_max 敏感性 — ✅ 已完整，无需补

LIAR 已有 minmax sweep（fixed_topk 即 minmax k_k，含 5/7/9 三个固定点 + 5_10/3_10/7_12 三个区间）：

| policy | acc | macro_f1 | 说明 |
|---|---|---|---|
| minmax5_5 (=fixed_topk k=5) | 0.3325 | 0.3391 | 固定 5 |
| minmax7_7 (=fixed_topk k=7) | 0.3413 | 0.3403 | 固定 7 |
| minmax9_9 (=fixed_topk k=9) | 0.3477 | 0.3481 | 固定 9 |
| minmax3_10 | 0.3357 | 0.3386 | 区间 [3,10] |
| **minmax5_10（主）** | **0.3597** | **0.3666** | 区间 [5,10] |
| minmax7_12 | 0.3325 | 0.3450 | 区间 [7,12] |
| budget1024 | 0.3445 | 0.3503 | token 预算 |
| two_pass_uncertainty | 0.3469 | 0.3504 | 两阶段不确定性 |

resolve_stop 已与 minmax 结合（minmax 本身就是 resolve_stop 的退化形式，max_evidence_count 兜底），不单独报告。**直接可用**。

### 2.2 ρ_target — ⚠️ 先改论文，sweep 低优先级

**问题**：论文写 $\rho_{\mathrm{target}}=0.80$（默认），但所有 `mrec_v0.2/` 配置实际都是 `target_resolved_rate: 1.0`。

**处理**：
- **P0（立即）**：把论文方法节里的默认值从 0.80 改为 1.0，并说明"实际采用更严格的解析目标 ρ_target=1.0"。零实验成本。
- **P3（低优先级）**：补 ρ_target ∈ {0.5, 0.66, 0.80, 1.0} 的 sweep 作为敏感性补充。需新建 4 个配置（改 `trace.target_resolved_rate`），重跑 build + infer（selector 权重可复用）。

### 2.3 backbone — ⚠️ 部分缺

- ✅ RAWFC：Ministral(LoRA) + Llama-3.1-8B(LoRA + FullFT) 有。
- ❌ LIAR：缺 Llama-3.1-8B 主方法 run（只有 Qwen3-4B 偏弱）。
- **补做**：LIAR 上 Llama-3.1-8B(LoRA) 主方法 run。属 P2。

---

## 三、优先级排序

### P0（立即处理）
1. **改论文 ρ_target 0.80 → 1.0**，消除论文与实现的硬不一致。

### P1（补齐核心消融表，均为 LIAR）
2. **Atom-Union 候选池消融**（1.3）：4 变体，证明 union 各组件价值。
3. **Evidence Map 消融**（1.4）：4 变体，证明 LLM 标注价值（需改代码）。

### P2（补齐敏感性）
4. **LIAR Llama-3.1-8B backbone**（2.3）。

### P3（低优先级）
5. **ρ_target sweep**（2.2）：4 值 × LIAR。

### 不需要补（已完整）
- selector_mech s0–s6（LIAR）：✅
- LIAR chunking 消融（5 种）：✅
- LIAR k_min/k_max sweep（含 fixed_topk 5/7/9）：✅
- LIAR policy 对比（minmax/budget/two_pass）：✅

---

## 四、实验量估算

| 实验 | 配置数 | 需重训 selector | 需重训 verifier | 需重跑 build | 估算 |
|---|---|---|---|---|---|
| ρ_target 改论文 | 0 | 否 | 否 | 否 | 0 |
| Atom-Union 消融 | 4 | 否 | 否 | 是（4 套 build） | 中 |
| Evidence Map 消融 | 4 | 是（4 套权重） | 否（复用） | 否 | 中 |
| LIAR Llama backbone | 1 | 否 | 是 | 否 | 中 |
| ρ_target sweep (P3) | 4 | 否 | 否 | 是 | 中 |

其中 Evidence Map 消融需要改代码（`extract_marginal_features` 加 mask 参数），其余主要是配置与流水线调用。
