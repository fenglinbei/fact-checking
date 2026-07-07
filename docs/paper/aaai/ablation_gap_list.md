# 组件消融与超参敏感性：待补实验清单

基于对 `configs/experiment/mrec_v0.2/`、`outputs/sentence_trace_method/` 的逐项核实，列出当前**已有 / 缺失**的实验，以及需要补做的具体配置与产物。

---

## 一、组件消融

### 1.1 Selector 机制消融 — ✅ 已完整，无需补

核实结论：s0–s6 全部有 test metrics（之前报告漏看了带 `_lora_ebs16...` 后缀的 case 目录）。

| 变体 | acc | macro_f1 | 说明 |
|---|---|---|---|
| s0 no_evidence | 0.3070 | 0.3102 | 无证据下界 |
| s1 random_top5 | 0.3125 | 0.3168 | 随机选择 |
| s2 hybrid_top5 | 0.3381 | 0.3451 | claim 整体检索 |
| s3 hybrid_mmr_top5 | 0.3181 | 0.3177 | + MMR 去冗余 |
| s4 atom_union_source_score_top5 | 0.3501 | 0.3537 | atom-union 但无学习 |
| s5 map_quality_greedy | 0.3485 | 0.3492 | 有 map 无学习 |
| s6 trace_shuffle | 0.3165 | 0.3290 | 破坏顺序 |
| **主方法 learned_marginal_proxy** | **0.3597** | **0.3666** | — |

梯度清晰（0.31→0.37），s6 shuffle 证明有序性。**直接可用**。

### 1.2 Chunking 消融 — ⚠️ LIAR 完整，RAWFC/HoVer 缺

- ✅ LIAR：`v0_7/` 组 5 种 chunking（sentence / ctx_window / semantic / abc_claim_aware / raw）+ `chunk_*_s4_union_*` 产物均有 test。
- ❌ **RAWFC 缺**：目前 RAWFC 只有 `abc_tight` 变体，缺 sentence / semantic / raw 的对照。
- ❌ **HoVer 缺**（HoVer 整体无产物）。

**补做**：
1. 新建 `configs/experiment/v0.7/rawfc_*_chunk_{sentence,semantic07,abc_claim_aware,report}.yaml`（基于现有 RAWFC 配置改 chunking）。
2. 跑 build + verifier，产出 test metrics。
3. （HoVer 待 HoVer 主流程跑通后同步）

### 1.3 Atom-Union 候选池消融 — ❌ 完全缺失

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
- 复用同一 selector 权重与 verifier checkpoint，只需重跑 build + infer（不需重训 selector）。
- 数据集：LIAR + RAWFC。

### 1.4 Evidence Map 消融 — ❌ 完全缺失

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
- 数据集：LIAR（主），RAWFC 可选。

---

## 二、超参敏感性

### 2.1 Prompt Evidence Policy — ⚠️ LIAR 基本完整，state_budget + RAWFC 缺

**LIAR 现状**（核实确认）：

| policy | acc | macro_f1 | 状态 |
|---|---|---|---|
| fixed_topk (top5) | 0.3445 | 0.3501 | ✅ |
| minmax5_5 | 0.3325 | 0.3391 | ✅ |
| **minmax5_10（主）** | **0.3597** | **0.3666** | ✅ |
| minmax7_7 | 0.3413 | 0.3403 | ✅ |
| minmax9_9 | 0.3477 | 0.3481 | ✅ |
| minmax3_10 | 0.3357 | 0.3386 | ✅ |
| minmax7_12 | 0.3325 | 0.3450 | ✅ |
| budget1024 | 0.3445 | 0.3503 | ✅ |
| resolve_stop | 0.3054 | 0.2995 | ✅ |
| two_pass_uncertainty | 0.3469 | 0.3504 | ✅ |
| **state_budget** | — | — | ❌ **test 未跑** |

**补做**：
1. **LIAR state_budget**：配置 `learned_marginal_proxy_fullpool_state_budget.yaml` 已存在，case 目录已有 train 产物，只需补跑 **infer（test）阶段**。
2. **RAWFC policy sweep**：目前 RAWFC 仅 `minmax5_10`(+baseline20) 有 test，缺 fixed_topk / budget / resolve_stop / state_budget / two_pass 的对照。需新建对应 RAWFC 配置并跑全流程。

### 2.2 ρ_target sweep — ❌ 完全缺失，且配置与论文不一致

**关键问题**：论文写 $\rho_{\mathrm{target}}=0.80$（默认），但所有 `mrec_v0.2/` 配置实际都是 `target_resolved_rate: 1.0`。这是**论文与实现的硬不一致**，必须处理。

**两条路（二选一）**：
- **路径 A（改论文）**：把论文方法节里的默认值从 0.80 改为 1.0，并解释"实际采用更严格的解析目标"。零实验成本。
- **路径 B（做 sweep）**：补 ρ_target ∈ {0.5, 0.66, 0.80, 1.0} 的 sweep，证明方法对解析率目标不敏感，同时把 0.80 作为推荐默认。

**建议路径 B**（更有说服力，且能补上"超参敏感性"实验）：
- 新建 4 个配置（LIAR 主方法基础上改 `trace.target_resolved_rate`）。
- 注意：ρ_target 影响 trace 生成（build 阶段），需重跑 build + infer（selector 权重可复用，因为 proxy 训练不依赖 ρ_target）。
- 数据集：LIAR（主）。

### 2.3 k_min/k_max sweep — ✅ LIAR 完整，RAWFC 缺

LIAR 已有 6 组 minmax sweep（见 2.1 表）。RAWFC 缺，建议至少补 minmax{3_10, 7_7, 9_9} 三组作为敏感性证据。

### 2.4 backbone 与训练方式 — ⚠️ 部分

- ✅ RAWFC：Ministral(LoRA) + Llama-3.1-8B(LoRA + FullFT) 配置与部分产物有。
- ❌ LIAR：缺 Llama-3.1-8B 主方法 run（只有 Qwen3-4B 偏弱）。
- **补做**：LIAR 上 Llama-3.1-8B(LoRA) 主方法 run。

---

## 三、优先级排序

### P0（影响正确性，必须处理）
1. **ρ_target 配置与论文不一致**：先决定路径 A（改论文）还是 B（做 sweep）。最低限度路径 A 零成本；建议路径 B。
2. **LIAR state_budget 补 test**：配置和 train 产物都在，只差 infer。

### P1（补齐核心消融表）
3. **Atom-Union 候选池消融**（1.3）：4 变体 × LIAR，证明 union 各组件价值。
4. **Evidence Map 消融**（1.4）：4 变体 × LIAR，证明 LLM 标注价值。
5. **RAWFC policy sweep**（2.1）：补 fixed_topk / budget / resolve_stop / state_budget。

### P2（补齐敏感性）
6. **ρ_target sweep**（2.2，若选路径 B）：4 值 × LIAR。
7. **RAWFC k_min/k_max sweep**（2.3）：3 组。
8. **RAWFC chunking 消融**（1.2）：4 种 chunking。
9. **LIAR Llama-3.1 backbone**（2.4）。

### 不需要补（已完整）
- selector_mech s0–s6（LIAR）：✅
- LIAR chunking 消融：✅
- LIAR k_min/k_max sweep：✅
- LIAR policy sweep（除 state_budget）：✅

---

## 四、实验量估算

| 实验 | 配置数 | 需重训 selector | 需重训 verifier | 需重跑 build | 估算 |
|---|---|---|---|---|---|
| ρ_target 路径 A | 0 | 否 | 否 | 否 | 0 |
| LIAR state_budget test | 0 | 否 | 否 | 否（仅 infer） | 小 |
| Atom-Union 消融 | 4×2=8 | 否 | 否 | 是（4 套 build） | 中 |
| Evidence Map 消融 | 4×1=4 | 是（4 套权重） | 否（复用） | 否 | 中 |
| RAWFC policy sweep | 4 | 否 | 否 | 是 | 中 |
| ρ_target sweep（路径 B） | 4 | 否 | 否 | 是 | 中 |
| RAWFC k sweep | 3 | 否 | 否 | 否 | 小 |
| RAWFC chunking | 4 | 否 | 否 | 是 | 中 |
| LIAR Llama backbone | 1 | 否 | 是 | 否 | 中 |

其中 Evidence Map 消融需要改代码（`extract_marginal_features` 加 mask 参数），其余主要是配置与流水线调用。
