# 可解释性实验规划（AAAI）

本规划基于对 trace 产物字段、状态转移分布、现有评估脚本、人工标注基础设施的全面探查，确认可解释性实验的**数据已就绪、评估脚本大部分可复用**，核心缺口是人工 gold 标注（已设计但未执行）。

规划原则：**每个实验都标注"基于什么数据 / 用什么脚本 / 需不需要新标注"**，避免规划落不了地。

---

## 0. 可解释性的三层论证结构

论文需要回答"为什么证据链是可解释的"，拆成三层，每层有对应实验：

| 层次 | 问题 | 实验类型 |
|---|---|---|
| **L1 过程可读性** | 证据链本身是否结构清晰、状态转移是否合理？ | 内部统计 + 可视化（无需 gold） |
| **L2 与外部标准一致** | 证据链选的证据、标的 relation 是否和人类/gold 一致？ | 对照评估（需 gold 或弱 gold） |
| **L3 对判别的贡献** | 证据链的不同部分对 verifier 判定各贡献多少？ | 消融/扰动（无需 gold，看性能变化） |

---

## L1. 过程可读性（内部统计，无需 gold）

这一层证明"证据链本身是一个合理的、人类可读的中间过程"。

### L1-1 证据链结构统计
**目标**：展示证据链 $\mathcal{T}$ 的宏观形态——长度分布、操作构成、解析率。

| 指标 | 数据来源 | 现状 |
|---|---|---|
| trace 长度 $T$ 分布（双峰：T=1 占 54%、T=10 占 22%） | `mrec_diagnostics.step_count` | ✅ 已有 |
| $K^\ast$（截断后）分布 | 当前 $K^\ast=T$，需补 prompt policy 截断后的对比 | ⚠️ 需对比 |
| operation 分布（BRIDGE 2859/OPEN 1405/CORROBORATE 119/CONTRAST 69/FALLBACK 1） | `mrec_diagnostics.operation_counts` | ✅ 已有 |
| resolved_atom_rate（mean 0.774，71.7% 达标） | `mrec_diagnostics.resolved_atom_rate` | ✅ 已有 |
| stop_reason 分布 | `mrec_diagnostics.stop_reason` | ✅ 已有 |

**产出**：一张"证据链构成"统计表/图。
**脚本**：新写 `scripts/phase5_selectors/eval/summarize_trace_statistics.py`（聚合 `mrec_diagnostics`，约 50 行）。
**待办**：编写聚合脚本，跑 LIAR/RAWFC val+test。

### L1-2 状态转移合理性
**目标**：证明状态机转移是合法且有意义的（无非法回退、冲突状态确有立场冲突）。

| 指标 | 数据来源 | 现状 |
|---|---|---|
| 状态转移分布（U→S 1028, U→R 340, S→C 13, R→C 4 等） | `mrec_steps` 的 state_before→after | ✅ 已有 |
| 非法转移率（如 S→U 回退） | 期望为 0 | ✅ 已有（当前 0） |
| 进入 C（conflicted）时是否确有 S↔R 立场冲突 | `mrec_steps` 中 state_after=C 的 step 的 relation | ⚠️ 需校验脚本 |
| BRIDGE 后 atom 仍未解析的比例（U→U 占 2114） | 现状偏高，需解释或改进 | ⚠️ 焦点 |

**产出**：状态转移桑基图/热力图 + 非法转移率 = 0 的声明。
**脚本**：新写 `validate_state_transitions.py`（扫描 `mrec_steps`，校验合法性 + 统计 C 状态的 relation 一致性）。
**待办**：编写校验脚本；U→U 偏高需在论文中讨论（BRIDGE 提供背景但不直接解析，属设计预期，或作为 selector 改进方向）。

### L1-3 Case Study 可视化
**目标**：用具体案例展示证据链的"逐 atom 解释"过程。

| 内容 | 工具 | 现状 |
|---|---|---|
| 证据链图（atoms → steps → 状态转移） | `render_evidence_chain_graph_html.py` | ✅ 已存在 |
| 选取典型 case：true/false/pants-fire + 有 CONTRAST 的 + 多 atom 的 | 按 gold_label + operation 筛选 | ⚠️ 需选 case |
| evidence map 标注可视化（relation/directness/key_spans） | `render_evidence_map_claim_html.py` | ✅ 已存在 |

**产出**：论文 figure，3-5 个 case 的证据链图。
**待办**：筛选代表性 case（如一个含 S→C 冲突的、一个纯 OPEN 解析的、一个 BRIDGE 主导的），渲染成图。

---

## L2. 与外部标准一致（需 gold 或弱 gold）

这一层证明"证据链选的证据和标的立场，和人类判断一致"。**这是最硬的可解释性证据，也是当前最大缺口**。

### L2-1 人工 gold 标注（核心待办，基础设施已就绪）
**目标**：获得人工 gold，度量 evidence map 标注质量与 atom 拆分质量。

**现状**：`docs/paper/aaai/annotation_project/` 已完整设计——指导书（413 行）、200+250 条任务已生成、Label Studio 已部署，**但 DB 为空，标注未开始**。

| 标注项 | 规模 | 度量什么 | 数据 |
|---|---|---|---|
| **实验1：Atom 质量** | 200 claim（LIAR 100+RAWFC 100） | atom 的 faithfulness/completeness/atomicity 三维 | `data/exp1_tasks.jsonl` ✅ |
| **实验2：Evidence Map** | 250 pair（每数据集 125） | relation/directness/confidence 与 LLM 标注的一致性 | `data/exp2_tasks.jsonl` ✅ |

**度量指标**（指导书已定义）：
- relation：Cohen's κ（标注者间）+ 与 LLM 的一致率（acc/macro-F1）
- directness：Spearman ρ（标注者间 + 与 LLM）
- confidence：ECE（校准误差，LLM 的 map_confidence 是否与正确率匹配）
- atom 三维：faithfulness（无幻觉）/completeness（无遗漏）/atomicity（不可再分）的比例

**待办（按顺序）**：
1. 把已生成的 `exp1_tasks_flat_zh.jsonl`（257 条）、`exp2_tasks.jsonl`（250 条）**导入 Label Studio**（当前 DB 空）。
2. 完成双盲双标注（A/B 两位标注者）。
3. 仲裁分歧，导出 `results/exp{1,2}_annotations_{A,B}.json`。
4. 编写评估脚本 `evaluate_annotation_agreement.py`：计算 κ/ρ/ECE + LLM vs human 指标。

**预估工作量**：200+250 = 450 条，双标注，每条 1-2 分钟，约 15-30 人时。

### L2-2 状态转移与人工 relation 的一致性
**目标**：证明 selector 推的 atom 状态（S/R/Q/C）与人类标的 relation 一致。

**依赖**：L2-1 的 evidence map gold（250 pair）。

| 指标 | 计算 |
|---|---|
| 状态-relation 一致率 | gold relation=support → trace state∈{S}；gold=refute → state∈{R}；gold=mixed/conflict → state∈{Q,C} 的比例 |
| 状态转移方向正确率 | OPEN 后 state 是否与 gold relation 方向一致 |

**脚本**：`evaluate_state_relation_consistency.py`（join trace 的 mrec_steps 与 gold annotation）。
**待办**：依赖 L2-1 完成。

### L2-3 证据覆盖与弱 gold 对照（无需新标注）
**目标**：在有 gold evidence 的数据集上，度量 trace 选中的证据是否覆盖了 gold。

| 数据集 | gold 来源 | 指标 | 现状 |
|---|---|---|---|
| RAWFC | `evidence` 列表（相关证据句） | trace 选中集 ∩ gold / gold（recall） | ⚠️ 需 join |
| HoVer | `supporting_facts`（title+句号） | trace 选中集 ∩ gold / gold | ❌ HoVer 未跑 |
| LIAR-RAW | `explain`（自然语言 rationale，非结构化） | 仅做定性对照 | 弱 |

**脚本**：复用 `scripts/phase3_oracle_evidence/evaluate_candidate_pool_recall.py` 的逻辑，改为评估 `selected_evidence_ids` 对 gold 的 recall。
**待办**：编写 `evaluate_trace_evidence_recall.py`；HoVer 需先跑通主方法。

---

## L3. 对判别的贡献（消融/扰动，无需 gold）

这一层证明"证据链的每个组成部分都对 verifier 判定有贡献"。**这组实验无需 gold，直接看性能变化，最易执行**。

### L3-1 证据链组件消融（verifier 性能驱动）
**目标**：逐步移除证据链的成分，看 verifier 性能下降多少。

| 消融条件 | 做法 | 预期 |
|---|---|---|
| **full trace**（主方法） | 原样 | baseline |
| 去掉 evidence map 标注 | selector 特征 mask 掉 map 维度，重跑 trace | 性能↓ |
| 去掉状态转移信息 | verifier prompt 里只给证据列表，不给 operation/atom/state | 性能↓ |
| 去掉 atom 关联 | prompt 里去掉 cue_text/atom proposition | 性能↓ |
| shuffle trace（s6 已有） | 打乱证据顺序 | 性能↓（已验证 f1 0.329 vs 0.367） |

**做法**：在 `build_trace_verifier_data.py` 的 prompt 渲染层加开关（`--no-map / --no-state / --no-atom / --shuffle`），生成多套 build jsonl，跑同一 verifier。
**待办**：实现渲染开关；跑 LIAR/RAWFC test。

### L3-2 单步贡献分析（step-level ablation）
**目标**：度量证据链中每一步（尤其是 BRIDGE/CONTRAST）对判别的边际贡献。

| 分析 | 做法 |
|---|---|
| 移除所有 BRIDGE 步 | 从 `mrec_steps` 过滤 operation=BRIDGE，重渲染 prompt，看 verifier 性能 |
| 移除所有 CONTRAST 步 | 同上，过滤 CONTRAST |
| 只保留 OPEN 步 | 只留 OPEN，看性能（最小证据链） |
| 逐步 leave-one-out | 对每步移除后看性能变化，画"步贡献曲线" |

**做法**：复用 verifier build 产物 + 一个后处理脚本过滤 `mrec_prompt_steps`，重新渲染 + 推理。
**脚本**：新写 `ablate_trace_operations.py`。
**待办**：实现过滤脚本；跑 leave-one-out（计算成本较高，可抽样 200 条）。

### L3-3 evidence map confidence 校准
**目标**：证明 `map_confidence` 是有意义的（高置信度标注更准、对判别更有用）。

| 分析 | 做法 |
|---|---|
| confidence 分桶下的 verifier 性能 | 按 trace 中证据的 mean map_confidence 分桶，看各桶 verifier acc |
| 去掉低置信证据 | 只保留 map_confidence > 阈值的证据，看性能变化 |
| 与 L2-1 的 ECE 对照 | 人工 gold 验证 confidence 校准度 |

**待办**：依赖 verifier 产物 + L2-1（后者可选）。

---

## 待办总表（按优先级 + 依赖排序）

### P0 — 论文必须，数据已就绪（无需新标注）
| # | 待办 | 数据 | 脚本 | 工作量 |
|---|---|---|---|---|
| 1 | 编写 `summarize_trace_statistics.py` 聚合 trace 统计 | `mrec_trace_*.jsonl` ✅ | 新写（~50 行） | 0.5 天 |
| 2 | 编写 `validate_state_transitions.py` 校验状态转移 | `mrec_steps` ✅ | 新写（~80 行） | 0.5 天 |
| 3 | 筛选 3-5 个 case 渲染证据链图（L1-3） | `render_evidence_chain_graph_html.py` ✅ | 复用 | 0.5 天 |
| 4 | shuffle trace 已有（s6），补 no-map/no-state/no-atom 消融（L3-1） | verifier build ✅ | 加渲染开关 | 1-2 天 |
| 5 | leave-one-out step ablation（L3-2，抽样） | verifier build ✅ | 新写过滤脚本 | 1 天 |

### P1 — 需要 gold，标注基础设施已就绪
| # | 待办 | 依赖 | 工作量 |
|---|---|---|---|
| 6 | **导入 450 条标注任务到 Label Studio 并完成双标注** | `exp{1,2}_tasks.jsonl` ✅ | 15-30 人时 |
| 7 | 编写 `evaluate_annotation_agreement.py`（κ/ρ/ECE） | #6 完成 | 1 天 |
| 8 | 编写 `evaluate_state_relation_consistency.py` | #6 完成 + trace ✅ | 0.5 天 |

### P2 — 需要数据集/流程补全
| # | 待办 | 依赖 | 工作量 |
|---|---|---|---|
| 9 | HoVer 主方法跑通后，做 evidence recall（L2-3） | HoVer 配置 ❌ | 依赖 HoVer |
| 10 | RAWFC evidence recall（L2-3） | RAWFC trace ✅ | 0.5 天 |
| 11 | confidence 校准分析（L3-3） | verifier 产物 ✅ + #6 可选 | 1 天 |

---

## 论文可解释性章节结构建议

```
4.5 Interpretability Analysis
  4.5.1 Evidence Chain Structure（L1-1, L1-2：统计 + 状态转移合法性）
  4.5.2 Case Study（L1-3：3-5 个证据链图）
  4.5.3 Alignment with Human Judgment（L2-1, L2-2：κ/ρ/ECE + 状态-relation 一致率）
  4.5.4 Contribution of Chain Components（L3-1, L3-2, L3-3：消融 + step 贡献）
```

**关键卖点**：
- 4.5.1 用数据证明证据链是"结构合理的中间过程"（非法转移率 0、71.7% 解析达标）。
- 4.5.3 是最硬的——用人工 gold 证明 LLM 标注和 selector 状态与人类一致。
- 4.5.4 用消融证明证据链每个部分都有用（shuffle 已证顺序重要，no-state 证状态有用）。

---

## 关键文件路径

| 用途 | 路径 |
|---|---|
| Trace 产物（LIAR val） | `outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec/mrec_trace_val.jsonl` |
| Verifier build 产物 | `outputs/selectors/atom_anchor/liar_raw_abc_v0_1/06_verifier_data/build/build_val.jsonl` |
| Claim atoms（完整字段） | `outputs/selectors/atom_anchor/liar_raw_abc_v0_1/01_claim_atoms/claim_atoms_val.jsonl` |
| LLM evidence map 标注 | `outputs/selectors/atom_anchor/liar_raw_abc_v0_1/04_evidence_map/deepseek_evidence_map_annotations_val.jsonl` |
| 人工标注项目 | `docs/paper/aaai/annotation_project/`（README.md, annotation_guideline.md, data/, config/） |
| 标注任务数据 | `docs/paper/aaai/annotation_project/data/exp1_tasks_flat_zh.jsonl`（257 条） |
|  | `docs/paper/aaai/annotation_project/data/exp2_tasks.jsonl`（250 条） |
| 标注导出脚本 | `docs/paper/aaai/annotation_project/scripts/export_tasks.py` |
| 证据链可视化 | `scripts/phase5_selectors/visualize/render_evidence_chain_graph_html.py` |
| 候选池 recall 评估（可复用） | `scripts/phase3_oracle_evidence/evaluate_candidate_pool_recall.py` |
