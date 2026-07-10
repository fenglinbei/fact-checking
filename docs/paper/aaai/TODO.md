# AAAI 论文总 TODO（单一事实来源）

> 本文件汇总并统一了 `docs/paper/aaai/` 下所有规划文档（`experiment_plan.md`、`ablation_gap_list.md`、`api_reliability_experiments.md`、`evidence_map_ablation_plan.md`、`interpretability_experiment_plan.md`）的待办项，**消除了它们之间的冲突**。以本文件为准。
>
> 状态图例：✅ 已完成｜🔧 进行中｜❌ 缺失（需补做）｜📝 论文写作

---

## 一、状态校准：哪些实验其实已经做完

> ⚠️ 旧版 `experiment_plan.md` 高估了缺口。经 `ablation_gap_list.md` 逐项核实，以下已**实际完成**，无需补跑：

| 实验 | 状态 | 关键数字（LIAR test） |
|---|---|---|
| Selector 机制消融 s0–s6 | ✅ 全部有 test | 主方法 acc 0.3597 / f1 0.3666；shuffle(s6) f1 0.3290 证有序性 |
| Chunking 消融（5 种） | ✅ LIAR 完整 | acc 0.33–0.34 |
| k_min/k_max 敏感性 sweep | ✅ LIAR 完整 | minmax5_10 主，含 5/7/9 固定点 + 区间 |
| Policy 对比（minmax/budget/two_pass） | ✅ LIAR 完整 | — |

**已删除/合并（不再单列）**：state_budget（不报告）、fixed_topk（=minmax k_k）、resolve_stop（已与 minmax 结合）、RAWFC 机制消融（数据集小不做）。

---

## 二、P0 — 论文成立的硬前提（必须先做）

### P0-1 ⚙️ 修复影响实验正确性的代码缺陷（前置阻塞项）
> 在所有可信度实验开跑前必须落地，否则测得的方差/稳定性被人为压低。

- [ ] **R-A** `src/fact_checking/selectors/claim_atomization.py:129-143` — `seed` 放入 HTTP payload（同时改 `question_decomp.py`）
- [ ] **R-B** `src/fact_checking/selectors/evidence_map_selector.py:84-96` — evidence map 缓存键纳入 `temperature/top_p/max_tokens/thinking_type` + prompt sha256（**最关键**）
- [ ] **R-D** `scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py:361` — schema 失败重试上限 `min(2,attempts)` → `max_retries`
- [ ] 修复 R-B 后，**重跑受影响的 evidence map 标注 + 下游 selector/verifier**，确保结果一致

### P0-2 📝 对齐 ρ_target（零实验成本，消除论文/实现硬不一致）
- [ ] 论文方法节把默认值从 0.80 改为 **1.0**，说明"实际采用更严格解析目标 ρ_target=1.0"（所有 `mrec_v0.2/` 配置实际都是 `target_resolved_rate: 1.0`）

### P0-3 ❌ HoVer 数据集全流程（主表必需，当前完全缺失）
- [ ] 新建 HoVer 数据集加载器（2 类标签 schema + claim/evidence 加载）
- [ ] 新建 `mrec_v0.2/hover_*` 主方法配置
- [ ] 跑通 HoVer 主方法 + 基础基线（chunking → atom → union → map → selector → verifier）

---

## 三、P1 — 主表与核心消融

### P1-1 ❌ Atom-Union 候选池消融（核心缺口，证明 union 各组件价值）
> 控制 selector/verifier 不变，只换候选池来源。复用同一 selector 权重与 verifier checkpoint，只需重跑 build + infer。数据集：LIAR-RAW。

- [ ] A: baseline_only（仅整 claim 检索 top-k）
- [ ] B: atom_route_only（仅 atom 各自检索 + RRF 聚合）
- [ ] C: union_no_mmr（A+B 融合但不做最终 MMR 去冗余）
- [ ] D: union_full（=主方法，复用已有产物）
- [ ] 实现方式：build 脚本加 `pool_mode ∈ {baseline_only, atom_only, union_no_mmr, union_full}` 开关，生成 4 套 build jsonl

### P1-2 ✅ Evidence Map 消融（已完成，结果写入 v0.4.1）
> 已修复 B1（`_DIRECTNESS_FACTOR` 缺 medium 键）+ B2（`_operation_for_transition` 忽略 directness）两个 bug，载体从 fixed_topk k=5 改为 **minmax5_10**（让 K\* 成为变量，map 价值才显现）。4 变体 + full_map 基准全部跑完（LIAR-RAW test），结果已写入 `writing_outline_v0.4.1.md` 的 Evidence Map Ablation 子节。

- [x] 代码：`extract_marginal_features` 加 `map_ablation_mode`（full/no_map/no_directness/no_confidence/no_relation）+ B1/B2 bug 修复
- [x] 变体 A: no_map（8 个 map 相关特征置 0）— f1 0.343
- [x] 变体 B: no_directness（$d_{ij}$ 退化为 medium=0.4）— f1 0.342，解析率正确降为 0（B2 修复生效）
- [x] 变体 C: no_confidence（$c_{ij}$ 恒为 1.0）— f1 0.332（最低，uniform 高置信引入噪声）
- [x] 变体 E: no_relation（$r_{ij}$ 退化为 background）— f1 0.355（异常韧性，待跨数据集验证）
- [x] 变体 D: full_map = minmax5_10 主方法产物（f1 0.367，最优）
- [x] 每个 no_* 变体：重训 selector 权重 → build trace → verifier 训练 + 推理
- [ ] 可选：RAWFC / HoVer 上补做，厘清 no_relation 韧性是否数据集相关

---

## 四、P2 — 可解释性与敏感性

### P2-1 可解释性实验（三层论证，详见 `interpretability_experiment_plan.md`）

**L1 过程可读性（无需 gold，数据已就绪）**

- [ ] 编写 `scripts/phase5_selectors/eval/summarize_trace_statistics.py`：聚合 `mrec_diagnostics`（长度分布、operation 分布、解析率、stop_reason）
- [ ] 编写 `validate_state_transitions.py`：校验状态转移合法性（非法转移率应=0）+ C 状态的 relation 一致性；讨论 U→U 偏高（2114）
- [ ] Case study：筛选 3–5 个代表性 case（含 S→C 冲突 / 纯 OPEN / BRIDGE 主导），用 `render_evidence_chain_graph_html.py` 渲染证据链图

**L3 对判别的贡献（无需 gold，看性能变化）**

- [ ] L3-1 证据链组件消融：`build_trace_verifier_data.py` 加渲染开关 `--no-map / --no-state / --no-atom`（shuffle 已有 s6），跑 LIAR/RAWFC test
- [ ] L3-2 step-level ablation：新写 `ablate_trace_operations.py`（移除 BRIDGE/CONTRAST、只留 OPEN、leave-one-out，抽样 200 条）
- [ ] L3-3 confidence 校准分析：按 mean map_confidence 分桶看 verifier acc（依赖 verifier 产物，#B-7 可选）

**L2 与外部标准一致（需 gold，依赖第五节标注）**

- [ ] 编写 `evaluate_annotation_agreement.py`：κ/ρ/ECE + LLM vs human（依赖标注完成）
- [ ] 编写 `evaluate_state_relation_consistency.py`：join trace 与 gold map
- [ ] 编写 `evaluate_trace_evidence_recall.py`：RAWFC evidence recall（HoVer 依赖 P0-3）

### P2-2 ❌ LIAR Llama-3.1-8B backbone
- [ ] LIAR 上 Llama-3.1-8B(LoRA) 主方法 run（RAWFC 已有，统一三 backbone 两数据集结果）

### P2-3 📝 RAWFC policy sweep 补全（次要）
- [ ] RAWFC 目前仅 minmax5_10，其余 policy 缺。可选补 budget/two_pass（优先级低于 LIAR）

---

## 五、人工标注（可信度实验，关键路径，周期长，尽早启动）

> 基础设施已就绪：指导书（413 行）、257+250 条任务已生成、Label Studio 已部署（fc.fenglin.pro），**但 DB 标注未开始**。详见 `annotation_project/`。

### B-1 启动人工标注
- [ ] 把 `exp1_tasks_flat_zh.jsonl`（257 条）+ `exp2_tasks_zh.jsonl`（250 条）导入 Label Studio（当前 DB 空）
- [ ] 2 位标注者培训 + 20 条 calibration（κ ≥ 0.6 方可进入正式标注）
- [ ] 完成双盲双标注（实验1 atom 三维：faithfulness/completeness/atomicity；实验2 map：relation/directness/confidence）
- [ ] 项目作者仲裁分歧，导出 `results/exp{1,2}_annotations_{A,B}.json`
- [ ] 预估：450 条 × 双标注，约 15–30 人时

### B-2（依赖标注完成）下游可信度分析
- [ ] **可信度实验 3（confidence 校准）**：复用实验2 gold，reliability diagram + ECE，对照 max(c,0.5) clip
- [ ] **可信度实验 4（噪声鲁棒性 + gold 上界）**：注入 relation 噪声 X%{10,20,30,40} + confidence 扰动；gold map 子集跑下游看增益

---

## 六、API 可靠度实验（依赖 P0-1 代码修复）

> 两条正交主线：流程可靠度（重跑一致吗）+ 标注可信度（标得对吗）。详见 `api_reliability_experiments.md`。

### R-1 可靠度实验 1：标注稳定性量化（200 claim × K=5）
- [ ] 固定 200 claim（LIAR 100 + RAWFC 100），关缓存，K=5 重复调用 atomization + evidence map
- [ ] 测：atom 数一致率/proposition exact-match/Jaccard；map relation 一致率/Cohen's κ/directness Spearman/confidence σ
- [ ] 判断标准：κ ≥ 0.7 视为基本稳定

### R-2 可靠度实验 2：标注方差对下游的端到端影响（复用 R-1 的 5 套标注）
- [ ] 5 套标注各自跑完整下游（selector → verifier），各得 5 个 macro-F1
- [ ] 报 final F1 μ±σ；trace 重叠率；增益-方差对比
- [ ] 判断标准：σ < 方法与 baseline F1 差距的 1/3

---

## 七、工程稳健性（P3，建议做）

- [ ] **R-C** `infer/deepseek.py:81-98` `_stable_request_key` 加 temperature/max_tokens
- [ ] **R-E** `infer/api.py` `OpenAICompletionsClient.complete()` 加 2–3 次指数退避 retry（针对 URLError/Timeout/5xx）
- [ ] **R-F** `generate_claim_atom_cache.py` 复用 evidence map 的 RateLimiter（rpm≈2048）
- [ ] **R-G** 产物固化：每个 LLM 标注产物 manifest 固化 model/prompt_version/prompt_sha256/temperature/seed/api_base_url/时间戳；论文声明可复现性依赖冻结缓存；开源清单：atom cache + evidence_map annotations + learned_marginal_weights + verifier checkpoint + train.resolved.yaml + 缓存 checksum

---

## 八、论文写作（贯穿，📝）

- [ ] **Related Work**：当前只有占位 + 文献地图（`related_work.md`），需补全为完整段落（早期 BERT 前 / BERT 类 / LLM 类：辩护、agent、图、内在表示），收束到"复杂度高、黑箱化"
- [ ] **符号约定**：补全核心符号表（$c, \mathcal{A}, \mathcal{R}, \mathcal{T}, u_j, a_i, h_i^{(t)}$ 等）
- [ ] 方法节各处 **XXX 填充**（chunking 实验引用、selector 对照引用、policy 消融引用）
- [ ] **Experimental Setup**：数据集/指标/baselines/实现细节（LLM 版本、seed、缓存策略）
- [ ] **ρ_target 对齐**（见 P0-2）
- [ ] **Reproducibility 小节**：确定性分层声明 + 残余方差不掩盖增益 + 标注可信度四数字 + map 是软先验 + 闭源 caveat + 产物发布清单（依赖 R-1/R-2/B-2 数字）
- [ ] 多份写作大纲收敛（现有 `writing_outline.md`/`v0.2`/`v0.3`/`v0.3.1`/`v0.3.2` 五份）→ 定稿保留一份
- [ ] 实验章节结构：4.1 Setup / 4.2 Main / 4.3 Ablation（组件 + policy）/ 4.4 Generalization / 4.5 Interpretability / 4.6 Reproducibility

---

## 九、依赖关系与建议执行顺序

```
P0-1 代码修复 R-A/R-B/R-D ──┬─→ R-1 可靠度1 ──→ R-2 可靠度2
P0-2 改论文 ρ_target         │
P0-3 HoVer 全流程 ───────────┼─→ 主表(4.2) ──→ L2 evidence recall(HoVer)
                              │
P1-1 Atom-Union 消融 ────────┼─→ 组件消融表(4.3)
P1-2 Evidence Map 消融 ──────┘

B-1 人工标注(尽早启动) ──→ B-2 可信度3/4 ──┬─→ L2 κ/ρ/ECE
                                           └─→ Reproducibility 小节

P2-1 L1/L3(无需gold) ─→ 可解释性(4.5)
P2-2 LIAR Llama backbone ─→ Generalization(4.4)
P3 工程稳健性 ─→ Reproducibility(4.6)
```

**关键路径**：B-1 人工标注（周期最长，应立即招募标注者）；P0-1 代码修复（阻塞所有可靠度实验）；P0-3 HoVer（阻塞主表完整性）。

---

## 十、待决策/悬而未决问题

1. **外部 SOTA 是否复现**：当前全靠文献数值，无复现——表注需标明数据来源与是否同 split。是否要补 1–2 个复现？
2. **ρ_target 最终值**：论文改 1.0（当前方向），还是补 sweep 证明 0.80 更优？（sweep 降为 P3）
3. **多份写作大纲归并**：五份 outline 保留哪一份作主稿？
