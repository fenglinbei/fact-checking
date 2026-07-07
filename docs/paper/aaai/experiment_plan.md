# 实验展示规划（AAAI）

本规划基于对 `configs/experiment/`、`outputs/runs/`、`outputs/sentence_trace_method/` 与 LLM 调用代码的全面探查，对照论文方法的核心贡献（有序证据链 $\mathcal{T}$），列出需要展示的实验、当前缺口，以及 API 调用可靠度/可复现性的补充实现。

---

## 0. 核心贡献与对应的实验主线

| 核心贡献 | 对应实验 | 现状 |
|---|---|---|
| ① 有序证据链驱动判别（vs 无序/无选择/复杂组织） | 主表 + selector 机制消融 | 主表 LIAR/RAWFC 有，HoVer 缺；selector_mech s0–s4 test 缺 |
| ② atom-conditioned 全流程（chunking→union→map→selector） | 组件消融（chunking / union / map / selector） | chunking 消融有；union/map 消融缺 |
| ③ 学习型边际选择器（vs 启发式/oracle） | selector 对照（learned vs proxy-rule vs reward vs greedy） | proxy 主方法有；reward/greedy 部分有 |
| ④ prompt evidence policy（动态截断） | policy 消融 + k_min/k_max sweep | LIAR 齐；RAWFC/state_budget 部分 |
| ⑤ 可解释性（证据链人类可读） | case study + 标注一致性 | 未做 |
| ⑥ 跨数据集/跨 backbone 通用性 | LIAR/RAWFC/HoVer × 多 backbone | HoVer 全缺；backbone 部分 |

---

## 1. 主实验：与外部基线对比（主表）

### 目标
证明"简单编排的证据链 + LLM 微调"超越复杂证据组织方法，且与引入外部证据/闭源模型的 agent 方法具竞争性。

### 实验设计
- **数据集**：LIAR-RAW（6 类）、RAWFC（3 类）、HoVer（2 类）三个公开基准。
- **本文方法**：`mrec_v0.2` 主配置（learned_marginal_proxy + minmax5_10，Ministral-3-8B + LoRA）。
- **对比组**：
  1. **无证据组织**：直接 top-k 证据 → verifier（对应 b0/b3_mmr_topk_sweep）。
  2. **复杂证据组织（文献复现/数值）**：CofCED（证据特征级联）、L-defense（辩护）、G-defense（辩护图）、FactLLaMA、DeReC、FFRR、KG-CRAFT、HiSS、RAFTS。
  3. **Agent / 闭源模型**：DelphiAgent；GPT-4o / DeepSeek-V4 直接 zero/few-shot（含/不含检索证据）。
  4. **本文方法变体**：去除证据链的 atom-union top5（`v0_7_*_s4_union_top5`）。
- **指标**：Accuracy、Macro-F1（主）、per-class P/R/F1、混淆矩阵。LIAR 额外报 ordinal MAE 与 extreme error rate（六类有序性）。

### 现状与缺口
- ✅ LIAR/RAWFC 主方法 test 已有（LIAR acc 0.360/f1 0.367；RAWFC acc 0.635/f1 0.638，baseline20 版 0.660/0.661）。
- ✅ DeepSeek-V4 闭源对照有 RAWFC val（需补 test）。
- ❌ **HoVer 完全没有配置/产物**——需新建 `mrec_v0.2/hover_*` 配置（2 类标签 schema、HoVer 的 claim+evidence 加载）并跑通全流程。
- ❌ **外部 SOTA 全靠文献数值**，无复现——可接受，但需在表注标明数据来源与是否同 split。
- ❌ **闭源模型仅 DeepSeek 一项且仅 val**——补 GPT-4o（或 Claude）在 LIAR/RAWFC/HoVer test 上的 zero-shot + RAG 结果。
- ❌ **LIAR 6 类下内部 baseline（b0/b3）无完整 test**——需补 b0、b3_label_token_ce_1024 的 test 评估，构成同设定对照。

---

## 2. 组件消融：各模块的必要性

### 2.1 Chunking 消融（证据粒度）
- **对比**：sentence / ctx_window(k=1) / semantic(θ=0.7) / abc_claim_aware（主）/ raw(整 report)。
- **控制**：固定 atom-union top5 + verifier，只换 chunking。
- **现状**：✅ LIAR 已跑（`v0_7/` 组 + `chunk_*_s4_union_*` 产物，acc 0.33–0.34）。
- **缺口**：补 RAWFC 上的 chunking 消融（RAWFC 现仅有 abc_tight 变体）；HoVer 同步。

### 2.2 Atom-Union 候选池消融
- **对比**：① 仅 baseline 池（整 claim 检索）；② 仅 atom-route 池；③ Atom-Union（主，baseline+atom 融合）；④ + MMR 去冗余。
- **现状**：❌ 缺——需要新建配置或后处理脚本，从同一批 chunks 出发分别构造四类候选池，喂同一 selector/verifier。
- **意义**：证明"atom 各自检索 + 融合"比"整 claim 一次检索"召回更高，是 union 的核心论点。

### 2.3 Evidence Map 消融
- **对比**：① 无 map（selector 仅用检索分数+新颖性）；② map 但无 directness/confidence；③ 完整 map（主）。
- **现状**：❌ 缺——需在 selector 特征计算里 mask 掉 map 相关维度（`phi_conf/phi_map/phi_res` 退化为 0 或常数），跑同一 verifier。
- **意义**：证明 LLM 结构化标注 $M(u_j,a_i)$ 对 selector 决策的价值。

### 2.4 Selector 机制消融（核心，已有部分）
- **对比**（selector_mech s0–s6）：
  - s0 无证据 / s1 随机 / s2 hybrid / s3 hybrid+MMR / s4 atom-union source-score / **s5 map-quality-greedy（无学习）** / **s6 trace shuffle（破坏顺序）** / **主方法 learned-marginal-proxy**。
- **现状**：✅ s5/s6 + 主方法有 test；❌ **s0–s4 仅有 val，test 目录为空**——需补跑 test 评估。
- **关键补充**：s6（shuffle）已证明顺序重要（test f1 0.329 vs 主方法 0.367），需在论文里作为"有序性"的直接证据重点展示。

---

## 3. 超参敏感性分析

### 3.1 prompt evidence policy（已有，需补全）
- **对比**：fixed_topk / minmax(k_min,k_max) / budget / resolve_stop / state_budget / two_pass_uncertainty。
- **sweep**：k_min/k_max ∈ {(5,5),(5,10),(7,7),(9,9),(3,10),(7,12)}。
- **现状**：✅ LIAR 齐；❌ state_budget test 未完成；❌ RAWFC 仅 minmax5_10，其余 policy 缺。
- **缺口**：补 state_budget(LIAR) test；补 RAWFC 的 policy sweep。

### 3.2 ρ_target sweep（缺）
- **现状**：配置里 `target_resolved_rate` 固定 1.0，论文方法写默认 0.80——**配置与论文不一致**，需对齐并 sweep ρ_target ∈ {0.5, 0.66, 0.80, 1.0}。
- **意义**：证明"解析率目标"对证据链长度与性能的权衡。

### 3.3 backbone 与训练方式
- **对比**：Ministral-3-8B（LoRA）/ Llama-3.1-8B（LoRA + FullFT）/ Qwen3-4B。
- **现状**：RAWFC Llama-3.1 配置有（`rawfc_llama31_*`），产物待确认；LIAR 跨 backbone 部分（Qwen3-4B 偏弱）。
- **缺口**：补 LIAR 上 Llama-3.1-8B 的主方法 run；统一三 backbone 在两数据集上的结果。

---

## 4. 可解释性实验（定性 + 定量）

这是论文立意的差异化点，必须做。

### 4.1 Case Study（定性）
- 选取 LIAR/RAWFC 各 3–5 个典型案例（true/false/pants-fire 各一），展示：
  - claim → atoms 分解结果
  - evidence map 标注（关系/directness/confidence）
  - 完整证据链 $\mathcal{T}$（每步的 operation、atom 状态转移 $U→S/R/Q$）
  - verifier 判定与证据链的对应
- **重点**：展示"冲突证据如何触发 C 状态""CORROBORATE 如何强化""BRIDGE 如何补背景"。

### 4.2 证据链质量量化（定量）
- **atom 覆盖率**：$\mathcal{T}$ 覆盖的 atom 数 / 总 atom 数。
- **状态转移合理性**：人工抽检 100 条 trace，统计状态转移是否合理（如无 $S→U$ 回退、$C$ 是否真有立场冲突）。
- **与人类标注的一致性**：抽 50–100 条，让标注者独立选出"最相关证据 top-5"，计算与 selector 选择的重叠率（top-k precision）。
- **现状**：❌ 全缺——需新建评估脚本（可放 `scripts/phase5_selectors/eval/`）。

### 4.3 证据链长度分布
- 统计 $\mathcal{T}$ 的 $T$ 分布、$K^\ast$（截断后）分布、atom 解析率随步数变化曲线。
- 现有 trace 产物（`mrec_trace_*.jsonl`）已含 `trace_state.resolved_atom_rate`，可直接统计。

---

## 5. API 调用可靠度与可复现性补充

### 5.1 当前薄弱点（探查确认）

| # | 位置 | 问题 | 严重度 |
|---|---|---|---|
| 1 | `claim_atomization.py:129-143` | `seed` 字段在 settings/fingerprint 中存在但**未放入 HTTP payload**，闭源 API 实际采样不受控 | 高 |
| 2 | `evidence_map_selector.py:86-96` | evidence map 缓存键**不含 temperature/top_p/thinking_type**，改采样参数会复用旧标注（静默不一致） | 高 |
| 3 | `evidence_map_selector.py` | prompt 文本改动**不触发缓存失效**（只靠 prompt_version 字符串，无 sha 兜底，不像 atomization） | 中 |
| 4 | `annotate_evidence_maps_deepseek.py:361` | schema 解析失败重试上限被硬编码 `min(2, attempts)`，远小于 max_retries=4 | 中 |
| 5 | `infer/deepseek.py:81-98` | DeepSeek 推理缓存键**不含 temperature/max_tokens** | 中 |
| 6 | `infer/api.py:81-132` | vLLM 推理**无 retry、无请求级缓存**，server OOM 即整轮重跑 | 中 |
| 7 | `claim_atomization.py` | atomization **无 RateLimiter**（并发 128，仅靠服务端限流 + transient retry） | 低 |
| 8 | 闭源 API | `deepseek-v4-flash` 模型版本可能被服务商静默更新，无法 bit-level 复现 | 不可消除 |

### 5.2 需要补充的实现

#### (A) seed 真正传入 payload（修复 #1）
- `claim_atomization.py:129-143` 的 `generate()`：在 payload 加 `"seed": self.settings.seed`（DeepSeek/OpenAI 兼容端点支持 seed 字段）。
- `question_decomp.py` 同步修复。
- 注意：闭源 API 即便传 seed 也不保证 bit-level 一致，但能提高近似可复现性，且零成本。

#### (B) evidence map 缓存键纳入采样参数（修复 #2、#3）
- `evidence_map_selector.py:86-96` 的 `evidence_map_annotation_key`：把 `temperature/top_p/max_tokens/thinking_type` 与 `system_prompt_sha256/user_prompt_sha256` 纳入 fingerprint（对齐 atomization 的 `atom_config_fingerprint` 设计）。
- 同时在 manifest 里记录这些值，便于审计。
- **这是最关键的一致性修复**——避免"改了 thinking_type 但数据没变"的静默 bug。

#### (C) DeepSeek 推理缓存键补全（修复 #5）
- `infer/deepseek.py:81-98` 的 `_stable_request_key`：加入 `temperature/max_tokens`。

#### (D) evidence map schema 失败重试上限对齐（修复 #4）
- `annotate_evidence_maps_deepseek.py:361`：把 `min(2, attempts)` 改为 `max_retries`（与 transient 失败同等对待），并在最终失败时发出显式告警（log warning + 统计失败率写入 manifest）。

#### (E) vLLM 推理增加轻量 retry（修复 #6）
- `infer/api.py` 的 `OpenAICompletionsClient.complete()`：加 2–3 次指数退避 retry（针对 URLError/Timeout/5xx），避免单次网络抖动导致整轮重跑。
- 不需要请求级缓存（vLLM 本地 + temperature=0 基本确定）。

#### (F) atomization 增加 RateLimiter（修复 #7）
- `generate_claim_atom_cache.py`：复用 evidence map 的 `RateLimiter`（默认 rpm 可设高些，如 2048），与并发数解耦。

#### (G) 可复现性声明与产物固化（应对 #8）
- 在每个 LLM 标注产物（atom_cache / evidence_map_annotations）的 manifest 里固化：`model`、`prompt_version`、`prompt_sha256`、`temperature`、`seed`、`api_base_url`、生成时间戳。
- 论文里明确声明：**可复现性依赖已保存的标注缓存**（raw_responses 已落盘），而非重新调用闭源 API；提供缓存文件的 checksum 供审稿验证。
- 开源时发布：① 标注缓存（atom + evidence_map 的 jsonl）；② selector 训练得到的权重（`learned_marginal_weights`）；③ verifier checkpoint；④ 完整 `train.resolved.yaml`。

### 5.3 优先级
- **必须做（影响实验正确性）**：B（evidence map 缓存键）、D（schema 重试）、A（seed 入 payload）。
- **建议做（提升稳健性）**：C、E、G。
- **可选**：F。

---

## 6. 实验执行优先级（建议顺序）

### P0 — 论文成立的前提
1. 补 HoVer 配置与全流程 run（主方法 + 基础基线）。
2. 补 selector_mech s0–s4 的 test 评估。
3. 修复 evidence map 缓存键（5.2-B），重跑受影响的标注与下游 selector/verifier，确保结果一致。
4. 对齐 ρ_target（配置 1.0 → 论文 0.80，或更新论文为 1.0 并说明）。

### P1 — 主表与核心消融
5. 补 LIAR b0/b3（6 类）test，构成同设定内部对照。
6. 补闭源模型（GPT-4o 或 DeepSeek）在 LIAR/RAWFC/HoVer test 的结果。
7. 补 state_budget(LIAR) test 与 RAWFC policy sweep。
8. 补 atom-union 候选池消融（2.2）。

### P2 — 可解释性与敏感性
9. Case study + 证据链质量量化（4.1、4.2）。
10. evidence map 消融（2.3）。
11. ρ_target sweep（3.2）。
12. backbone 消融补全（3.3）。

### P3 — 工程稳健性
13. API 可靠度修复 5.2-A/C/D/E/F。
14. 产物固化与可复现性声明 5.2-G。

---

## 7. 论文实验章节结构建议

```
4. Experiments
  4.1 Experimental Setup（数据集、指标、基线、实现细节含 LLM 版本/seed/缓存策略）
  4.2 Main Results（主表：三数据集 × 本文 vs 外部基线 vs agent/闭源）
  4.3 Ablation Study
    4.3.1 Component Ablation（chunking / atom-union / evidence-map / selector 机制）
    4.3.2 Prompt Evidence Policy（policy 对比 + k_min/k_max sweep + ρ_target sweep）
  4.4 Generalization（跨 backbone、跨数据集）
  4.5 Interpretability Analysis（case study + 证据链质量量化 + 与人类一致性）
  4.6 Reproducibility（API 缓存策略、seed、产物固化、可复现性声明）
```
