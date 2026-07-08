# API 标注可信度与流程可靠度：完整实验计划

本文方法的整条证据链 $\mathcal{T}$ 建立在两处 LLM 标注之上：claim atomization（拆原子命题）与 evidence map（$M(u_j,a_i)=(r_{ij},d_{ij},c_{ij})$）。最终判别器（verifier）是本地微调的开源模型，vLLM 推理 temperature=0 取 argmax，给定 checkpoint 完全确定；闭源 API 仅用于生成中间标注与作为对照基线。因此论证分两条正交主线：

- **流程可靠度（reliability）**：重跑一不一致、残余方差多大——回答"会不会出错"。
- **标注可信度（credibility）**：LLM 生成的 atoms 与 map 内容对不对——回答"标注能不能信"。

两者都要做、且都要在代码修复后跑：只做可靠度会被质疑"稳定地错"；只做可信度会被质疑"对一次不代表每次对"。

---

## 前置：必须先完成的代码修复

以下修复影响实验测量的正确性，必须在所有实验开跑前落地：

| 编号 | 位置 | 修复 | 影响的实验 |
|---|---|---|---|
| R-A | `src/fact_checking/selectors/claim_atomization.py:129-143` | `seed` 字段放入 HTTP payload | 可靠度实验 1、2 |
| R-B | `src/fact_checking/selectors/evidence_map_selector.py:84-96` | evidence map 缓存键纳入 `temperature/top_p/max_tokens/thinking_type` 与 prompt sha256 | 可靠度实验 1、2 |
| R-D | `scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py:361` | schema 失败重试上限从 `min(2, attempts)` 改为 `max_retries` | 可靠度实验 3（fallback 统计） |

未修复时，R-B 会导致"改 thinking_type 却静默复用旧标注"，使可靠度实验测得的方差被人为压低，结论失真。

---

## 第一部分：流程可靠度实验

### 可靠度实验 1：标注稳定性 / 残余非确定性量化

**问题**：闭源 API 即便 temperature=0，因服务端 batching/路由仍可能返回不同结果。要量化这个"残余方差"有多大。

**数据与设置**：
- 固定 200 条 claim（LIAR-RAW 100 + RAWFC 100），从现有验证集采样。
- **关闭缓存**（或用独立 cache 目录），对每条 claim 重复调用 $K=5$ 次。
- 同时覆盖 atomization 与 evidence map 两个标注点。
- 固定 `temperature=0, top_p=1.0, seed=20260526`（修复 R-A 后 seed 真正生效）。

**测量指标**：

| 标注点 | 指标 | 计算方式 |
|---|---|---|
| atomization | atom 数量一致率 | 5 次调用 atom 数相同的 claim 占比 |
| atomization | proposition exact-match 率 | 5 次 proposition 文本集合完全相同的 claim 占比 |
| atomization | (atom, query) 对集合 Jaccard | 5 次两两 Jaccard 的均值 |
| evidence map | relation 一致率 | 每个 (evidence, atom) pair 5 次 relation 众数一致占比 |
| evidence map | relation Cohen's κ | 5 次两两 κ 的均值（排除随机一致） |
| evidence map | directness Spearman 相关 | 5 次两两 Spearman ρ 的均值 |
| evidence map | confidence 标准差 | 每个 pair 5 次 $c_{ij}$ 的平均标准差 |

**预期产出**：一张表给出上述 7 个指标，如"atomization exact-match 98.2%；evidence map relation 一致率 95.4%（κ=0.89）"。

**判断标准**：κ ≥ 0.7 视为标注基本稳定；若 <0.7 需在论文中作为 caveat 说明并讨论。

**作用**：给"temperature=0 下标注基本稳定"一个数字证据，而非空口声明。

---

### 可靠度实验 2：标注方差对下游的端到端影响

**问题**：审稿人真正关心的是——标注抖动会不会让最终 F1 漂得没边？

**设置**：
- 用可靠度实验 1 得到的 5 套标注变体，各自跑完整下游：selector 构链 → verifier 微调 → 推理。
- verifier 用同一 checkpoint 初始化、同一 seed、同一数据划分，唯一变量是中间标注。
- 在 LIAR-RAW 与 RAWFC 的 test set 上各得 5 个 macro-F1。

**测量指标**：
1. **final F1 均值 ± 标准差**：5 次重复的 macro-F1 的 $\mu \pm \sigma$，如 $0.367 \pm 0.003$。
2. **trace 重叠率**：对同一 claim，5 套标注产出的证据序列 $\mathcal{T}$ 的平均重叠率（按位置/按集合分别算）。
3. **增益-方差对比**：把方法相对 baseline 的增益（如 +3 F1）与残余标准差（如 ±0.3 F1）并列，直观显示"增益远大于抖动"。

**判断标准**：$\sigma < $ 方法与 baseline 的 F1 差距的 1/3，则残余非确定性不掩盖方法增益。

**作用**：直接回答"结果可复现吗"。这是可靠性论证的核心一招——若 ±0.003，则方法增益可信。

---

## 第二部分：标注可信度实验

### 可信度实验 1：Atom 忠实性与完整性（人工评测）

**问题**：LLM 拆出的 atoms 会不会凭空编造、会不会漏掉可验证事实？

**数据**：抽 200 条 claim（LIAR-RAW 100 + RAWFC 100），从训练/验证集采样。

**标注协议**：
- 由 2 位标注者独立逐条评，每条 claim 的每个 atom 在三个维度打分：
  - **忠实性（faithfulness）**：atom 是否能从 claim 推出，有无引入外部知识/幻觉（binary：是/否）。
  - **完整性（completeness）**：atoms 是否覆盖 claim 中所有可独立验证的事实断言（记录漏检的断言数）。
  - **原子性（atomicity）**：atom 是否"最小可验证"，有无把多个事实黏在一起（binary：是/否）。
- 标注前给统一指导书与 20 条 calibration 样本。

**测量指标**：
- atom 幻觉率 = 含幻觉的 atom 数 / 总 atom 数。
- claim 漏检率 = 至少漏一个断言的 claim 数 / 总 claim 数。
- 整体可接受率 = 三维度全通过的 atom 占比。
- **标注者间一致性（IAA）**：两位标注者在每个维度的 Cohen's κ。

**作用**：给"atoms 可作为状态变量"提供基础证据。若幻觉率 <5%、漏检率 <10%，则 atoms 可信。

---

### 可信度实验 2：Evidence Map 标注准确率（最关键）

**问题**：$M(u_j,a_i)=(r_{ij},d_{ij},c_{ij})$ 里的 relation/directness 标得对不对？这直接驱动 selector 决策。

**数据**：抽 200–300 个 (evidence, atom) pair（从可信度实验 1 的 claim 对应的候选池中采样，覆盖 support/refute/qualify/background 各类）。

**标注协议**：
- 2 位标注者独立标每个 pair 的 gold relation（support / refute / qualify / background / irrelevant）与 gold directness（1–5 有序等级）。
- 与 LLM 标注比对。

**测量指标**：
- relation **整体准确率** + **per-relation 准确率**（support/refute/qualify/background 分别报，refute 最易错重点看）。
- relation **Cohen's κ**（去掉随机一致后的真实一致度）。
- relation **混淆矩阵**（哪些类最易混淆，如 qualify↔background）。
- directness **Spearman ρ**。
- 标注者间 IAA（两位人工标注者的 κ）。

**判断标准**：κ > 0.7 基本可接受；refute 准确率若偏低需针对性说明。

**作用**：这是"selector 决策依据可不可信"的直接证据。selector 的 $\phi_{\mathrm{res}}$、$\phi_{\mathrm{tension}}$、$\phi_{\mathrm{corr}}$ 都依赖 relation，标错会直接误导构链。

---

### 可信度实验 3：Confidence 校准

**问题**：$c_{ij}$ 被 selector 当真实概率用——代码里 `max(confidence, 0.5)` 进入解析增益 $g_{ij}$、立场张力、佐证增益（见 `mrec_learned_marginal.py:252,264,266`）。它校准了吗？

**数据**：复用可信度实验 2 的 (evidence, atom) pair 与 gold relation。

**测量指标**：
- **reliability diagram**：按 $c_{ij}$ 分箱（如 0.5–0.6, 0.6–0.7, …, 0.9–1.0），每箱画"LLM 标对 gold 的频率"。
- **ECE（Expected Calibration Error）**：分箱加权绝对误差。
- 分层看：高 confidence（≥0.8）pair 的准确率是否显著高于低 confidence。

**对照分析**：
- 现有代码用 `max(c, 0.5)` 对低 confidence 做下界 clip。对比"用原始 $c$"vs"用 clip 后 $c$"vs"温度缩放校准后的 $c$"三种设定下，selector 在子集上的构链质量（与 gold trace 的重叠率）。
- 若原始 $c$ 欠校准（高 c 但准确率低），证明 clip 机制的必要性；若 clip 后接近校准，证明 clip 是有效的轻量补偿。

**作用**：证明 confidence 不是噪声，或证明 `max(c,0.5)` clip 机制补偿了失准。把代码里一个看似 ad-hoc 的设计变成有实验支撑的设计选择。

---

### 可信度实验 4：下游对 map 噪声的敏感性（"垃圾进"检验）

**问题**：map 不完美没关系，但影响有多大？这是可信度论证的收尾闭环。

**设置（两个对照）**：

**(i) 噪声注入（鲁棒性下界）**：
- 在已有 evidence map 上注入受控噪声：
  - 随机翻转 X% 的 relation 标签（X ∈ {10%, 20%, 30%, 40%}）。
  - 随机扰动 confidence（加 $\mathcal{N}(0, 0.15)$ 噪声后 clip 到 [0,1]）。
- 对每个噪声水平跑完整 selector 构链 + verifier，画 final macro-F1 vs 噪声率曲线。

**(ii) Gold map 上界（天花板）**：
- 在可信度实验 2 的人工 gold map 覆盖的 claim 子集上，用 gold map 替换 LLM map，跑下游，对比 LLM map 的 F1。
- 增益 = gold_F1 − LLM_F1，即"完美 map 能带来多少上限"。

**测量指标**：
- 噪声曲线：各噪声水平的 F1 下降幅度，如"注入 20% relation 噪声，F1 仅降 1.2%"。
- gold 增益：如"gold map 相对 LLM map 仅 +0.8 F1"。

**判断标准**：
- 若噪声 20% 只掉 <1.5 F1 且 gold 增益 <1 F1 → 方法对 map 误差鲁棒，当前 map 已"够好"。两个数字一夹，闭环可信度问题。
- 若 gold 增益大（如 +3 F1）→ 说明 map 质量是瓶颈，诚实报告并作为未来工作。

**作用**：把"标注不完美"从弱点转化为"方法鲁棒"的论点。这是审稿人攻可信度时的终极防御。

---

## 第三部分：论文说明要点

### Reproducibility 小节应包含的声明

1. **确定性分层声明**（论证框架）：
   - 最终 verifier = 本地开源模型 + temperature=0 argmax → 完全确定；
   - 中间标注 = 闭源 API + temperature=0 + seed → 近似确定（引用可靠度实验 1 的 κ）；
   - 主结果可复现性 = 标注缓存冻结（同一缓存重跑 bit-identical）。

2. **残余方差不掩盖增益**：引用可靠度实验 2 的 $\mu \pm \sigma$，说明方法与 baseline 差距远大于残余抖动。

3. **标注可信度证据**：引用可信度实验 1（atom 幻觉率/漏检率）、实验 2（map κ）、实验 3（ECE）、实验 4（噪声鲁棒性 + gold 增益）四个硬数字。

4. **"map 是软先验而非硬门"的设计论证**：selector 用 12 维特征聚合，map 相关项（$\phi_{\mathrm{res}},\phi_{\mathrm{conf}},\phi_{\mathrm{map}}$）只是其中一部分，且 `max(c,0.5)` 做下界 clip。用可信度实验 4 证明这种设计天然抑制 map 误差。

5. **闭源模型 caveat**：明确"闭源 API 可能静默更新快照，无法 bit-level 复现；我们发布带时间戳的 raw API responses 供审计，主结果依赖冻结的标注缓存而非重新调用"。

6. **产物发布清单**：atom cache、evidence map annotations（含 raw_responses）、learned_marginal_weights、verifier checkpoint、train.resolved.yaml、缓存 checksum。

---

## 第四部分：实验执行顺序

```
阶段 0  代码修复 R-A / R-B / R-D
         ↓
阶段 1  可靠度实验 1（稳定性量化，K=5 重复标注）
         ↓
阶段 2  可靠度实验 2（端到端方差，5 套标注跑下游）
         ↓  （与阶段 1 衔接，复用其 5 套标注）
阶段 3  可信度实验 1（atom 人工评测，200 claim）
阶段 4  可信度实验 2（map 人工评测，200-300 pair）
         ↓  （阶段 3/4 可并行，人工标注周期长，尽早启动）
阶段 5  可信度实验 3（confidence 校准，复用阶段 4 gold）
阶段 6  可信度实验 4（噪声注入 + gold 上界，复用阶段 4 gold）
         ↓
阶段 7  汇总数字，撰写 Reproducibility 小节
```

**关键依赖**：
- 阶段 2 复用阶段 1 的 5 套标注，不重复调 API。
- 阶段 5、6 复用阶段 4 的 gold map，人工标注只做一次。
- 阶段 3、4 人工标注周期长，应在阶段 0/1 期间就启动标注者招募与指导书编写。

---

## 第五部分：所需资源估算

| 实验 | 标注量 | API 调用 | GPU 跑下游 | 周期 |
|---|---|---|---|---|
| 可靠度 1 | — | 200 claim × 2 标注点 × 5 次 = 2000 次调用 | — | 1–2 天（API） |
| 可靠度 2 | — | 复用可靠度 1 | 5 套 × 2 数据集 × (selector+verifier) | 3–5 天（GPU） |
| 可信度 1 | 200 claim × 2 标注者 | — | — | 5–7 天（人工） |
| 可信度 2 | 200–300 pair × 2 标注者 | — | — | 5–7 天（人工） |
| 可信度 3 | 复用可信度 2 | — | 子集 selector 跑 | 1–2 天 |
| 可信度 4 | 复用可信度 2 | — | 4 噪声水平 + gold = 10 次下游 | 3–5 天（GPU） |

人工标注是关键路径，建议立即启动可信度实验 1、2 的标注者招募与指导书编写。
