# Figure 1 视觉方案：Structure-Induced Evidence Organization

对应方法稿：`writing_outline_v0.4.2_structure_only.md`

## 一句话任务

主图需要让读者在十秒内看懂：**typed Evidence Map 的 prefix-conditioned 状态变化如何在不使用 verdict label、gold evidence order 或 verifier feedback 的情况下产生 selector supervision，并最终投影为 verifier 可见的、保持原顺序的证据前缀。**

## 推荐叙事

采用 AAAI 双栏通栏的横向三层结构，而不是将方法画成七步等权流水线：

1. **Upstream Input Construction（蓝色，支撑层）**：claim-only atomization 与 source-linked report chunking 形成 atoms、queries 和 evidence units；claim route 与 atom routes 融合为 Atom-Union candidate pool。该层只构造候选池，不决定最终 evidence order。
2. **Structure-Induced Evidence Organization（橙色，核心层）**：Typed Evidence Map 提供 candidate--atom relations 与 map-induced state；硬结构优先级在每个 prefix state 生成 winner-vs-rest pairs，再将这些离散约束蒸馏为连续的 state-conditioned scorer。冻结 scorer 在完整候选池上循环执行 re-score、select、state update，输出 source-linked ordered audit trace。
3. **Prefix Projection & Verification（绿色，下游层）**：独立 prefix policy 只沿 audit ordering 截取前缀，并经 tokenizer context guard 做 tail deletion；它不重新评分、不重新排序。Renderer 仅暴露 claim、atom cues 与 raw evidence text，label-token verifier 输出最终标签。

视觉权重建议为上游 22%、核心 52%、下游 26%。视觉中心应是 **Evidence Map → preference distillation → state-conditioned re-scoring loop**。

## 主图中的四个关键表达

### 1. Evidence Map 是 selection state，不是 verdict graph

用小型二部映射连接 atoms 与 evidence units，边上直接标注 `S / R / Q`，并同时保留 source ID。图内公式只保留：

\[
M(u_j,a_i)=(r_{ij},d_{ij},\mathrm{conf}_{ij}).
\]

状态条写作 `H^(t)={h_i^(t)} ≠ y`：`h_i^(t)` 可承载 `S/U/R/Q/C` 等 evidence-acquisition state，但不把有限示例节点误画成固定维度，也避免用 `c` 同时指 claim 和 confidence。

### 2. 规则只产生训练监督

在核心区单独画虚线训练支路：

`Structural priorities → winner vs. rest → pairwise distillation → learned θ`

其中只概括三档优先级：direct progression、new coverage/relation、quality/retrieval tie-break。核心区顶部的排除带明确写：

`No verdict y · No gold/teacher order · No verifier feedback`

这避免把硬规则误画成推理阶段的最终 selector，也避免把 learned score 描述为 true utility。

### 3. Audit trace 与 visible prefix 是两个不同 artifact

核心区输出完整 `T^audit` 卡片栈，每步保留 source、state transition、score 与 provenance；下游区复制该顺序并只高亮前 `K*` 项，灰化尾部，得到：

\[
T^{vis}=T^{audit}_{1:K^*}.
\]

图内用 `prefix · fixed order` 保证最小字号，caption 再完整说明 `prefix only · no re-scoring · no re-ordering`。图中的 5 个高亮 evidence unit 只是最小前缀示意，实际长度仍由 `K*` 的 count / cover / context guard 决定。不要把 `T^audit` 称为 reasoning chain、proof path 或 faithful explanation。

### 4. Gold label 只进入 verifier training

只在 verifier 卡片旁画一条来自 `gold y` 的蓝色虚线训练箭头，标注 `training only`；不得出现 verifier 向 selector、Evidence Map 或 preference generation 回流的箭头。

## 主图保留与省略

主图最多保留以下四个符号；当前紧凑版将第二项压缩为 `Winner vs. rest` 文本：

- `M(u,a)`：typed candidate--atom alignment；
- `P_t={(u_t^+,u^-)}`：winner-vs-rest supervision；
- `S_θ^str(u | T_<t,H^(t))`：state-conditioned structural score；
- `T^vis=T^audit_{1:K*}`：order-preserving prefix projection。

以下内容留在正文、算法或附录，不进入主图：完整 RankNet loss、12 个标量特征、BM25/RRF/MMR 公式、概念性 `J(π)`、完整 `K*` 定义、具体模型/checkpoint/temperature、prompt/schema 细节。

## 视觉规范

- 画布：`1600 × 680`，白底，约 2.35:1；按 `\textwidth` 通栏插入。
- 字体：Helvetica/Arial；层级标题 34、模块标题 31、所有其余可见标签不低于 29。按 AAAI 双栏约 7 英寸缩放后，最小标签约为 9.1 pt。
- 配色：上游 `#EAF2F8 / #285F8F`，核心 `#FFF3E4 / #9A4D08`，下游 `#EAF7F1 / #1F684F`，audit/supervision `#F2EDF8 / #6F5A95`。
- 实线表示推理/数据流，紫色虚线表示 selector supervision 或参数蒸馏，蓝色虚线只表示 verifier label supervision。
- 不使用渐变、阴影、外部图标或位图；relation 由文字和线型共同编码，不能只依赖颜色。

## 建议 caption

> **Figure 1: Structure-induced, label- and verifier-independent evidence organization.** Claim atoms and source-linked evidence units are fused into an Atom-Union candidate pool and aligned through a typed Evidence Map. Prefix-conditioned map transitions induce winner-vs-rest preferences, which are distilled into a continuous state-conditioned selector without verdict labels, gold evidence order, or verifier feedback. The frozen selector produces a source-linked full-pool-access audit ordering; a separate capacity and context policy exposes only an order-preserving prefix to the label-token verifier. Gold labels are used solely for downstream verifier training.

投稿时从 draw.io 导出已裁边且嵌入字体的 PDF，再以 `figure*` 和 `\includegraphics[width=\textwidth]{...}` 通栏插入；SVG 仅作为仓库预览与后续编辑参照，不直接作为 AAAI 投稿插图格式。

## 概念红线

- Atom-Union 是 candidate construction，不是最终排序器。
- Evidence Map relation 是局部 candidate--atom relation，不是最终 verdict。
- `U/S/R/Q/C` 是 evidence-acquisition state，不是 label state。
- full-pool access 仅指访问 Atom-Union 后的候选池；重复/无效项仍可跳过，因此不保证严格全排列。
- structural coverage proxy 不等于人工或逻辑意义上的 evidence sufficiency。
- Audit metadata 不进入 verifier prompt。
- Selector 不读取 verdict、gold evidence order 或 verifier feedback。
