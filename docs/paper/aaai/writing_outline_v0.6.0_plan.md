# AAAI 论文初稿计划 v0.6.0：Verifier-Grounded Marginal Evidence Selection

> **文档状态：方法重构提案，尚未冻结。**  本文档在 `writing_outline_v0.5.0.md` 的候选池、claim atomization、Evidence Map 与 verifier pipeline 基础上，将核心方法由冻结的 ordinal coverage objective 改为 **Verifier-Grounded Marginal Evidence Selection（VGMES）**。本文中的 teacher utility cache、cross-fitting、marginal predictor、adaptive stopping 与相应实验均为 v0.6.0 待实现内容，不能作为现有结果表述。BACES v0.3 和 learned marginal proxy v0.2 保留为主要 baseline，而不是被直接删除或改名。

## 0. 版本重构结论

### 0.1 v0.5.0 的核心问题

v0.5.0 将 Evidence Map 投影为每个 atom 的 ordinal coverage：

\[
q_{ij}\in\{0,1,2\},\qquad
U(S)=\sum_i w_i\max_{e_j\in S}q_{ij}.
\]

该目标具有清晰、可审计和可精确求解的优点，但只保留 direct/partial coverage。连续 confidence、support/refute direction、独立佐证、反向证据、qualify、background/context 及证据互补性没有进入主目标；同一 atom 达到 direct 后立即饱和。因此，一个手工的 direct/partial atom-coverage 策略可能近似解决该问题，限制了方法贡献及 Evidence Map 多维标注的实际价值。

早期 learned marginal proxy v0.2 已经加入 resolution、new relation、stance tension、corroboration、source novelty、map confidence 等前缀条件特征，但其监督来自人工 proxy preference，本质上学习复现人工排序，而不是学习证据对最终 fact-checking 判别的真实影响。

### 0.2 v0.6.0 的核心转向

v0.6.0 不再预先规定 direct、corroboration、opposite 和 context 各自值多少，而是将一条证据加入当前前缀后对冻结 verifier 正确判别能力的改善定义为训练阶段的边际效用：

\[
\Delta_\phi(e\mid S)
=
\log P_\phi(y^*\mid c,S\cup\{e\})
-
\log P_\phi(y^*\mid c,S).
\]

随后训练一个部署时不读取 gold label 的 marginal predictor：

\[
g_\theta(c,M,S,e)\approx \Delta_\phi(e\mid S).
\]

Evidence Map 不再直接决定人工 utility，而是作为预测反事实下游收益的结构化状态表示。选择器根据预测净边际收益依次选择证据，并在没有候选具有足够正收益时自适应停止。

### 0.3 一句话方法定义

> Given a claim, an Evidence-Map-annotated candidate pool, and a frozen fact-checking teacher, VGMES learns to predict how much each candidate would improve the teacher's correct-label evidence-conditioned decision at the current prefix, and constructs a budget-feasible evidence slate by repeatedly selecting the candidate with the largest predicted net marginal gain until no worthwhile acquisition remains.

对应中文：

> 给定声明、带 Evidence Map 标注的候选池和冻结的事实核查 teacher，VGMES 学习预测每条候选证据在当前前缀下能够带来的正确标签判别改善，并反复选择预测净边际收益最大的候选，直至没有值得继续获取的证据或预算耗尽。

## 1. 论文中心论点与贡献口径

### 1.1 中心论点

论文不再主张某个手工 Evidence Map 映射是“客观证据效用”。更稳健的主张是：

1. Evidence Map 提供了描述证据—atom 关系、方向、直接性、置信度、来源和上下文角色的结构化接口；
2. 不同维度在不同前缀下是否有用，应该由其对下游判别的实际边际影响学习，而不是由固定线性权重指定；
3. 证据容量应由预测收益与注意力/token 成本共同决定，而不是在真实部署中永远固定为同一个 K；
4. learned utility 是相对于 teacher verifier 的 decision-grounded utility，不是模型无关的绝对事实充分性。

### 1.2 计划贡献

若实验证据成立，v0.6.0 计划形成以下三项贡献：

1. **Typed fact-checking evidence-bundle utility。** 将已有的 downstream conditional-gain supervision 实例化为 atom × role × direction × provenance 条件化的多证据核查问题，系统建模重复、异源佐证、反向证据、context 和互补效应。
2. **Evidence-Map-conditioned utility surrogate with adaptive acquisition。** 使用多维 Evidence Map 和当前集合状态预测候选净边际收益，在 count/token constraints 下以同一个 candidate-specific 估计联合完成下一条选择与实例级容量分配。
3. **Leakage-controlled, interaction-aware, and transfer-oriented evaluation。** 通过 train-only utility labeling、cross-fitting 或 held-out teacher、固定候选池、singleton/stateful/coalition 近邻对照、teacher-to-student transfer 和 fixed-K/adaptive paired controls，检验收益是否来自结构化证据效用学习，而非 teacher memorization 或同模型迎合。

`add-one likelihood gain`、`stateful selector`、`teacher-to-student utility distillation` 和 `threshold stopping` 均有直接先例，不得单独写为首创贡献。详细 novelty boundary 见 `related_work_v0.6.0.md`。

若跨 verifier transfer 不成立，贡献口径必须降级为 verifier-specific evidence adaptation，不能声称通用 evidence utility learning。

## 2. 系统范围与符号

给定 claim \(c\)、标签空间 \(\mathcal Y\)、claim atoms：

\[
A(c)=\{a_1,\ldots,a_m\},
\]

以及由 claim route 与 atom routes 构造的 union candidate pool：

\[
E(c)=\{e_1,\ldots,e_n\}.
\]

Evidence Map 为 candidate--atom pair 提供：

\[
M_{ij}=(r_{ij},d_{ij},p_{ij},s_{ij},\rho_{ij}),
\]

其中 \(r\) 为 relation，\(d\) 为 directness/role，\(p\) 为 map confidence，\(s\) 为 key spans，\(\rho\) 表示 provenance/source metadata。候选另具有 token cost \(c(e)\)、retrieval scores 与 duplicate group。

输出为有序 evidence slate：

\[
\pi=(e_{\pi_1},\ldots,e_{\pi_T}),\qquad T\le K_{\max},
\]

其前缀集合为：

\[
S_t=\{e_{\pi_1},\ldots,e_{\pi_t}\},\qquad S_0=\varnothing.
\]

主文应继续使用 `ordered evidence slate`，不将其描述为 multi-hop reasoning chain。VGMES 学习的是 verifier-facing evidence acquisition policy，不显式求解事实世界中的逻辑证明。

## 3. Verifier-grounded teacher utility

### 3.1 冻结 teacher

设冻结的 fact-checking teacher 为：

\[
V_\phi(y\mid c,S).
\]

对训练样本 gold label \(y^*\)，定义 evidence-conditioned decision utility：

\[
F_\phi(c,S,y^*)=\log P_\phi(y^*\mid c,S).
\]

主定义使用 gold-label log probability，而非 0/1 accuracy，因为后者过于离散。稳健性实验可使用 gold-logit margin：

\[
m_\phi(S)=z_{y^*}(S)-\max_{y\ne y^*}z_y(S).
\]

### 3.2 单步边际效用

候选 \(e\) 在前缀 \(S\) 下的 teacher marginal gain 为：

\[
\Delta_\phi(e\mid S)
=F_\phi(c,S\cup\{e\},y^*)-F_\phi(c,S,y^*).
\]

为控制异常 log probability，对标签进行裁剪或标准化：

\[
\bar\Delta_\phi(e\mid S)
=\operatorname{clip}(\Delta_\phi(e\mid S),-\delta_{\max},\delta_{\max}).
\]

加入 token 与注意力成本后的净收益定义为：

\[
R_\phi(e\mid S)
=\bar\Delta_\phi(e\mid S)
-\lambda_{tok}\frac{c(e)}{B}
-\lambda_{step}.
\]

这里 \(\lambda_{step}\) 表示每增加一个 evidence unit 的固定注意力成本；\(\lambda_{tok}\) 表示长度成本。二者只能在 validation split 上选择。

### 3.3 序列效用

采用折扣前缀效用：

\[
J_\phi(\pi)
=\sum_{t=1}^{T}\gamma^{t-1}
R_\phi(e_{\pi_t}\mid S_{t-1}),
\qquad 0<\gamma\le1.
\]

当 \(\gamma=1\) 且不考虑 clipping 时，teacher gain 具有 telescoping 性：

\[
\sum_{t=1}^{T}\Delta_\phi(e_{\pi_t}\mid S_{t-1})
=F_\phi(c,S_T,y^*)-F_\phi(c,\varnothing,y^*).
\]

当 \(\gamma<1\) 时，目标额外奖励有用证据更早出现。主实验建议先使用 \(\gamma=1\) 隔离集合选择，再将 discounted/prefix-AUC 作为顺序扩展，避免在第一轮同时混入集合效用与位置假设。

## 4. Evidence-Map-conditioned marginal predictor

### 4.1 部署约束

测试时未知 \(y^*\)，因此不能直接计算 teacher marginal gain。训练一个不读取 gold label 的预测器：

\[
g_\theta(c,M,S,e)\approx R_\phi(e\mid S).
\]

其输入只能包含部署时可用的信息：claim、atoms、候选文本、Evidence Map、当前已选前缀、source/duplicate metadata、成本，以及可选的 teacher 当前无标签分布。gold label、gold evidence 和 test-time verifier reward 均禁止进入输入。

### 4.2 前缀状态表示

对每个 atom \(a_i\)，从已选集合聚合：

\[
h_i(S)=
[q_i^{sup},q_i^{ref},q_i^{qual},q_i^{ctx},
n_i^{sup},n_i^{ref},n_i^{src}],
\]

其中强度聚合、来源数和 duplicate-aware counts 由 Evidence Map 生成。该状态不是人工效用函数，只是 marginal predictor 的输入。候选表示包括：

- candidate--atom relation/directness/confidence/key-span features；
- 与当前 \(h(S)\) 的交互，如新 atom、新方向、同向独立来源、context complement；
- source novelty、duplicate/text similarity；
- retrieval/map quality 与 token cost；
- 可选的 claim/atom/candidate text embeddings。

### 4.3 第一版模型

第一版使用可审计的结构化模型：

\[
g_\theta(e\mid S)=\operatorname{MLP}_\theta(x(c,M,S,e)),
\]

并以 LightGBM/线性模型作为低容量对照。文本 cross-encoder 只作为第二阶段扩展。这样可以先回答“Evidence Map 多维状态能否预测真实边际收益”，而不是让大型文本模型掩盖结构贡献。

## 5. Marginal utility dataset construction

### 5.1 前缀采样

只从当前 policy rollout 采样会造成 on-policy selection bias。对每条 train claim，从以下 mixture 采样前缀：

\[
P(S)=
\rho_rP_{random}
+\rho_bP_{retrieval}
+\rho_mP_{map\text{-}greedy}
+\rho_pP_{current\text{-}policy}.
\]

初始建议为 \((\rho_r,\rho_b,\rho_m,\rho_p)=(0.25,0.20,0.25,0.30)\)，实际比例按 prefix-state coverage audit 调整。必须包含空前缀、已覆盖 direct 的过饱和前缀、存在冲突的前缀及缺少 context 的前缀。

### 5.2 反事实标签缓存

对每个采样前缀 \(S\)，先缓存：

\[
F_\phi(c,S,y^*),
\]

再对未选候选 \(e\) 缓存：

\[
F_\phi(c,S\cup\{e\},y^*).
\]

形成训练记录：

\[
\mathcal D_{marginal}
=\{(c,M,S,e,R_\phi(e\mid S))\}.
\]

缓存必须记录 teacher checkpoint、prompt contract、candidate UID、prefix UIDs、label schema、tokenization、logits/log-probabilities 与配置 hash，保证可重放。

### 5.3 防泄漏方案

最低要求：utility labels 只在 train split 生成，validation 只用于模型与停止阈值选择，test 不读取 gold label。

推荐要求：使用 cross-fitting。将 train 分为 \(L\) 折，第 \(l\) 折的 utility labels 由未在该折训练的 teacher \(V_{\phi^{-l}}\) 生成：

\[
\Delta^{(l)}(e\mid S)
=F_{\phi^{-l}}(S\cup\{e\})-F_{\phi^{-l}}(S).
\]

若 cross-fitting 成本过高，至少使用独立冻结 teacher，并明确报告其是否见过 selector-labeling 样本。没有完成这一审计前，不得使用“causal”或“unbiased utility”表述。

## 6. 学习目标

### 6.1 回归目标

\[
\mathcal L_{reg}
=\sum_{S,e}w_{S,e}\operatorname{Huber}
\left(g_\theta(e\mid S)-R_\phi(e\mid S)\right).
\]

### 6.2 前缀内排序目标

若：

\[
R_\phi(e^+\mid S)>R_\phi(e^-\mid S)+\epsilon,
\]

构造 pairwise preference，并使用：

\[
\mathcal L_{rank}
=\sum_{S,e^+,e^-}
\log\left(1+\exp[-(g_\theta(e^+\mid S)-g_\theta(e^-\mid S))]\right).
\]

### 6.3 停止监督

定义候选是否具有正净收益：

\[
z_{S,e}=\mathbf 1[R_\phi(e\mid S)>0].
\]

加入 sign classification：

\[
\mathcal L_{sign}=\operatorname{BCE}(\sigma(g_\theta(e\mid S)),z_{S,e}).
\]

联合损失为：

\[
\mathcal L
=\mathcal L_{reg}+\alpha\mathcal L_{rank}+\beta\mathcal L_{sign}.
\]

主选择依据是同一前缀内排序，回归值主要服务于 adaptive stopping，因此必须分别报告 ranking 与 calibration 指标。

## 7. 推理与自适应容量

初始 \(S_0=\varnothing\)。第 \(t\) 步从预算可行且未重复的候选中选择：

\[
e_t=\arg\max_{e\in E\setminus S_{t-1}}g_\theta(c,M,S_{t-1},e).
\]

随后更新 \(S_t=S_{t-1}\cup\{e_t\}\)。满足任一条件时停止：

1. \(t=K_{max}\)；
2. token budget 耗尽；
3. 最佳预测净收益不高于 validation-calibrated threshold：

\[
\max_e g_\theta(e\mid S_t)\le\tau.
\]

为降低过早停止风险，可以设置软下限 \(K_{min}\)，但真实部署主配置应允许 \(K_{min}=0\) 或 1；固定 K=5 仅保留为公平比较控制。需要报告 adaptive policy 的 mean/P50/P90 K、token 数、stop reason 和性能—成本曲线。

## 8. 二阶互补性扩展

一步 marginal 可能遗漏“context 单独无用、与 direct evidence 组合后有用”的互补关系。定义 pair interaction：

\[
I_\phi(e_i,e_j\mid S)
=F(S\cup\{e_i,e_j\})-F(S\cup\{e_i\})-F(S\cup\{e_j\})+F(S).
\]

第一阶段不对所有候选对做 \(O(n^2)\) 标注，只针对 Evidence Map 指示的高可能互补对采样：

- 同 atom 的 direct/context；
- support/refute；
- partial/independent corroboration；
- qualify/direct；
- 单步收益接近零但 map-relevant 的候选。

二阶模型写为：

\[
Q_\theta(e\mid S)
=g_\theta(e\mid S)+\max_{e'\ne e}h_\theta(e,e'\mid S).
\]

该扩展只有在一阶模型的 interaction diagnostic 明确显示显著互补缺口后才进入主方法；否则保留为 future work，避免方法复杂度失控。

## 9. 算法描述

### Algorithm 1：Train-time marginal utility labeling

1. 在 train split 上构造冻结的 atom-union pools 与 Evidence Maps；
2. 使用 mixture policies 为每个 claim 采样多个 evidence prefixes；
3. 对每个 prefix 运行冻结 teacher，缓存 base log probabilities；
4. 对每个 prefix--candidate intervention 运行 teacher；
5. 计算 clipped marginal gain、成本和净收益；
6. 进行 leakage、duplicate、prompt visibility 和 cache provenance audit；
7. 输出 marginal utility dataset。

### Algorithm 2：Marginal predictor learning

1. 从 Evidence Map 与 prefix 构造结构化状态；
2. 在同一 prefix 内构造 ranking pairs；
3. 联合优化 Huber regression、pairwise ranking 与 sign loss；
4. 在 validation 上选择模型、成本系数和停止阈值；
5. 冻结 selector，不在 test 上重新拟合。

### Algorithm 3：Adaptive evidence acquisition

1. 初始化空 slate；
2. 过滤已选、duplicate 和预算不可行候选；
3. 预测每个候选相对于当前 prefix 的净边际收益；
4. 若最佳收益低于阈值则停止；
5. 否则选择最佳候选、更新 Evidence Map prefix state；
6. 重复直至停止或达到硬预算。

该算法是 learned greedy policy，不具有对任意非次模 teacher utility 的全局最优保证。论文必须明确这一点。可在小候选池上通过 exhaustive/beam oracle 测量 policy regret，而不能沿用 BACES 的 exact-optimality 表述。

## 10. 实验问题

### RQ1：teacher marginal gain 是否是可学习信号？

在 held-out prefixes 上报告：

- Spearman/Kendall correlation；
- pairwise accuracy/NDCG；
- positive-gain AUROC/AUPRC；
- predicted-vs-observed calibration；
- 按 relation/directness/role、prefix K 和 atom saturation 分桶的性能。

### RQ2：Evidence Map 多维信息是否真正有用？

进行输入消融：

- no map；
- directness only；
- no confidence；
- no relation/direction；
- no source/duplicate；
- no context/background；
- no prefix state（static scoring）；
- full Evidence Map。

消融评价既包括 marginal prediction，也包括最终 fact-checking。仅展示最终 F1 不足以说明 selector 学到了对应维度。

### RQ3：是否优于人工 proxy 与结构覆盖？

主要 baselines：

1. retrieval top-K；
2. random matched-K；
3. manual direct/partial greedy；
4. learned marginal proxy v0.2；
5. BACES exact v0.3；
6. role-rescue `cor/opp/ctx/full`；
7. VGMES fixed-K；
8. VGMES adaptive。

必须在相同 candidate pool、相同 Evidence Map、相同 verifier input contract 和 matched K/token budget 下比较。

### RQ4：自适应容量是否改善性能—成本折中？

比较 fixed K=3/5/7/10、legacy minmax、resolve-stop 和 VGMES adaptive。报告：

- Macro-F1/accuracy；
- 平均 K 与 token cost；
- performance-cost Pareto frontier；
- excess evidence harm；
- early-stop false-negative rate；
- 不同 claim complexity/atom count 下的容量分布。

### RQ5：收益是否迁移到未参与标注的 verifier？

至少区分：

- in-teacher evaluation：生成 utility labels 的 teacher；
- held-out verifier：未参与 utility labeling 的同架构 checkpoint；
- cross-backbone verifier；
- 可行时的人类 evidence sufficiency/utility evaluation。

若只在 teacher 上改善而不迁移，必须将结果解释为 teacher adaptation，而不是通用 evidence quality 提升。

### RQ6：一阶边际选择遗漏多少互补性？

在小规模审计集上计算二阶 interaction，报告：

- 正/负 interaction 比例；
- direct/context、support/refute 等角色对的 interaction 分布；
- greedy 与 beam/exhaustive oracle regret；
- 是否值得启用二阶模型。

## 11. 关键实验设计与数据隔离

建议的数据职责如下：

- **Train-A：** teacher/verifier 训练；
- **Train-B 或 cross-fit folds：** marginal utility generation；
- **Train-C：** marginal predictor fitting（可与 cross-fit labels 合并）；
- **Validation：** \(\lambda_{tok},\lambda_{step},\tau,K_{max}\) 和模型选择；
- **Test：** 一次性冻结评价，不生成 gold-conditioned utility。

若数据量不足以物理拆成 A/B/C，则使用 K-fold cross-fitting，并完整记录每条 utility label 对应的 teacher training exclusion。所有 selector 对比必须固定 retrieval 和 Evidence Map artifacts，防止候选池变化混入方法收益。

## 12. 实施阶段与验收门槛

### Phase 0：可行性审计

- 冻结 teacher prompt 与 checkpoint；
- 验证能够稳定导出 label log probabilities；
- 对少量 claims 枚举 prefix--candidate interventions；
- 检查 marginal gain 的方差、正负比例和对 evidence order 的敏感性。

**Go/No-Go：** 若绝大多数 \(\Delta\) 接近零、重复运行不稳定或主要由 prompt formatting 决定，则先修复 teacher/calibration，不启动大规模标签生成。

### Phase 1：一阶 utility cache

- 设计 prefix mixture；
- 建立可恢复、可校验的 teacher inference queue；
- 生成 train-only marginal labels；
- 完成 leakage、candidate identity、visibility 与 duplicate audit。

**验收：** cache 可按 event/prefix/candidate 唯一重放，teacher provenance 完整，正负/近零样本均有足够覆盖。

### Phase 2：结构化 marginal predictor

- 线性/LightGBM baseline；
- MLP full-map model；
- regression/ranking/sign objectives；
- held-out-prefix prediction diagnostics。

**Go/No-Go：** full-map 必须稳定优于 no-map、static 和旧 proxy；否则不能进入主方法叙事，应先检查 utility noise 或 map information ceiling。

### Phase 3：fixed-K downstream validation

- 先固定 K=5，隔离排序质量；
- 与 BACES、learned marginal、role rescue 进行 paired comparison；
- 同时在 teacher 与 held-out verifier 上评价。

**Go/No-Go：** 只有 teacher 内收益而无任何 transfer 时，不启动“通用 selector”叙事。

### Phase 4：adaptive stopping

- 在 validation 上校准成本与阈值；
- 生成 fixed-K 与 adaptive paired traces；
- 报告 Pareto curve 和容量分布。

### Phase 5：二阶互补与顺序扩展

- 只在 interaction audit 支持时实现 pair model；
- 比较 greedy、beam 与小池 exhaustive oracle；
- 决定二阶模型进入正文、附录或 future work。

## 13. 论文表格计划

### Table 1：Marginal prediction quality

模型 × correlation、pairwise accuracy、NDCG、positive-gain AUPRC、calibration error。

### Table 2：Fixed-budget selector comparison

相同 pool、K=5/token-matched 条件下，比较 retrieval、manual proxy、learned marginal、BACES、role rescue 与 VGMES。

### Table 3：Adaptive capacity

方法 × Macro-F1、mean K、P90 K、tokens、latency、Pareto efficiency。

### Table 4：Evidence Map ablation

同时报告 marginal prediction 指标和 downstream 指标。

### Table 5：Teacher transfer

selector-label teacher × evaluation verifier matrix，区分同 teacher、held-out checkpoint、cross-backbone。

### Figure 1：方法流程

Claim/atoms → union pool → Evidence Map → prefix interventions with frozen teacher → marginal utility dataset → learned marginal predictor → adaptive slate → downstream verifier。

### Figure 2：性能—容量曲线

显示 fixed K 与 adaptive policy 的 Macro-F1/token Pareto frontier。

### Figure 3：案例研究

展示同一 atom 下 primary、corroboration、opposite、context 在不同 prefix 中的 teacher marginal gain 如何变化，以及 VGMES 与 BACES/旧 proxy 的选择差异。

## 14. 风险与降级方案

1. **Teacher hacking。** selector 可能学习 teacher 的表面偏好。通过 cross-verifier transfer、teacher ensemble 和人工证据评价检测。
2. **Teacher calibration。** log probability 未必可跨样本比较。主训练以 prefix 内 ranking 为主，回归与停止使用校准后的净收益。
3. **Circular training。** 若最终 verifier 与 utility teacher 是同一模型且共享样本，可能产生循环。优先 cross-fitting 或独立 teacher；至少做 held-out verifier evaluation。
4. **Non-monotonic utility。** 好证据可能暂时降低 gold confidence，teacher 也可能错误处理真实冲突。保留负边际、做 interaction audit，并避免宣称 teacher utility 等于客观证据质量。
5. **Computational cost。** prefix--candidate interventions 昂贵。采用候选上限、prefix sampling、base-prefix cache、batch inference 和分阶段扩展。
6. **Greedy regret。** teacher utility 非次模，greedy 无全局保证。小池上用 exhaustive/beam oracle 量化 regret。
7. **Context invisibility。** 一阶标签可能低估条件性 context。先做二阶 interaction audit，再决定是否加入 pair model。

## 15. 与 v0.5.0 的继承和退役关系

### 继续继承

- claim atomization 与 query rendering；
- claim/atom union retrieval；
- provenance-preserving candidate pool；
- Evidence Map schema 与人工对齐实验；
- verifier rendering、context guard 和 matched-pool evaluation；
- BACES evaluator 作为结构覆盖诊断与 baseline。

### 从主方法退役

- 将 \(q\in\{0,1,2\}\) ordinal max coverage 视为最终证据效用；
- 对冻结 structural objective 的 exact solver 作为唯一核心贡献；
- verifier-agnostic 作为 v0.6.0 主方法属性；
- fixed K 作为真实部署的唯一容量设置。

### 保留为重要 baseline

- manual direct/partial greedy；
- learned marginal proxy v0.2；
- BACES exact v0.3；
- role-rescue full；
- resolve-stop/minmax capacity controls。

## 16. 当前需要冻结的决策

在开始实现前，需要按顺序冻结：

1. teacher checkpoint 与 prompt/input contract；
2. gold utility 使用 log probability 还是 logit margin；
3. cross-fitting 或独立 teacher 的数据隔离方案；
4. prefix mixture、每条 claim 的 prefix 数和 candidate intervention 上限；
5. 第一版 predictor 是否固定为结构化 MLP；
6. fixed-K Phase 3 的主 verifier 与 held-out verifier；
7. adaptive cost normalization 与 validation protocol。

在这些决策和 Phase 0 utility audit 完成前，不建议直接启动大规模 selector 训练。v0.6.0 的第一关键证据不是最终 Macro-F1，而是：teacher marginal gain 是否稳定、可学习、由 Evidence Map 多维状态解释，并能迁移到未参与标注的判别器。
