# MREC v0.2 Learned Marginal Proxy Top-5 方法描述

## 1. 方法概览

本文将事实核查建模为一个两阶段过程：首先构造一条能够逐步解析 claim 原子事实的有序证据链，然后由 verifier 基于 claim 与该证据链预测最终真实性标签。给定 claim \(c\)、报告集合 \(R\) 和标签空间 \(\mathcal{Y}\)，最终预测为：

$$
\hat{y}
= \arg\max_{y \in \mathcal{Y}}
p_{\psi}(y \mid \pi(c, Z_{1:k})),
$$

其中 \(Z_{1:k}\) 是有序证据链的前 \(k\) 步，\(\pi(\cdot)\) 表示将证据链渲染为 verifier 输入的函数。在 top-5 设定下，\(k=5\)。需要注意的是，selector 内部可以构造更长的链，但 verifier 只接收前五条证据。

整体流程可写为：

$$
A = \operatorname{Atomize}(c),
$$

$$
C = \operatorname{Retrieve}(c, R),
$$

$$
M = \operatorname{Map}(A, C),
$$

$$
Z = \operatorname{MREC}_{\theta}(c, A, C, M),
$$

$$
\hat{y}
= \operatorname{Verifier}_{\psi}\left(\pi(c, Z_{1:5})\right).
$$

其中 \(A=\{a_1,\ldots,a_m\}\) 是 claim 的原子核查单元集合，\(C=\{e_1,\ldots,e_n\}\) 是候选证据池，\(M\) 是 atom-evidence map，\(\operatorname{MREC}_{\theta}\) 是 learned marginal minimal-resolving evidence-chain selector，\(\operatorname{Verifier}_{\psi}\) 是 label-token verifier。

该方法的核心不是简单扩大 evidence 数量，而是在有限上下文预算下选择一条有顺序、有覆盖目标、低冗余的证据链。

## 2. 原子事实与证据映射

方法首先将 claim 分解为若干原子核查单元：

$$
A=\{a_1,a_2,\ldots,a_m\}.
$$

每个 atom 表示 claim 中一个相对独立、可被证据支持或反驳的事实成分，例如实体关系、数量、时间条件、归因关系或比较关系。

对于候选证据 \(e_j\)，evidence map 记录其与 atom \(a_i\) 的关系：

$$
M(e_j,a_i)
= (r_{ij}, d_{ij}, q_{ij}),
$$

其中 \(r_{ij}\) 表示语义关系，\(d_{ij}\) 表示直接性，\(q_{ij}\) 表示置信度或质量分数。关系可归一化为：

$$
g(r_{ij}) \in \{S,R,Q,B,I\},
$$

其中 \(S\) 表示 support，\(R\) 表示 refute，\(Q\) 表示 qualify，\(B\) 表示 background，\(I\) 表示 irrelevant。

selector 在证据选择过程中维护每个 atom 的解析状态：

$$
h_t(a_i) \in \{U,S,R,Q,C\},
$$

其中 \(U\) 表示 unresolved，\(C\) 表示当前已有证据之间形成 conflict。若一个 atom 的状态属于 \(\{S,R,Q,C\}\)，则认为该 atom 已被解析。第 \(t\) 步后的 atom resolved rate 定义为：

$$
\rho_t
=
\frac{1}{|A|}
\sum_{a_i \in A}
\mathbf{1}\left[h_t(a_i)\in\{S,R,Q,C\}\right].
$$

这个状态空间使 selector 能够显式追踪：哪些 claim 成分已被支持、反驳、限定或置于冲突状态，哪些成分仍缺少有效证据。

## 3. Learned Marginal Proxy Utility

与固定规则或静态 top-\(k\) 排序不同，MREC v0.2 使用一个前缀条件化的 learned marginal utility 来评估候选证据。给定已选证据前缀 \(Z_{<t}\) 和当前 atom 状态 \(h_{t-1}\)，对任一未选候选证据 \(e\)，计算：

$$
\phi_t(e)
=
\phi(e \mid Z_{<t}, h_{t-1}, M).
$$

\(\phi_t(e)\) 是候选证据相对于当前证据链前缀的边际特征向量：

$$
\phi_t(e)=
\left[
\Delta_{\mathrm{res}},
\Delta_{\mathrm{ent}},
\Delta_{\mathrm{cov}},
\Delta_{\mathrm{rel}},
\Delta_{\mathrm{ten}},
\Delta_{\mathrm{cor}},
\Delta_{\mathrm{src}},
\Delta_{\mathrm{text}},
q_{\mathrm{map}},
q_{\mathrm{qual}},
s_{\mathrm{ret}},
c_{\mathrm{cost}}
\right].
$$

各项含义如下：

| 符号 | 含义 |
|---|---|
| \(\Delta_{\mathrm{res}}\) | 对 atom 解析状态的边际增益 |
| \(\Delta_{\mathrm{ent}}\) | atom 状态不确定性的下降 |
| \(\Delta_{\mathrm{cov}}\) | 新覆盖的 atom 数量或比例 |
| \(\Delta_{\mathrm{rel}}\) | 对已跟踪 atom 引入的新关系类型 |
| \(\Delta_{\mathrm{ten}}\) | 与当前证据前缀形成的有效 stance tension |
| \(\Delta_{\mathrm{cor}}\) | 对已有解析关系的 corroboration |
| \(\Delta_{\mathrm{src}}\) | 来源新颖性 |
| \(\Delta_{\mathrm{text}}\) | 文本新颖性 |
| \(q_{\mathrm{map}}\) | atom-evidence map 置信度 |
| \(q_{\mathrm{qual}}\) | evidence-map 质量分 |
| \(s_{\mathrm{ret}}\) | 检索相关性分数 |
| \(c_{\mathrm{cost}}\) | 归一化证据成本 |

learned marginal proxy 使用非负线性函数计算候选证据的边际效用：

$$
u_{\theta}(e \mid Z_{<t})
= b
+ \sum_{\ell=1}^{L-1}\theta_{\ell}\phi_{t,\ell}(e)
- \beta c_{\mathrm{cost}}(e),
$$

其中 \(\theta_{\ell}\ge 0\)，\(\beta\ge 0\)。正向特征奖励能够解析新 atom、降低状态不确定性、引入有效新关系、提供非冗余支持或冲突信息、具有较高 map 质量和检索相关性的证据；成本项惩罚过长或预算消耗较高的证据。

该效用函数是 prefix-conditioned 的，因此同一条证据在不同选择前缀下可能有不同价值：

$$
u_{\theta}(e \mid Z_{<t})
\ne
u_{\theta}(e).
$$

这使方法能够区分“本身相关”与“相对于当前证据链仍有新增价值”的证据。

## 4. Proxy Pairwise 学习目标

learned marginal proxy 在 verifier 训练之前学习。它不直接使用测试集标签，也不直接以最终 verifier 正确性作为训练目标，而是从候选证据之间的 proxy preference 学习局部排序。

对每个训练样本，方法在候选池上模拟短 rollout。在第 \(t\) 步，proxy preference 给出一个更优候选 \(e_i^+\)，并将其与较弱候选 \(e_i^-\) 构成 pairwise comparison：

$$
\mathcal{D}_{\mathrm{pair}}
=
\{(e_i^+, e_i^-, Z_{<t_i}, h_{t_i-1})\}_{i=1}^{N}.
$$

模型通过 logistic pairwise loss 学习让正例候选的边际效用高于负例候选：

$$
\mathcal{L}_{\mathrm{proxy}}(\theta,\beta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\log
\left(
1+
\exp
\left(
-
\left[
u_{\theta}(e_i^+ \mid Z_{<t_i})
-
u_{\theta}(e_i^- \mid Z_{<t_i})
\right]
\right)
\right).
$$

这个目标的作用是让 selector 学习一种轻量、可解释的局部证据偏好，而不是手工固定所有边际增益权重。

## 5. Greedy Minimal-Resolving Evidence Chain

学得 \(u_{\theta}\) 后，证据选择采用序列式 greedy 决策。初始时所有 atom 均为 unresolved：

$$
h_0(a_i)=U,\quad \forall a_i\in A.
$$

第 \(t\) 步的候选集合为：

$$
\mathcal{C}_t
=
C \setminus Z_{<t},
$$

并排除重复证据或明显无效候选。下一条证据由最大边际效用决定：

$$
e_t
=
\arg\max_{e\in\mathcal{C}_t}
u_{\theta}(e \mid Z_{<t}).
$$

选中 \(e_t\) 后，selector 根据 evidence map 中该证据与 atom 的关系更新 atom 状态：

$$
h_t
=
\operatorname{Update}(h_{t-1}, e_t, M).
$$

每个证据链步骤表示为：

$$
z_t
=
\left(
e_t,\ a_t,\ o_t,\ h_{t-1}(a_t),\ h_t(a_t),\ q_t
\right),
$$

其中 \(a_t\) 是当前步骤聚焦的 atom，\(o_t\) 是状态转移动作，\(q_t\) 是一个自然语言 verification cue，用于说明该证据正在检查什么。

转移动作只用于 selector 诊断和链结构构造，不直接暴露给 verifier：

| 动作 | 含义 |
|---|---|
| OPEN | 为尚未解析的 atom 引入第一条有效证据 |
| CONTRAST | 引入与当前状态形成冲突或反向约束的证据 |
| CORROBORATE | 补充支持已有解析状态的证据 |
| BRIDGE | 提供上下文桥接信息 |
| FALLBACK | 没有强 atom transition 时的兜底证据 |

链长度受最小步数和最大步数约束：

$$
k_{\min} \le |Z| \le k_{\max}.
$$

当已达到最小步数后，如果剩余候选的最高边际效用不再为正，则可以提前停止：

$$
\text{stop if }
|Z|\ge k_{\min}
\quad\text{and}\quad
\max_{e\in\mathcal{C}_t}
u_{\theta}(e\mid Z_{<t})
\le \tau_{\mathrm{stop}}.
$$

在当前 top-5 方法中，selector 内部构造的链满足 \(k_{\min}=5\)、\(k_{\max}=10\)。这一设计避免了旧式早停带来的证据不足，同时仍允许 selector 在低边际收益时停止扩展。

## 6. Top-5 证据链渲染

MREC selector 可以产生至多 10 步的内部链，但 verifier 只接收前五步：

$$
Z_{1:5}=(z_1,\ldots,z_5).
$$

对每个步骤 \(z_t\)，只保留 verification cue 与原始证据文本：

$$
\tilde{e}_t
=
\operatorname{Render}(q_t,e_t)
=
\text{``Check: }q_t\text{''}
\oplus
\operatorname{text}(e_t).
$$

因此 verifier 输入为：

$$
\pi(c,Z_{1:5})
=
\operatorname{Prompt}
\left(
c,\tilde{e}_1,\ldots,\tilde{e}_5
\right).
$$

其可见形式为：

```text
Claim:
<claim>

Evidence:
[1] Check: <verification cue>
<evidence text>

[2] Check: <verification cue>
<evidence text>

...
```

状态转移动作、atom 状态、relation、directness、utility features 等结构化信息不进入 verifier prompt。它们只用于选择和排序证据。这样可以让 verifier 知道每条证据的局部核查目标，同时降低模型利用内部标签或结构字段走 shortcut 的风险。

## 7. Label-Token Verifier

给定渲染后的 prompt \(\pi(c,Z_{1:5})\)，verifier 通过 label token 的条件概率预测真实性标签：

$$
p_{\psi}(y\mid\pi)
=
\operatorname{softmax}
\left(
s_{\psi}(y,\pi)
\right).
$$

主分类损失为：

$$
\mathcal{L}_{\mathrm{cls}}(\psi)
=
-\log p_{\psi}(y\mid\pi(c,Z_{1:5})).
$$

对于有序事实核查标签，可以加入 ordinal penalty：

$$
\mathcal{L}_{\mathrm{verifier}}
=
\mathcal{L}_{\mathrm{cls}}
+ \lambda_{\mathrm{ord}}\mathcal{L}_{\mathrm{ord}}.
$$

最终预测为：

$$
\hat{y}
=
\arg\max_{y\in\mathcal{Y}}
p_{\psi}(y\mid\pi(c,Z_{1:5})).
$$

在 LIAR-RAW 上，标签空间为：

$$
\mathcal{Y}_{\mathrm{LIAR}}
=
\{\text{pants-fire},\text{false},\text{barely-true},
\text{half-true},\text{mostly-true},\text{true}\}.
$$

## 8. 方法贡献表述

这一路线可以概括为 learned marginal minimal-resolving evidence-chain selection。与静态 top-\(k\) retrieval 相比，它显式追踪 claim atoms 的解析状态；与手工加权的 graph selector 相比，它学习 prefix-conditioned 的边际效用；与复杂 reasoning prompt 相比，它只向 verifier 暴露最小自然语言核查 cue 和证据文本。

从论文角度，可以将其贡献写为：

1. 将 fact-checking evidence selection 表述为 atom-state resolution process，而不是无结构 evidence ranking。
2. 使用 learned marginal proxy 学习 prefix-conditioned evidence utility，使证据价值依赖当前已选证据链。
3. 将内部结构化证据链压缩为 top-5 minimal prompt，使 verifier 获得每条证据的核查目标，同时避免暴露过多 selector 元数据。

## 9. 当前实例化参数

| 项目 | 设置 |
|---|---|
| 候选池大小 | 每个 claim 取前 20 条候选证据进入 MREC |
| selector 类型 | minimal-resolving evidence chain |
| utility 模型 | pairwise learned marginal proxy |
| 内部链长度 | 最少 5 步，最多 10 步 |
| verifier 可见证据数 | top 5 |
| target resolved-atom rate | 0.8 |
| stop threshold | 最小步数之后阈值为 0.0 |
| prompt 形式 | `Check: <cue>` + evidence text |
| verifier 标签空间 | LIAR-RAW 六分类 |

## 10. 可直接用于论文的简短描述

We introduce a learned marginal minimal-resolving evidence-chain selector for fact checking. The method decomposes each claim into atomic verification units and maps candidate evidence to these atoms with relation, directness, and confidence signals. A prefix-conditioned marginal utility model is trained from proxy pairwise preferences to score the additional value of each candidate evidence unit given the current selected prefix. During inference, the selector greedily constructs a bounded evidence chain that improves atom resolution, reduces uncertainty, adds useful relation diversity, and penalizes redundant or costly evidence. The verifier receives only a compact top-5 realization of the chain, where each evidence unit is preceded by a short verification cue. This separates evidence organization from final label prediction: the selector builds an atom-resolving evidence chain, while the label-token verifier predicts the veracity label from a minimal natural-language rendering of that chain.
