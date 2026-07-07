下面按当前实现讲完整过程：`learned marginal selector` 的核心是把 evidence selection 写成一个“状态条件化的边际效用最大化”问题。它不是一次性把 \(\mathcal R\) 静态排序，而是在每一步根据已经选过的 evidence trace 和当前 atom 状态，重新计算每个候选证据的边际贡献。

**1. 输入**

selector 的输入是 evidence-map 之后的候选池：

\[
\mathcal R=\{u_1,u_2,\ldots,u_n\},
\]

以及 claim atom 集合：

\[
\mathcal A=\{a_1,a_2,\ldots,a_m\}.
\]

每个候选证据 \(u_j\) 已经带有 atom-evidence map 标注：

\[
M(u_j,a_i)=(r_{ij}, d_{ij}, c_{ij}),
\]

其中：

- \(r_{ij}\)：relation，例如 support / refute / qualify / insufficient / background / irrelevant；
- \(d_{ij}\)：directness，例如 direct / partial / context / none；
- \(c_{ij}\)：map confidence；
- 候选本身还带有 retrieval score、map quality、source、duplicate group、token cost 等信息。

在当前 `learned_marginal_proxy_fullpool` 配置里，`candidate_top_n=0` 表示不截断候选池，使用 fullpool；`max_steps=0` 表示 selector 最多可以走完整个候选池长度，而不是 0 步。配置见 [learned_marginal_proxy_fullpool_policy.yaml](/data/liaozijie/fact-checking/configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_policy.yaml:16)。

**2. Atom 状态建模**

selector 把每个 atom 看成一个待验证状态变量：

\[
h_i^{(t)}\in\{U,S,R,Q,C\},
\]

含义是：

- \(U\)：unresolved，尚未被解析；
- \(S\)：supported；
- \(R\)：refuted；
- \(Q\)：qualified / partially resolved；
- \(C\)：conflicted。

初始状态是：

\[
h_i^{(0)}=U,\quad \forall a_i\in\mathcal A.
\]

实现中同时维护 hard state 和 soft state。hard state 用于最终 trace 记录；soft state 用于计算边际特征，例如当前 atom 仍未解析的概率质量 \(p_i^{(t)}(U)\)。入口在 [minimal_resolving_chain.py](/data/liaozijie/fact-checking/src/fact_checking/selectors/minimal_resolving_chain.py:72)，learned marginal 选择循环在 [minimal_resolving_chain.py](/data/liaozijie/fact-checking/src/fact_checking/selectors/minimal_resolving_chain.py:317)。

**3. 每一步的边际特征**

在第 \(t\) 步，已有 trace 为：

\[
\mathcal T_{<t}=\{u_{1},\ldots,u_{t-1}\}.
\]

对每个未选择候选 \(u_j\)，selector 计算一组状态条件化特征：

\[
\phi_t(u_j)=
[
\phi_{\text{res}},
\phi_{\text{ent}},
\phi_{\text{cov}},
\phi_{\text{new-rel}},
\phi_{\text{tension}},
\phi_{\text{corr}},
\phi_{\text{src}},
\phi_{\text{text}},
\phi_{\text{conf}},
\phi_{\text{map}},
\phi_{\text{ret}},
\phi_{\text{cost}}
].
\]

其中最核心的是 resolution gain。实现中 directness 被转成权重：

\[
\delta(d)=
\begin{cases}
1.0, & d=\text{direct/full}\\
0.65, & d=\text{partial}\\
0.25, & d=\text{context}\\
0, & d=\text{none}
\end{cases}
\]

如果 \(u_j\) 对 atom \(a_i\) 给出可解析关系 support/refute/qualify，则该 pair 的解析增益近似为：

\[
g_{ij}^{(t)}
=
p_i^{(t)}(U)\cdot \delta(d_{ij})\cdot \max(c_{ij},0.5).
\]

候选 \(u_j\) 的整体 resolution gain 是对 atom 归一化后的增益：

\[
\phi_{\text{res}}^{(t)}(u_j)
=
\frac{1}{m}
\sum_{i=1}^{m}
\max_{M(u_j,a_i)} g_{ij}^{(t)}.
\]

其它特征的直观含义是：

- `entropy_reduction`：选择该证据后 atom 状态不确定性的下降；
- `new_atom_coverage`：是否覆盖了此前未覆盖的 atom；
- `new_relation_for_atom`：是否为某个 atom 引入新的 relation 类型；
- `stance_tension`：是否引入 support/refute 之间的张力或冲突；
- `corroboration_gain`：是否从新来源或新文本提供同向佐证；
- `source_novelty` / `text_novelty`：来源或文本是否重复；
- `map_confidence` / `map_quality`：evidence map 自身质量；
- `retrieval_score`：上游检索分数；
- `cost_ratio`：证据长度代价。

这些特征定义在 [mrec_learned_marginal.py](/data/liaozijie/fact-checking/src/fact_checking/selectors/mrec_learned_marginal.py:217)。

**4. Learned marginal utility**

候选的边际效用是一个线性模型：

\[
U_\theta(u_j\mid \mathcal T_{<t},H^{(t)})
=
b+
\sum_{\ell\neq \text{cost}}
w_\ell \phi_{\ell}^{(t)}(u_j)
-
w_c \phi_{\text{cost}}^{(t)}(u_j).
\]

也就是说，正向特征越高越好，长度代价越高越差。实现中所有正向权重和 cost 权重都被约束为非负。打分函数在 [mrec_learned_marginal.py](/data/liaozijie/fact-checking/src/fact_checking/selectors/mrec_learned_marginal.py:296)。

当前 LIAR-RAW 权重文件显示，权重来自 `proxy_pairwise` 训练，fingerprint 是 `73e064c851af`。较大的权重包括：

- `resolution_delta`: 2.5881
- `new_relation_for_atom`: 2.4089
- `entropy_reduction`: 1.6916
- `retrieval_score`: 0.9772
- `corroboration_gain`: 0.9132
- `stance_tension`: 0.8934
- `cost_weight`: 0.7087

见 [weights.json](/data/liaozijie/fact-checking/outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy/weights/weights.json:1)。

**5. 权重是怎么学习的**

权重不是从最终 verifier 的真假分类 loss 端到端学出来的，而是在 selector 层用 proxy pairwise ranking 学出来的。

训练输入是 evidence-map features：

\[
\{x^{(k)}\}_{k=1}^{N},
\]

每个 row 里包含 claim atoms、candidate pool、candidate-evidence-map features，以及可能存在的 oracle/proxy ordering。当前 LIAR-RAW 权重训练用的是：

- train rows: 10065
- val rows: 1274
- candidate_top_n: 20
- rollout_steps: 5
- epochs: 30
- learning_rate: 0.05
- pair_count: 689274

见 [manifest.json](/data/liaozijie/fact-checking/outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy/weights/manifest.json:1) 和 [train_metrics.json](/data/liaozijie/fact-checking/outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy/weights/train_metrics.json:1)。当前 `val_metrics.json` 的 `scored_row_count=0`，所以它不是一个有效的验证准确率面。

训练过程如下。

第一步，对每个训练样本取候选池前 \(k\) 个候选：

\[
\mathcal R_k=\{u_1,\ldots,u_k\}.
\]

第二步，在当前 soft atom state 下构造 proxy 排序。若有 `oracle_ordered_keys`，优先使用 oracle 顺序；否则使用启发式 proxy：优先 direct resolving，其次 partial resolving，再看新 atom 覆盖、新 relation、map quality、retrieval score。对应实现是 [mrec_learned_marginal.py](/data/liaozijie/fact-checking/src/fact_checking/selectors/mrec_learned_marginal.py:305)。

第三步，把 proxy 排序转成 pairwise preference。若当前最好候选是 \(u^+\)，其它候选是 \(u^-\)，则构造：

\[
(u^+,u^-)\in\mathcal P.
\]

实现上是：

\[
\mathcal P_t
=
\{(u_{\text{best}},u_j):u_j\neq u_{\text{best}}\}.
\]

第四步，对正负样本都抽取同一状态下的边际特征：

\[
\phi_t(u^+),\quad \phi_t(u^-).
\]

第五步，把 proxy winner 加入模拟 trace，并更新 soft atom state。这样下一轮 rollout 的特征不是静态的，而是会受“已经选过什么证据”影响。

第六步，用 pairwise logistic loss 训练权重：

\[
\mathcal L(\theta)
=
\frac{1}{|\mathcal P|}
\sum_{(u^+,u^-)\in\mathcal P}
\log
\left(
1+\exp
\left(
-\left[
U_\theta(u^+)-U_\theta(u^-)
\right]
\right)
\right).
\]

实现中使用 Adam 优化，并用 softplus 参数化保证权重非负：

\[
w_\ell=\mathrm{softplus}(\theta_\ell),\quad
w_c=\mathrm{softplus}(\theta_c).
\]

训练函数在 [mrec_learned_marginal.py](/data/liaozijie/fact-checking/src/fact_checking/selectors/mrec_learned_marginal.py:374)，训练入口在 [train_mrec_learned_marginal_proxy.py](/data/liaozijie/fact-checking/scripts/phase5_selectors/train/train_mrec_learned_marginal_proxy.py:26)。

**6. 推理时如何选择 evidence trace**

推理阶段加载训练好的 \(\theta\)，然后贪心构建 trace。

第 \(t\) 步：

\[
u_t
=
\arg\max_{u_j\in \mathcal R\setminus \mathcal T_{<t}}
U_\theta(u_j\mid \mathcal T_{<t},H^{(t)}).
\]

排序时主键是 `utility_score`，tie-breaker 依次看 `resolution_delta`、`entropy_reduction`、`stance_tension`、`new_atom_coverage`、`new_relation_for_atom`、`map_confidence`，再看 token cost 和 candidate index。实现见 [minimal_resolving_chain.py](/data/liaozijie/fact-checking/src/fact_checking/selectors/minimal_resolving_chain.py:1044)。

选中 \(u_t\) 后，selector 会生成一个 MREC step：

\[
s_t=
(u_t,a_{i_t},h_{i_t}^{(t-1)}\rightarrow h_{i_t}^{(t)}).
\]

这个 step 里会记录：

- `atom_id`
- `state_before`
- `state_after`
- `operation`
- `cue_text`
- `transition_reason`
- `utility_score`
- `utility_features`
- `trace_state`

然后更新 atom hard state 和 soft state：

\[
H^{(t+1)}=\mathrm{Update}(H^{(t)},M(u_t,a_{i_t})).
\]

如果 relation 是 support/refute/qualify，则状态进入 \(S/R/Q\)；如果后续出现 support 和 refute 的冲突，则可能进入 \(C\)。

最终输出的不是单纯 top-k evidence list，而是一个 ordered evidence trace：

\[
\mathcal T=
[s_1,s_2,\ldots,s_T].
\]

trace row 会包含 `selected_indices`、`selector_ordered_indices`、`selected_candidates`、`mrec_steps`、`atom_states_initial`、`atom_states_final`、`mrec_diagnostics` 等字段。构造位置在 [minimal_resolving_chain.py](/data/liaozijie/fact-checking/src/fact_checking/selectors/minimal_resolving_chain.py:103)。

**7. 和 prompt top-k 的关系**

这一点很关键：learned marginal selector 负责产生一个状态感知的 trace ordering；后面的 prompt evidence policy 再决定 verifier 实际看到几条证据。

所以：

\[
\text{learned marginal selector}
:
\mathcal R \rightarrow \mathcal T
\]

\[
\text{prompt evidence policy}
:
\mathcal T \rightarrow \text{verifier-visible evidence}.
\]

在当前 `fullpool_policy` 里 prompt 层是 `fixed_topk=5`；在 `minmax5_10` 这类变体里，selector 仍然先产出 trace ordering，然后 prompt 层按 min/max 规则截取。这也是为什么不要把 learned marginal selector 本身解释成“直接选 top5”。它更准确的表述是：学习一个状态条件化的边际效用函数，用它贪心地产生 MREC evidence trace；最终展示给 verifier 的证据数量由 prompt evidence policy 控制。