你的段落方向是对的，但建议改三点：

1. “Verifier” 部分不要只说 LLM 推理能力提升，要明确它的输入、输出和训练目标：输入是 claim + prompt-visible evidence trace，输出是事实核查标签。
2. “token 预算”和“resolve stop/minmax”要分开。当前实现里 `budget` 是按 token 停；`resolve_stop` / `minmax` 是满足最小证据数后，遇到 `trace_state.target_resolved=True` 停。它们不是同一个停止条件。
3. “\(h_{i_t}^{(t)}\) 仍处于 unresolved”这句不够准确，因为停止判断不是只看当前 step 关联的单个 atom，而是看整体 atom resolved rate 是否达到目标阈值。

可以改成下面这一版，更适合论文方法部分。

## Verifier

在得到 ordered evidence trace \(\mathcal T=[u_1,u_2,\ldots,u_T]\) 后，我们使用一个指令微调后的 LLM 作为最终 verifier。Verifier 接收 claim 及其 prompt-visible evidence trace，并输出事实核查标签：

\[
\hat y
=
\arg\max_y p_\theta(y\mid x),
\]

其中 \(x=\mathrm{Render}(c,\mathcal T_{\mathrm{prompt}})\) 表示由 claim \(c\) 和被截断后的证据 trace \(\mathcal T_{\mathrm{prompt}}\) 构造的输入 prompt。训练阶段采用监督微调，最小化标准交叉熵损失：

\[
\mathcal L_{\mathrm{verifier}}
=
-\log p_\theta(y^\ast\mid x),
\]

其中 \(y^\ast\) 为样本的 gold fact-checking label。

由于 MREC selector 生成的完整 trace \(\mathcal T\) 长度存在显著差异，直接将完整 trace 输入 verifier 可能导致长上下文噪声增加、有效证据信号被稀释，并显著提高训练与推理成本。因此，我们在 verifier 之前引入 prompt evidence policy，将完整 ordered trace 映射为 verifier 可见的 evidence prefix：

\[
\mathcal T_{\mathrm{prompt}}
=
[u_1,u_2,\ldots,u_{K^\ast}],
\quad
K^\ast \le T.
\]

具体地，对于 `resolve_stop` / `minmax` 类型策略，我们沿 selector 给出的顺序逐步加入 evidence step，并在满足最小证据数 \(k_{\min}\) 后检查当前 trace 是否已经达到 atom-level resolution target。令

\[
\rho_t
=
\frac{
|\{a_i:h_i^{(t)}\in\{S,R,Q,C\}\}|
}{
|\mathcal A|
},
\]

其中 \(\rho_t\) 是第 \(t\) 步后的 resolved atom rate。若

\[
\rho_t \ge \rho_{\mathrm{target}},
\]

则认为当前 trace 已经达到目标解析状态，并停止继续加入证据。因此截断位置定义为：

\[
K^\ast
=
\min
\left\{
t:
t\ge k_{\min}
\land
\rho_t\ge \rho_{\mathrm{target}}
\right\}.
\]

若不存在满足条件的 \(t\)，则退化为最大证据数约束：

\[
K^\ast=\min(k_{\max},T).
\]

在 `minmax` 设置下，\(k_{\min}\) 和 \(k_{\max}\) 分别控制 verifier 至少看到多少条证据、最多看到多少条证据；在 `resolve_stop` 设置下，通常设置较小的 \(k_{\min}\) 和较大的 \(k_{\max}\)，使 atom resolution signal 主导停止位置。

对于纯 token-budget 策略，我们进一步约束 prompt-visible trace 的 token cost：

\[
\sum_{t=1}^{K^\ast}\mathrm{cost}(u_t)\le B,
\]

并在满足最小证据数后，当加入下一条 evidence 会超过预算 \(B\) 时停止。该策略用于研究不同上下文预算下 verifier 性能的变化。

整体上，Verifier 阶段并不重新执行 evidence selection，而是消费 selector 生成的 ordered trace，并通过 prompt evidence policy 在证据充分性和上下文长度之间取得折中。我们在实验中比较不同 prompt evidence budgets 与截断策略，以分析 verifier 对证据容量的敏感性。

如果你想更贴近你当前 `minmax5_10` 主实验，可以把最后两段简化成只讲：

\[
k_{\min}=5,\quad k_{\max}=10,
\]

即 verifier 至少接收 5 个 MREC steps；若 5 步后 claim atoms 已达到目标解析状态，则停止，否则继续补充，最多到 10 步。这样表述会比“token 预算 \(B\)”更贴合 `minmax5_10` 的实际机制。