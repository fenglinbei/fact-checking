# Oracle Evidence Set 上界分析与 Gap 诊断

> 最后更新：2026-05-16  
> Oracle 实验结果：`outputs/oracle_evidence/20260516_135632/`  
> MMR Baseline：`outputs/runs/b3_mmr_topk_sweep_1024/...top_k-5__b23a0bbe/infer/val/`  
> 参考文档：`docs/analysis/202605141045_RL_MMR_research_review.md`、`docs/analysis/202605151453_RL_MMR_direction_summary.md`

---

## 1. 任务定义

### 1.1 Fact-Checking Evidence Retrieval

给定一个政治声明（claim）\(c\)，以及从关联报告中抽取的候选证据集合（candidate pool）\(C = \{d_1, d_2, \ldots, d_N\}\)，目标是选择一个大小为 \(K\) 的证据子集（evidence set）：

\[
S_K = \{d_1^*, d_2^*, \ldots, d_K^*\} \subseteq C, \quad |S_K| = K
\]

使得下游 verifier（经过 SFT 的 LLM）能够基于该证据集给出正确的 6 类 veracity 判断：

\[
\hat{y} = \text{Verifier}(c, S_K) = y^*
\]

其中 \(y^* \in \{\text{pants-fire}, \text{false}, \text{barely-true}, \text{half-true}, \text{mostly-true}, \text{true}\}\)。

### 1.2 与普通 IR 的区别

事实核查 evidence retrieval 的本质不同于标准 document retrieval：

1. **单文档高相关 ≠ 证据充分**：即使每个候选都与 claim 相关，子集可能只覆盖 claim 的一个侧面
2. **多低冗余证据 > 多近重复高相关证据**：冗余证据浪费上下文预算且不增加信息量
3. **支持与反驳证据可能同时存在**：需要暴露冲突信息，而非仅选择单向证据
4. **下游 verifier utility > 传统 IR relevance**：最终目标是 verdict 正确，而非检索相关性

### 1.3 数据集

LIAR-RAW：包含 6 类标签的政治声明，每条关联多篇 fact-checking reports。报告中的句子被标注 `is_evidence`（约 61% 的训练样本有 gold evidence 标注）。当前系统**未使用** gold evidence 标注进行候选筛选——所有句子均参与检索。

### 1.4 系统 Pipeline

```
原始 Reports → Sentence Extraction → Semantic Chunking → Hybrid Scoring
(dense + lexical + BM25) → MMR Selection → Prompt Construction → Verifier (Qwen2.5-7B + LoRA)
```

---

## 2. Oracle Evidence Set：定义与动机

### 2.1 为什么需要 Oracle 上界

当前研究中 evidence selection 策略的比较对象包括：

| 系统 | 描述 |
|---|---|
| System 1: reranker-only | 用 cross-encoder 对每个候选打分，取 top-K |
| System 2: fixed-MMR | 全局 λ=0.7 的 MMR 选择 |
| System 3: learned-λ MMR | 监督学习预测 per-claim λ |
| System 4: RL/DPO/GRPO learned-λ MMR | 用 RL/preference learning 优化 λ-policy |
| System 5/6: reranker + learned/RL-MMR | 混合方案 |

要评估这些策略的有效性，需要一个**理论上界**：如果有一个完美的 evidence selector，在同样的候选池和 verifier 下，能达到多高的准确率？这个上界——

- 量化了 evidence selection 的**最大可能改进空间**（gap = oracle − baseline）
- 同时暴露了 verifier 本身的**能力上限**（即使最优证据也无法修复的样本比例）
- 为选择下一步研究方向提供依据：gap 大则优化 selection，bottleneck 大则优化 verifier

### 2.2 形式化定义

Oracle evidence set 定义为在候选池中最大化 verifier 对正确标签概率的 K-子集：

\[
S_K^* = \arg\max_{S \subseteq C, |S|=K} P_{\text{verifier}}(y^* \mid c, S)
\]

这是一个组合优化问题——搜索空间为 \(\binom{N}{K}\)，在典型场景下（N≈16, K=5）约 4,368 个组合，但 N 可达 40 时则约 658,008 个。

### 2.3 搜索方法

由于候选空间有限（semantic chunking 下 N 中位数 16，最大值 40），在 N≤15 时可使用穷举搜索（\(\binom{15}{5}=3003\)），N>15 时使用**贪婪前向选择**：

1. 初始 \(S_0 = \emptyset\)
2. 第 \(t\) 步：对每个剩余候选 \(d \in C \setminus S_{t-1}\)，评估 verifier 对 prompt \((c, S_{t-1} \cup \{d\})\) 的正确标签 log-probability
3. 选择 logprob 最大的候选加入 \(S_t\)
4. 重复直到 \(|S| = K\)

复杂度：\(O(NK)\) 次 verifier 调用（每样本约 235 次，N≈50）。

评分使用 vLLM 离线推理（`prompt_logprobs=0`），与现有 `compute_oracle_lambda.py` 的实现模式一致。实现位于 `scripts/oracle_evidence/search_optimal_evidence.py`。

### 2.4 与 Oracle λ 的区别

现有 oracle λ 管线（`scripts/learned_lambda/compute_oracle_lambda.py`）在 λ ∈ {0.0, 0.05, …, 1.0} 网格上搜索最优 λ，再通过 MMR 选择 evidence。这有两个局限：

1. 搜索结果仍受 MMR greedy selection 约束——最优 evidence set 可能无法被任何单一 λ 的 MMR 选中
2. MMR 的 objective（λ·Rel − (1−λ)·Red）只是 set utility 的一个粗糙代理

Oracle evidence set search 直接搜索 K-子集，跳过了 MMR 的 intermediate proxy，因此是真正的理论上界。

---

## 3. 实验配置

### 3.1 Pipeline 参数

| 组件 | 配置 | 值 |
|---|---|---|
| Chunking | Semantic (θ=0.5) | 合并相邻语义相似句子 |
| Embedder | BGE-base-en-v1.5 | 768-dim dense embeddings |
| Scoring | Hybrid | α_dense=0.70, α_lexical=0.20, α_bm25=0.10 |
| Selection (MMR) | fixed λ=0.7 | baseline |
| Selection (Oracle) | Greedy forward | argmax verifier correct-label logprob |
| Top-K | 5 | 证据数 |
| Prompt | label_only, letter format | max_length=1024, ChatML template |
| Verifier | Qwen2.5-7B-Instruct + LoRA | r=16, α=32, 2 epochs |
| Inference | vLLM 0.8.5.post1, 4×L20 (48GB) | tensor_parallel=4, max_model_len=1024 |

### 3.2 候选池统计（val split, semantic chunking θ=0.5）

| 指标 | 值 |
|---|---|
| 样本数 | 1,274 |
| N 范围 | 1 – 40 |
| P25 / Median / P75 | 9 / 16 / 22 |
| 均值 | 15.7 |
| N ≤ 15 | 611 (48.0%) |
| N ≤ 20 | 880 (69.1%) |

Semantic chunking 下 N 中位数仅 16，远小于 sentence chunking 的 ~51。这说明语义分块显著减少了候选池，但也可能丢失细粒度信息。

---

## 4. 实验结果

### 4.1 总览

| 指标 | Oracle (greedy) | MMR (λ=0.7) | Gap |
|---|---|---|---|
| Accuracy | **48.43%** | 29.67% | **+18.76 pp** |
| Macro Precision | 57.65% | 30.39% | +27.26 pp |
| Macro Recall | 49.22% | 30.40% | +18.82 pp |
| Macro F1 | **43.03%** | 30.03% | **+13.00 pp** |

**核心结论**：如果有一个完美的 evidence selector，verifier 准确率从 29.67% → 48.43%，绝对提升 18.76 个百分点（相对提升 63%）。Gap 很大，**evidence selection 质量是当前系统的主要瓶颈**。

### 4.2 事件级 Gap 分解

将 1,274 个 val 样本按 Oracle 和 MMR 的预测正确性交叉分为四类：

| 类别 | 样本数 | 占比 | 含义 |
|---|---|---|---|
| Both correct | 222 | 17.4% | 两者都选对了证据 → 当前 MMR 已足够 |
| **Oracle only correct** | **395** | **31.0%** | **Oracle 对、MMR 错 → evidence selection 可直接修复** |
| MMR only correct | 156 | 12.2% | Oracle 反而错 → 目标函数（logprob）与 accuracy 不一致 |
| Neither correct | 501 | 39.3% | 两者都错 → Verifier 瓶颈，selector 无法解决 |

**关键数字**：

- **31.0%（395 样本）可被更好的 evidence selection 修复**——这是 learned-λ / RL-MMR 的直接改进空间
- **12.2%（156 样本）Oracle 目标函数缺陷**——最大化正确标签 logprob 并不等价于最大化 accuracy。Verifier calibration 问题：高 logprob ≠ argmax 正确
- **39.3%（501 样本）是 verifier 硬瓶颈**——候选池缺乏有效证据，或 verifier 本质上无法处理这类 claim

### 4.3 按 Label 分桶

| Class | N | Oracle Acc | MMR Acc | Gap | Verifier Bottleneck |
|---|---|---|---|---|---|
| **pants-fire** | 115 | **88.7%** | 40.0% | **+48.7 pp** | 11.3% |
| **false** | 259 | **89.6%** | 27.0% | **+62.5 pp** | 10.4% |
| barely-true | 236 | 40.7% | 34.7% | +5.9 pp | 59.3% |
| half-true | 244 | 53.3% | 28.7% | +24.6 pp | 46.7% |
| mostly-true | 251 | 21.9% | 27.1% | **−5.2 pp** | 78.1% |
| true | 169 | 1.2% | 24.9% | **−23.7 pp** | 98.8% |

**发现 1 — "假"类别受益巨大**：pants-fire 和 false 的 oracle 召回率接近 90%，说明候选池中存在强力反驳证据，只要能选出来 verifier 就能判对。这两个类别的 bottleneck 仅 10-11%。

**发现 2 — "真"类别 Oracle 反而更差**：mostly-true（−5.2pp）和 true（−23.7pp）的 oracle 准确率低于 MMR baseline。Oracle 选出的证据集让 verifier 更倾向判假——说明 verifier 有强烈的 **false bias**。

**发现 3 — half-true 是唯一既显著提升又有中等上限的类别**（+24.6pp, 上限 53.3%）：多样性证据对"半真半假"的 mixed claim 确实有帮助。

**发现 4 — Verifier bottleneck 分布极不均衡**：pants-fire 仅 11.3% 样本无法修复，true 高达 98.8%。提升上限的关键在于让 verifier 更好地利用支持性证据判真。

### 4.4 Oracle Logprob 分析

| Class | Mean Logprob | Median Logprob |
|---|---|---|
| pants-fire | −0.69 | −0.00 |
| false | −0.57 | −0.00 |
| barely-true | −1.33 | −0.21 |
| half-true | −2.14 | −0.39 |
| mostly-true | −7.04 | −7.62 |
| true | −8.50 | −7.60 |

**pants-fire 和 false 的 median logprob 几乎为 0**（即 verifier 对正确标签接近 100% 确信），而 **true 的 median logprob 为 −7.6**（即正确标签概率极低）。即使最优证据集，verifier 也不相信 claim 是真的。

**Verifier 存在严重的 calibration 偏差**：对"假"类过度自信，对"真"类过度不自信。这解释了为什么 Oracle 在 true 类上比 MMR 更差——Oracle 贪婪地最大化正确标签 logprob，但该标签的 logprob 即使被最大化后仍远低于其他标签，argmax 仍选错。

---

## 5. 与现有研究框架的关系

### 5.1 Oracle Set 在方法体系中的位置

参考 `RL_MMR_direction_summary.md` 中定义的探索顺序：

| 阶段 | 方法 | 与 Oracle Set 的关系 |
|---|---|---|
| 1. fixed λ=0.7 | baseline | 下界参照 |
| 2. log(n) heuristic | 极简 adaptive | — |
| 3. sensitivity-gated | 无监督 adaptive | — |
| 4. soft-label λ policy | 修复 System 3 | Oracle set 可作为更精确的 utility 来源 |
| 5. DPO step-wise λ | 主线 | Oracle set 可构造 S⁺，MMR set 为 S⁻ |
| 6. multi-weight MMR | 扩展 | Oracle set 提供 set-level supervision |
| 7. GRPO refinement | 最终 | Oracle set 可用于 reward shaping |

### 5.2 此前 Oracle λ 方法的局限

`RL_MMR_direction_summary.md` 中诊断了 hard oracle λ predictor 失败的五个原因，其中最关键的是：

> Oracle λ 曲面高度平坦。大量 claim 中，最优 λ 与次优 λ 的 logprob 差异很小。此时 hard argmax λ 只是平坦曲面上的一个不稳定点。

Oracle evidence set search 直接跳过了 λ 这个中间变量，直接优化最终目标（evidence subset）。这避免了将平坦 utility 曲面压缩为噪声 hard label 的问题。

---

## 6. 此前 Adaptive λ 方法的失败记录

在 Oracle Evidence Set 搜索之前，已有四个自适应 λ 方向的实验全部失败或仅获微弱收益。这些失败直接构成了转向 Oracle Set Search 的动机。

### 6.1 Learned-λ Predictor（Hard Oracle λ 回归）— 已停止

**实验**：训练神经网络从 claim + candidate pool 特征预测最优 λ，监督信号为 `compute_oracle_lambda.py` 在 21 个 λ 值 (0.00, 0.05, …, 1.00) 中 argmax 得到的 oracle λ。

**三个变体全失败**：

| 变体 | Val MAE | Val RMSE | Target Std |
|---|---|---|---|
| Chunk embedding (256-dim attention, regression) | 0.256 | 0.294 | 0.296 |
| Handcrafted 73 features (regression) | 0.250 | 0.283 | 0.299 |
| Handcrafted 73 features (classification) | 0.250 | 0.282 | 0.299 |

**对比基线**：

| 基线方法 | MAE | RMSE |
|---|---|---|
| 均值预测 (λ=0.445) | 0.262 | 0.296 |
| 最优固定 λ=0.45 | 0.262 | 0.296 |
| log(n_candidates) 线性回归 | **0.253** | **0.290** |

- R² vs 均值基线 ≈ 0.01（最高仅 0.058）——模型几乎等价于预测均值
- 预测标准差 = 0.057，oracle 标准差 = 0.296 → **方差比仅 19%**
- 预测值全部集中在 0.41–0.50 区间，6 个 label 的预测分布完全一致

**两个修复实验也全失败**：

1. **高 margin 过滤**（仅保留 margin ≥ 0.05 的 2761 样本）：MAE=0.267, RMSE=0.319, R² vs 均值 = **−0.167**（比均值预测更差），预测值坍缩至 0.30–0.32
2. **3-bin 粗粒度分类**：准确率=0.336（随机基线=0.333），全部 10,065 样本被预测为同一类别

**四重根因**：

1. **Oracle λ 信号极弱**（核心问题）。72.6% 的 claim 的 margin（最优与次优 λ 的 logprob 差异）< 0.05，38.5% < 0.01，中位数 margin 仅 0.0185。每个 claim 平均有 7.9/21 个 λ 值在最优值 0.1 logprob 范围内。Oracle λ 本质上是在平坦 logprob 曲面上人为挑出的噪声点。

2. **Tie-break 偏差**。`compute_oracle_lambda.py` 中当多个 λ 在 0.01 logprob 范围内时，优先选离 default_lambda=0.7 最近的，这使 oracle λ 向 0.7 收缩，降低了自然方差。

3. **Oracle λ 与 claim 语义无关**。6 个 label 的 oracle λ 均值全在 0.42–0.50，标准差全在 0.28–0.30。无论 claim 真假难易，分布完全相同。

4. **特征–目标语义鸿沟**。预测器只能看到 BGE 文本 embedding，但 oracle λ 的实际决定因素是 SFT 模型对不同 evidence 排序的内部响应——文本语义中不存在"什么 λ 能让我做对这道题"的信息。

**唯一可观察到的系统模式**：候选数量与 oracle λ 的弱相关（corr(log(n), λ) = −0.13）：候选越多 → 越需 diversity（λ 越低）。但这不足以支撑有效预测。

### 6.2 Oracle λ 的 Verifier Utility 验证

尽管 hard λ predictor 失败，oracle λ 的存在本身验证了自适应方向的价值：

- 在 2013 个 val 样本上，oracle λ vs fixed λ=0.7：accuracy **+3.08%**（30.40% → 33.48%），macro F1 **+3.24%**（30.60% → 33.84%）
- 所有 6 个类别均有改善
- **结论**：自适应 λ 方向正确，但 supervised regression of hard λ label 是错误的手段

### 6.3 Soft-Label λ Policy — 已停止

**实验**：将 hard oracle λ 替换为 soft target（utility curve 经 softmax 平滑后的概率分布），用 weighted cross-entropy 训练分类器。

**结果**：修复 oracle logprob 计算 bug 后，soft target 的熵接近均匀分布（1.5968 vs 均匀分布 1.6094）。三种模型（LightGBM、Logistic Regression、MLP）的 `expected` 推理模式全部退化为固定 λ≈0.5，`argmax/sample` 模式比 fixed λ=0.7 更差。

| 模型 | Fixed utility | Expected ∆ utility | Argmax ∆ | Val KL |
|---|---|---|---|---|
| LightGBM | −1.5518 | −0.0001 | −0.0291 | 0.0157 |
| MLP | −1.5518 | −0.0001 | −0.0284 | 0.0150 |

**根因**：utility curve 本身近乎平坦时，softmax 自然产生接近均匀的分布。模型正确地学到了"大多数 claim 对 λ 不敏感"这一事实——但这对优化 evidence selection 没用。该方向已终止。

### 6.4 DPO Step-wise λ Policy — 已停止

**实验**：对同一 claim 生成多条 λ trajectory，构造 preference pairs（winner/loser），用 DPO 训练 StepLambdaPolicy（MLP，hidden_dims=[64, 32]）。

**偏好对统计**（val split）：

- 78,510 train pairs（来自 7,851/10,065 claim）
- 11,030 val pairs（来自 1,103/1,274 claim）
- Utility gap 均值=4.54, std=3.53
- 63.2% 的 claim 存在优于 fixed λ=0.7 的 λ schedule
- 最优 λ 分布：0.3 (20.9%), 0.5 (39.9%), 0.7 (39.1%)

**三次训练全部坍缩至 λ=0.7**：

| 版本 | 特征 | β | Best val loss | Accuracy | Entropy | argmax=0.7 占比 |
|---|---|---|---|---|---|---|
| V1 | 20-dim (pool+step) | 1.0 | 0.7215 | 0.528 | 1.319 | **99.97%** |
| V2 | 20-dim, 过滤 tail | 3.0 | 0.6545 | 0.529 | 1.567 | **99.87%** |
| V3 | 13-dim (仅 step) | 3.0 | 0.7011 | 0.533 | 1.563 | **98.43%** |

**四重根因**：

1. **Pool features 是纯噪声**。8 维 pool features 在同一 claim 的 winner/loser 之间完全相同——对 DPO 损失无贡献
2. **Step features 存在内生性问题**。step features 的差异是 λ 选择的**结果**而非**原因**（如选了 λ=0.3 → 选了不同的 evidence → max_sim 不同），模型无法从中学习因果关系
3. **Reference policy 的吸引域太强**。reference（λ=0.7 偏好的 policy）已与多数 winner 的 λ 选择一致，DPO 更新的梯度不足以让 policy 离开 reference 的吸引域
4. **信噪比太低**。utility gap 中位数仅 2.34，相对于 logprob 的自然方差，足够强的偏好信号太少

**结论**：经过 4 轮训练全部坍缩至 fixed λ=0.7。scalar λ 方向（含 claim-level 和 step-wise）已完整探索，该方向已终止。

### 6.5 当前实验状态总览

| 实验 | 结论 | 状态 |
|---|---|---|
| fixed λ=0.7 | 稳定基线, test accuracy=0.2702 | **locked** |
| log(n) heuristic | test +0.0064, 弱自适应基线 | 保留 |
| sensitivity-gated MMR | test +0.0040, 弱自适应基线 | 保留 |
| learned-λ predictor (hard) | R²≈0.01, 预测坍缩为均值 | **已停止** |
| soft-label λ policy | expected 退化, argmax/sample 更差 | **已停止** |
| DPO step-wise λ | 4 次训练全部坍缩至 λ=0.7 | **已停止** |

**下一步唯一未探索的方向**：multi-weight MMR policy——将 scalar λ 扩展为多维权重向量 w_t = (w_rel, w_red, w_cov, w_src, w_stance, w_cost)，每步给出不同的权重组合。

### 6.6 Oracle Evidence Set Search 的定位

此前所有 adaptive λ 方法的根本困境是：**λ 是 utility 的间接代理，而非 utility 本身**。在 utility curve 平坦时，λ 信号退化。

Oracle evidence set search 直接跳过了 λ 这个中间变量：
- 不再问"什么 λ 能选出好的证据集"
- 直接问"哪个证据集最好"，然后在集合空间中搜索

这产生了两个数量级的差异：
- Oracle λ vs fixed λ：**+3.1% accuracy**（已验证）
- Oracle evidence set vs fixed MMR：**+18.8% accuracy**（本次实验）

差距从 +3pp 扩大到 +19pp，说明 MMR 框架本身（即使配合最优 λ）限制了 evidence selection 的上界。最优 K-子集可能无法被任何单一 λ 的 MMR 选中。

---

## 7. 对后续研究的指导

### 6.1 三个改进方向

**方向 A：提升 Verifier 本身（ROI 最高，Bottleneck = 39.3%）**

Oracle 48.4% 的硬上限表明 verifier 是主要瓶颈。具体措施：

- 修复 false bias：调整 logit_adjust（当前 τ=1.0 不足以纠正类别不平衡），对 true/mostly-true 类别加训练权重
- 从 LoRA（r=16）升级到 full fine-tune
- 尝试 explanation_label 模式：要求模型在判断标签前先生成解释
- 增加候选池质量：混入 reranker 评分扩充候选

**方向 B：让 Evidence Selector 逼近 Oracle（Gap = 18.8pp）**

利用已产出的 `oracle_results_val.jsonl`：

- 每条样本有最优 K-子集的索引 → 作为 learned-λ predictor 的精确监督信号
- 可在 train split 上运行 oracle search，直接构造 (claim, candidate_pool) → oracle_set 的监督数据
- DPO 可直接用 oracle set 作为正例（S⁺），MMR set 作为负例（S⁻）
- 分析 oracle 选出的 evidence 特征：是否更偏好 diversity？是否跨 source？是否包含冲突 stance？

**方向 C：改善候选池质量**

- 当前 semantic chunking N 中位数仅 16，可能丢失关键候选
- 可用 reranker 从更大候选池中召回 top-M，再做 MMR 选择（reranker + MMR 范式）
- 可尝试不同 chunking 策略（sentence / ctx_window / semantic with different θ）的上界差异

### 6.2 建议实验优先级

```
1. ✅ 计算 gap → 已完成（Oracle 48.4% vs MMR 29.7%, gap 18.8pp）
2. 🔲 修复 verifier false bias
     调整 logit_adjust / 类别权重，目标是提升 oracle 上限（当前 48.4%）
3. 🔲 在 train split 上跑 oracle search（~2000 样本，约 1.5-2h）
     产出 oracle_results_train.jsonl 作为监督信号
4. 🔲 用 oracle set 构造 DPO preference pairs
     S⁺ = oracle set, S⁻ = MMR λ=0.7 set
5. 🔲 比较 DPO-trained selector vs fixed-MMR vs oracle
     量化向 oracle 逼近了多少
6. 🔲 混合 reranker：扩大候选池后用 learned-λ MMR 选择
7. 🔲 终极目标：reranker + learned diversity policy > reranker-only
```

### 6.3 期望的论文证据链

最终的实证目标不仅是 "oracle > baseline"，而是一条完整的证据链：

1. Oracle set search 量化 evidence selection 的理论上界（**已完成**）
2. Per-class / per-bucket gap 分析揭示 diversity 收益集中在哪些 claim 类型（**已完成**）
3. Learned policy 在 gap 大的子集上显著优于 fixed-MMR
4. Reranker + learned policy 进一步逼近 oracle 上限
5. 解释：什么时候 diversity policy 重要？→ 高冗余候选池、冲突证据、多子事实 claim

---

## 7. 数据文件索引

| 文件 | 内容 |
|---|---|
| `outputs/oracle_evidence/20260516_135632/oracle_metrics_val.json` | Oracle 搜索汇总指标 |
| `outputs/oracle_evidence/20260516_135632/oracle_results_val.jsonl` | 每条样本的最优证据集 |
| `outputs/runs/...b23a0bbe/.../api/metrics.json` | MMR baseline 指标 |
| `outputs/runs/...b23a0bbe/.../api/val_predictions.jsonl` | MMR baseline 逐条预测 |
| `scripts/oracle_evidence/search_optimal_evidence.py` | Oracle search 脚本 |
| `docs/analysis/202605141045_RL_MMR_research_review.md` | 研究综述（18 节，含文献索引） |
| `docs/analysis/202605151453_RL_MMR_direction_summary.md` | 后续方向与方法 gate 设计 |
