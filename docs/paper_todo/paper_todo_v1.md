# 本文档记录该项目在正式成文前需要解答的事情

## 关于Candidate Evidence Retrieval

### 整体流程

系统先将 report collection 切分为候选 evidence units，例如 sentence 或 semantic chunk。我们最初评估了三类 claim-evidence 相关性信号：

```text
dense similarity
lexical overlap
BM25-like score
```

这三类信号并不是最终都进入主方法。我们基于 full-pool oracle evidence 做 candidate-pool recall 消融后发现，dense-only 在固定候选池预算下最稳定；lexical overlap 与 BM25-like 线性混入后没有带来一致的 oracle recall 提升，部分设置下反而引入噪声。因此最终主方法采用 dense-only relevance score：

$$s(e, c) = s_{\mathrm{dense}}(e, c)$$

随后使用 MMR 控制相关性与多样性，得到初始 candidate evidence pool。当前 v0.6c 默认消费 v0.6b evidence-map features，候选池大小为 top-20。实现上可以通过设置 `alpha_dense=1.0, alpha_lexical=0.0, alpha_bm25=0.0` 复用原有 `hybrid_score` 字段；此时该字段等价于 min-max 归一化后的 dense score。

需要解答：
1. 为何使用这种方式(Candidate Evidence Retrieval)
2. 为何评估 dense / lexical / BM25-like 三类信号
3. 为何最终采用 dense-only，而不是三路线性加权
4. BM25-like 为什么不是标准 BM25

解答：
1. Candidate Evidence Retrieval是作为证据选择器的前置筛选器使用，尽管数据集已经为每条claim提供了其相关的原始report，但是其为粗粒度证据，若直接喂给evidence selector，会直接导致两个后果：(1) 检索得到的语义较粗，对于判别器来说噪声大于有用信息; (2) 大段文本占用大量的prompt空间，不仅导致训练推理成本上升，同时噪声淹没了有用证据导致性能下降，为此本文还做了补充实验，我们定义一个证据单元，该单元用于检索以及充当后续的构图节点，我们测试三种粒度的证据单元，分别为：(1) 粗粒度：原始report文档作为证据单元；(2)中粒度：语义段，即根据report中两两句子的相似度决定是否断句，证据单元为多句子级；(3) 细粒度(当前使用): 把report按句子切分，证据单元为句子级。
2. dense similarity：使用BAAI/bge-base-en-v1.5作为基础embedding模型，dense similarity分数由claim与证据单元的embedding余弦相似度得到，负责捕捉语义近似、改写、同义表达；lexical overlap：根据英文规则将文本分词，然后统计 claim 与 evidence 的内容词重叠。分数是 overlap F1：既看 evidence 中有多少词命中 claim，也看 claim 中有多少核心词被 evidence 覆盖；BM25-like score：基于同一套内容词 token counter，但对 query term 在 evidence 中的出现频次做 BM25 风格的 TF 饱和和长度归一化。三类信号的评估动机是检验 sparse exact-match 线索是否能补足 dense retrieval 对实体、数字、政策名、地点、人名等硬约束的不足。
3. 最终采用 dense-only 的原因是实验结果不支持三路线性加权作为主方法。我们在 full deduplicated evidence pool 上先搜索 oracle evidence，再用 dense-only、lexical-only、BM25-like-only、dense+sparse、equal 3-way 和 0.70/0.20/0.10 hybrid 等候选检索方式构建同等大小的 candidate pool，并比较这些候选池对 full-pool oracle evidence 的 recall。结果显示 dense-only 在 val/test 上整体最稳；lexical overlap 与 BM25-like 单独使用明显弱于 dense-only，线性混入后也没有稳定提升 candidate-pool recall。因此论文中不再将 sparse 信号表述为主方法的有效组成，而是将其作为被检验但未采用的检索信号消融。主方法采用 dense-only，是由 full-pool oracle recall 实验选择出的保守方案，而不是拍脑袋指定的权重。
4. BM25-like 不是标准 BM25。标准 BM25 需要 corpus-level document frequency / IDF 和真实语料平均文档长度；当前实现只在本地 content-token counter 上使用 BM25 风格的 TF 饱和与长度归一化，并使用固定的 `avgdl=18.0` 与启发式 IDF。因此它应被称为 BM25-like score，而不是完整 BM25 ranker。

### 要补充的实验证明

在固定候选池预算下，前期检索是否更稳定地把有用证据放进后续 selector 可见范围。

当前已完成的 Qwen3-4B full-pool oracle candidate-pool recall 结果显示，dense-only 是更稳的候选池构建方式。MMR λ=0.70、top-32 时：

| split | dense-only recall@32 | full hybrid recall@32 | dense-only hit@32 | full hybrid hit@32 |
|---|---:|---:|---:|---:|
| val | 0.9220 | 0.9240 | 0.9900 | 0.9850 |
| test | 0.9060 | 0.9010 | 0.9900 | 0.9850 |

解释方式：val 上 full hybrid 的 recall 与 dense-only 非常接近，但 hit rate 略低；test 上 dense-only 在 recall 和 hit rate 上均更好。因此不能声称 sparse 线性加权稳定提升前期检索。更合适的论文表述是：我们系统评估了 dense、lexical、BM25-like 及其组合，发现 dense-only 在 full-pool oracle recall 上最稳定，因此主方法采用 dense-only；sparse 信号仅作为诊断性消融保留。

# 关于Question Decomposition Retrieval and Union

### 整体流程

在 claim-level dense retrieval 之后，系统还会运行 question decomposition (QD) 扩展检索。该步骤将原始 claim 分解为若干面向核查的子问题：

$$Q = \{q_1, q_2, \ldots, q_m\}$$

每个子问题会在同一批 report evidence units 上单独执行检索，并沿用与 claim-level retrieval 相同的 dense-only relevance recipe：

$$s(e, q_i) = s_{\mathrm{dense}}(e, q_i)$$

代码字段中如果仍出现 `hybrid_score` / `baseline_hybrid_score` / `qd_max_question_hybrid`，应理解为兼容旧 pipeline 的字段名；在 dense-only 设置下，它们实际表示归一化后的 dense relevance score。

每个 question route 默认保留若干高分候选，再通过 RRF-style route merging 合并为 QD candidate pool。合并时会考虑：

```text
question route rank
max question dense score (stored in legacy hybrid-score fields)
question hit count
question focus
```

随后系统将两类候选做 union：

```text
baseline claim-MMR candidates
QD merged candidate pool
```

union 阶段会对重复 evidence 做 canonical text 去重，并为每条候选记录来源特征，例如：

```text
from_baseline
baseline_rank
baseline_hybrid_score  # legacy field name; dense-only setting means dense relevance score
from_qd
qd_pool_rank
qd_rrf_score
qd_question_hit_count
qd_max_question_hybrid  # legacy field name; dense-only setting means max question dense score
union_pool_rank
```

最终 evidence-map stage 消费的是 `union_candidate_pool_<split>.jsonl`，并按 `union_pool_rank` 取 `candidate_top_n=20` 作为 map annotation 的候选证据池。因此，完整流程应理解为：

```text
DenseRetrieve + ClaimMMR
    +
QuestionDecompose + QDRetrieve + route merge
    ->
Union candidate pool
    ->
Evidence Map Construction
```
