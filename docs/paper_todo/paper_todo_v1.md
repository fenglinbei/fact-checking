# 本文档记录该项目在正式成文前需要解答的事情

## exp list

1. 不同分块方法[x]
2. 不同score[x]
3. 不使用QD[ ]
4. 为什么证据选择重要？full evidence v.s. selected evidence v.s 与selected evidence相同证据预算 [ ]


## 关于Candidate Evidence Retrieval

### 整体流程

系统先将 report collection 切分为候选 evidence units，例如 sentence 或 semantic chunk。我们最初评估了三类 claim-evidence 相关性信号：

```text
dense similarity
lexical overlap
BM25-like score
```

这三类信号并不是都必须进入主方法。我们基于 full-pool oracle evidence 做 candidate-pool recall 消融后发现，dense-only 在固定候选池预算下最稳定；lexical overlap 与 BM25-like 线性混入后没有带来一致的 oracle recall 提升，部分设置下反而引入噪声。因此，如果按候选池覆盖质量选择前期检索策略，当前 dense-only 方案使用如下 relevance score：

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
3. 从 candidate-pool recall 角度选择 dense-only 的原因是实验结果不支持三路线性加权作为主方法。我们在 full deduplicated evidence pool 上先搜索 oracle evidence，再用 dense-only、lexical-only、BM25-like-only、dense+sparse、equal 3-way 和 0.70/0.20/0.10 hybrid 等候选检索方式构建同等大小的 candidate pool，并比较这些候选池对 full-pool oracle evidence 的 recall。结果显示 dense-only 在 val/test 上整体最稳；lexical overlap 与 BM25-like 单独使用明显弱于 dense-only，线性混入后也没有稳定提升 candidate-pool recall。因此论文中不再将 sparse 信号表述为主方法的有效组成，而是将其作为被检验但未采用的检索信号消融。若主文采用 dense-only，它的依据是 full-pool oracle recall 实验，而不是拍脑袋指定的权重；但下游 verifier test 指标仍需单独报告，不能由 candidate-pool recall 直接推出。
4. BM25-like 不是标准 BM25。标准 BM25 需要 corpus-level document frequency / IDF 和真实语料平均文档长度；当前实现只在本地 content-token counter 上使用 BM25 风格的 TF 饱和与长度归一化，并使用固定的 `avgdl=18.0` 与启发式 IDF。因此它应被称为 BM25-like score，而不是完整 BM25 ranker。

### dense-only 与 hybrid 的方法对照

我们在 RAW-FC 上对 Llama3.1-8B-Instruct 和 Qwen3-4B-Instruct-2507 的 FullFT verifier 做 dense-only 与原 hybrid retrieval 的配对比较。下游指标只报告 test Accuracy、Macro-Precision、Macro-Recall 和 Macro-F1：

| model | variant | test acc | macro P | macro R | macro F1 |
|---|---|---:|---:|---:|---:|
| Llama3.1-8B FullFT | dense-only | 0.6550 | 0.6681 | 0.6549 | 0.6582 |
| Llama3.1-8B FullFT | hybrid | 0.6700 | 0.6771 | 0.6702 | 0.6723 |
| Qwen3-4B-2507 FullFT | dense-only | 0.6450 | 0.6504 | 0.6452 | 0.6474 |
| Qwen3-4B-2507 FullFT | hybrid | 0.6700 | 0.6738 | 0.6704 | 0.6711 |

paired bootstrap 按 `sample_idx` 对齐 test predictions，重采样 10,000 次。下表中 `delta = dense-only - hybrid`；负值表示 hybrid 点估计更高，正值表示 dense-only 点估计更高。`P(delta > 0)` 是 bootstrap 样本中 dense-only 优于 hybrid 的比例，`two-sided p` 由两侧符号概率估计。

| model | metric | delta | 95% paired bootstrap CI | P(delta > 0) | two-sided p | interpretation |
|---|---|---:|---:|---:|---:|---|
| Llama3.1-8B FullFT | accuracy | -0.0150 | [-0.0800, +0.0500] | 0.301 | 0.602 | not significant |
| Llama3.1-8B FullFT | macro_precision | -0.0090 | [-0.0757, +0.0578] | 0.390 | 0.779 | not significant |
| Llama3.1-8B FullFT | macro_recall | -0.0153 | [-0.0812, +0.0503] | 0.325 | 0.650 | not significant |
| Llama3.1-8B FullFT | macro_f1 | -0.0141 | [-0.0803, +0.0516] | 0.336 | 0.671 | not significant |
| Qwen3-4B-2507 FullFT | accuracy | -0.0250 | [-0.0750, +0.0250] | 0.144 | 0.289 | not significant |
| Qwen3-4B-2507 FullFT | macro_precision | -0.0234 | [-0.0772, +0.0298] | 0.198 | 0.395 | not significant |
| Qwen3-4B-2507 FullFT | macro_recall | -0.0252 | [-0.0752, +0.0240] | 0.163 | 0.326 | not significant |
| Qwen3-4B-2507 FullFT | macro_f1 | -0.0237 | [-0.0747, +0.0262] | 0.184 | 0.367 | not significant |

证据集合确实被 retrieval variant 改变，而不是同一证据上的随机训练波动：

| model | evidence changed | avg evidence Jaccard | prediction changed | both correct | dense-only correct only | hybrid correct only | both wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama3.1-8B FullFT | 183/200 (91.5%) | 0.6873 | 51/200 (25.5%) | 110 | 21 | 24 | 45 |
| Qwen3-4B-2507 FullFT | 183/200 (91.5%) | 0.6873 | 31/200 (15.5%) | 118 | 11 | 16 | 55 |

结论：
1. dense-only 与 hybrid 会显著改变 evidence set；RAW-FC test 上 91.5% 的样本证据集合或顺序发生变化，平均 Jaccard 约 0.687。
2. 证据变化进一步造成部分 prediction flip，但收益和损失是混合的。Llama 上 dense-only 独有判对 21 条、hybrid 独有判对 24 条；Qwen 上 dense-only 独有判对 11 条、hybrid 独有判对 16 条。
3. test 点估计在两个模型的 Accuracy、Macro-Precision、Macro-Recall 和 Macro-F1 上都偏向 hybrid；其中 Llama 的 dense-only macro-F1 低 1.41 个点，Qwen 的 dense-only macro-F1 低 2.37 个点。
4. paired bootstrap CI 均覆盖 0，two-sided p 均未达到常用显著性阈值，因此不能写成 hybrid 显著优于 dense-only；更稳妥的说法是：RAW-FC 下游结果不支持 dense-only 带来稳定 verifier 性能提升，点估计更偏向原 hybrid retrieval。
5. 如果正文篇幅有限，该分析适合放入附录，用来限定主文中的 retrieval 结论：dense-only 的依据是 full-pool oracle recall 与候选池稳定性，而不是 end-to-end verifier 指标显著更高。若最终主实验以 verifier test metric 为最高优先级，则 RAW-FC 上原 hybrid retrieval 目前更稳。


### 不同chunking方式的比较

为了说明不同 chunking 方式对 verifier 判断结果的真实影响，我们在 RAW-FC 上对 Qwen3-4B-Instruct-2507 LoRA verifier 做 chunking 粒度对照。由于 raw report、semantic chunk 和 sentence 的 evidence unit 长度不同，固定 top-5 会让细粒度 chunking 看到更少信息，因此这里使用 `adaptive5_10` prompt-budget matched 设置：先按同一套 hybrid MMR 排序，取 `candidate_pool_k=32`，再以 `adaptive5_10` reference build 中同一 `event_id` 的 `prompt_token_count` 作为目标，约束 `min_k=5, max_k=20`，选择接近同等 prompt 信息量的 evidence。后续不使用 map selector，仅比较 chunking 粒度本身。

| chunking | acc | macro P | macro R | macro F1 |
|---|---:|---:|---:|---:|
| raw report | 0.6450 | 0.6487 | 0.6451 | **0.6464** |
| semantic chunk | 0.6300 | 0.6319 | 0.6300 | 0.6308 |
| sentence | 0.6400 | 0.6436 | 0.6400 | 0.6409 |

prompt 统计显示 budget matching 基本消除了固定 top-5 下的信息量偏差。`adaptive ref` 是文中主方法的prompt预算信息

| setting | prompt mean | prompt median | prompt p25 | prompt p75 | prompt min | prompt max | evidence mean | evidence median | trunc rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive ref | 775.5 | 824.0 | 633 | 936 | 323 | 1019 | 7.9 | 9.0 | 36.0% |
| raw report | 746.4 | 773.5 | 617 | 901 | 322 | 1020 | 8.1 | 8.0 | 34.0% |
| semantic chunk | 748.4 | 772.5 | 632 | 895 | 322 | 1020 | 8.3 | 8.0 | 33.5% |
| sentence | 761.5 | 798.0 | 635 | 905 | 326 | 1016 | 14.1 | 14.5 | 7.5% |

paired bootstrap 按同一 200 条 RAW-FC test 样本的 `sample_idx` 对齐预测，重采样 10,000 次。下表中差值均为前者减后者的 Macro-F1：

| comparison | macro-F1 delta | 95% paired bootstrap CI | P(delta > 0) | two-sided p | interpretation |
|---|---:|---:|---:|---:|---|
| raw report - semantic chunk | +0.0157 | [-0.0237, +0.0566] | 0.7805 | 0.4389 | not significant |
| raw report - sentence | +0.0055 | [-0.0482, +0.0602] | 0.5847 | 0.8305 | not significant |
| sentence - semantic chunk | +0.0102 | [-0.0353, +0.0558] | 0.6620 | 0.6761 | not significant |

结论：
1. budget-matched 后，raw report、semantic chunk 和 sentence 三种粒度的 verifier 指标接近，点估计上 raw report 最高，sentence 次之，semantic chunk 最低。
2. paired bootstrap 的 95% CI 均覆盖 0，且对比均不显著，即在 w/o map selector、同等 prompt budget 下，RAW-FC verifier 对 chunking 粒度不高度敏感；但粒度的差别主要体现在后续map select的选择口径上

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

### API / open-weight LLM 表述与验证口径

需要解答：
1. QD 由 DeepSeek-V4-Flash API 生成，是否会被审稿人理解为依赖闭源黑箱
2. 如何证明这些 generated verification questions 对候选证据召回和最终 verifier 有用
3. 如何证明 API 生成的问题与人类核查思路对齐

处理方式：
1. 不再写成 `closed-source API`。更准确的表述应为：`an open-weight / open-source LLM accessed through a hosted API`。也就是说，DeepSeek-V4-Flash 本身应作为开源或 open-weight teacher model 表述；真正需要控制的是 hosted API 带来的 endpoint drift、服务端推理实现差异和 generation reproducibility，而不是模型权重不可见。
2. 在方法部分将 QD 定位为 `synthetic retrieval-query generation` 或 `verification-question guided retrieval expansion`，而不是 human rationale 或 gold decomposition。QD 的输出只用于扩展检索 route，不直接作为最终 label、rationale gold 或 verifier prompt 中的事实依据。
3. 在 reproducibility 部分记录并释放以下信息：model id、API provider、调用日期、prompt version、JSON schema、temperature / top_p / max_tokens、retry / parse policy、question cache fingerprint、生成后的 `questions_<split>.jsonl` 和 `union_candidate_pool_<split>.jsonl`。论文中应说明 API 只是调用方式，核心实验以缓存后的中间产物和固定代码路径复现。
4. 在 limitation 中保留一句谨慎表述：虽然 teacher model 是 open-weight / open-source，但 hosted API 仍可能随时间发生行为漂移；因此所有 QD 相关结论都通过候选池召回、下游消融和人工抽样验证支撑，而不是把 API 输出当作不可质疑的标注。

### QD 有用性的实验设计

主张边界：
QD 不是最终 evidence selector，也不应声称它直接选出的 top-k 一定优于 claim-level MMR。它要证明的是：在固定候选池预算下，verification-question routes 能把更多潜在有效证据带入 evidence-map / graph selector 的可见范围。

建议报告的对比：

| setting | 目的 |
|---|---|
| claim-MMR only | 无 QD 的前期检索 baseline |
| QD only | 检查 question routes 本身的召回能力 |
| claim-MMR + QD union | 当前完整候选池 |
| random / paraphrase questions | 排除收益只是来自“多检索几轮” |
| human-written questions | 人类 QD 上界或对齐参照 |

建议指标：
1. candidate-pool 层面：`oracle_pool_recall@20`、`any_oracle_hit@20`、`all_oracle_hit@20`、`QD-only rescued oracle evidence rate`。
2. selector 层面：最终 evidence chain 中 `qd-only`、`baseline+qd`、`baseline-only` 证据占比，以及各来源证据的 oracle overlap / atom coverage。
3. verifier 层面：在相同 evidence-map / graph / verifier 设置下比较 Accuracy、Macro-F1、class-wise F1，并对关键差距做 paired bootstrap。
4. 消融解释：如果 QD only top-k 不强，但 union pool recall 明显提升，则应写成“QD improves candidate coverage for downstream structured selection”，而不是“QD itself is a better selector”。

最终论文主表应优先使用与最终 dense-only retrieval 设置一致的 QD rerun artifact。历史 QD artifact 可作为 sanity check，但如果其 `hybrid_score` 实际来自早期 dense+lexical+BM25-like 配置，不应直接混入主结论表。

### 与人类核查问题的对齐评估

QD 与人类结果对齐不能用 exact string match 评估，因为同一核查意图可以有多种自然语言问法。建议用 facet-level / utility-level 对齐：

1. 抽取 100-200 个 claim，按 label、claim 长度、实体/数字/时间/比较/归因复杂度分层。
2. 由两名以上人工标注者写出每个 claim 需要核查的 verification facets 或 retrieval questions，并标注 focus 类型：`overall`、`entity`、`quantity`、`time`、`comparison`、`causal`、`attribution`、`policy`、`other`。
3. 将 API QD 与 human QD 映射到同一 facet schema，计算 facet precision / recall / F1、question count agreement、focus distribution agreement、redundancy rate、vague-question rate、unsupported-facet / hallucinated-facet rate。
4. 用 human-written questions 跑同一套 `QDRetrieve`，与 API QD、claim-MMR only、random questions 比较候选池 oracle recall 和下游 verifier 指标。若 API QD 接近 human QD，则说明 API 生成与人类核查思路足够对齐；若 human QD 显著更强，则应把 API QD 写为可替换的自动近似模块，并在 limitation 中承认仍有改进空间。
5. 对人工评分报告 inter-annotator agreement，例如 Cohen's kappa 或 Krippendorff's alpha；对主观评分可报告平均分和置信区间，维度包括 necessity、specificity、coverage、non-redundancy、no outside facts。

### 同类 API-generated intermediate data 的论文处理方式

在相关使用 LLM API 生成中间数据或 synthetic supervision 的论文中，通常不会把生成结果包装成人类标注，而是采用如下处理：
1. 明确披露 teacher model、API provider、prompt、decoding 参数和生成时间。
2. 释放生成后的数据缓存、过滤脚本和 schema validator，使实验不依赖未来重新调用同一个 API endpoint。
3. 将生成数据称为 synthetic / weak supervision / teacher-generated intermediate representation。
4. 加入人工抽样验证、规则过滤、去重、invalid JSON / schema violation 统计。
5. 做 no-LLM、random、human upper-bound 或不同 teacher model 的对照实验。

因此本文的稳妥写法是：DeepSeek-V4-Flash 不是闭源黑箱 teacher；它是 open-weight / open-source teacher model，但 QD 仍属于 LLM-generated intermediate retrieval queries。论文需要证明的是这些 queries 在候选召回、下游选择和人工 facet 对齐上有效，而不是仅凭模型开源身份假定其生成结果可靠。
