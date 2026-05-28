# Oracle Set Case 调查到 v0.5 Evidence Selector 路线记录

日期：2026-05-28

## 目的

本文记录从实际 Oracle evidence set case 调查出发，到 v0.1 - v0.5 系列 selector diagnostic 的方法演化、关键指标和阶段性结论。

这条路线的核心问题一直是：

```text
能否用一个可解释的 evidence selector，在不读取 gold label / oracle 字段的前提下，
选出更接近 oracle set、且更有利于最终 LIAR-RAW 分类的 top5 evidence？
```

截至 v0.5b，结论比较清楚：

1. stance bucket 和 directness teacher 能解释一部分现象，但 candidate-level oracle overlap 信号较弱。
2. provenance / rank / source 特征仍是最稳定的 oracle-likelihood 信号。
3. off-the-shelf text-only direct evidence cross-encoder 信号弱，但融合后有很小增益。
4. evidence map 显著提升可解释性、atom coverage 和 direct/partial rate，但会损失 oracle set overlap。
5. 旧 oracle-direct verifier 并没有直接吃到 map-aware prompt 的收益；v0.5b 最强仍是 base-only evidence prompt。

## Stage2 Oracle Set 来源说明

本文讨论的 Oracle set 来自 sentence-level Stage2 margin oracle search，而不是人工 evidence relevance 标注。本文使用的主分析对象是 val split，共 `1274` 条 claim；同一套口径也可用于 train split，但本文的 case 与指标均以 val split 为准。

Stage2 Oracle search 的候选池语义是：

```text
raw report sentences
-> sentence chunks
-> dense / lexical / BM25 hybrid scoring
-> canonical dedup
-> hybrid_score descending effective top15
-> greedy oracle search selects ordered top5
```

关键配置与坐标系：

| item | value |
| --- | --- |
| `chunk_mmr_fingerprint` | `432dfc970e75` |
| chunking | sentence-level chunks |
| effective candidate pool | dedup 后按 hybrid score 排序的 top15 |
| hybrid score | dense `0.70` + lexical `0.20` + BM25 `0.10` |
| search method | greedy |
| search objective | margin |
| top_k | 5 |
| `selected_indices` 坐标 | 每条 row 的 effective `candidate_pool` 内 index，不是全局 source index |

margin objective 定义为：

```text
margin(claim, S) =
  log P(gold label | claim, S)
  - max_{wrong label} log P(wrong label | claim, S)
```

因此 `selected_indices` 表示：在当前 verifier / label-token scorer 下，greedy search 找到的、能最大化 gold-label margin 的 evidence 顺序。它是 verifier-utility supervision，不等同于人工 rationale，也不等同于传统 IR relevance label。

每条 oracle row 保存了：

- `claim`、`gold_label`、`gold_id`
- `candidate_pool`：selector 需要面对的 effective top15 候选池
- `candidate_scores`：候选的 dense / lexical / BM25 / hybrid 分数与 hybrid rank
- `selected_indices`、`selected_texts`：oracle greedy top5
- `search_steps`：每一步 greedy 选择后的 logprob / margin 信息
- `candidate_pool_metadata`、`candidate_pool_fingerprint`：候选池口径审计字段

在本文所有 selector 实验中，`gold_label`、`oracle_selected`、`selected_indices`、`oracle_step`、oracle margin/logprob 只能用于离线训练标签、case 分析和 metrics；它们不能进入 teacher prompt、selector scoring 或部署时 inference 输入。

当前 val oracle search 的整体指标仅作为这批 Oracle set 的健康度参考：

| metric | value |
| --- | ---: |
| n_samples | 1274 |
| oracle search accuracy | 0.6593 |
| oracle search macro-F1 | 0.6620 |
| effective candidate pool mean size | 14.6193 |
| parse error rate | 0.0000 |

## QD Union Pool 来源说明

v0 - v0.2 中使用的 `QD union pool` 来自 question decomposition retrieval v0。它不是 Oracle set，也不是人工标注数据；它是一个只根据 claim 生成检索问题、再用这些问题召回候选 evidence 的候选池扩展机制。

QD 生成与检索流程如下：

```text
claim
-> 生成 1-5 个自包含检索问题
-> 每个问题分别检索 sentence-level candidates
-> 每个 question route 保留 top candidates
-> 按 canonical text 去重合并
-> 用 RRF / route hit count / max question hybrid score 排序
-> 与 baseline retrieval pool 做 union
```

问题生成阶段只读取 claim，不读取 `gold_label`、Oracle selected indices、oracle margin/logprob 或任何 selector 输出。问题 schema 中保留：

| field | meaning |
| --- | --- |
| `question_id` | 问题编号，`q1` 固定是原 claim 的整体核验问题 |
| `question` | 自包含 retrieval question |
| `focus` | 问题侧重点，例如 `overall`、`entity`、`quantity`、`time`、`comparison`、`causal`、`attribution`、`policy`、`other` |
| `priority` | 问题优先级 |

检索阶段对每个问题单独计算候选句的 hybrid score：

```text
question-candidate hybrid score =
  0.70 * dense_score
+ 0.20 * lexical_score
+ 0.10 * BM25_score
```

每个问题 route 默认保留最多 `20` 条候选。多个 question route 合并时使用 RRF 风格分数，`q1` 作为 broad verification question 权重略高；同一候选如果被多个问题命中，会累积 `question_hit_count` 和 route coverage。合并后的 QD pool 默认保留 `15` 条候选。

本文中的 `QD union pool` 已经把两类候选放在同一 candidate list 中：

| source flag | meaning |
| --- | --- |
| `from_baseline` | 来自原始 baseline retrieval / Stage2-style hybrid pool |
| `from_qd` | 来自 question decomposition routes |
| `from_baseline && from_qd` | 原始检索和 QD route 都命中的候选 |
| `qd_rrf_score` | 多 question route 合并后的 QD RRF 分数 |
| `qd_question_hit_count` | 命中该候选的不同问题数量 |
| `qd_question_routes` | 命中该候选的具体 question route、focus、rank 和 question-level score |
| `union_pool_rank` | baseline 与 QD 合并去重后的候选排序 |

因此 v0 - v0.2 的候选池不是只看 Stage2 effective top15，而是做了进一步合并：

```text
analysis candidate pool =
  dedup(Stage2 original candidate_pool ∪ QD union pool)
```

这一步的目的，是在保留 Stage2 Oracle supervision 坐标的同时，补足原始检索可能没有覆盖到的 claim aspect，例如比较对象、时间范围、数量口径、归因来源或反例。代价是会引入更多 qd-only noise 和近重复 evidence，所以后续 selector 才需要 source dedup、question route weight、directness / stance / map 结构来控制候选质量。

## 起点：Oracle Set Case 调查

最早的启发来自对 Stage2 Oracle set 的人工观察，尤其是 `4855.json`、`11447.json`、`10443.json` 三个 case。下面直接展开这三个 case 的 effective candidate pool 与 Oracle greedy top5，作为本文后续方法演化的自足上下文。

### 三个 Case 的主要观察

三例的摘要观察如下：

| event_id | claim 摘要 | gold label | oracle selected indices | v0.1 命中 |
| --- | --- | --- | --- | --- |
| `4855.json` | Romney 曾支持 Obama health care plan 但现在反对 | `barely-true` | `[13, 8, 1, 9, 2]` | O3, O5 |
| `11447.json` | We have the highest tax rate anywhere in the world | `false` | `[1, 13, 10, 2, 7]` | O2 |
| `10443.json` | 美国领导力和军事力量正在阻止 ISIS 推进 | `half-true` | `[13, 2, 11, 3, 9]` | O3, O5 |

| event_id | label | Oracle set 特征 | v0.1 结果 | 关键教训 |
| --- | --- | --- | --- | --- |
| `4855.json` | `barely-true` | Oracle 五条全部落在 ambiguous bucket，围绕 Romneycare / Obamacare 的支持、反对与历史关系链 | overlap `2/5`，v0.1 从 oppose 桶额外选入一条非 Oracle evidence | 这个 case 需要关系链和时间立场转变，不需要强行做 stance 两端平衡 |
| `11447.json` | `false` | Oracle 同时包含关键反证、原始错误说法、碎片数字和背景片段 | overlap `1/5`，stance 覆盖改善，但没选到最关键反证 O1 | 多桶平衡能救回方向，但桶内 scorer 不会自动识别最强可判定反证 |
| `10443.json` | `half-true` | Oracle 混合 oppose / ambiguous，包含军事手段不足、ISIS 背景、地区安全上下文 | overlap `2/5`，v0.1 引入 support-side 美国军事存在背景证据 | stance 极性相关不等于 claim-specific factuality |

### Case 1：`4855.json`

- Claim: Says Mitt Romney once supported President Obamas health care plan but now opposes it.
- Gold label: `barely-true`
- Oracle selected indices: `[13, 8, 1, 9, 2]`

Oracle greedy top5：

| oracle order | candidate idx | hybrid rank | hybrid score | evidence text |
| --- | ---: | ---: | ---: | --- |
| O1 | 13 | 13 | 0.6354 | Still , Romney also never withdraw his support from the program . `` After a backlash Thursday , Romney try to walk that line again , post on Facebook that he still oppose Obamacare because `` it have fail , `` `` drive up premium `` and `` take insurance away from people `` . |
| O2 | 8 | 8 | 0.6907 | He argue that without `` Romneycare , `` the universal health care plan he sign into law a Massachusetts governor , Obamacare would never have become law . |
| O3 | 1 | 1 | 0.9061 | Romney say if elect president , he would allow state to opt out of the health care law . |
| O4 | 9 | 9 | 0.6714 | In the speech , Romney wo n't spend much time talk about Massachusetts , and the plan he sign that now require the state 's citizen to buy health insurance — an individual mandate that be include in the federal law and drive Republican fury . |
| O5 | 2 | 2 | 0.7745 | ConsensusRead in appMitt Romney spoke in Washington on Thursday after the Supreme Court rule on President Obama 's health care law . |

完整 effective candidate pool：

| candidate idx | oracle order | hybrid rank | hybrid score | evidence text |
| ---: | --- | ---: | ---: | --- |
| 0 | - | 0 | 0.9335 | As Republican presidential candidate , Romney oppose the federal health care law that do the same thing . |
| 1 | O3 | 1 | 0.9061 | Romney say if elect president , he would allow state to opt out of the health care law . |
| 2 | O5 | 2 | 0.7745 | ConsensusRead in appMitt Romney spoke in Washington on Thursday after the Supreme Court rule on President Obama 's health care law . |
| 3 | - | 3 | 0.7671 | Despite his party ’ s unify attack on the health care law , Romney , whose own health insurance reform in Massachusetts be a model for Obama ’ s plan , have recently hint at willingness to compromise on some of it politically popular element . |
| 4 | - | 4 | 0.7530 | He admit that the health care plan he institute a governor of Massachusetts be the precursor to Obamacare . `` What you [ President Obama ] do instead be to push through a plan without a single Republican vote . |
| 5 | - | 5 | 0.7181 | Story highlight Mitt Romney say the health insurance mandate be a tax , even though he disagree His campaign have be criticize for inconsistency on the issue President Barack Obama also disagree that the mandate be a tax Republicans want to galvanize support around their desire to repeal Obamacare As Massachusetts governor , Mitt Romney impose a penalty on people who could afford health insurance but choose to go without it . |
| 6 | - | 6 | 0.7169 | Mitt Romney sign the Massachusetts plan into law in 2006 . |
| 7 | - | 7 | 0.7005 | President Obama 's signature health law draw heavily on Romney 's own health-care reform effort in Massachusetts when he be governor there . |
| 8 | O2 | 8 | 0.6907 | He argue that without `` Romneycare , `` the universal health care plan he sign into law a Massachusetts governor , Obamacare would never have become law . |
| 9 | O4 | 9 | 0.6714 | In the speech , Romney wo n't spend much time talk about Massachusetts , and the plan he sign that now require the state 's citizen to buy health insurance — an individual mandate that be include in the federal law and drive Republican fury . |
| 10 | - | 10 | 0.6464 | He admit that the health-care plan he institute a governor of Massachusetts be the precursor to Obamacare . `` What you [ President Obama ] do instead be to push through a plan without a single Republican vote . |
| 11 | - | 11 | 0.6409 | Romney would encourage more people to buy health plan in the individual market by make the tax treatment of individually purchase coverage similar to that now accord to employer-based plan . |
| 12 | - | 12 | 0.6397 | Elected governor of Massachusetts in 2002 , Romney help develop and later sign a health care reform law ( commonly call `` Romneycare `` ) that provide near-universal health insurance access through state-level subsidy and individual mandate to purchase insurance . but support Nixon 's ongoing Cambodian Incursion a a sincere attempt to end the war . |
| 13 | O1 | 13 | 0.6354 | Still , Romney also never withdraw his support from the program . `` After a backlash Thursday , Romney try to walk that line again , post on Facebook that he still oppose Obamacare because `` it have fail , `` `` drive up premium `` and `` take insurance away from people `` . |
| 14 | - | 14 | 0.6260 | The public plan have command enormous public attention , and Romney use to it frame Masscare a a conservative reform rely on private health insurance , and against Obama ’ s proposal to create a government plan that , Romney claim , would balloon into a massive entitlement . |

Case 观察：最高 hybrid 候选 C0 是“Romney oppose federal health care law”，方向上很强，但 Oracle 没选；Oracle 更偏向把“曾经支持 / Romneycare 先例 / 现在反对”这条关系链拼起来。这个 case 说明 stance 极性强不等于 oracle utility 高。

### Case 2：`11447.json`

- Claim: We have the highest tax rate anywhere in the world.
- Gold label: `false`
- Oracle selected indices: `[1, 13, 10, 2, 7]`

Oracle greedy top5：

| oracle order | candidate idx | hybrid rank | hybrid score | evidence text |
| --- | ---: | ---: | ---: | --- |
| O1 | 1 | 1 | 0.8112 | Our verdict Incorrect , a number of European country have high income tax rate than Scotland . “ They ( Scotland ) have the high tax anywhere in Europe ” Boris Johnson , 4 September 2019 ( 23 . |
| O2 | 13 | 13 | 0.7173 | 35 ) At his first Prime Minister ’ s Questions , Boris Johnson say that Scotland have the high tax anywhere in Europe . |
| O3 | 10 | 10 | 0.7356 | The tax rate for 2021 range from 14 . |
| O4 | 2 | 2 | 0.7943 | The tax we love to hate today . |
| O5 | 7 | 7 | 0.7469 | 19 percent , more than anywhere else in the world . |

完整 effective candidate pool：

| candidate idx | oracle order | hybrid rank | hybrid score | evidence text |
| ---: | --- | ---: | ---: | --- |
| 0 | - | 0 | 0.9601 | remove ( ) ; } ( ) ) ; Which country have the high tax rate ? . |
| 1 | O1 | 1 | 0.8112 | Our verdict Incorrect , a number of European country have high income tax rate than Scotland . “ They ( Scotland ) have the high tax anywhere in Europe ” Boris Johnson , 4 September 2019 ( 23 . |
| 2 | O4 | 2 | 0.7943 | The tax we love to hate today . |
| 3 | - | 3 | 0.7778 | Countries With the Highest Income Tax for Single People Let ’ s look at the country with the high all-in average personal income tax rate at the average wage for a single person with no child . |
| 4 | - | 4 | 0.7625 | If Trump be talk about the federal income tax rate that individual pay , Americans still do not face the high tax rate in the world . |
| 5 | - | 5 | 0.7600 | Where do taxpayer pay the high income tax ? . |
| 6 | - | 6 | 0.7534 | If you could live anywhere in the world , wouldn ’ t you want to know the potential income tax before move and how that compare to the U . |
| 7 | O5 | 7 | 0.7469 | 19 percent , more than anywhere else in the world . |
| 8 | - | 8 | 0.7403 | As a share of the economy , the United States be nowhere close to the highest-taxed country in the world and do not raise nearly as much tax revenue a other developed country , many of which be in Europe . tax revenue amount to 26 percent of GDP in 2014 , with about one-third of that come from state and local government tax . |
| 9 | - | 9 | 0.7360 | In general , income tax be high in the Nordic country . |
| 10 | O3 | 10 | 0.7356 | The tax rate for 2021 range from 14 . |
| 11 | - | 11 | 0.7301 | The high tax be in Denmark , Finland , and Iceland with respectively , 55 . |
| 12 | - | 12 | 0.7288 | Countries around the world usually implement one of four type of tax system . |
| 13 | O2 | 13 | 0.7173 | 35 ) At his first Prime Minister ’ s Questions , Boris Johnson say that Scotland have the high tax anywhere in Europe . |
| 14 | - | 14 | 0.7131 | 10 state with the high personal income tax rate A comparison of 2020 tax rate compile by the Tax Foundation rank California a the top taxer with a 12 . |

Case 观察：Oracle O1 是最关键的直接反证；但 Oracle 同时选了 O4/O5 这类碎片或上下文句。它显示 Oracle set 不是干净的人工 rationale，而是当前 verifier 在 margin objective 下觉得有用的组合。

### Case 3：`10443.json`

- Claim: In Iraq and Syria, American leadership, including our military power, is stopping (the Islamic States) advance.
- Gold label: `half-true`
- Oracle selected indices: `[13, 2, 11, 3, 9]`

Oracle greedy top5：

| oracle order | candidate idx | hybrid rank | hybrid score | evidence text |
| --- | ---: | ---: | ---: | --- |
| O1 | 13 | 13 | 0.5570 | Little suggest these group can be defeat by military mean alone , yet they espouse goal hard to accommodate in negotiated settlement . the largely nationalist Afghan Taliban , resurgent a foreign troop draw down from Afghanistan , and Pakistani group include sectarian movement , tribal militant fight the central state and Kashmir- or Afghanistan-focused element align to it military establishment . |
| O2 | 2 | 2 | 0.7612 | These militant , know a the Islamic State in Iraq and the Levant ( ISIL , or the Islamic State ) , declare an Islamic state or caliphate in this captured territory and claim political and theological authority over the world ’ s Muslims . |
| O3 | 11 | 11 | 0.5808 | And official have say they believe Iran be behind the October drone attack at the military outpost in southern Syria where American troop be base . |
| O4 | 3 | 3 | 0.7232 | Its leadership be mostly Iraqi but the movement be protean . |
| O5 | 9 | 9 | 0.6347 | People seek to travel to engage in terrorist activity in Syria or Iraq should be in no doubt that the UK will take the strong possible action to protect our national security , include prosecute those who break the law . |

完整 effective candidate pool：

| candidate idx | oracle order | hybrid rank | hybrid score | evidence text |
| ---: | --- | ---: | ---: | --- |
| 0 | - | 0 | 1.0000 | Loss of territory do not equal defeat According to RAND expert , the Islamic State have lose around 36 per cent of it former territory in Iraq and Syria since the beginning of coalition military operation in 2014 . |
| 1 | - | 1 | 0.7891 | In this section Parliamentary LibraryAbout the Parliamentary LibraryResearch PublicationsParliamentary LibraryMonthly Statistical BulletinFlagPostBills DigestBrowse by TopicParliamentary HandbookParliament then and now Nathan Church , Foreign Affairs , Defence and Security Key Issues The nature and scope of military operation in Iraq and Syria be significant issue for Australia give the evolving , long-term threat from the Islamic State . |
| 2 | O2 | 2 | 0.7612 | These militant , know a the Islamic State in Iraq and the Levant ( ISIL , or the Islamic State ) , declare an Islamic state or caliphate in this captured territory and claim political and theological authority over the world ’ s Muslims . |
| 3 | O4 | 3 | 0.7232 | Its leadership be mostly Iraqi but the movement be protean . |
| 4 | - | 4 | 0.7168 | There be no military solution to the situation in the country , he stress , reaffirm commitment to advance a Syrian-led and Syrianowned political process . |
| 5 | - | 5 | 0.7143 | will keep the current 2,500 troop in Iraq for the foreseeable future , despite their shift to a non-combat role , and they will still provide air support and other military support for Iraq 's continue fight against the Islamic State . force to a non-combat role in Iraq , they will still provide air support and other military aid for Iraqs fight against the Islamic State . |
| 6 | - | 6 | 0.6926 | reveal the extent to which the US military be once again engage in intense combat in Iraq . |
| 7 | - | 7 | 0.6383 | While acknowledge military force be not the principal answer to the region challenge , our presence in the region provide advantage , opportunity and leverage for U . diplomat to operate from a position of strength , prevents lose ground to our global competitor , and protect the security of the American people by meet challenge abroad from state and non-state adversary who threaten the U . |
| 8 | - | 8 | 0.6379 | And where gap open , China and Russia pursue steady economic and military measure that encroach on U . diplomatic facility or Iraqi military base host U . national power to address the underlying condition threaten stability . |
| 9 | O5 | 9 | 0.6347 | People seek to travel to engage in terrorist activity in Syria or Iraq should be in no doubt that the UK will take the strong possible action to protect our national security , include prosecute those who break the law . |
| 10 | - | 10 | 0.5857 | Sniper work , capture the enemy , undermine armored syrian solider kill isis gopro war in syria combat footage heavy attack /a > Post navigation . security in iraq and isi insurgent - isi fighter stock video & royalty-free footage . security in iraq and isi insurgent - isi stock video & royalty-free footage . |
| 11 | O3 | 11 | 0.5808 | And official have say they believe Iran be behind the October drone attack at the military outpost in southern Syria where American troop be base . |
| 12 | - | 12 | 0.5606 | This include key city in Iraq like Fallujah and Ramadi . |
| 13 | O1 | 13 | 0.5570 | Little suggest these group can be defeat by military mean alone , yet they espouse goal hard to accommodate in negotiated settlement . the largely nationalist Afghan Taliban , resurgent a foreign troop draw down from Afghanistan , and Pakistani group include sectarian movement , tribal militant fight the central state and Kashmir- or Afghanistan-focused element align to it military establishment . |
| 14 | - | 14 | 0.5504 | Little suggest they can be defeat by military mean alone . |

Case 观察：hybrid rank 0 的 C0 直指“territory loss does not equal defeat”，看起来比若干 Oracle 句更像人类 rationale；但 Oracle 选择了更分散的背景、限制和上下文组合。这个 case 强化了一个判断：Oracle set 是 verifier-utility set，而不是单纯 direct evidence set。

这些 case 给了两个初始假设：

1. Oracle evidence 往往更像 verifier utility set，而不是人工 rationale set。它有时会选择背景、碎片、原始说法或上下文锚点。
2. stance 与 label region 有统计关系，但单条 evidence 是否 oracle-worthy，还取决于 claim-specific directness、事实要素覆盖、检索/provenance 位置和 verifier 的使用方式。

## 初始方案：v0 / v0.1 Count-Amplified Stance Bucket

### 方法要点

初始方案可以概括为：

```text
original Stage2 pool + QD union pool
-> 去重合并
-> DeepSeek teacher 标注 stance / completeness
-> stance soft buckets
-> effective_count
-> count-amplified top5
```

核心假设是：如果某个 stance bucket 中出现更多相对独立、语义完整、与 claim 相关且立场一致的 evidence，则该 bucket 的 selector mass 应该超线性增加。

### v0 指标

| selector | recall@5 | jaccard@5 | top1 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| original_pool_order_top5 | 0.3435 | 0.2294 | 0.1028 | 0.2872 |
| qd_union_source_score_top5 | 0.3408 | 0.2271 | 0.1028 | 0.2879 |
| count_amplified_stance_bucket_top5 | 0.2985 | 0.1960 | 0.0714 | 0.2433 |
| linear_stance_bucket_count_top5 | 0.3005 | 0.1974 | 0.0706 | 0.2436 |

v0 的 observation metrics：

| signal | value |
| --- | ---: |
| completeness selected lift | 0.0046 |
| oracle vs pool stance alignment lift | 0.0194 |

结论：stance alignment 和 completeness 都有方向性，但非常弱；超线性 count 不如 linear control，也远低于 original/QD order。

### v0.1 调整

v0.1 的方向是压低 ambiguous bucket 的支配性，增加 selected-count penalty 和 ambiguous penalty。case 级观察显示它确实改善 stance 覆盖：

| event_id | v0 overlap | v0 buckets | v0.1 overlap | v0.1 buckets |
| --- | ---: | --- | ---: | --- |
| `4855.json` | 2 | amb:5 | 2 | amb:4, opp:1 |
| `11447.json` | 0 | amb:5 | 1 | amb:2, opp:2, sup:1 |
| `10443.json` | 2 | amb:5 | 2 | amb:2, opp:2, sup:1 |

但 aggregate metrics 仍没有突破：

| n | count selector recall@5 | count selector jaccard@5 | top1 | NDCG@5 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.2980 | 0.1956 | 0.0754 | 0.2469 |
| 5 | 0.3013 | 0.1979 | 0.0769 | 0.2492 |
| 7 | 0.2985 | 0.1960 | 0.0714 | 0.2433 |

v0.1 的关键结论：

```text
压平 ambiguous 能改善 stance entropy，但没有解决桶内排序。
错误不在“选哪个桶”本身，而在“同一 stance / provenance band 内选哪条 evidence”。
```

## v0.2：Directness-Aware Stance Bucket

### 方法要点

v0.2 不再只让 teacher 给 stance 和 completeness，而是加入更细的 directness / role 字段：

- `direct_evidence_score`
- `claim_specificity_score`
- `key_fact_overlap_score`
- `background_only_score`
- `claim_directness_score`
- `role_evidence_score`

同时加入 adaptive polar quota，试图让 selector 在 false/true-side evidence 上更主动。

### 关键指标

| n | selector | recall@5 | jaccard@5 | top1 | NDCG@5 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | original_pool_order_top5 | 0.3435 | 0.2294 | 0.1028 | 0.2872 |
| 3 | count_amplified_stance_bucket_top5 | 0.2984 | 0.1944 | 0.1115 | 0.2711 |
| 5 | count_amplified_stance_bucket_top5 | 0.2987 | 0.1952 | 0.1193 | 0.2775 |
| 7 | count_amplified_stance_bucket_top5 | 0.3042 | 0.1986 | 0.0965 | 0.2710 |

Teacher signal 很弱：

| feature | oracle lift | AUROC |
| --- | ---: | ---: |
| semantic_completeness | 0.0003 | 0.5013 |
| direct_evidence | 0.0102 | 0.5095 |
| key_fact_overlap | 0.0124 | 0.5113 |
| retrieval_score | 0.0053 | 0.5233 |

Bucket behavior 有改善但没转化成 overlap：

| n | ambiguous share | single-bucket collapse | forced hit rate |
| ---: | ---: | ---: | ---: |
| 3 | 0.568 | 0.1690 | 0.2952 |
| 5 | 0.495 | 0.1502 | 0.2929 |
| 7 | 0.219 | 0.0102 | 0.2638 |

v0.2 结论：

```text
n=7 能机械性降低 bucket collapse；
directness teacher 对 top-rank/order 有轻微帮助；
但 set overlap 仍输给 original/QD controls。
不应继续只调 bucket amplification。
```

## v0.3 / v0.3.1：Oracle-Likelihood Calibrated Selector

### 方法要点

v0.3 转向学习 candidate 的 oracle-likelihood score。stance bucket 不再是主驱动，只作为轻量 diversity constraint。

默认使用 event-level 5-fold OOF val analysis，特征包括：

- retrieval/rank/source
- semantic completeness / directness teacher fields
- stance probs / entropy / region
- route/source diversity helpers

禁止把 `oracle_selected`、`oracle_step`、`gold_label`、`event_id` identity、raw text、candidate_key 作为模型特征。

### v0.3 关键 sweep

| setting | recall@5 | jaccard@5 | top1 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| `ANCHOR_K=0` | 0.3576 | 0.2379 | 0.1028 | 0.2900 |
| `ANCHOR_K=1` | 0.3548 | 0.2355 | 0.1028 | 0.2874 |
| `ANCHOR_K=2` | 0.3576 | 0.2375 | 0.1028 | 0.2880 |
| source penalty `0.10`, stance penalty `0` | 0.3603 | 0.2406 | 0.1028 | 0.2897 |

### v0.3.1 Feature Ablation

| run | objective | feature set | AUROC | AUPRC | recall@5 | jaccard@5 | top1 | NDCG@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pointwise_all_features` | pointwise | all features | 0.6339 | 0.3716 | 0.3642 | 0.2445 | 0.1028 | 0.2996 |
| `pointwise_provenance_rank_only` | pointwise | provenance/rank only | 0.6364 | 0.3757 | 0.3582 | 0.2388 | 0.1036 | 0.2944 |
| `pairwise_all_features` | pairwise | all features | 0.6244 | 0.3431 | 0.3568 | 0.2386 | 0.0871 | 0.2869 |
| `pointwise_teacher_directness_stance_only` | pointwise | teacher + stance only | 0.5029 | 0.2849 | 0.3171 | 0.2108 | 0.1122 | 0.2856 |

与 controls 对比：

| selector | recall@5 | jaccard@5 | top1 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| original_pool_order_top5 | 0.3435 | 0.2294 | 0.1028 | 0.2872 |
| qd_union_source_score_top5 | 0.3408 | 0.2271 | 0.1028 | 0.2879 |
| count_amplified_stance_bucket_top5 | 0.3042 | 0.1986 | 0.0965 | 0.2710 |
| oracle_likelihood_top5 | 0.3642 | 0.2445 | 0.1028 | 0.2996 |

v0.3.1 结论：

```text
纯 learned ranking 是目前第一个明确超过 original/QD controls 的 selector。
但提升主要来自 provenance/rank/source 特征；
teacher directness/stance alone 几乎没有 candidate-level 区分能力。
```

## v0.4a / v0.4a.1：Text-Only Direct Evidence Cross-Encoder

### 方法要点

v0.4a 试图避免继续依赖 provenance/rank，改用更强的 off-the-shelf text-only direct evidence scorer：

```text
claim + evidence_text -> Qwen/Qwen3-Reranker-8B CrossEncoder -> direct_ce_score
```

模型输入只包含 claim 和 evidence text，不包含 rank/source/oracle/gold metadata。

v0.4a.1 修复了 Qwen3-Reranker 与 `sentence_transformers.CrossEncoder` 的 prompt/tokenization 问题，最终以 `default_query` 跑通。

### 关键指标

| metric | value |
| --- | ---: |
| candidate AUROC | 0.5162 |
| AUPRC | 0.2927 |
| same-source hard-negative pairwise acc | 0.5212 |
| within-event pairwise acc | 0.5336 |
| oracle selected score lift | 0.0246 |

Selector 对比：

| selector | recall@5 | jaccard@5 | top1 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| direct_ce_text_only_top5 | 0.3262 | 0.2180 | 0.1107 | 0.2975 |
| direct_ce_light_source_diverse_top5 | 0.3229 | 0.2157 | 0.1107 | 0.2914 |
| original_pool_order_top5 | 0.3435 | 0.2294 | 0.1028 | 0.2872 |
| qd_union_source_score_top5 | 0.3408 | 0.2271 | 0.1028 | 0.2879 |

v0.4a.1 结论：

```text
off-the-shelf Qwen3 text-only scorer 有一点 directness 排序能力，
top1/NDCG 不差，但 oracle overlap 和 candidate AUROC 不足。
不能替代 v0.3.1。
```

## v0.4d：Light Fusion Diagnostic

### 方法要点

v0.4d 不重新跑 DeepSeek / Qwen3，只融合已有：

```text
v0.3.1 oracle_likelihood_score
v0.4a.1 direct_ce_score
```

目标不是替代 v0.3.1，而是确认 direct CE 是否提供可测增量。

### 关键指标

| selector | recall@5 | jaccard@5 | top1 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| oracle_likelihood_top5 | 0.3642 | 0.2445 | 0.1028 | 0.2996 |
| direct_ce_text_only_top5 | 0.3262 | 0.2180 | 0.1107 | 0.2975 |
| fusion_refit_all_features_plus_direct_ce_top5 | 0.3694 | 0.2485 | 0.1036 | 0.2977 |

Candidate-level：

| score | AUROC | AUPRC |
| --- | ---: | ---: |
| oracle_likelihood_score | 0.6339 | 0.3716 |
| direct_ce_score | 0.5162 | 0.2957 |
| fusion_refit_score | 0.6357 | 0.3741 |

v0.4d 结论：

```text
direct CE 对 learned selector 有小增量：
jaccard +0.0040，recall +0.0052。
但没有达到预设 +0.005 jaccard go 条件，且 direct CE feature weight 较小。
因此 direct CE 可保留为 diagnostic / weak feature，不应成为主线。
```

## v0.5a：Claim-Atom Evidence Map Selector

### 方法要点

v0.5a 从“追 oracle overlap”转向“构造可解释 evidence map”：

```text
claim -> claim_atoms
candidate evidence -> covered_atom_ids / relation / directness / role / key_spans / duplicate_group
coverage-aware greedy selector -> top5
```

Teacher prompt 只包含 claim 和候选 evidence text；prompt 内候选用 `E01/E02/...` 标识，不包含 gold/oracle/rank/source/model scores。

### 关键指标

| selector | recall@5 | jaccard@5 | top1 | NDCG@5 | weighted atom cov | direct/partial | background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fusion_refit_all_features_plus_direct_ce_top5 | 0.3694 | 0.2485 | 0.1036 | 0.2977 | 0.5528 | 0.3283 | 0.6740 |
| v0_5a_base_only_top5 | 0.3557 | 0.2377 | 0.1130 | 0.3053 | 0.6655 | 0.4444 | 0.5587 |
| v0_5a_coverage_only_top5 | 0.3349 | 0.2235 | 0.1146 | 0.3026 | 0.7237 | 0.6500 | 0.3523 |
| v0_5a_evidence_map_top5 | 0.3438 | 0.2297 | 0.1217 | 0.3109 | 0.7205 | 0.6200 | 0.3825 |

相对 v0.4d fusion refit：

| metric | delta |
| --- | ---: |
| weighted atom coverage | +0.1677 |
| direct/partial selected rate | +29.17pp |
| background-only selected rate | -29.15pp |
| top1_match | +0.0181 |
| NDCG@5 | +0.0132 |
| recall@5 | -0.0256 |
| jaccard@5 | -0.0188 |

v0.5a 结论：

```text
作为可解释 selector，v0.5a 是有效的：
减少 background、提高 directness、提高 atom coverage。

作为 oracle-overlap selector，v0.5a 还不能替代 v0.4d fusion：
它选出的 evidence 更像人可读 rationale，但不完全等同于 oracle verifier utility set。
```

## v0.5b：Map-Aware Verifier 分类实验

### 方法要点

v0.5b 不重新训练 selector，而是把 v0.5a 的 selected evidence + map 信息渲染成 verifier 输入，比较不同 selector / checkpoint 在最终分类上的表现。

评估 selector：

- `v0_5a_base_only_top5`
- `v0_5a_evidence_map_top5`
- `fusion_refit_all_features_plus_direct_ce_top5`

评估 checkpoint：

- `checkpoint-450`
- `checkpoint-500`
- `checkpoint-550`
- `checkpoint-600`
- `best`

### 关键指标

| selector | checkpoint | acc | macro-F1 | true-side F1 | selection score |
| --- | --- | ---: | ---: | ---: | ---: |
| v0_5a_base_only_top5 | best / checkpoint-600 | 0.2943 | 0.2842 | 0.3295 | 0.4489 |
| v0_5a_evidence_map_top5 | checkpoint-500 | 0.2747 | 0.2638 | 0.3181 | 0.4229 |
| fusion_refit_all_features_plus_direct_ce_top5 | checkpoint-500 | 0.2755 | 0.2666 | 0.3300 | 0.4315 |

对照：同一 oracle-direct checkpoint 在 Oracle evidence 上的先验表现：

| checkpoint | original oracle acc | original oracle macro-F1 |
| --- | ---: | ---: |
| checkpoint-600 | 0.7125 | 0.7183 |
| checkpoint-550 | 0.7039 | 0.7092 |
| checkpoint-500 | 0.7039 | 0.7099 |

v0.5b 结论：

```text
在当前旧 verifier prompt / checkpoint 下，
map-aware evidence 不会自动提升最终分类指标。

最强 eval-only evidence selector 是 v0_5a_base_only_top5；
v0_5a_evidence_map_top5 更适合作为解释层和后续 verifier 训练数据结构，
而不是直接替换分类用 evidence selector。
```

## 阶段性总表

| version | 主问题 | 方法核心 | 最好 jaccard@5 / recall@5 | 分类指标 | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| v0 | stance bucket 超线性计数是否有效 | count-amplified stance buckets | 0.1960 / 0.2985 | - | No-Go，低于 original/QD |
| v0.1 | 压平 ambiguous 是否改善 | ambiguous penalty + selected-count penalty | 0.1979 / 0.3013 | - | stance 覆盖改善，桶内排序仍失败 |
| v0.2 | directness teacher 是否增强 | richer teacher schema + polar quota | 0.1986 / 0.3042 | - | directness 信号太弱，仍输 controls |
| v0.3.1 | oracle-likelihood 是否可学 | OOF logistic pointwise all-features | 0.2445 / 0.3642 | - | 首个稳定超过 original/QD 的 selector，但依赖 provenance/rank |
| v0.4a.1 | text-only direct CE 是否能替代 | Qwen3-Reranker-8B CrossEncoder | 0.2180 / 0.3262 | - | text-only 信号弱，不能替代 |
| v0.4d | direct CE 是否有增量 | v0.3.1 + direct CE fusion refit | 0.2485 / 0.3694 | - | 小增量，但未达强 go |
| v0.5a | 可解释 map selector 是否改善 evidence quality | atom coverage + directness + duplicate control | 0.2297 / 0.3438 | - | 解释性强，overlap 下降 |
| v0.5b | map-aware evidence 是否提升最终分类 | selected top5 + evidence map verifier prompt | - | acc 0.2943 / macro-F1 0.2842 | 旧 verifier 未吃到 map 收益；base-only 最强 |

## 当前判断

### 对 selector 主线

当前 oracle-overlap 方向的 selector-of-record 仍是 v0.4d fusion：

```text
fusion_refit_all_features_plus_direct_ce_top5
```

它在 oracle overlap 上最好：

```text
recall@5 = 0.3694
jaccard@5 = 0.2485
```

但这个提升仍然偏小，且主要信号来自 provenance/rank/source。

### 对可解释主线

当前最强解释结构是 v0.5a evidence map：

```text
v0_5a_evidence_map_top5
```

它的价值不在 oracle overlap，而在：

```text
weighted atom coverage = 0.7205
direct/partial selected rate = 0.6200
background-only selected rate = 0.3825
top1_match = 0.1217
NDCG@5 = 0.3109
```

这说明 map 能让 evidence 更像“可检查的事实覆盖结构”，但和旧 oracle set / 旧 verifier 的偏好存在错位。

### 下一步建议

下一阶段不建议继续单独调 stance bucket、direct CE lambda 或 v0.5a greedy 权重。更可行的方向是：

1. 保留 v0.4d fusion selector 作为 overlap-oriented baseline。
2. 保留 v0.5a evidence map 作为解释结构和 verifier 输入结构。
3. 在 train split 上构建同构 evidence map 数据。
4. 训练 map-aware verifier，而不是直接把旧 oracle-direct verifier 套到 map prompt 上。
5. 分类目标从只拟合 oracle evidence 转向同时利用：
   - selected evidence text
   - claim atoms
   - atom coverage
   - support/refute/qualify relation
   - direct spans
   - missing atoms
   - duplicate/background flags

这样才能避免之前 selector 路线的问题：只优化 oracle overlap，却没有让最终 classifier 学会使用可解释 evidence structure。
