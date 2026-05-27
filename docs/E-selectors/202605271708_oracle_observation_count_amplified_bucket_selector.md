# Oracle Set 观察驱动的超线性立场分桶 Evidence Selector 方案

日期：2026-05-27

## 目标

本文单独记录一个新的 selector 方向：先用原始 candidate pool 与设问机制得到的 QD 检索 evidence 构造更完整的候选池，再基于 Oracle set 观察，将 evidence 按语义完整性、claim-evidence 立场强度、以及与最终分类类别同构的立场桶进行组织，最后用超线性的桶级证据数量响应来做 top-k evidence selection。

核心假设：

```text
如果某个立场桶中出现了更多相对独立、语义完整、与 claim 相关且立场一致的 evidence，
那么 selector 从该桶选择 evidence 的概率应显著升高；
这种升高应当超过线性计数，而不是简单地多一条 evidence 就加一份权重。
```

## 1. Candidate Pool 选择：Original Pool + QD Evidence Union

### 候选池来源

该方案不直接在原始 Stage2 top15 上训练 selector，而是先构造一个更完整的候选池：

```text
analysis_candidate_pool
= dedup(original_stage2_pool ∪ qd_union_pool)
```

两个来源分别是：

1. `original_stage2_pool`：从 Stage2 oracle results 中每条样本的 `candidate_pool` 字段读取。这是当前 oracle search 与 selector 对齐的原始候选空间。
2. `qd_union_pool`：从 question decomposition retrieval 产出的 `union_candidate_pool_<split>.jsonl` 读取。这部分 evidence 来自多个问题 / 子断言检索，目标是补足原始检索未覆盖的 claim aspect。

默认 val 路径：

```text
outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl
outputs/selectors/question_decomp_retrieval/qwen_v0_val/union_candidate_pool_val.jsonl
```

### 为什么用并集

原始 pool 的优点是与已有 Stage2 oracle supervision 完全对齐；缺点是检索意图单一，可能没有覆盖 claim 的关键比较项、定义口径、时间范围或反例。

QD pool 的优点是由多个问题 route 召回，能扩大 evidence coverage；缺点是可能引入重复、噪声和不稳定的 qd-only evidence。

因此前期处理使用并集，而不是二选一：

```text
原始 pool 负责保留 oracle-aligned supervision anchor；
QD pool 负责补足更完整的相关证据覆盖；
并集候选池作为后续完整性打分、立场分桶和 selector 的统一输入。
```

### 去重与对齐

合并时必须保留来源，而不是简单覆盖：

```json
{
  "event_id": "...",
  "candidate_uid": "...",
  "text": "...",
  "source_pools": ["original_stage2_pool", "qd_union_pool"],
  "original_candidate_idx": 3,
  "qd_candidate_idx": 8,
  "source_index": 124,
  "question_ids": [0, 2],
  "retrieval_scores": {
    "original_hybrid_score": 0.82,
    "qd_source_score": 0.77
  },
  "oracle_selected": true
}
```

去重优先级：

1. `source_index` / `original_candidate_idx` 等可稳定对齐的 chunk id。
2. canonical text hash：lowercase、空白归一化、去掉首尾标点后的文本。
3. embedding 或 token overlap 近重复，v0 可先只记录，不强行删除。

Oracle 信息只能作为离线分析或评估标签使用：

```text
oracle_selected、gold_label、oracle rank 不能作为部署时 selector 输入。
```

## 2. Oracle Set 观察

人工观察 Stage2 Oracle set 后，目前有两个关键现象。

### 观察 A：Oracle evidence 偏向语义完整句

Oracle selected evidence 往往是可以独立承载事实判断的句子，而不是：

- 标题或导航文本。
- 孤立数字、孤立日期、列表残片。
- 被截断的从句。
- 只有背景名词但没有谓词的片段。
- 与 claim 有词面重合但不能形成完整语义判断的句子。

这说明 selector 应该给语义完整句更高优先级。完整性不是最终目标，但它是 verifier 能否有效使用 evidence 的前置条件。

### 观察 B：Oracle evidence 的立场强度与最终分类类别同构

Oracle set 不只是选择 topical relevance 高的句子，也倾向选择与 claim 呈现特定立场关系的 evidence。这个立场关系可以与最终分类类别做粗粒度同构：支持 claim 的证据对应 true-side 类别，反对 claim 的证据对应 false-side 类别，模糊 / 限定 / 部分成立的证据对应 mixed 类别。

v0 的 `n=3` 粗略对应关系：

| final label region | stance bucket | oracle evidence tendency |
| --- | --- | --- |
| `pants-fire` / `false` | `oppose_claim_bucket` | 更常出现明确反驳、纠错、contradiction 或 negative evidence |
| `barely-true` / `half-true` | `ambiguous_claim_bucket` | 更常出现限定、模糊、partial support、context dependent evidence |
| `mostly-true` / `true` | `support_claim_bucket` | 更常出现直接支持、确认、统计一致或 source-confirmed evidence |

因此，candidate-level selector 不应只判断“相关不相关”，还应预测：

```text
给定 claim + evidence，这条 evidence 对 claim 呈现哪一类立场桶。
```

更准确地说，这里的 bucket 是 claim-evidence 的立场桶，而不是任意主题桶或语义簇。v0 直接使用 DeepSeek v4 flash API 标注的 `stance_score` 做本地分桶；v1 才训练 stance classifier 蒸馏这一信号。整个过程中不能读取 gold label。

## 3. 完整五步实现方案

### Step 1：构造 union candidate pool

输入：

```text
oracle_results_<split>.jsonl
union_candidate_pool_<split>.jsonl
```

输出：

```text
union_analysis_candidate_pool_<split>.jsonl
```

每条 candidate 保留：

- `pool_membership`：来自 original、QD，或二者都有。
- `source_index`、`original_candidate_idx`、`qd_candidate_idx`。
- `question_ids`、`question_coverage_score`。
- original / QD retrieval scores。
- canonical text 与 dedup key。
- `oracle_selected`，仅用于离线分析和评估。

这一步的目标不是排序，而是构造更完整的候选空间。

### Step 2：语义完整性与相关性门控

为每条 candidate 计算：

```text
semantic_completeness_score
relevance_gate_score
text_fragment_flags
```

初始规则：

```text
semantic_completeness_score =
  0.25 * length_ok
+ 0.20 * has_predicate
+ 0.15 * sentence_boundary_ok
+ 0.15 * not_fragment
+ 0.15 * entity_or_keyword_overlap
+ 0.10 * lexical_or_embedding_relevance
```

其中 `relevance_gate_score` 用于防止长背景句被完整性分数误伤性提权。v0 可由 hybrid score、claim lexical overlap 和 embedding similarity 组合得到。

### Step 3：DeepSeek API 标注与本地 stance bucket 后处理

立场桶数量不是固定设计。后续实验应把 `n_stance_buckets` 作为可调超参数，比较不同桶数对 selection overlap、oracle order 和 verifier utility 的影响。

v0 初始设置：

```text
n_stance_buckets = 3
```

对应三个粗粒度立场桶：

| stance bucket | mapped labels | meaning |
| --- | --- | --- |
| `oppose_claim_bucket` | `pants-fire`, `false` | evidence 对 claim 呈现反对 / 反驳 / contradiction 立场 |
| `ambiguous_claim_bucket` | `barely-true`, `half-true` | evidence 与 claim 的立场关系模糊、限定、部分成立或依赖上下文 |
| `support_claim_bucket` | `mostly-true`, `true` | evidence 对 claim 呈现支持 / entailment / confirmation 立场 |

后续桶数实验可以使用有序立场刻度，而不是固定这三个名称。例如：

```text
n=5:
  strong_oppose / weak_oppose / ambiguous / weak_support / strong_support

n=7:
  strong_oppose / moderate_oppose / weak_oppose /
  ambiguous /
  weak_support / moderate_support / strong_support
```

无论 `n` 如何变化，bucket 都应保持从 oppose 到 support 的有序关系，并在 prompt 中明确每个 bucket 的定义。

每条 candidate 的 teacher 标注与本地后处理输出：

```json
{
  "n_stance_buckets": 3,
  "teacher_annotation": {
    "stance_score": 2.0,
    "semantic_completeness": 7.6
  },
  "teacher_stance_probs": {
    "oppose_claim_bucket": 0.72,
    "ambiguous_claim_bucket": 0.21,
    "support_claim_bucket": 0.07
  },
  "stance_bucket_derived": "oppose_claim_bucket",
  "stance_strength": 0.51,
  "stance_entropy": 0.74,
  "semantic_completeness_score": 0.76
}
```

监督与对照来源：

1. DeepSeek v4 flash teacher 数值标注：用 `deepseek-v4-flash` 对 `claim + evidence` 标注 `stance_score` 和 `semantic_completeness` 两个 0-10 数值。`stance_score` 经本地后处理映射为 `n_stance_buckets` 维 `teacher_stance_probs`，v0 selector 直接使用该分布。
2. API token 证据记录：标注请求必须开启并保存 `logprobs` / `top_logprobs`，用于检查 teacher 输出、定位异常样本和后续审计。
3. Verifier proxy 只作对照：若 oracle search 或 verifier proxy 保存了单条 evidence 的 `label_logprobs`，只作为 sanity check / agreement analysis，不进入 v0 主监督。
4. Oracle selected set 只用于分析：如果只有 selected set，可把 oracle selected candidate 作为正例分析，不直接构造硬分类标签。
5. QD-only evidence 先通过 canonical text / `source_index` 对齐原始 pool；无法对齐的样本在 v0 中直接调用 DeepSeek v4 flash 标注。v1 才考虑用训练后的 classifier 补打标签。

#### DeepSeek v4 flash 标注 schema

DeepSeek v4 flash API 标注时固定使用：

```json
{
  "model": "deepseek-v4-flash",
  "response_format": {"type": "json_object"},
  "logprobs": true,
  "top_logprobs": 20,
  "temperature": 0
}
```

如果服务端或 SDK 对 `top_logprobs=20` 有兼容问题，可降到 `top_logprobs=5`，但必须在 manifest 中记录实际参数。

#### Prompt 设计

标注 prompt 只提供 `claim`、单条 `candidate evidence`、可选 source metadata 和 bucket schema。不能提供 `gold_label`、`oracle_selected`、oracle order 或 verifier prediction。

System prompt：

```text
You are a stance annotation model for fact-checking evidence selection.
Given a political claim and one candidate evidence sentence, estimate how the evidence relates to the claim.
Return JSON only.
Do not use external knowledge. Judge only the relation between the claim and the evidence text.
Return exactly two numeric fields:
1. stance_score: integer or float from 1 to 10, where 1 means the evidence strongly opposes the claim, 5 to 6 means ambiguous/partial/context-dependent, and 10 means the evidence strongly supports the claim.
2. semantic_completeness: integer or float from 0 to 10, where 0 means fragmentary or unusable as evidence, and 10 means a complete, self-contained sentence.
Do not include the gold veracity label or any hidden oracle information in the answer.
```

User prompt template：

```text
Annotate the claim-evidence stance score and semantic completeness.

claim:
{claim}

candidate_evidence:
{evidence_text}

Return only this JSON object:
{
  "stance_score": <number from 1 to 10>,
  "semantic_completeness": <number from 0 to 10>
}
```

#### 请求 payload

每条 candidate 一个请求，payload 示例：

```json
{
  "model": "deepseek-v4-flash",
  "messages": [
    {
      "role": "system",
      "content": "You are a stance annotation model for fact-checking evidence selection. Return JSON only..."
    },
    {
      "role": "user",
      "content": "Annotate the claim-evidence stance score and semantic completeness.\n\nclaim:\n...\n\ncandidate_evidence:\n...\n\nReturn only this JSON object:\n{\"stance_score\": <number from 1 to 10>, \"semantic_completeness\": <number from 0 to 10>}"
    }
  ],
  "response_format": {"type": "json_object"},
  "logprobs": true,
  "top_logprobs": 20,
  "temperature": 0,
  "max_tokens": 64,
  "user_id": "stance_bucket_v0"
}
```

解析时必须做 schema validation：

1. JSON 必须只包含 `stance_score` 与 `semantic_completeness` 两个字段。
2. `stance_score` 必须是有限数值，并 clamp 到 `[1, 10]`；若发生 clamp，记录 `stance_score_clamped=true`。
3. `semantic_completeness` 必须是有限数值，并 clamp 到 `[0, 10]`；若发生 clamp，记录 `semantic_completeness_clamped=true`。
4. 本地后处理负责把 `stance_score` 映射为 `n_stance_buckets` 维 `teacher_stance_probs`，把 `semantic_completeness` 归一化为 `semantic_completeness_score=semantic_completeness/10`。

#### Resume 机制

标注脚本必须支持可中断续跑。推荐以 candidate-level deterministic key 作为幂等键：

```text
annotation_key = sha1(event_id + candidate_uid + prompt_version + model)
```

因为 teacher 只输出连续的 `stance_score`，所以不同 `n_stance_buckets` 实验可以复用同一份 teacher 标注；桶数只影响本地后处理。

输出采用 append-only JSONL：

```text
deepseek_teacher_annotations_<split>.jsonl
deepseek_teacher_raw_responses_<split>.jsonl
deepseek_teacher_errors_<split>.jsonl
deepseek_teacher_progress.json
```

Resume 流程：

```text
1. 启动时读取已有 annotations / raw_responses / errors。
2. 建立 completed_keys、failed_retryable_keys、failed_terminal_keys。
3. 对 completed_keys 跳过请求。
4. 对 retryable errors 按 retry_count 和 backoff 策略重试。
5. 每完成一个请求立即 flush 一行 JSONL，避免进程中断丢失批次结果。
6. 每 N 条样本更新 progress summary。
```

错误分类：

| error type | action |
| --- | --- |
| HTTP 429 / rate limit | exponential backoff retry |
| 5xx / insufficient_system_resource | retry |
| timeout / connection reset | retry |
| finish_reason=`length` | retry with larger `max_tokens` once |
| invalid JSON | retry once with stricter repair prompt, then write errors |
| schema validation fail | retry once, then write errors |
| content_filter | terminal error, skip from training |

#### 并发请求与限流

标注脚本应使用 async worker pool 加速请求，但必须受限流和 checkpoint 约束。

建议 CLI 参数：

```text
--concurrency 8
--requests-per-minute 120
--tokens-per-minute 200000
--max-retries 5
--retry-base-sleep 2.0
--retry-max-sleep 60.0
--resume true
--flush-every 1
```

并发策略：

```text
producer: 读取未完成 candidates，按 annotation_key 去重
workers: 受 semaphore 控制并发请求
rate limiter: 同时限制 RPM 和估算 TPM
writer: 单独串行写 JSONL，保证 append-only 文件不交错
progress: 周期性统计 completed / failed / retrying / token usage
```

TPM 估算先用 prompt 字符数粗估；收到 response 后用 API 返回的真实 `usage` 修正。若连续触发 429，自动降低并发，例如：

```text
concurrency = max(1, floor(concurrency * 0.7))
```

为了减少成本，标注前应先按 canonical text 去重。同一个 `claim + canonical evidence + prompt_version` 的结果可以复用到 original pool 与 QD union pool 中的重复 candidate。

这里需要区分两种 probability：

1. token logprobs：API 返回的生成 token 概率，适合检查 JSON 数值 token 和定位异常输出。
2. local stance probabilities：本地将 teacher 返回的 `stance_score` 映射为 `n_stance_buckets` 维软标签；DeepSeek 本身不再直接输出 `stance_probs`。

teacher 标注的核心输出只有两个字段：

```json
{
  "stance_score": 2,
  "semantic_completeness": 7.6
}
```

`stance_score` 的定义：

```text
1 = strongly oppose claim
2-4 = weak to moderate oppose
5-6 = ambiguous / qualified / partial / context-dependent
7-9 = weak to moderate support
10 = strongly support claim
```

`semantic_completeness` 的定义：

```text
0 = fragmentary, noisy, or unusable as standalone evidence
5 = partially interpretable but incomplete or context-dependent
10 = complete, self-contained, and usable as evidence
```

本地后处理将 `stance_score` 转成 `teacher_stance_probs`。一种简单可实现方式是把 n 个桶均匀放在 `[1, 10]` 上，并用温度控制的 RBF/softmax 生成软标签：

```text
bucket_center_j = 1 + 9 * (j - 1) / (n_stance_buckets - 1)
teacher_stance_probs_j =
  softmax_j(-((stance_score - bucket_center_j)^2) / tau)
```

其中 `tau` 是可调温度，v0 可设为 `tau=2.0`。当 `n=3` 时，三个中心分别接近 oppose / ambiguous / support；当 `n=5` 或 `n=7` 时，同一个连续分数自然映射到更细的有序立场桶。

语义完整性门控改为：

```text
semantic_completeness_score = semantic_completeness / 10
```

注意：模糊立场桶不能吸收所有无关 evidence。语义不完整的 evidence 应通过 `semantic_completeness_score` 被门控出去，而不是作为 ambiguous 的高质量样本。

#### V0：直接使用 API 立场分数

v0 不训练 stance classifier。selector 直接使用从 DeepSeek v4 flash `stance_score` 后处理得到的 `teacher_stance_probs`：

```text
membership_prob_i,B = teacher_stance_probs_i,B
stance_bucket = argmax(teacher_stance_probs_i)
stance_strength = max_prob - second_prob 或 1 - normalized_entropy
```

理由：

- v0 的目标是先验证 Oracle set 观察和超线性 stance bucket prior 是否有效，不引入 student classifier 的额外误差。
- API 的连续 `stance_score` 可以直接映射到 `n=3/5/7`，支持桶数 sweep。
- `teacher_stance_probs` 不是人工 gold truth，只是 v0 selector 的外部 teacher signal。

这也是使用软标签的主要意义：当后续把 `n_stance_buckets` 从 3 扩展到 5、7 或其他粒度时，不需要把 evidence 强行压到固定三类；一条 evidence 可以按 `membership_prob_i,B` 同时给多个立场桶贡献有效证据数。

#### V1：训练 stance classifier

如果 v0 证明有效，再训练 student stance classifier 蒸馏 API 标注，降低后续大规模推理成本。

模型形式：

```text
Input:  [claim] [SEP] [candidate evidence]
Output: n-way stance bucket logits
```

训练目标：

```text
loss =
  KL(teacher_stance_probs || student_stance_probs)
```

可选 ablation 才加入硬标签 CE、1-10 score regression 或 verifier-derived auxiliary target；它们不属于 v1 主监督。

#### Token 用量与成本审计

落地标注时必须统计实际 token 用量，用于后续成本计算、重跑预算估计和审计。每个 API response 至少记录：

```json
{
  "model": "deepseek-v4-flash",
  "request_id": "...",
  "system_fingerprint": "...",
  "prompt_tokens": 512,
  "completion_tokens": 96,
  "total_tokens": 608,
  "prompt_cache_hit_tokens": 128,
  "prompt_cache_miss_tokens": 384,
  "finish_reason": "stop",
  "logprobs_saved": true,
  "top_logprobs": 20
}
```

聚合产物中需要额外输出：

```text
teacher_api_usage_summary.json
teacher_api_cost_estimate.json
teacher_api_audit_manifest.json
```

其中 `teacher_api_cost_estimate.json` 只按实际记录的 token usage 计算，不用样本数粗略估算。

### Step 4：计算单层 stance bucket 的有效 evidence 数

v0 先不做多层分桶，只在第一层 stance bucket 上计算有效证据数：

```text
stance buckets:
  B_1 ... B_n, ordered from strongest oppose to strongest support
```

strength bucket、semantic faction bucket、source group bucket 暂不进入 v0 的 slot allocation。它们可以作为后续 v1 扩展或分析字段保留。

每个桶使用 `effective_count`，而不是 raw count：

```text
effective_count_B =
  sum_i [
    membership_prob_i,B
    * quality_gate_i
    * source_dedup_weight_i
    * question_route_weight_i
  ]
```

其中：

```text
quality_gate_i =
  I(semantic_completeness_score_i >= tau_c)
  * I(relevance_gate_score_i >= tau_r)

source_dedup_weight_i = 1 / sqrt(n_same_source_in_bucket)
question_route_weight_i = min(1.0, 0.5 + 0.25 * n_question_routes_i)
```

这一步的目标是估计：

```text
某个立场桶中，有多少相对独立、完整、相关且非重复的 evidence 正在形成群体信号。
```

### Step 5：超线性单层 stance bucket 选择 top-k

只对 stance bucket 计算超线性 mass：

```text
bucket_mass_B = (effective_count_B + alpha) ^ gamma
```

推荐初值：

```text
alpha = 0.5
gamma_stance = 1.6 或 1.8
rho = 0.6
```

其中 `gamma > 1` 是关键。它使证据数从 4 增加到 8 时，桶权重不是 2 倍，而是：

```text
8^1.7 / 4^1.7 ≈ 3.25
```

top-k 选择过程：

```text
for t in 1..top_k:
  choose stance bucket S by slot_score_S
  choose best remaining candidate i inside S
  update selected counts and diversity state
```

每层 slot score 都带有已选择惩罚，防止单桶完全坍塌：

```text
slot_score_B(t) =
  bucket_mass_B / (1 + selected_count_B) ^ rho
```

候选级最终排序分数：

```text
candidate_score_i =
  0.30 * retrieval_score_i
+ 0.30 * semantic_completeness_score_i
+ 0.20 * membership_prob_i,S
+ 0.10 * qd_question_coverage_i
+ 0.10 * source_diversity_bonus_i
```

桶间主要由 `effective_count` 的超线性 prior 决定；桶内才用 `candidate_score` 选择代表 evidence。

## 4. 超线性单层选择的直观含义

这个 selector 不是“把所有 evidence 全局打分后取前 5”，而是：

```text
先看哪个立场桶聚集了最多有效证据；
再在该立场桶内部选择语义完整、相关、且不过度同源重复的代表句。
```

它与当前 roundtable-QD 的区别：

| method | primary signal | risk |
| --- | --- | --- |
| roundtable-QD | faction/source/aspect diversity | 可能为了多样性选到立场关系不合适的 evidence |
| stance-bucket selector | single candidate quality | 容易忽略某个立场中的群体证据数量 |
| count-amplified stance-bucket selector | stance bucket 的有效 evidence 数 + 超线性 prior | 需要防止重复 evidence 抬高 raw count |

因此 v0 必须同时使用：

- semantic completeness gate
- source dedup weight
- QD question route coverage
- nonlinear bucket mass
- selected-count diminishing return

多层分桶暂不进入 v0。后续如果单层 stance bucket 有效，再考虑加入 strength bucket、semantic faction 或 source group 作为第二层 slot allocation。

## 5. 建议实现文件

新增模块：

```text
src/fact_checking/selectors/evidence_quality.py
src/fact_checking/selectors/stance_buckets.py
src/fact_checking/selectors/count_amplified_stance_bucket_selector.py
```

新增脚本：

```text
scripts/phase5_selectors/build/build_union_analysis_candidate_pool.py
scripts/phase5_selectors/build/annotate_stance_buckets_deepseek.py
scripts/phase5_selectors/build/postprocess_stance_scores_to_buckets.py
scripts/phase5_selectors/eval/eval_count_amplified_stance_bucket_selector.py
scripts/phase5_selectors/run/run_count_amplified_stance_bucket_selector_v0.sh
```

v0 不训练神经模型，只做 API 标注驱动的 analysis/selector artifact：

```text
original candidate_pool + qd_union_pool
-> union + dedup
-> semantic completeness heuristic
-> DeepSeek v4 flash stance_score + semantic_completeness annotation
-> save logprobs / top_logprobs and token usage
-> postprocess stance_score into teacher_stance_probs for n=3/5/7
-> effective_count and bucket_mass
-> deterministic top5 selection
-> compare with original order / QD order / roundtable-QD
```

## 6. 产物

```text
outputs/selectors/count_amplified_stance_bucket_selector/v0_<split>/
  union_analysis_candidate_pool_<split>.jsonl
  candidate_quality_labels_<split>.jsonl
  candidate_stance_buckets_<split>.jsonl
  stance_bucket_count_metrics.json
  stance_bucket_selection_trace_<split>.jsonl
  deepseek_teacher_annotations_<split>.jsonl
  deepseek_teacher_raw_responses_<split>.jsonl
  deepseek_teacher_errors_<split>.jsonl
  deepseek_teacher_progress.json
  teacher_api_usage_summary.json
  teacher_api_cost_estimate.json
  teacher_api_audit_manifest.json
  selector_metrics.json
  oracle_observation_metrics.json
  analysis.md
  manifest.json
```

`stance_bucket_selection_trace_<split>.jsonl` 应记录每个 slot 的决策：

```json
{
  "event_id": "...",
  "slot": 1,
  "chosen_stance_bucket": "oppose_claim_bucket",
  "bucket_mass": 6.31,
  "slot_score": 6.31,
  "selected_candidate_uid": "...",
  "selected_text": "...",
  "oracle_selected": true
}
```

## 7. 评估指标

Oracle observation validation：

- `mean_completeness_oracle_selected`
- `mean_completeness_non_selected`
- `completeness_selected_lift`
- `completeness_selected_auroc`
- `oracle_selected_stance_label_alignment`
- `pool_stance_label_alignment`
- `oracle_vs_pool_stance_alignment_lift`

Selector metrics：

- `recall@5`
- `precision@5`
- `jaccard@5`
- `top1_match`
- `oracle_rank_ndcg@5`
- `mean_semantic_completeness@5`
- `source_entropy@5`
- `stance_bucket_entropy@5`
- `oracle_stance_bucket_recall@5`
- `n_stance_buckets`
- `stance_soft_entropy_mean`
- `stance_expected_score_mean`

Ablations：

```text
original_pool_order_top5
qd_union_pool_order_top5
roundtable_qd_union_top5
completeness_only_top5
linear_stance_bucket_count_top5
count_amplified_stance_bucket_top5
count_amplified_stance_bucket_roundtable_top5
n_stance_buckets_sweep: 3 / 5 / 7
```

关键对比：

```text
linear_stance_bucket_count_top5
vs count_amplified_stance_bucket_top5
```

如果超线性 prior 成立，应看到：

- `jaccard@5`、`top1_match` 或 `oracle_rank_ndcg@5` 至少一个提升。
- `mean_semantic_completeness@5` 不下降。
- `source_entropy@5` 不明显坍塌。
- 大桶被更多选择，但不是所有样本都退化成单桶 top5。
- `n_stance_buckets` 从 3 增加到 5/7 时，若细粒度 soft buckets 提升 order metrics 或降低 ambiguous 桶坍塌，则保留更细配置；否则回退到 n=3。

## 8. Go / Stop 条件

Go：

```text
1. union candidate pool 相比 original pool 提供更高 oracle coverage 或 question-aspect coverage；
2. oracle selected evidence 的 completeness 明显高于非 selected evidence；
3. oracle selected evidence 的 stance soft distribution 比全 pool 更贴近由 gold_label 映射出的同构 stance region；
4. count_amplified_stance_bucket_top5 优于 qd_union_pool_order_top5 或 roundtable_qd_union_top5；
5. 超线性版本优于 linear stance bucket count control。
```

Analysis-only：

```text
completeness signal 成立，但 DeepSeek stance_score 标注或本地桶映射噪声较大；
或 stance bucket count 能解释 oracle set，但 deterministic top5 暂时未超过 control。
```

Stop：

```text
semantic completeness 与 oracle_selected 无关；
stance soft distribution 与由 final label 映射出的同构 stance region 无明显对应关系；
effective_count 去重后信号消失；
超线性 prior 导致单桶坍塌且 selection metrics 同时下降。
```

## 9. 风险与边界

- Oracle set 是 verifier-utility supervision，不是人工 rationale；该方案是在学习当前 oracle/verifier 的选择偏好。
- `gold_label` 只能用于离线分析或评估，不能作为 v0 selector 输入。
- stance bucket 不能被解释为 evidence 自身真假，只表示 `claim + evidence` 中 evidence 对 claim 的支持、模糊或反对关系。
- raw bucket size 不能直接作为强信号，必须经过完整性门控、相关性门控和去重权重。
- ambiguous claim bucket 不是低质量桶；它对 `barely-true` / `half-true` claim 可能正是关键 evidence 类型。
