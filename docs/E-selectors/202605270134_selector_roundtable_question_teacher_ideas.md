# Selector New Direction Notes: Question Decomposition, Roundtable Evidence, and Teacher Rationale

日期：2026-05-26

## 背景

`utility_listwise` v0 / v0.1 表明：在固定 Stage2 candidate pool 上，让冻结 DeBERTa + set head 直接从 claim-candidate text 学 `delta_margin`，很难学到可用 scorer。v0.1 消除了 rank/position shortcut 后，row-level utility correlation 仍接近随机。

因此，后续方向应从“在已有 15 个候选里直接学 scorer”转向“改变候选池生成、证据结构表示、以及 teacher signal 的形态”。

本文记录三个新想法：

1. 圆桌会议机制：把 evidence chunk 或 report 组织成多个观点派别。
2. 设问机制：从 claim 出发生成多个问题 / 子断言，并分别检索证据。
3. 可解释 teacher：调用强能力模型输出结构化选择理由，蒸馏给小模型。

## 总体判断

| idea | 可行性 | 建议优先级 | 主要作用 |
| --- | --- | --- | --- |
| 设问机制 | 高 | P0 | 改善候选池覆盖，降低 selector 直接猜 utility 的难度 |
| 圆桌会议机制 | 中高 | P1 | 给 evidence 增加结构：观点簇、stance、source diversity、派别强度 |
| 可解释 teacher | 中 | P2 | 生成结构化中间标签和短理由，辅助小模型学习选择策略 |

推荐融合路线：

```text
claim
-> question / subclaim decomposition
-> per-question retrieval
-> evidence roundtable clustering and stance profiling
-> structured teacher labels
-> selector / classifier with evidence-structure features
```

如果只能先做一个方向，优先做设问机制。它最接近现有 pipeline，验证成本低，且直接影响 candidate pool 质量。

## Idea 1: 圆桌会议机制

### 核心概念

把候选池或更大的可检索 evidence space 组织成多个“派别”。每个派别不是简单的支持 / 反对二分，而是由语义相近、来源相近、立场相近、或关注同一 claim aspect 的 evidence 组成。

一个派别可以表示为：

```text
faction = {
  cluster_id,
  representative_chunks,
  stance_to_claim,
  covered_questions,
  source_domains,
  report_ids,
  evidence_strength,
  conflict_with_other_factions
}
```

这比 L-defense 的正反两方更细：同一个 claim 可能有时间口径派、统计定义派、人物引用派、反例派、背景事实派等。

### 可用信息

可用于分派别的信号：

- sentence / chunk embedding similarity
- report-level embedding similarity
- report metadata: domain, report_id, source_index, publication/source family
- lexical overlap with claim or subquestions
- stance / entailment / contradiction / neutral labels
- numeric/time/entity overlap
- verifier-derived saved scores, if available

### 输出标签

建议先输出结构化标签，而不是自然语言长解释：

| field | meaning |
| --- | --- |
| `faction_id` | 派别编号 |
| `faction_label` | 简短派别名，如 `definition_scope`, `counterexample`, `same_source_support` |
| `stance_to_claim` | support / refute / qualify / background / unclear |
| `strength_score` | 派别证据强度 |
| `coverage_score` | 覆盖 claim aspects 或 generated questions 的程度 |
| `source_diversity` | report/domain 多样性 |
| `representative_candidate_indices` | 代表证据 |
| `redundant_candidate_indices` | 同质重复证据 |
| `conflicting_faction_ids` | 相互冲突的派别 |

### 用法

圆桌机制可以有三种使用方式：

1. 作为 selector 特征：candidate 加入 `faction_id`、`faction_strength`、`stance`、`source_diversity`。
2. 作为 rerank 约束：top-k 不只选单点高分证据，还要覆盖主要派别和关键问题。
3. 作为 classifier 输入：把 evidence 按派别组织后喂给 verifier，让 verifier 看到观点结构。

### 风险

- 聚类质量可能不稳定，尤其是短 sentence chunk。
- stance label 若由弱模型产生，可能引入噪声。
- 派别强度不能简单等同于证据数量，否则会奖励重复报道。

### 低成本 v0

先不训练新模型，只做 analysis artifact：

```text
candidate_pool / expanded_pool
-> sentence embedding clustering
-> optional report/domain grouping
-> stance classifier or LLM batch labeling
-> faction summary JSONL
-> evaluate faction coverage vs oracle selected evidence
```

验收指标：

- oracle selected evidence 是否覆盖多个 faction
- retrieval control 是否过度集中在单一 faction
- faction-aware top-k 是否提升 recall/jaccard 或 verifier margin

## Idea 2: 设问机制

### 核心概念

从原始 claim 生成多个问题或子断言，每个问题对应一个检索意图。对每个问题单独检索 evidence，再合并去重，形成更有覆盖度的候选池。

例子：

```text
claim:
  "We have less Americans working now than in the 70s."

questions:
  1. What is the employment level now?
  2. What was the employment level in the 1970s?
  3. Is the comparison about absolute workers, employment rate, or labor force participation?
  4. Which data source defines the relevant employment statistic?
```

### 为什么优先

当前 selector 失败不一定说明“选择模型完全不可能”，也可能是 candidate pool 没有显式覆盖 claim 的关键问题。设问机制先改善候选池和证据组织，避免小模型在信息不足的 pool 里硬学 utility。

它还更容易诊断：

- question generation 错了
- retrieval per question 错了
- merge/rerank 错了
- verifier 不受益

### 初始实现形态

设问 v0 可以不训练：

```text
claim
-> generate 3-5 questions
-> retrieve top-m chunks per question from associated reports
-> deduplicate by chunk uid / source_index / text hash
-> add question_coverage features
-> rerank by hybrid_score + coverage + diversity
-> evaluate oracle coverage and verifier metrics
```

候选特征：

| feature | meaning |
| --- | --- |
| `covered_question_ids` | 该证据被哪些问题召回 |
| `question_retrieval_rank_min` | 多问题检索中最优 rank |
| `question_retrieval_count` | 被多少问题召回 |
| `question_type` | numeric / temporal / definition / entity / causal 等 |
| `subclaim_overlap` | 与子断言的语义或词面重合 |

### 可选 question generator

低成本顺序：

1. prompt strong LLM 离线生成问题。
2. 使用规则模板补充数值、时间、比较类问题。
3. 以后再训练小模型生成 questions。

### 验收指标

第一阶段不看 selector 训练，只看 retrieval / candidate pool 是否改善：

- oracle evidence coverage@pool
- oracle selected recall@expanded_pool
- question coverage entropy
- source diversity
- final verifier accuracy / macro-F1 / gold margin
- ablation: original pool vs question-expanded pool

## Idea 3: 可解释 Teacher / Reasoning Distillation

### 核心概念

调用强能力闭源模型，为 evidence selection 输出结构化解释和选择标签，再蒸馏给小模型。这里 teacher 不应只输出长 chain-of-thought，而应输出可评估、可训练的结构化中间变量。

### 推荐 teacher 输出

```json
{
  "questions": ["..."],
  "candidate_roles": [
    {
      "candidate_idx": 0,
      "role": "definition_scope",
      "stance": "qualify",
      "covered_questions": [1, 3],
      "utility_label": "medium",
      "short_reason": "Clarifies the statistic's definition."
    }
  ],
  "factions": [
    {
      "faction_id": "F1",
      "label": "labor_force_participation_definition",
      "stance": "qualify",
      "representatives": [0, 4],
      "strength": "high"
    }
  ],
  "selected_indices": [0, 4, 8, 2, 11],
  "missing_evidence": ["1970s baseline statistic"],
  "selection_summary": "..."
}
```

### 蒸馏对象

小模型不一定要学习长解释文本。更建议学习：

- question generation
- candidate role classification
- stance / relation classification
- utility bucket classification
- selected mask / ordered pair preference
- short rationale generation as auxiliary objective

### 风险

- 闭源模型解释可能合理化错误选择。
- 长文本 teacher 难以与 `jaccard@5` / verifier margin 对齐。
- API 成本和模型版本漂移会影响复现。
- 若 teacher 没有结构化 schema，后续训练和评估会很难。

### 低成本 v0

先做小规模标注，不训练：

```text
sample 100-300 val/train claims
-> provide claim + candidate pool + optional question-expanded pool
-> LLM outputs structured JSON
-> validate JSON schema
-> compare teacher selected_indices with oracle / saved-score ranker / verifier metric
```

只有当 teacher 本身在 selection gate 上优于 retrieval control，才进入 distillation。

## 推荐融合方案

### Phase A: Question-Decomposition Retrieval

目标：改善候选池覆盖。

产物：

- `questions.jsonl`
- `question_retrieval_trace.jsonl`
- `expanded_candidate_pool.jsonl`
- `coverage_metrics.json`

核心比较：

```text
original Stage2 pool
vs question-expanded pool
vs question-expanded + diversity merge
```

通过条件：

- expanded pool oracle coverage 明显高于原 pool
- verifier 在 oracle-like rerank 或 simple rerank 下有提升
- 不显著扩大噪声导致 verifier 退化

### Phase B: Roundtable Evidence Map

目标：将 expanded pool 组织成多派别证据地图。

产物：

- `faction_map.jsonl`
- `candidate_role_labels.jsonl`
- `faction_metrics.json`

核心比较：

```text
hybrid top-k
vs coverage-aware top-k
vs faction-aware top-k
```

通过条件：

- faction-aware top-k 提升 set overlap 或 verifier margin
- selected evidence source diversity / aspect coverage 上升
- order metrics 不明显退化

### Phase C: Structured Teacher Distillation

目标：用强模型产生结构化 supervision，再训练小模型。

产物：

- `teacher_labels.jsonl`
- `teacher_schema_errors.jsonl`
- `teacher_vs_oracle_metrics.json`
- distillation train/eval artifacts

训练目标：

```text
candidate role CE
stance CE
utility bucket CE / pairwise preference
selected mask BCE
short rationale auxiliary loss
```

通过条件：

- teacher labels 本身优于 retrieval control
- 小模型能复现 teacher 的 role/stance/utility labels
- selection gate 至少超过 v0.x 和 retrieval control，且 order metrics 不低于 control

## 推荐下一步

最推荐先做：

```text
Question-Decomposition Retrieval v0
```

原因：

1. 它直接改变候选池，绕开当前 frozen scorer 学不到 `delta_margin` 的瓶颈。
2. 它能为圆桌机制提供更丰富的 evidence space。
3. 它的失败原因容易拆解，工程成本低于训练新 selector。
4. 它可以自然产出 teacher 所需的 questions 和 candidate roles。

暂不建议：

- 直接训练 Qwen3/3.5 生成长解释后选择 evidence。
- 直接把圆桌机制做成复杂多智能体辩论。
- 在没有 teacher gate 验证前大规模调用闭源模型生成训练集。

最小实验定义：

```text
Input: claim + associated reports / existing retrievable chunks
Generate: 3-5 questions
Retrieve: top-m chunks per question
Merge: dedup + coverage/diversity rerank
Evaluate: oracle coverage, jaccard@5, oracle_rank_ndcg@5, verifier margin
```

如果该阶段不能改善 candidate coverage，再训练任何 selector 都很可能是治标不治本。
