# QEC-MIN / QEC-MAP 方法设计与实验矩阵

## 0. 目标

当前主方法已经确定为 `v0_7_budgeted_marginal_chain_adaptive5_10`。该 selector 能构造较好的 evidence chain，但默认 verifier 使用 `plain` prompt，只看到 claim 与 evidence text，不能显式看到“这条 evidence 在检查什么”。

本设计的目标是，在不先改变 selector 的前提下，增加两个轻量 prompt/view 变体：

```text
QEC-MIN = Question-guided Evidence Chain, minimal prompt
QEC-MAP = Question-guided Evidence Chain, minimal prompt + compact map tags
```

两者的共同原则是：

```text
不生成 Answer；Evidence 本身就是 Answer。
```

每一步 chain 只保留关键链信息：

```text
step order + verification cue + evidence text
```

QEC-MAP 在此基础上额外显示少量 evidence-map 标签，用于测试显式结构信息是否帮助 verifier。

---

## 1. 方法定位

### 1.1 当前 v0.7 的问题

v0.7 adaptive5_10 已经能选择一组较优 evidence，但其结构信息主要停留在 selector 内部。默认 verifier prompt 是：

```text
Claim:
<claim>

Evidence:
[1] <evidence text>
[2] <evidence text>
...
```

这会把 evidence chain 退化成一个 top-k evidence list。QEC-MIN / QEC-MAP 要解决的是：

```text
让 verifier 知道每条 evidence 是在检查 claim 的哪个问题或哪个 claim component。
```

但不要引入额外 answer generation，也不要把 prompt 写成复杂的 reasoning trace。

### 1.2 QEC 的核心定义

每个 selected evidence 被表示为一个 chain step：

```text
z_t = (check_t, evidence_t)
```

其中：

```text
check_t   = 用于说明 evidence_t 正在检查什么的 verification cue
evidence_t = 原始 evidence sentence / chunk
```

`check_t` 的来源优先级为：

```text
1. QD question route
2. covered claim atom
3. fallback claim-level check
```

因此，QEC 不是严格的 `Question → Answer → Evidence`，而是：

```text
Question or Atom → Evidence-as-Answer
```

更准确的论文表述可以是：

```text
Question-guided evidence chain with evidence-as-answer.
```

---

## 2. 共同输入

QEC-MIN 和 QEC-MAP 都复用现有 pipeline 的三个结构来源。

### 2.1 QD retrieval route metadata

当前 question decomposition retrieval 已经为 candidate 保存了 route 信息。每个 route 通常包含：

```text
question_id
question
focus
rank
dense_score
lexical_score
bm25_score
hybrid_score
```

union 阶段会把 QD candidate 的 route 信息合并到 candidate 上，常见字段包括：

```text
from_qd
qd_pool_rank
qd_rrf_score
qd_question_hit_count
qd_max_question_hybrid
qd_question_routes
```

这些字段用于选择 `check_t`。

### 2.2 Evidence-map atom alignment

Evidence-map 已经为每条 candidate 提供：

```text
covered_atom_ids
map_relation
map_directness
map_evidence_role
key_spans
duplicate_group
map_confidence
evidence_map_quality_score
```

这些字段用于 fallback cue、QEC-MAP 标签，以及后续 diagnostics。

### 2.3 v0.7 adaptive5_10 selected evidence

第一阶段不改变 selector，直接复用：

```text
selector_name = v0_7_budgeted_marginal_chain_adaptive5_10
min_top_k = 5
max_top_k = 10
selection_mode = trace
top_k = 10
```

也就是说，QEC-MIN / QEC-MAP 首先是 **prompt/view ablation**，不是新的 retrieval 或 selector。这样可以隔离检验：

```text
显式 evidence-chain cue 是否提升 verifier？
```

---

## 3. QEC-MIN

## 3.1 方法定义

QEC-MIN 是最小证据链提示格式。每条 evidence 前只加一个 verification cue。

最终 evidence text 被重写为：

```text
Check: <verification cue>
<original evidence text>
```

因为底层 prompt builder 已经会把候选 evidence 渲染成：

```text
Evidence:
[1] <candidate text>
[2] <candidate text>
```

所以每条 candidate text 不再额外写 `Evidence:`，避免冗余。

### 3.2 Prompt 示例

```text
Claim:
<claim>

Evidence:
[1] Check: Did the unemployment rate fall below 5% in 2016?
The unemployment rate was 4.9 percent in October 2016, according to ...

[2] Check: Did the statement refer to national unemployment?
The claim referred to the national unemployment rate, not state-level unemployment ...
```

### 3.3 Cue 选择规则

对每个 selected candidate `e_t`，选择一个 `check_t`。

优先使用 QD question：

```text
routes = e_t.qd_question_routes
```

从 routes 中选择 best route：

```text
sort key:
1. rank ascending
2. hybrid_score descending
3. focus priority
4. question_id stable order
```

建议的 focus priority：

```text
quantity / attribution / entity / comparison / policy / time / causal
> other
> overall
```

如果没有可用 QD route，则使用 evidence-map covered atom：

```text
covered_atom_ids = e_t.covered_atom_ids
```

从 covered atoms 中选一个：

```text
sort key:
1. atom importance descending
2. atom order ascending
```

如果没有 covered atom，则 fallback：

```text
Check: Verify the main factual claim.
```

### 3.4 QEC-MIN 的优点

QEC-MIN 只增加一个 `Check:` cue，不引入 relation、directness、confidence、answer 等中间判断。因此它最适合作为主线候选，因为它回答的问题很干净：

```text
只告诉 verifier 每条 evidence 在检查什么，是否就能提升分类？
```

### 3.5 QEC-MIN 的风险

主要风险是 QD question 与 evidence-map atom 不完全一致。QD question 是为了检索生成的，不一定总是最适合作为 verification cue。因此需要记录：

```text
qec_cue_type = qd_question / claim_atom / fallback
qec_qd_route_rate
qec_atom_fallback_rate
qec_fallback_rate
```

如果 `qd_question` cue 明显伤性能，应测试 `atom-first` 版本。

---

## 4. QEC-MAP

## 4.1 方法定义

QEC-MAP 在 QEC-MIN 的基础上，额外把 evidence-map 的少量结构标签放到 cue 行中。

候选 evidence text 被重写为：

```text
Check: <verification cue> [covers=<atom ids>; relation=<relation>; directness=<directness>]
<original evidence text>
```

例如：

```text
Check: Did the unemployment rate fall below 5% in 2016? [covers=A2; relation=support; directness=direct]
The unemployment rate was 4.9 percent in October 2016, according to ...
```

### 4.2 QEC-MAP 显示哪些字段

第一版只显示三个字段：

```text
covers
relation
directness
```

不默认显示：

```text
confidence
key_spans
route_rank
hybrid_score
qd_rrf_score
atom importance
source information
```

原因是：

```text
QEC-MAP 只测试最小结构标签是否有帮助；
不是把 selector debug log 暴露给 verifier。
```

### 4.3 字段格式

`covers`：

```text
covered_atom_ids joined by comma
empty -> none
```

`relation`：

```text
support / refute / qualify / mixed / background / irrelevant / unknown
```

`directness`：

```text
direct / partial / context / none / unknown
```

### 4.4 QEC-MAP 的优点

QEC-MAP 可以检验一个重要问题：

```text
显式告诉 verifier 每条 evidence 的 local stance 和 directness，是否能帮助细粒度真假判断？
```

如果 QEC-MAP 明显优于 QEC-MIN，说明上游 evidence-map 的结构标签不仅可用于选择 evidence，也值得进入 verifier。

### 4.5 QEC-MAP 的风险

QEC-MAP 的最大风险是 verifier 过度依赖 `relation` 标签，而不是认真阅读 evidence。特别是：

```text
relation=refute
relation=support
```

可能成为强提示。因此 QEC-MAP 不建议直接作为唯一主方法，除非配套做 leakage/shortcut diagnostics。

建议至少做一个控制实验：

```text
QEC-MAP-no-relation:
Check: <cue> [covers=<atom ids>; directness=<directness>]
<evidence text>
```

以及一个压力测试：

```text
QEC-MAP-shuffled-relation:
在同一 split 内随机打乱 relation/directness 标签，但保持 evidence text 不变。
```

如果 shuffled relation 仍然提高很多，说明模型可能把标签当噪声正则或格式 cue；如果 shuffled relation 明显伤性能，说明 relation 标签确实被使用，但仍需谨慎解释。

---

## 5. Trace schema 建议

为了后续分析，每条 build row 建议增加：

```json
{
  "trace_prompt_style": "qec_min",
  "qec_steps": [
    {
      "step": 1,
      "candidate_idx": 3,
      "evidence_id": "E04",
      "selector_rank": 1,
      "cue_type": "qd_question",
      "check": "Did ...?",
      "question_id": "q2",
      "question_focus": "quantity",
      "question_route_rank": 3,
      "question_route_hybrid_score": 0.71,
      "covered_atom_ids": ["A2"],
      "map_relation": "support",
      "map_directness": "direct"
    }
  ],
  "qec_diagnostics": {
    "cue_type_counts": {
      "qd_question": 7,
      "claim_atom": 2,
      "fallback": 1
    },
    "qd_cue_rate": 0.7,
    "atom_fallback_rate": 0.2,
    "fallback_rate": 0.1,
    "map_relation_counts": {
      "support": 5,
      "refute": 2,
      "qualify": 1
    }
  }
}
```

这些字段不一定进入 prompt，但必须进入 build/eval report，方便解释 QEC 何时有效、何时失败。

---

## 6. 实现建议

### 6.1 最小改动路径

第一阶段只改 verifier data build，不改 selector。

需要修改：

```text
scripts/phase5_selectors/build/build_trace_verifier_data.py
scripts/sentence_trace_method/run_one.sh
scripts/sentence_trace_method/run_lora_matrix.sh
scripts/sentence_trace_method/run_qec_v1_ministral3_prompt_matrix.sh
```

可选修改：

```text
src/fact_checking/selectors/evidence_chain_graph.py
src/fact_checking/selectors/evidence_map_selector.py
```

### 6.2 修改 build_trace_verifier_data.py

把 prompt style 增加为：

```python
TRACE_PROMPT_STYLES = (
    "plain",
    "trace_lite",
    "rawfc_boundaries",
    "qec_min",
    "qec_map",
)
```

在 `_build_split()` 中，当前逻辑是：

```python
if trace_prompt_style == "trace_lite":
    claim, candidates = _apply_trace_lite_prompt_fields(...)
else:
    claim = sample.claim
```

建议改成：

```python
if trace_prompt_style == "trace_lite":
    claim, candidates = _apply_trace_lite_prompt_fields(...)
elif trace_prompt_style in {"qec_min", "qec_map"}:
    claim, candidates, qec_payload = _apply_qec_prompt_fields(
        claim=sample.claim,
        candidates=candidates,
        claim_atoms=trace.get("claim_atoms") or [],
        style=trace_prompt_style,
    )
else:
    claim = sample.claim
```

然后把 `qec_payload` 写入 `training_row`：

```python
training_row["qec_steps"] = qec_payload["steps"]
training_row["qec_diagnostics"] = qec_payload["diagnostics"]
```

### 6.3 新增 helper

建议 helper：

```python
def _apply_qec_prompt_fields(
    *,
    claim: str,
    candidates: list[dict[str, Any]],
    claim_atoms: list[dict[str, Any]],
    style: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    ...
```

内部逻辑：

```text
for each candidate:
    cue = best_qd_question(candidate)
    if no cue:
        cue = best_covered_atom(candidate, claim_atoms)
    if no cue:
        cue = "Verify the main factual claim."

    if style == qec_min:
        candidate.text = f"Check: {cue}\n{original_text}"

    if style == qec_map:
        candidate.text = f"Check: {cue} [covers={covers}; relation={relation}; directness={directness}]\n{original_text}"
```

### 6.4 新增 run_one.sh 环境变量

当前 `run_one.sh` 里 build 阶段把 `--trace-prompt-style plain` 写死。建议新增：

```bash
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-plain}"
```

并把 build command 改成：

```bash
--trace-prompt-style "$TRACE_PROMPT_STYLE"
```

这样实验脚本可以直接：

```bash
TRACE_PROMPT_STYLE=qec_min bash scripts/sentence_trace_method/run_lora_matrix.sh
TRACE_PROMPT_STYLE=qec_map bash scripts/sentence_trace_method/run_lora_matrix.sh
```

第一版固定入口：

```bash
bash scripts/sentence_trace_method/run_qec_v1_ministral3_prompt_matrix.sh
```

该入口只展开新增 QEC cases，不重跑 B0/B3 plain baseline。

### 6.5 确保 QD route 字段不丢失

理论上，v0.7 trace 的 `candidate_pool` 从原始 candidate 复制字段，`qd_question_routes` 应能保留。但建议显式检查 build output：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/.../build/build_val.jsonl')
row = json.loads(p.open().readline())
print(row['candidates'][0].keys())
print(row['candidates'][0].get('qd_question_routes'))
print(row['candidates'][0]['text'][:500])
PY
```

如果 `qd_question_routes` 在 trace compact output 中缺失，需要在以下函数的 output keys 中补充：

```text
_candidate_trace_output(...)
```

建议加入：

```text
qd_question_routes
qd_rrf_score
qd_question_hit_count
qd_max_question_hybrid
from_qd
from_baseline
baseline_rank
qd_pool_rank
```

---

## 7. 实验原则

第一阶段实验只回答一个问题：

```text
在相同 selector、相同 selected evidence、相同训练参数下，QEC prompt 是否优于 plain prompt？
```

因此必须固定：

```text
selector = v0_7_budgeted_marginal_chain_adaptive5_10
candidate source = same staged trace
train/val/test split = same
model = same
LoRA config = same within each dataset policy
training params = fixed by dataset policy
tau selection rule = fixed by dataset policy
```

只改变：

```text
trace_prompt_style
```

当前第一版口径里，B0/B3 不按新口径重跑，而是复用已有 plain baseline。解释结果时要把 B3 写成
`historical reused baseline`，不能误读为 RAWFC ep10/eval50 同批 plain baseline。

---

## 8. 核心实验矩阵

### 8.1 Phase 1: Prompt-only core matrix

| ID | Dataset | Prompt | Training/Eval 口径 | Run policy |
|---|---|---|---|---|
| B0 | LIAR-RAW | plain | 现有 `ep12/eval100/pat8`，采用 `tau0p75` | 复用已有 |
| M1 | LIAR-RAW | qec_min | `ebs16/lr2e-5/ep12/eval100/pat8`，采用 `tau0p75` | 新跑 |
| M2 | LIAR-RAW | qec_map | `ebs16/lr2e-5/ep12/eval100/pat8`，采用 `tau0p75` | 新跑 |
| B3 | RAWFC | plain | 现有 baseline 原口径 | 复用已有，`historical reused baseline` |
| M4 | RAWFC | qec_min | `ebs16/lr1e-5/ep10/eval50/pat8` | 新跑 |
| M5 | RAWFC | qec_map | `ebs16/lr1e-5/ep10/eval50/pat8` | 新跑 |

推荐第一轮：

```text
单 seed val 快速筛选：M1 / M2 / M4 / M5
B0 / B3 只作为 reused baseline 汇总进入表格
```

如果 `qec_min` 或 `qec_map` 任一超过 plain，再做：

```text
3 training seeds
```

### 8.2 推荐训练设置

当前固定训练参数：

```text
model: ministral3_8b
selector: v0.7 adaptive5_10
LoRA: r=16, alpha=32, dropout=0.05
LIAR-RAW QEC: ebs16, lr=2e-5, ep=12, eval_steps=100, save_steps=100, patience=8
RAWFC QEC: ebs16, lr=1e-5, ep=10, eval_steps=50, save_steps=50, patience=8
LIAR-RAW tau policy: report main metric from label_token_logit_adjust_tau0p75
RAWFC tau policy: report main metric from label_token; tau sweep, if any, is appendix only
```

RAWFC 如果验证集波动大，最终候选建议扩到：

```text
5 training seeds
```

---

## 9. 诊断实验矩阵

这些实验不一定第一轮全跑，但建议在 QEC-MAP 明显有效或结果难解释时补充。

| ID | Dataset | Selector | Prompt style | Change | Purpose |
|---|---|---|---|---|---|
| D1 | LIAR-RAW | v0.7 adaptive5_10 | qec_min_atom_first | 优先 atom cue，其次 question | 检查 QD question 是否噪声过大 |
| D2 | RAWFC | v0.7 adaptive5_10 | qec_min_atom_first | 优先 atom cue，其次 question | 同上 |
| D3 | LIAR-RAW | v0.7 adaptive5_10 | qec_map_no_relation | covers + directness，不显示 relation | 检查 support/refute 标签是否造成 shortcut |
| D4 | RAWFC | v0.7 adaptive5_10 | qec_map_no_relation | covers + directness，不显示 relation | 同上 |
| D5 | LIAR-RAW | v0.7 adaptive5_10 | qec_map_shuffled | split 内随机打乱 relation/directness | 检查 MAP tag 是否被模型机械利用 |
| D6 | RAWFC | v0.7 adaptive5_10 | qec_map_shuffled | split 内随机打乱 relation/directness | 同上 |
| D7 | LIAR-RAW | v0.7 adaptive5_10 | qec_span | Check + key span + Evidence | 检查 key_span 是否比 relation 更安全 |
| D8 | RAWFC | v0.7 adaptive5_10 | qec_span | Check + key span + Evidence | 同上 |

解释规则：

```text
qec_min > plain:
    evidence chain cue 本身有效。

qec_map > qec_min:
    explicit map structure 有额外收益。

qec_map_no_relation ≈ qec_map:
    relation 不是主要收益来源，covers/directness 已足够。

qec_map >> qec_map_shuffled:
    relation/directness 标签提供有效结构信号。

qec_map_shuffled 仍显著高于 plain:
    需要警惕格式效应或 prompt regularization，而不是结构语义收益。

qec_min_atom_first > qec_min:
    QD question cue 可能噪声较大，主 prompt 应改成 atom-first。
```

---

## 10. Selector 交互实验矩阵

如果 Phase 1 显示 QEC prompt 有效，再测试 QEC 是否依赖 v0.7 adaptive5_10。

| ID | Dataset | Selector | Prompt style | Purpose |
|---|---|---|---|---|
| S1 | LIAR-RAW | old adaptive5_10 | plain | 旧 selector baseline |
| S2 | LIAR-RAW | old adaptive5_10 | qec_min | 检查 QEC 是否能增强旧 selector |
| S3 | LIAR-RAW | v0.7 adaptive3_10 | qec_min | 检查短链是否足够 |
| S4 | LIAR-RAW | v0.7 adaptive5_10 | qec_min | 主方法 |
| S5 | RAWFC | old adaptive5_10 | plain | 旧 selector baseline |
| S6 | RAWFC | old adaptive5_10 | qec_min | 检查 QEC 是否能增强旧 selector |
| S7 | RAWFC | v0.7 adaptive3_10 | qec_min | 检查短链是否伤 RAWFC |
| S8 | RAWFC | v0.7 adaptive5_10 | qec_min | 主方法 |

这个矩阵回答：

```text
QEC 是 prompt representation 的收益，还是只有 v0.7 adaptive5_10 选出来的 evidence 才能受益？
```

---

## 11. 报告指标

除了现有 verifier 指标，建议增加 QEC-specific diagnostics。

### 11.1 Verifier metrics

主指标路径：

```text
LIAR-RAW: eval/val/best/label_token_logit_adjust_tau0p75/metrics.json
RAWFC: eval/val/best/label_token/metrics.json
```

汇总表必须包含：

```text
prompt_style
dataset
model
effective_batch_size
learning_rate
epoch
eval_cadence
patience
tau_policy
baseline_reused
baseline_training_policy
```

B0/B3 复用时：

```text
baseline_reused = true
baseline_training_policy = existing_plain_run
```

```text
accuracy
macro-F1
selection score
per-class F1
true-side F1, if used
confusion matrix
```

### 11.2 Prompt/build metrics

```text
prompt_token_count mean / p95
prompt_truncation_rate
evidence_count_before
evidence_count_after
evidence_text_truncated_rate
```

### 11.3 QEC diagnostics

```text
qec_qd_cue_rate
qec_atom_cue_rate
qec_fallback_cue_rate
mean_check_token_count
relation distribution in selected chain
directness distribution in selected chain
coverage distribution in selected chain
question focus distribution
```

### 11.4 Robustness diagnostics

```text
result by cue_type majority:
    mostly qd_question vs mostly atom fallback

result by selected relation mix:
    support-heavy / refute-heavy / mixed / qualify-present

result by prompt length bucket:
    short / medium / long
```

---

## 12. 推荐结论标准

### 12.1 QEC-MIN 可以作为主方法的条件

QEC-MIN 适合作为主方法，如果满足：

```text
1. 在至少一个数据集上稳定优于 plain；
2. 另一个数据集不明显退化；
3. 多 seed 平均收益为正；
4. prompt_truncation_rate 没有明显上升；
5. case study 能显示 check cue 让 evidence chain 更可解释。
```

### 12.2 QEC-MAP 可以作为主方法增强版的条件

QEC-MAP 适合作为增强版，如果满足：

```text
1. QEC-MAP > QEC-MIN；
2. QEC-MAP-no-relation 不能完全解释收益；
3. QEC-MAP-shuffled 明显弱于 QEC-MAP；
4. 结果不是由单一类别或单一 seed 驱动。
```

如果 QEC-MAP 只在 relation 标签可见时大幅提升，但 shuffled/randomized diagnostics 也高，应该谨慎把它作为主方法，而应作为：

```text
map-visible verifier ablation
```

---

## 13. 推荐第一批运行顺序

### Step 1: 实现 qec_min / qec_map prompt styles

只改 build，不改 selector。

### Step 2: 单 seed smoke test

```text
LIAR-RAW:
    qec_min
    qec_map

RAWFC:
    qec_min
    qec_map
```

检查：

```text
build rows 是否一致
selected_indices 是否一致
evidence_count_before 是否一致
最终 evidence_count 是否因 QEC prompt 变长而下降
prompt token 是否合理
truncation 是否增加
candidate text 是否正确包含 Check
qec_map 是否正确包含 covers/relation/directness
```

### Step 3: 单 seed full eval

比较 val/test：

```text
plain vs qec_min vs qec_map
```

第一轮只补 val；test 在 val 筛选后再决定是否补跑。

### Step 4: 多 seed confirmation

只对以下候选做多 seed：

```text
plain baseline
best QEC variant
second-best QEC variant, if close
```

### Step 5: diagnostic ablation

如果 qec_map 明显强：

```text
qec_map_no_relation
qec_map_shuffled
```

如果 qec_min 不稳定：

```text
qec_min_atom_first
```

---

## 14. 推荐论文表述

### QEC-MIN

```text
QEC-MIN exposes the selected evidence as a compact verification chain. Each evidence item is preceded by a single verification cue, derived from either the question-decomposition route that retrieved the evidence or the claim atom covered by the evidence. The evidence itself serves as the answer, avoiding intermediate answer generation.
```

中文：

```text
QEC-MIN 将选出的证据表示为紧凑的验证链。每条证据前只加入一个检查提示，该提示来自检索该证据的问题分解路径，或来自该证据覆盖的 claim atom。证据本身作为回答，因此不引入额外中间答案生成。
```

### QEC-MAP

```text
QEC-MAP further exposes compact evidence-map tags, including the covered atom ids, local relation, and directness. This variant tests whether the structured alignment used by the selector also benefits the downstream verifier when made visible in the prompt.
```

中文：

```text
QEC-MAP 在 QEC-MIN 基础上进一步暴露紧凑的 evidence-map 标签，包括覆盖的 atom、局部 relation 和 directness。该变体用于检验 selector 使用的结构化对齐信息在进入 verifier prompt 后是否仍能带来额外收益。
```

---

## 15. 当前建议

优先把 QEC-MIN 作为主方法候选，因为它最简洁、最不容易被质疑为 label shortcut。

QEC-MAP 作为增强版和消融，用来回答：

```text
结构化 evidence-map 标签是否应显式进入 verifier？
```

如果 QEC-MIN 已经稳定提升，论文主叙事可以是：

```text
Question-guided evidence chain with evidence-as-answer.
```

如果 QEC-MAP 进一步稳定提升并通过 shuffled/no-relation diagnostics，论文可以把最终方法写成：

```text
Map-aware question-guided evidence chain.
```
