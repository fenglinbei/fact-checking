# Sequential Pointer Selector Step4 实施记录

## 目标

实现第一版 Step4 supervised Sequential Pointer Selector，直接建模 Stage2 oracle 的 ordered `selected_indices`。本版只使用 deep semantic interaction：

```text
h_i_pair
H_i_ctx
P_t
H_i_ctx * P_t
abs(H_i_ctx - P_t)
cos(H_i_ctx, P_t)
bilinear(H_i_ctx, P_t)
```

不使用 rank/index prior，不使用位置、长度、词面 overlap、数字 overlap，也不使用 claim aspect / stance / graph-lite targeted features。后续特征扩展通过 profile 接口接入。

## 新增文件

```text
src/fact_checking/selectors/base.py
src/fact_checking/selectors/sequential.py
src/fact_checking/selectors/test_sequential.py
scripts/selectors/train_sequential_selector.py
scripts/selectors/eval_sequential_selector.py
scripts/selectors/run_sequential_step4.sh
configs/experiment/b3_sequential_stage2_sentence_1024.yaml
```

## 核心实现

`src/fact_checking/selectors/base.py` 定义通用 selector 接口：

```text
SelectorCandidateGroup
SelectorPrediction
EvidenceSelector Protocol
selector type registry
```

这为后续 targeted-feature selector、不同 neural selector 或 LLM selector 预留统一输出契约：输入 claim + candidates + candidate_scores，输出 ordered indices、scores、step trace、metadata。

`src/fact_checking/selectors/sequential.py` 实现：

```text
SequentialPointerSelectorModel
DeepInteractionPointerHead
SequentialSelector
teacher_forcing_sequential_logits
sequential_teacher_forcing_loss
predict_sequential_examples / groups
select_candidates_sequential
```

训练时使用 oracle prefix teacher forcing：

```text
prefix_t = selected_indices[:t]
target_t = selected_indices[t]
```

推理时使用 greedy pointer decode，并在每步 mask 掉已选 candidate 和 padding candidate，保证输出无重复。

## Feature Profile

第一版 profile 被显式锁定：

```text
semantic_feature_profile = deep
targeted_feature_profile = none
shallow_feature_profile = off
```

`targeted_feature_profile` 和 `shallow_feature_profile` 当前只做元数据和 CLI 契约，不参与特征构造。这样后续加入 aspect / stance_utility / graph_lite 时，不需要重写训练、评估和 build pipeline 接口。

## Build 接入

`src/fact_checking/build/candidates.py` 新增 selection method：

```text
build.retrieval.selection_method=sequential_selector
```

对应配置：

```yaml
build:
  retrieval:
    selection_method: sequential_selector
    sequential_selector:
      model_dir: outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep
      candidate_pool_size: 15
      max_length: 384
      batch_size: 8
      device: cuda
      dump_trace: true
      strict_fingerprint: true
```

build trace 输出：

```text
outputs/cache/build/<build_id>/sequential_selector_trace_<split>.jsonl
```

## 训练与评估

默认 wrapper：

```bash
scripts/selectors/run_sequential_step4.sh
```

等价主命令：

```bash
PYTHONPATH=src python scripts/selectors/train_sequential_selector.py \
  --model-name /data/models/deberta-v3-base/ \
  --train-oracle-results outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl \
  --val-oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --output-dir outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep \
  --semantic-feature-profile deep \
  --targeted-feature-profile none \
  --shallow-feature-profile off
```

selection-only eval 输出：

```text
selection_metrics.json
selection_trace.jsonl
control_hybrid_trace.jsonl
control_candidate_pool_trace.jsonl
```

新增 step-wise diagnostics：

```text
step accuracy
step entropy
first_wrong_step_mean
```

## 验证

已通过：

```bash
PYTHONPATH=src python -m compileall -q src/fact_checking/selectors scripts/selectors/train_sequential_selector.py scripts/selectors/eval_sequential_selector.py src/fact_checking/build/candidates.py
PYTHONPATH=src python -m unittest src/fact_checking/selectors/test_metrics.py src/fact_checking/selectors/test_sequential.py
PYTHONPATH=src python scripts/selectors/train_sequential_selector.py --help
PYTHONPATH=src python scripts/selectors/eval_sequential_selector.py --help
bash -n scripts/selectors/run_sequential_step4.sh
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b3_sequential_stage2_sentence_1024 --cfg job
git diff --check
```

尚未运行完整训练；本记录只覆盖代码实现、接口接入和轻量静态/单元验证。
