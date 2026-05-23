# LLM Action Selector with VIG Soft Supervision

## Goal

基于 saved-score VIG-lite 结果，新增一条大模型端到端解码式 selector 实验链路。第一版只做 selection-only gate，不接 full pipeline verifier。

## Method

每个训练样本对应一个 `(claim, step)`：

```text
输入 = claim + selected prefix + remaining candidates
输出 = 下一条 evidence action，例如 E
```

训练目标：

```text
loss = hard_action_ce + 0.3 * soft_listwise_ce
soft target = softmax(delta_margin / 0.2)
```

`delta_margin` 来自 `outputs/selectors/vig_utility/saved_step_train/vig_records_train.jsonl`。这些 VIG 分数只进入 loss，不写入 prompt。prompt 中不包含 `gold_label`、`selected_indices` 或 oracle target。

## Entry Point

```bash
bash scripts/selectors/run_llm_action_selector_vig_soft.sh
```

默认路径：

```text
MODEL_NAME=/home/fenglin/project/hateSpeechDetection/models/base/Qwen2.5-7B-Instruct
OUTPUT_DIR=outputs/selectors/llm_action_selector/qwen25_7b_vig_soft
TRAIN_ORACLE_RESULTS=outputs/oracle_evidence/stage2_margin_train_stepscores/oracle_results_train.jsonl
VAL_ORACLE_RESULTS=outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl
TRAIN_VIG_CACHE=outputs/selectors/vig_utility/saved_step_train/vig_records_train.jsonl
VAL_VIG_CACHE=outputs/selectors/vig_utility/saved_step_val/vig_records_val.jsonl
```

常用 smoke：

```bash
TRAIN_SAMPLE_LIMIT=32 VAL_SAMPLE_LIMIT=32 EVAL_SAMPLE_LIMIT=32 \
NPROC_PER_NODE=1 EPOCHS=1 EVAL_EVERY=20 \
bash scripts/selectors/run_llm_action_selector_vig_soft.sh
```

## Outputs

```text
data/action_samples_train.jsonl
data/action_samples_val.jsonl
selector_metadata.json
training_metrics.json
eval_val/selection_metrics.json
eval_val/selection_trace.jsonl
eval_val/analysis.md
```

主判定：

```text
Go: val Jaccard@5 > single_margin_step0_static Jaccard@5 0.3761
Strong-Go: Jaccard@5 >= 0.39 且 NDCG/order 指标同步提升
```

若未超过 `single_margin_step0_static`，下一步不进入 OPD/GRPO，先做错误分析或检查是否需要 verifier rerank / stronger oracle-margin distillation。
