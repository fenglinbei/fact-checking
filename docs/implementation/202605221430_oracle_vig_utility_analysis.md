# Oracle VIG Utility Analysis Implementation

## 目的

在继续训练 selector 前，先分析 Stage2 margin oracle 为什么选择这些 evidence。

这版不把 VIG 直接当黑盒 imitation target，而是把 verifier margin 变化拆成可审计的机制信号：

```text
prefix marginal utility:
u_t(i) = margin_verifier(claim, prefix_t + candidate_i) - margin_verifier(claim, prefix_t)

final-set counterfactual:
contribution(i) = margin_verifier(claim, oracle_final_set)
                - margin_verifier(claim, oracle_final_set without / replaced i)
```

## 新增入口

```text
scripts/selectors/generate_oracle_vig_cache.py
scripts/selectors/analyze_oracle_vig_utility.py
scripts/selectors/run_oracle_vig_utility_analysis.sh
```

## 产物

默认输出目录：

```text
outputs/selectors/vig_utility/stage2_margin_val/
```

主要文件：

```text
vig_records_val.jsonl
vig_final_counterfactuals_val.jsonl
vig_event_summaries_val.jsonl
analysis/vig_utility_analysis.json
analysis/analysis.md
```

`vig_records_val.jsonl` 是 step-wise cache。对每个 oracle prefix，记录所有 remaining candidates 的 verifier score：

```text
base_margin
after_margin
delta_margin
delta_gold_logprob
delta_best_wrong_logprob
base_pred_letter
after_pred_letter
target
oracle_remaining_selected
retrieval features
text overlap / prefix novelty features
single-evidence verifier features
```

`vig_final_counterfactuals_val.jsonl` 是 final-set counterfactual。默认对每条样本额外评分：

```text
remove_selected: oracle final set 去掉某条 selected evidence
replace_selected: 用每个 non-selected candidate 替换某条 selected evidence
```

在 `top_k=5, n_candidates=15` 时，每条样本约产生：

```text
65 prefix marginal rows
55 final counterfactual rows
```

完整 val 约为 `1274 * 65 = 82810` 条 prefix rows，以及 `1274 * 55 = 70070` 条 final counterfactual rows。目标服务器 4 卡 vLLM 可承受，因此默认保留完整 counterfactual 分析。

## 运行方式

在目标服务器已配置好 vLLM、transformers 与项目依赖的 conda 环境中运行。环境名不在脚本中固定，由运行机器自行激活：

```bash
bash scripts/selectors/run_oracle_vig_utility_analysis.sh
```

默认沿用 Stage2 oracle 的 verifier 设置：

```text
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/
LORA_ADAPTER=outputs/runs/b3_label_token_ce_1024/label_token_ce_stage1__0ee9b55f/train/best
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=1032
SCORE_BATCH_SIZE=256
```

如果 CUDA 0 被占用，只使用 1/2/3 三张卡：

```bash
GPU_DEVICES=1,2,3 bash scripts/selectors/run_oracle_vig_utility_analysis.sh
```

`GPU_DEVICES` 会导出为 `CUDA_VISIBLE_DEVICES`。未显式设置 `TENSOR_PARALLEL_SIZE` 时，脚本会自动按 `GPU_DEVICES` 数量设置 tensor parallel size；例如 `GPU_DEVICES=1,2,3` 会使用 `TENSOR_PARALLEL_SIZE=3`。如需手动覆盖：

```bash
GPU_DEVICES=1,2,3 TENSOR_PARALLEL_SIZE=3 bash scripts/selectors/run_oracle_vig_utility_analysis.sh
```

建议先做 50 条 smoke：

```bash
SAMPLE_LIMIT=50 \
OUTPUT_DIR=outputs/selectors/vig_utility/stage2_margin_val_sample50 \
bash scripts/selectors/run_oracle_vig_utility_analysis.sh
```

如需多进程分片：

```bash
NUM_SHARDS=4 SHARD_INDEX=0 ANALYZE=false bash scripts/selectors/run_oracle_vig_utility_analysis.sh
NUM_SHARDS=4 SHARD_INDEX=1 ANALYZE=false bash scripts/selectors/run_oracle_vig_utility_analysis.sh
NUM_SHARDS=4 SHARD_INDEX=2 ANALYZE=false bash scripts/selectors/run_oracle_vig_utility_analysis.sh
NUM_SHARDS=4 SHARD_INDEX=3 ANALYZE=false bash scripts/selectors/run_oracle_vig_utility_analysis.sh

ONLY_ANALYZE=true NUM_SHARDS=4 bash scripts/selectors/run_oracle_vig_utility_analysis.sh
```

## 分析指标

### Oracle self-check

`true_delta_margin_oracle_probe` 用重新打分得到的 `delta_margin` 排序 remaining candidates。

如果 scorer、LoRA、prompt、max length 与原 Stage2 oracle 完全一致，oracle target 在每个 step 应接近 rank 1：

```text
true_delta step_top1_match >= 0.90
```

若该值明显低于 0.90，优先排查 cache 与原 oracle 的运行配置是否不一致，而不是解释 selector。

### Delta decomposition

报告 target vs non-target 的三类差异：

```text
delta_margin
delta_gold_logprob
delta_best_wrong_logprob
```

这可以区分 oracle 选中 evidence 是因为：

```text
提高 gold label logprob
压低 best wrong label logprob
二者同时发生
```

### Feature-group probe

`analyze_oracle_vig_utility.py` 先用可解释 feature groups 预测 `delta_margin`：

```text
retrieval
text_overlap
prefix_state
single_verifier
retrieval + single_verifier
all
```

模型是标准化后的 ridge regression，并输出：

```text
R2 / RMSE
Spearman
target AUROC
step top1 match
permutation group importance
top coefficients
```

这一步回答的是：oracle margin gain 能否由可解释特征组解释，而不是直接训练深层 selector。

### Final-set counterfactuals

final-set 分析输出：

```text
removal contribution mean
selected harmful final rate
best replacement delta mean
selected replaceable rate
replacement row improves rate
```

其中：

```text
selected harmful final rate
```

表示移除某条 oracle-selected evidence 后 final margin 反而更高的比例。

```text
selected replaceable rate
```

表示存在某个 non-selected candidate 替换该 selected evidence 后 final margin 更高的比例。

这些指标用于判断 oracle greedy order 是否只是局部最优，以及 selected set 中是否有冗余或可替代证据。

## Stop/Go

第一阶段只做机制分析，不直接训练 selector。

建议判定：

```text
若 true_delta step_top1_match < 0.90:
  修复 VIG cache / prompt / LoRA / max length 一致性

若 all feature group 明显超过 retrieval baseline:
  进入 utility feature distillation

若 single_verifier 或 prefix_state 是主要贡献:
  下一版 selector 优先接入 verifier-aware utility features

若 final-set replaceable/harmful rate 高:
  当前 greedy oracle 本身有局部最优/冗余问题，应考虑 oracle target smoothing 或 set-level utility，而不是更强 imitation
```

默认分析脚本中的保守 go 条件：

```text
true_delta step_top1_match >= 0.90
all_feature target AUROC >= 0.60
all_feature step_top1_match 比 hybrid-rank baseline 高 >= 3pp
```

## 已验证

本地已完成静态验证与小型 synthetic cache smoke：

```text
python3 -m compileall scripts/selectors/generate_oracle_vig_cache.py scripts/selectors/analyze_oracle_vig_utility.py
PYTHONPATH=src python3 scripts/selectors/generate_oracle_vig_cache.py --help
PYTHONPATH=src python3 scripts/selectors/analyze_oracle_vig_utility.py --help
bash -n scripts/selectors/run_oracle_vig_utility_analysis.sh
PYTHONPATH=src python3 synthetic VIG smoke
```

当前环境未初始化本地 vLLM，因此未在本机跑真实 Qwen verifier cache；真实运行应在目标服务器对应的 conda 环境中执行。
