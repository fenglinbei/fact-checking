# Learned Lambda 验证实验：Oracle λ vs 固定 λ 端到端对比

## 背景

Learned lambda 实验试图用神经网络预测器为每个 claim 预测最优 MMR λ 值，但训练出的 `ChunkEmbeddingLambdaEncoder` 预测器几乎只能预测均值（R² vs 均值基线 ≈ 0.01）。

在优化预测器之前，需要先验证一个前提：**即使能完美预测每个 claim 的最优 λ（oracle λ），这个最优 λ 是否能比固定 λ=0.7 带来更高的 fact-checking 准确率？** 如果 oracle λ 相比固定 λ 没有显著提升，整个 learned lambda 方向就不成立。

## 实验设计

### 数据

- 使用 train split 中 predictor 训练时 hold-out 的 **2013 个样本**（predictor 训练时按 80/20 划分，seed=42）
- 这 2013 个样本的 oracle λ 已在之前计算好（来自 `oracle_lambda_train.jsonl`）
- Oracle λ 使用 **fine-tuned LoRA 模型** 计算（checkpoint: `b3_mmr_topk_sweep_1024/...top_k-5/.../best`），确保 oracle 定义与模型一致

### 对比条件

| 条件 | λ 值 | Build JSONL |
|------|------|-------------|
| **Fixed（基线）** | 全局 λ=0.70 | `build_fixed_predictor_val.jsonl` |
| **Oracle（实验）** | 每个 claim 用其 oracle λ | `build_oracle_predictor_val.jsonl` |

两种条件下使用的 prompt template、tokenizer、chunking strategy、hybrid scoring weights 完全相同，**唯一变量是 MMR 阶段使用的 λ 值**。

### 推理设置

- **模型**：Qwen2.5-7B-Instruct + LoRA adapter（与 oracle 计算时相同的 checkpoint）
- **推理引擎**：vLLM 0.8.5，tensor_parallel_size=4，max_model_len=1024
- **解码策略**：temperature=0，max_tokens=1，guided_choice（限定输出 A-F letter tokens）
- **指标**：accuracy，macro precision/recall/F1，per-class F1

### 证据选择差异

48.4% 的样本在 oracle λ 和固定 λ 下产生了不同的 evidence 选择（prompt 不同），说明 λ 的变化确实影响了 MMR 的证据排序结果。

## 结果

### 总体对比

| 指标 | 固定 λ=0.70 | Oracle λ | Δ |
|------|------------|----------|---|
| **Accuracy** | 0.3040 | **0.3348** | **+0.0308** |
| **Macro F1** | 0.3060 | **0.3384** | **+0.0324** |
| **Macro Precision** | 0.3328 | 0.3684 | +0.0356 |
| **Macro Recall** | 0.2997 | 0.3311 | +0.0314 |
| **Parse Error Rate** | 0.0000 | 0.0000 | — |

### 各类别 F1 对比

| Class | Fixed λ=0.70 | Oracle λ | Δ |
|-------|-------------|----------|---|
| pants-fire | 0.3608 | 0.4093 | **+0.0485** |
| false | 0.3657 | 0.3992 | **+0.0335** |
| barely-true | 0.2561 | 0.2680 | +0.0119 |
| half-true | 0.2649 | 0.2965 | +0.0316 |
| mostly-true | 0.2932 | 0.3109 | +0.0177 |
| true | 0.2953 | 0.3468 | **+0.0515** |

**所有 6 个类别全部改善。** 改善最大的三个类别是 `true`（+5.15%）、`pants-fire`（+4.85%）、`false`（+3.35%）。

## 结论

### 核心发现

1. **Oracle λ（理论上界）能显著提升 accuracy（+3.08%）和 macro F1（+3.24%）**。这验证了 adaptive λ 方向的价值——如果预测器能达到 oracle 水平，预期可获得约 +3% 的准确率提升。

2. **Learned lambda 方向是正确的，瓶颈在预测器**。上一轮诊断确认预测器只学到了预测均值（R²≈0），现在知道了理论上界约为 +3%，值得继续优化预测器。

3. **改善是全面的而非局部的**。所有 6 个类别都有正向提升，说明 adaptive λ 的效果不是针对特定类别，而是一种通用的证据选择质量提升。

### 与预测器诊断的关联

之前的诊断分析（docs/learned_lambda/analysis.md）发现：
- Oracle λ 标签本身噪声很大（72.6% 的样本 margin < 0.05）
- 但高 margin 子集（~27% 样本）上 oracle λ 偏向低值（mean=0.282），特征相关性有所提升

结合本次验证结果，改进预测器的方向包括：
1. **在 margin >= 0.05 的高质量子集上训练**，提高训练信号纯度
2. **用 logprob 分布作为软标签**，让模型学到不确定性
3. **将候选数量特征加入模型**，利用已知的最强单特征信号

## 新增脚本

| 脚本 | 说明 |
|------|------|
| `scripts/learned_lambda/build_with_oracle_lambda.py` | 加载 chunk MMR cache + oracle λ JSONL，调用 `_mmr_phase_from_chunk_cache(lambda_overrides=...)` 生成带 per-claim oracle λ 的 build JSONL |
| `scripts/learned_lambda/compare_inference.py` | 使用 vLLM 离线 API 对两个 build JSONL 分别推理，对比 accuracy/F1 指标 |

## 复现步骤

```bash
conda activate /data/liaozijie/conda/accelerate-fc/

# 1. 提取 predictor val 样本的 event IDs（或使用已有文件）
PYTHONPATH=src python -c "
import json, numpy as np
from pathlib import Path
from fact_checking.build.candidates import _load_pickle
from fact_checking.learned_lambda.embedding_features import build_matched_chunk_embedding_arrays

with open('outputs/learned_lambda/oracle_lambda_train.jsonl') as f:
    oracle_by_eid = {}
    for line in f:
        rec = json.loads(line.strip())
        rec['oracle_lambda'] = float(rec['oracle_lambda'])
        oracle_by_eid[rec['event_id']] = rec

chunk_samples = _load_pickle(Path('outputs/cache/chunk_mmr/e0b01520364d/train.pkl'))
arrays, targets, _, _ = build_matched_chunk_embedding_arrays(
    chunk_samples, oracle_by_eid, candidate_top_k=None,
    alpha_dense=0.7, alpha_lexical=0.2, alpha_bm25=0.1)

rng = np.random.default_rng(42)
n = len(targets)
indices = rng.permutation(n)
n_val = max(1, int(n * 0.2))
val_eids = arrays['event_ids'][indices[:n_val]].tolist()
with open('outputs/learned_lambda/predictor_val_eids.json', 'w') as f:
    json.dump(val_eids, f)
"

# 2. 生成 oracle λ build
PYTHONPATH=src python scripts/learned_lambda/build_with_oracle_lambda.py \
    --oracle-lambdas outputs/learned_lambda/oracle_lambda_train.jsonl \
    --experiment b3_mmr_topk_sweep_1024 \
    --config-overrides "build.retrieval.top_k=5" \
    --split-name train \
    --output outputs/learned_lambda/build_oracle_predictor_val.jsonl \
    --event-id-file outputs/learned_lambda/predictor_val_eids.json \
    --top-k 5

# 3. 生成 fixed λ build（从已有 per-λ prompts 中过滤）
PYTHONPATH=src python -c "
import json
with open('outputs/learned_lambda/predictor_val_eids.json') as f:
    val_eids = set(json.load(f))
with open('outputs/learned_lambda/prompts/lambda_0.70_train.jsonl') as fin, \
     open('outputs/learned_lambda/build_fixed_predictor_val.jsonl', 'w') as fout:
    for line in fin:
        row = json.loads(line.strip())
        if row.get('event_id', '') in val_eids:
            fout.write(line)
"

# 4. 运行对比推理
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src python scripts/learned_lambda/compare_inference.py \
    --baseline-build outputs/learned_lambda/build_fixed_predictor_val.jsonl \
    --experiment-build outputs/learned_lambda/build_oracle_predictor_val.jsonl \
    --baseline-label "fixed_0.70" \
    --experiment-label "oracle" \
    --model /data/models/Qwen2.5-7B-Instruct \
    --lora-adapter outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 1024 \
    --output-dir outputs/learned_lambda/comparison/
```

## 关键文件索引

| 文件 | 说明 |
|------|------|
| `scripts/learned_lambda/build_with_oracle_lambda.py` | 新建：使用 oracle λ 生成 build JSONL |
| `scripts/learned_lambda/compare_inference.py` | 新建：对比两个 build JSONL 的推理指标 |
| `outputs/learned_lambda/build_oracle_predictor_val.jsonl` | Oracle λ build（2013 样本） |
| `outputs/learned_lambda/build_fixed_predictor_val.jsonl` | 固定 λ=0.70 build（2013 样本） |
| `outputs/learned_lambda/predictor_val_eids.json` | Predictor val 样本的 event ID 列表 |
| `outputs/learned_lambda/comparison/comparison_summary.json` | 对比结果摘要 |
| `outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best/` | 使用的 LoRA checkpoint |

## 日期

2026-05-13
