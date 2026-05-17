# Oracle Search 输出格式更新

生成日期: 2026-05-17

## 目的

为后续 pointwise selector / DPO / sequential selector 构造严格监督数据，`scripts/oracle_evidence/run_search.sh` 现在会在每条 oracle result 中保存有效候选池、retrieval 分数和候选池指纹。

旧输出只有 `selected_indices` / `selected_texts`，无法确认这些 index 属于哪个候选池。新输出把 index 的坐标系一起写入 JSONL。

## 修改文件

```text
src/fact_checking/oracle_evidence/search.py
scripts/oracle_evidence/search_optimal_evidence.py
scripts/oracle_evidence/run_search.sh
```

## 新增字段

每条 `oracle_results_<split>.jsonl` 增加:

```json
{
  "candidate_pool_fingerprint": "...",
  "candidate_pool_metadata": {
    "candidate_pool_version": "oracle-search-candidate-pool-v1",
    "chunk_mmr_fingerprint": "...",
    "pre_mmr_fingerprint": "...",
    "chunk_mmr_cache_path": "...",
    "n_original": 54,
    "n_scored": 54,
    "n_dedup": 51,
    "n_candidates": 15,
    "two_stage": true,
    "two_stage_limit": 15,
    "two_stage_multiplier": 3,
    "top_k": 5,
    "score_config": {
      "alpha_dense": 0.7,
      "alpha_lexical": 0.2,
      "alpha_bm25": 0.1
    },
    "candidate_order": "hybrid_score_desc"
  },
  "candidate_pool": [
    {
      "candidate_idx": 0,
      "candidate_uid": "...",
      "source_index": 27,
      "report_id": "...",
      "sent_idx": 0,
      "chunk_sent_indices": [0, 1],
      "text": "...",
      "source_report": {}
    }
  ],
  "candidate_scores": [
    {
      "candidate_idx": 0,
      "candidate_uid": "...",
      "source_index": 27,
      "hybrid_rank": 0,
      "dense_score": 0.73,
      "lexical_score": 0.12,
      "bm25_score": 2.31,
      "hybrid_score": 1.0
    }
  ]
}
```

`selected_indices` 现在明确表示:

```text
indices into per-row effective candidate_pool after deduplication and optional two-stage pruning
```

## Two-stage 行为修正

之前 two-stage 注释是按 `hybrid_score` 截断，但 Chunk-MMR cache 的候选 dict 不一定带 `hybrid_score`。现在 search 入口会先调用 `compute_hybrid_scores()`，为每个候选重新计算:

```text
dense_score
lexical_score
bm25_score
hybrid_score
```

然后再执行:

```text
dedup by canonical text -> sort by hybrid_score desc -> keep top_k * two_stage_multiplier
```

因此新 run 的 two-stage candidate pool 是实际 hybrid top-M，而不是依赖候选 dict 中是否已有分数字段。

## 新增运行开关

`run_search.sh` 新增环境变量:

```bash
SAVE_CANDIDATE_POOL=true        # 默认 true
SAVE_SEARCH_STEP_SCORES=false   # 默认 false
```

默认运行会保存候选池和 retrieval scores:

```bash
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/ \
MODEL_BASE_PATH=/data/models/ \
LORA_ADAPTER=outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
SPLIT=train \
bash scripts/oracle_evidence/run_search.sh
```

如需减少输出体积:

```bash
SAVE_CANDIDATE_POOL=false bash scripts/oracle_evidence/run_search.sh ...
```

如需保存 greedy search 每一步所有剩余候选的 gold-label logprob:

```bash
SAVE_SEARCH_STEP_SCORES=true bash scripts/oracle_evidence/run_search.sh ...
```

注意: `SAVE_SEARCH_STEP_SCORES=true` 会显著增大 JSONL，建议先在 `MAX_SAMPLES` 小样本上确认格式。

## 运行方式

### 1. 配置与 cache 检查

先确认当前 config 会解析到哪个 Chunk-MMR fingerprint，以及本地是否已有对应 cache:

```bash
PYTHONPATH=src python scripts/oracle_evidence/search_optimal_evidence.py \
  --config configs/experiment/b3_mmr_topk_sweep_1024.yaml \
  --verifier-model /data/models/Qwen2.5-7B-Instruct/ \
  --model-base-path /data/models/ \
  --split train \
  --verify-config-only
```

如果日志显示 `Chunk-MMR exists: False`，正式运行会尝试自动构建 cache；这会调用 embedding 阶段并需要可用 GPU / 模型路径。

### 2. 推荐 smoke test

先用很小的样本数验证新 JSONL 字段和 candidate pool 坐标系:

```bash
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/ \
MODEL_BASE_PATH=/data/models/ \
LORA_ADAPTER=outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
SPLIT=val \
MAX_SAMPLES=5 \
SAVE_CANDIDATE_POOL=true \
SAVE_SEARCH_STEP_SCORES=false \
bash scripts/oracle_evidence/run_search.sh \
  --output-dir outputs/oracle_evidence/smoke_output_contract_v2
```

检查输出:

```bash
jq '{
  event_id,
  n_candidates,
  selected_indices,
  candidate_pool_fingerprint,
  metadata: .candidate_pool_metadata,
  candidate_pool_len: (.candidate_pool | length),
  candidate_scores_len: (.candidate_scores | length),
  first_candidate: .candidate_pool[0],
  first_score: .candidate_scores[0]
}' outputs/oracle_evidence/smoke_output_contract_v2/oracle_results_val.jsonl | head -n 80

jq '.output_contract, .effective_candidate_pool_stats' \
  outputs/oracle_evidence/smoke_output_contract_v2/oracle_metrics_val.json
```

应满足:

```text
candidate_pool_len == candidate_scores_len == n_candidates
selected_indices 均在 [0, n_candidates) 内
candidate_pool_fingerprint 非空
candidate_pool_metadata.selected_indices_coordinate 语义见 metrics output_contract
```

### 3. Val 小规模正式验证

确认 smoke 输出无误后，建议先跑完整 val。val 成本远低于 train，适合检查指标和输出体积:

```bash
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/ \
MODEL_BASE_PATH=/data/models/ \
LORA_ADAPTER=outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
SPLIT=val \
SAVE_CANDIDATE_POOL=true \
SAVE_SEARCH_STEP_SCORES=false \
bash scripts/oracle_evidence/run_search.sh \
  --output-dir outputs/oracle_evidence/val_output_contract_v2
```

检查文件大小与核心字段:

```bash
ls -lh outputs/oracle_evidence/val_output_contract_v2

jq '.output_contract, .candidate_pool_stats, .effective_candidate_pool_stats' \
  outputs/oracle_evidence/val_output_contract_v2/oracle_metrics_val.json

python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/oracle_evidence/val_output_contract_v2/oracle_results_val.jsonl")
bad = 0
for line in p.open(encoding="utf-8"):
    r = json.loads(line)
    n = r["n_candidates"]
    if len(r.get("candidate_pool", [])) != n:
        bad += 1
    if len(r.get("candidate_scores", [])) != n:
        bad += 1
    if any(i < 0 or i >= n for i in r["selected_indices"]):
        bad += 1
print({"rows_checked": sum(1 for _ in p.open(encoding="utf-8")), "bad_rows": bad})
PY
```

### 4. Train 全量权威重跑

生成可用于后续 selector 训练的权威 train oracle set:

```bash
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/ \
MODEL_BASE_PATH=/data/models/ \
LORA_ADAPTER=outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
SPLIT=train \
SAVE_CANDIDATE_POOL=true \
SAVE_SEARCH_STEP_SCORES=false \
bash scripts/oracle_evidence/run_search.sh
```

默认输出目录为:

```text
outputs/oracle_evidence/<timestamp>/
```

若希望固定目录:

```bash
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/ \
MODEL_BASE_PATH=/data/models/ \
LORA_ADAPTER=outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
SPLIT=train \
SAVE_CANDIDATE_POOL=true \
SAVE_SEARCH_STEP_SCORES=false \
bash scripts/oracle_evidence/run_search.sh \
  --output-dir outputs/oracle_evidence/train_output_contract_v2
```

### 5. 保存 per-step oracle logprob

仅在需要分析 greedy search 每一步候选 logprob 曲面时开启:

```bash
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/ \
MODEL_BASE_PATH=/data/models/ \
LORA_ADAPTER=outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
SPLIT=val \
MAX_SAMPLES=20 \
SAVE_CANDIDATE_POOL=true \
SAVE_SEARCH_STEP_SCORES=true \
bash scripts/oracle_evidence/run_search.sh \
  --output-dir outputs/oracle_evidence/val_step_scores_smoke
```

该模式会在 `search_steps[*].candidate_logprobs` 中保存每一步剩余候选的 gold-label logprob。全量 train 开启前必须先估算 JSONL 体积。

### 6. 输出体积控制

如果只想计算 oracle 上界指标，不构造 selector 训练数据，可以关闭 candidate pool:

```bash
SAVE_CANDIDATE_POOL=false \
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/ \
MODEL_BASE_PATH=/data/models/ \
LORA_ADAPTER=outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
SPLIT=val \
bash scripts/oracle_evidence/run_search.sh
```

关闭后仍会保存:

```text
selected_indices
selected_texts
candidate_pool_fingerprint
candidate_pool_metadata
```

但不会保存完整 `candidate_pool` / `candidate_scores`，因此不适合作为严格 pointwise selector 监督源。

## metrics 新增字段

`oracle_metrics_<split>.json` 增加:

```json
{
  "effective_candidate_pool_stats": {},
  "output_contract": {
    "version": "oracle-results-v2",
    "save_candidate_pool": true,
    "save_search_step_scores": false,
    "selected_indices_coordinate": "indices into per-row effective candidate_pool after deduplication and optional two-stage pruning"
  }
}
```

## 验证

已执行:

```bash
PYTHONPATH=src python -m compileall \
  src/fact_checking/oracle_evidence/search.py \
  scripts/oracle_evidence/search_optimal_evidence.py

bash -n scripts/oracle_evidence/run_search.sh

PYTHONPATH=src python scripts/oracle_evidence/search_optimal_evidence.py \
  --config configs/experiment/b3_mmr_topk_sweep_1024.yaml \
  --verifier-model /data/models/Qwen2.5-7B-Instruct/ \
  --model-base-path /data/models/ \
  --split train \
  --verify-config-only
```

本轮未重跑 vLLM oracle search。
