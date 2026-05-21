# LLM Decomp+ Aspect Coverage Implementation

## 目的

验证 G-Defense `decomp+` 风格 claim decomposition 是否能给 selector 提供比 `rule_aspect_v1` 更强的 coverage 信号。

这版仍是前置上界诊断，不直接训练 `deberta_sequential_aspect`。

## 新增入口

```text
scripts/selectors/generate_llm_claim_decomp_aspects.py
scripts/selectors/run_llm_decomp_plus_aspect_coverage.sh
```

并扩展：

```text
scripts/selectors/analyze_oracle_aspect_coverage.py --claim-aspects-input
src/fact_checking/selectors/aspects.py::build_claim_aspect_bundle_from_texts
```

## 分解契约

`generate_llm_claim_decomp_aspects.py` 使用本地 vLLM 加载 Qwen2.5-7B-Instruct，提示模型输出 2-5 个 self-contained / atomic / verifiable sub-claims。prompt 采用 G-Defense `decomp+` 的约束：

```text
logical decomposition
causal / condition-result reasoning
hierarchical reasoning
fact-checkable propositions
attribution vs reality distinction
quantitative / temporal / comparative cues
who / what / when / where / why-how / consequence coverage
no outside facts
```

输出落为与现有 aspect coverage 兼容的缓存：

```text
claim_aspects.jsonl
raw_generations.jsonl
manifest.json
analysis.md
```

`claim_aspects.jsonl` 的 `extraction_version` 为 `llm_decomp_plus_v1`，aspect `source` 为 `llm_decomp_plus`，`type` 为 `llm_subclaim`。

脚本默认启用 `--guided-json`。实现同时兼容 vLLM 的两类接口：

```text
SamplingParams(guided_json=...)
SamplingParams(guided_decoding=GuidedDecodingParams(json=...))
```

目标服务器 vLLM 0.8.5 可走 guided JSON；若某个本地环境不支持，脚本会 warning 并回退到 prompt-only JSON parsing。

## 推荐命令

默认两步串跑：

```bash
bash scripts/selectors/run_llm_decomp_plus_aspect_coverage.sh
```

服务器多卡时：

```bash
TENSOR_PARALLEL_SIZE=4 \
QWEN_MODEL=/data/models/Qwen2.5-7B-Instruct \
ASPECT_ENCODER=BAAI/bge-base-en-v1.5 \
bash scripts/selectors/run_llm_decomp_plus_aspect_coverage.sh
```

先做 100 条 smoke：

```bash
SAMPLE_LIMIT=100 DEVICE=cuda bash scripts/selectors/run_llm_decomp_plus_aspect_coverage.sh
```

如果只生成 LLM sub-claims：

```bash
conda run -n cppo env PYTHONPATH=src python scripts/selectors/generate_llm_claim_decomp_aspects.py \
  --oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --split val \
  --output-dir outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_val \
  --model /data/models/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --generation-batch-size 128
```

随后可复用缓存跑不同 encoder：

```bash
conda run -n cppo env PYTHONPATH=src python scripts/selectors/analyze_oracle_aspect_coverage.py \
  --oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --split val \
  --output-dir outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_val_bge \
  --claim-aspects-input outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_val/claim_aspects.jsonl \
  --model-name BAAI/bge-base-en-v1.5 \
  --batch-size 128 \
  --max-length 128 \
  --device cuda
```

## Stop/Go

沿用现有 gate：

```text
uncovered_gain AUROC >= 0.57
或 oracle top5 coverage 比 hybrid top5 高 >= 3pp
```

若 LLM decomp+ 仍不过线，应停止 claim-aspect coverage 主线，不进入 selector 训练。

若过线，再实现 `targeted_feature_profile=aspect`，把 per-candidate aspect coverage / uncovered gain 特征接入 sequential selector。

## 已验证

当前交互环境不能初始化 CUDA vLLM，因此未本地跑 Qwen 生成；运行时应使用 `cppo` conda 环境。
目标服务器 vLLM 0.8.5 支持 guided JSON，正式运行会默认使用结构化解码。

已完成轻量验证：

```text
PYTHONPATH=src python -m unittest src/fact_checking/selectors/test_aspects.py
PYTHONPATH=src python -m unittest src/fact_checking/selectors/test_llm_decomp_parser.py
python -m compileall src/fact_checking/selectors/aspects.py scripts/selectors/generate_llm_claim_decomp_aspects.py scripts/selectors/analyze_oracle_aspect_coverage.py
conda run -n cppo env PYTHONPATH=src python scripts/selectors/generate_llm_claim_decomp_aspects.py --help
PYTHONPATH=src python scripts/selectors/analyze_oracle_aspect_coverage.py --help
bash -n scripts/selectors/run_llm_decomp_plus_aspect_coverage.sh
git diff --check
```
