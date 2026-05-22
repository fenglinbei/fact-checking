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

## 2026-05-22 同步结果核查

已核查：

```text
outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_val/
outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_val_coverage/
```

结果显示 coverage 未过线：

```text
uncovered_gain AUROC = 0.4760
oracle_vs_hybrid_coverage_lift_pp = -1.45
claims_with_no_local_aspects = 886 / 1274
```

但这不能作为 LLM decomp+ 路线的真实 no-go。生成缓存本身严重失效：

```text
parse_failed = 655 / 1274
fewer_than_min_subclaims = 179 / 1274
ok = 439 / 1274
n_local_aspects = 629
```

人工抽查 `raw_generations.jsonl` 后发现，大量 `ok` 行只是 JSON 解析成功，但 `sub_claims` 内容是 prompt/schema 残片或乱码，例如包含 `Logical decomposition`、`news claim`、`decompose`、`JSON`、`sub{` 等，而不是可核查子论断。按启发式粗筛，629 个 local aspects 中约 463 个带明显生成残片，只有约 111 条 claim 至少有 1 个看起来可用的 sub-claim。

因此当前结论是：

```text
不要进入 selector 训练；
也不要把这次 coverage no-go 视为 LLM decomp+ 语义路线失败；
需要先重跑 claim decomposition cache。
```

脚本已补充生成质量过滤：

```text
filter_valid_subclaims(...)
invalid_subclaims / fewer_than_min_valid_subclaims 状态
rejected_subclaims 记录
默认 max_tokens 从 512 下调到 256
wrapper 支持 GUIDED_JSON=0 显式关闭 guided JSON
```

建议下一次先跑 100 条对比：

```bash
SAMPLE_LIMIT=100 GUIDED_JSON=1 bash scripts/selectors/run_llm_decomp_plus_aspect_coverage.sh
SAMPLE_LIMIT=100 GUIDED_JSON=0 DECOMP_OUTPUT_DIR=outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_val_sample100_plain COVERAGE_OUTPUT_DIR=outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_val_sample100_plain_coverage bash scripts/selectors/run_llm_decomp_plus_aspect_coverage.sh
```

只有当生成缓存满足以下最低质量线时，coverage 指标才可作为 stop/go 依据：

```text
parse_failed <= 5%
claims_with_no_local_aspects <= 10%
valid local aspects mean >= 2.0
人工抽查 30 条中明显 prompt/schema 残片 <= 2 条
```

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
