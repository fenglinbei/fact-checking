# Oracle Evidence Selection 最优性能上界计算 — 代码计划

## Context

研究目标是学习 claim-adaptive 的 evidence diversity policy（learned-λ MMR / RL-MMR）。需要一个**理论上界**作为参照：固定 chunking + top-K + verifier，若完美 evidence selector 选出最优 K 个候选证据，verifier 能达到多高准确率？

现有 `scripts/learned_lambda/compute_oracle_lambda.py` 只在 λ 网格上搜索最优 λ，仍受 MMR greedy 约束。本计划直接搜索最优 K-subset。

## 核心思路

对每个 claim，从候选池 \(C = \{d_1, ..., d_N\}\) 中选出大小为 K 的子集 \(S_K\)，最大化 verifier 对正确标签的概率：

\[
S_K^* = \arg\max_{S \subseteq C, |S|=K} P_{\text{verifier}}(y^* \mid c, S)
\]

## 关键设计决策

### 1. 配置驱动的 Cache 自动解析

不再要求手动指定 `--chunk-cache-dir`。改为接受 Hydra experiment config 路径，自动：

1. 加载并解析配置（Hydra/OmegaConf 合并 experiment + build/default + pipeline/default）
2. 调用 `_chunk_mmr_config_fingerprint(cfg)` 计算缓存指纹
3. 按指纹查找 `outputs/cache/chunk_mmr/<sha1>/<split>.pkl`
4. **若缓存不存在，自动触发构建**：运行 pre-MMR + chunk-MMR 两个阶段，生成缓存
5. 若 pre-MMR 缓存也不存在，同样自动构建

缓存指纹计算（复用 `src/fact_checking/build/candidates.py:118-134`）：
```python
def _chunk_mmr_config_fingerprint(cfg: dict) -> str:
    retrieval = {k: cfg["retrieval"][k] for k in ("embedder_model", "device", "max_length", "precision")}
    payload = {
        "version": "chunk-text-embedding-v1",
        "data": cfg["data"],
        "retrieval": retrieval,
        "chunking": cfg["retrieval"]["chunking"],
    }
    return hashlib.sha1(stable_json(payload).encode()).hexdigest()[:12]
```

### 2. 与 b3 Pipeline 完全一致的 Prompt 构建

搜索过程中的 prompt 必须与训练/推理时的 prompt **完全相同**，包括分块策略、证据格式化、截断逻辑。

**默认配置**：`configs/experiment/b3_mmr_topk_sweep_1024.yaml`，关键参数：

| 参数 | 值 |
|---|---|
| `chunking.strategy` | `semantic` |
| `chunking.theta` | `0.5` |
| `prompt.max_length` | `1024` |
| `prompt.output_mode` | `label_only` |
| `prompt.label_format` | `letter` |
| `prompt.system_prompt` | `null`（使用默认） |

**Prompt 构建流程**（复用 `src/fact_checking/build/candidates.py` 中的函数）：

```
1. _build_system_message(None)
   → "You are a careful fact-checking assistant for LIAR-RAW claims. ..."

2. _build_user_content(claim, evidence_texts, output_mode="label_only", label_format="letter")
   → 构造包含 label 定义、规则、claim、evidence 的用户消息

3. _build_chat_prompt(tokenizer, system_msg, user_content)
   → 应用 Qwen2.5 ChatML template，生成最终 prompt

4. 若 prompt token 数 > max_length (1024)，调用 _auto_truncate_evidence()
   → 从尾部弹出低分证据 → 二分搜索截断最后一条证据
```

**最终 prompt 格式**（ChatML）：
```
<|im_start|>system
You are a careful fact-checking assistant for LIAR-RAW claims. Classify claims using only the claim and retrieved evidence supplied by the user.<|im_end|>
<|im_start|>user
Classify the claim into exactly one LIAR-RAW label.

Labels:
- A (pants-fire): completely false and implausible
- B (false): false based on the available evidence
- C (barely-true): mostly false, with only a small element of truth
- D (half-true): partly true and partly false
- E (mostly-true): mostly true, with minor missing context or caveats
- F (true): accurate based on the available evidence

Rules:
- Use the retrieved evidence as the primary source.
- Do not invent facts not supported by the evidence.
- Respond with exactly one line: Label: <a single letter from A-F>

Claim:
{claim_text}

Evidence:
[1] {evidence_1}
[2] {evidence_2}
...
<|im_end|>
<|im_start|>assistant
```

### 3. vLLM 离线推理

使用 `vllm.LLM` + `SamplingParams(prompt_logprobs=0)` 直接离线推理，与 `compute_oracle_lambda.py` 一致。不走 HTTP API server。

### 4. 模型路径

本机模型目录：`~/project/hateSpeechDetection/models/base/`。需要根据配置中的 model_name 映射到实际路径。

## 实现计划

### 文件 1：`scripts/oracle_evidence/search_optimal_evidence.py` — 主脚本

#### 命令行参数

```
--config               Hydra experiment 配置路径
                       （默认：configs/experiment/b3_mmr_topk_sweep_1024.yaml）
--config-overrides     额外的 Hydra 覆盖，逗号分隔
                       （例如：build.retrieval.top_k=5,build.retrieval.mmr_lambda=0.7）
--verifier-model       训练好的 verifier 模型路径（必需）
--lora-adapter         LoRA adapter 路径（可选，若 verifier 使用 LoRA）
--top-k                目标 evidence set 大小（默认 5）
--search-method        搜索方法：greedy / exhaustive / beam（默认 greedy）
--beam-width           beam search 宽度（默认 3）
--max-exhaustive-n     穷举搜索的最大候选池大小（默认 15）
--tensor-parallel-size GPU 数（默认 1）
--gpu-memory-utilization GPU 显存利用率（默认 0.95）
--max-model-len        最大序列长度（默认 1024）
--dtype                推理精度（默认 auto）
--score-batch-size     单次 llm.generate 的 prompt 数（默认 512）
--max-samples          最大处理样本数（默认全部）
--split                数据集划分：train / val / test（默认 val）
--output-dir           输出目录（默认 outputs/oracle_evidence/<timestamp>/）
--two-stage            启用两阶段剪枝（默认 true）
--two-stage-multiplier 两阶段保留倍数 M = top_k * m（默认 3）
```

#### 核心流程

```
1. 加载 Hydra 配置
   - 用 hydra.initialize_config_dir() + compose() 加载配置
   - 合并 --config + --config-overrides
   - OmegaConf.to_container(resolve=True) → 纯 dict

2. 解析模型路径
   - verifier_model: 从 --verifier-model 或配置的 train 部分推断
   - 若配置中的模型路径以 /data/models/ 开头，替换为实际路径
   - tokenizer 与 verifier 使用同一模型

3. 确保 chunk-MMR cache 存在
   a. 调用 _chunk_mmr_config_fingerprint(build_cfg) 计算指纹
   b. 检查 outputs/cache/chunk_mmr/<fingerprint>/<split>.pkl
   c. 若不存在：
      - 计算 pre-MMR fingerprint
      - 若 pre-MMR cache 也不存在，运行 _compute_pre_mmr_split()
      - 运行 _compute_chunk_mmr_split()
   d. 加载 ChunkMMRSample 列表

4. 搜索最优 evidence set（每个样本）
   for sample in chunk_mmr_samples:
       candidates = sample.candidates  # 候选池
       if two_stage:
           candidates = top_k * multiplier by hybrid_score
       result = search_fn(claim, candidates, top_k, gold_label, scorer)
       results.append(result)

5. 汇总指标并输出
```

#### 步骤 3 详细：Cache 自动构建

```python
def ensure_chunk_mmr_cache(build_cfg: dict, split: str) -> list[ChunkMMRSample]:
    """确保 chunk-MMR cache 存在，若不存在则自动构建"""
    chunk_fp = _chunk_mmr_config_fingerprint(build_cfg)
    chunk_dir = Path("outputs/cache/chunk_mmr") / chunk_fp
    cache_path = chunk_dir / f"{split}.pkl"
    
    if cache_path.exists():
        return _load_pickle(cache_path)
    
    # 确保 pre-MMR cache 存在
    pre_fp = _premmr_config_fingerprint(build_cfg)
    pre_dir = Path("outputs/cache/pre_mmr") / pre_fp
    pre_path = pre_dir / f"{split}.pkl"
    
    if not pre_path.exists():
        pre_dir.mkdir(parents=True, exist_ok=True)
        # 加载数据 + 计算句子/claim embedding
        samples = _compute_pre_mmr_split(build_cfg, split, pre_dir)
    else:
        samples = _load_pickle(pre_path)
    
    # 构建 chunk-MMR cache
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_samples = _compute_chunk_mmr_split(build_cfg, split, samples, chunk_dir)
    return chunk_samples
```

**关键复用**：
- `_compute_pre_mmr_batch()` (`candidates.py:989`) — 批量嵌入句子和 claim
- `_compute_chunk_mmr_batch()` (`candidates.py:326`) — 分块 + 批量嵌入 chunk
- `_build_chunk_candidate_rows()` (`candidates.py:254`) — 应用 SemanticChunking 策略
- `build_chunking_strategy()` (`chunking.py:415`) — 创建分块策略实例

### 文件 2：`src/fact_checking/oracle_evidence/scorer.py` — vLLM 离线评分

```python
@dataclass
class VerifierScorer:
    """封装 vLLM 离线评分，复用 compute_oracle_lambda.py 的模式"""
    llm: LLM
    tokenizer: AutoTokenizer
    label_token_ids: dict[str, int]       # {"A": token_id, ..., "F": token_id}
    prompt_sampling_params: SamplingParams  # max_tokens=1, prompt_logprobs=0
    gen_sampling_params: SamplingParams     # max_tokens=8, 用于最终预测
    
    def score_evidence_sets(
        self,
        claim: str,
        current_set: list[str],           # 已选证据文本列表
        candidate_texts: list[str],       # 候选证据文本列表
        gold_label_letter: str,           # 正确标签字母 "A"-"F"
    ) -> np.ndarray:
        """
        批量评分：对每个 (current_set + candidate) 组合，
        返回正确标签的 log-probability。
        
        内部步骤：
        1. 对每个候选构造完整 prompt（复用 _build_user_content + _build_chat_prompt）
        2. 拼接 "Label:" 后缀
        3. tokenize → prompt_token_ids
        4. llm.generate(prompt_token_ids=..., sampling_params=prompt_sampling_params)
        5. 从 prompt_logprobs[-1] 提取 gold_label_letter 对应的 logprob
        """
        ...
    
    def predict_label(self, prompt: str) -> int:
        """生成预测标签 id（用于最终评估）"""
        ...
```

**关键复用**：
- `_build_user_content()` (`candidates.py:703`) — 构造用户消息
- `_build_chat_prompt()` (`candidates.py:740`) — 应用 ChatML template
- `_build_system_message()` (`candidates.py:683`) — 系统消息
- `_extract_prompt_token_logprob()` 逻辑 (来自 `compute_oracle_lambda.py:218-242`)
- `_build_label_token_ids()` 逻辑 (来自 `compute_oracle_lambda.py:362`)

### 文件 3：`src/fact_checking/oracle_evidence/search.py` — 搜索算法

```python
@dataclass
class SearchResult:
    event_id: str
    claim: str
    gold_label: str
    gold_id: int
    n_candidates: int
    top_k: int
    selected_indices: list[int]
    selected_texts: list[str]
    final_logprob: float
    final_prediction: int
    is_correct: bool
    search_method: str
    search_steps: list[dict]

def greedy_search(
    claim: str,
    candidates: list[dict],
    top_k: int,
    gold_label_letter: str,
    scorer: VerifierScorer,
    build_prompt_fn: Callable,
    max_length: int = 1024,
) -> SearchResult: ...

def exhaustive_search(...) -> SearchResult: ...

def beam_search(...) -> SearchResult: ...
```

### 文件 4：`src/fact_checking/oracle_evidence/__init__.py`

```python
from .search import SearchResult, greedy_search, exhaustive_search, beam_search
from .scorer import VerifierScorer
```

### 文件 5：`scripts/oracle_evidence/run_search.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/experiment/b3_mmr_topk_sweep_1024.yaml}"
TOP_K="${TOP_K:-5}"
SEARCH_METHOD="${SEARCH_METHOD:-greedy}"
SPLIT="${SPLIT:-val}"
VERIFIER_MODEL="${VERIFIER_MODEL:-/data/models/Qwen2.5-7B-Instruct}"
TWO_STAGE="${TWO_STAGE:-true}"

python scripts/oracle_evidence/search_optimal_evidence.py \
  --config "$CONFIG" \
  --verifier-model "$VERIFIER_MODEL" \
  --top-k "$TOP_K" \
  --search-method "$SEARCH_METHOD" \
  --split "$SPLIT" \
  --two-stage "$TWO_STAGE" \
  "$@"
```

### 复用组件汇总

| 组件 | 来源 | 用途 |
|---|---|---|
| `_chunk_mmr_config_fingerprint()` | `candidates.py:118` | 计算 chunk-MMR 缓存指纹 |
| `_premmr_config_fingerprint()` | `candidates.py:101` | 计算 pre-MMR 缓存指纹 |
| `_compute_pre_mmr_batch()` | `candidates.py:989` | 批量嵌入句子+claim |
| `_compute_pre_mmr_split()` | `candidates.py:1087` | 单 split 的 pre-MMR 计算 |
| `_compute_chunk_mmr_batch()` | `candidates.py:326` | 批量嵌入 chunk |
| `_compute_chunk_mmr_split()` | `candidates.py:1198` | 单 split 的 chunk-MMR 计算 |
| `_build_chunk_candidate_rows()` | `candidates.py:254` | 应用分块策略构建候选 |
| `_build_system_message()` | `candidates.py:683` | 系统提示词 |
| `_build_user_content()` | `candidates.py:703` | 用户消息（含 claim+evidence） |
| `_build_chat_prompt()` | `candidates.py:740` | ChatML template |
| `_auto_truncate_evidence()` | `candidates.py:831` | 证据截断 |
| `_format_evidence_block()` | `candidates.py:689` | `[1] text\n[2] text\n...` |
| `_count_tokens()` | `candidates.py:672` | Token 计数 |
| `_build_label_token_ids()` 逻辑 | `compute_oracle_lambda.py:362` | label letter → token_id |
| `_extract_prompt_token_logprob()` 逻辑 | `compute_oracle_lambda.py:218` | 提取 logprob |
| `_compute_classification_metrics()` | `sft/metrics.py:4` | Accuracy/F1 |
| `build_chunking_strategy()` | `chunking.py:415` | 创建 SemanticChunking |
| `ChunkMMRSample` | `candidates.py:65` | 数据类型 |
| `TextEmbedder` | `retrieval/embedder.py:27` | 文本嵌入 |
| `LABEL_LETTERS`, `ID_TO_LABEL_LETTER` | `data/constants.py` | 标签映射 |

### 文件清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `scripts/oracle_evidence/search_optimal_evidence.py` | **新建** | 主搜索脚本 |
| `scripts/oracle_evidence/run_search.sh` | **新建** | Shell 启动脚本 |
| `src/fact_checking/oracle_evidence/__init__.py` | **新建** | 模块初始化 |
| `src/fact_checking/oracle_evidence/scorer.py` | **新建** | vLLM 离线批量评分 |
| `src/fact_checking/oracle_evidence/search.py` | **新建** | 搜索算法 |
| 现有文件 | **不修改** | 完全独立，import 复用 |

## 计算成本估算

### 硬件与模型配置

| 项目 | 值 |
|---|---|
| GPU | 4× NVIDIA L20 48GB |
| 显存利用率 | 0.95 |
| max_model_len | 1024 |
| 模型 | Qwen2.5-7B-Instruct（~14GB FP16） |
| 张量并行 | 4（跨 4 卡拆分模型） |

**单卡显存分配估算**（`max_model_len=1024`）：
- 模型权重：14GB / 4 ≈ 3.5 GB/卡
- 可用 KV cache：(48 × 0.95) - 3.5 ≈ 42 GB/卡
- Qwen2.5-7B 每 token 的 KV cache 约 4 KB（32 layers × 2 KV heads × 128 hidden × 2 bytes）
- 每条序列 KV cache：1024 × 4 KB ≈ 4 MB
- 单卡最大并发序列数：42 GB / 4 MB ≈ **10,000+ 条**
- 实际 vLLM 可批量数千条 prompt，吞吐瓶颈在 forward 计算而非显存

**vLLM 离线评分吞吐量估算**（`max_tokens=1, prompt_logprobs=0`）：
- Qwen2.5-7B prompt processing 在 L20 上约 50-100ms/forward（batch_size=512, seq_len≤1024）
- 保守估计：~800-1500 prompt/s
- 取保守值：**~800 prompt/s**

### 穷举搜索成本（搜索方法 = exhaustive）

穷举搜索须枚举 \(C(N, K)\) 个组合，每个组合构造一个 prompt 交 vLLM 评分。

| N（候选池大小） | C(N,5) 组合数 | 单样本耗时 | 100 样本耗时 | 1000 样本耗时 |
|---|---|---|---|---|
| 8 | 56 | < 0.1s | 7s | 1.2 min |
| 10 | 252 | 0.3s | 30s | 5 min |
| 12 | 792 | 1s | 1.7 min | 17 min |
| **15** | **3,003** | **3.8s** | **6.3 min** | **1 h** |
| 18 | 8,568 | 11s | 18 min | 3 h |
| 20 | 15,504 | 19s | 32 min | 5.4 h |
| 25 | 53,130 | 66s | 1.8 h | 18 h |
| 30 | 142,506 | 178s | 5 h | — |

**结论**：
- **N ≤ 15**：穷举完全可行，即使 1000 样本也仅需 ~1 小时
- **N ≤ 20**：小批量（≤200 样本）可行
- **N > 20**：不可行，必须用贪婪或 beam search

### 贪婪搜索成本（对比）

贪婪搜索每样本仅需 \(\sum_{i=0}^{K-1} (N-i) \approx NK\) 次评分，与 N 呈线性关系。

| N | 每样本评分数 | 1000 样本耗时 |
|---|---|---|
| 15 | ~63 | 1.3 min |
| 30 | ~138 | 2.9 min |
| 50 | ~235 | 4.9 min |
| 100 | ~485 | 10 min |

### 建议的搜索策略

```
if N <= 15:
    穷举搜索（精确最优）
elif N <= 30:
    beam search（beam_width=3，近似最优）
else:
    两阶段贪婪（先 hybrid_score 筛到 top-15，再贪婪搜索）
```

或统一使用混合策略：默认穷举 N≤15 的样本，其余用两阶段贪婪。

### 两阶段剪枝 + 贪婪搜索成本

| N（原始） | 筛后 M=15 | 每样本评分数 | 1000 样本耗时 |
|---|---|---|---|
| 30 | 15 | ~63 | 1.3 min |
| 50 | 15 | ~63 | 1.3 min |
| 100 | 15 | ~63 | 1.3 min |

**全量 val split（~1000 样本）最坏情况总耗时**：
- 穷举样本（N≤15，假设占 ~20%）：200 样本 × 3.8s = **12.7 min**
- 两阶段贪婪样本（N>15，占 ~80%）：800 样本 × 0.08s = **1.1 min**
- 总计：**~14 min**

首次运行构建 chunk-MMR cache（若不存在）：额外 ~10-15 分钟（需 GPU embedding + semantic chunking）。

## 验证步骤

### 本地验证（当前环境，无 GPU 要求）

1. **语法检查 + import 链路**：
   ```bash
   PYTHONPATH=src python -m compileall src/fact_checking/oracle_evidence/
   PYTHONPATH=src python -c "from fact_checking.oracle_evidence import VerifierScorer, greedy_search, SearchResult; print('imports OK')"
   ```
2. **配置加载 + 指纹计算测试**（纯 CPU，不需 GPU）：
   ```bash
   PYTHONPATH=src python scripts/oracle_evidence/search_optimal_evidence.py \
     --config configs/experiment/b3_mmr_topk_sweep_1024.yaml \
     --split val --max-samples 0 --verify-config-only
   ```
   增加 `--verify-config-only` 标志：仅加载配置、计算指纹、检查 cache 是否存在、打印候选池统计信息，不加载 vLLM。

3. **Prompt 一致性校验**（纯 CPU）：
   - 若 chunk-MMR cache 已存在：加载 cache，对 5 个样本用搜索脚本构造 prompt，与 b3 pipeline build 输出的 `prompt` 字段逐字节对比
   - 若 cache 不存在：先对比 `_build_user_content()` + `_build_chat_prompt()` 的输出与 pipeline build 产物的 `prompt` 字段

4. **候选池统计信息**：加载 chunk-MMR cache（若存在），输出每个 split 的候选池大小分布（min/P25/median/P75/P90/P95/max），确认 N≤15 的样本占比

### 目标服务器验证（4×L20 环境）

5. **小规模端到端测试**：`--max-samples 5 --split val`，验证 vLLM 加载 + 搜索 + 输出完整流程
6. **贪婪 vs 穷举 一致性**：对 N≤10 的样本同时跑贪婪和穷举，确认 greedy 也能找到最优解（oracle search 的合理性验证）
7. **全量运行**：`--split val`（~1000 样本），产出完整的 `oracle_results.jsonl` + `oracle_metrics.json`
8. **与 MMR baseline 对比**：计算 gap = oracle_accuracy - mmr_baseline_accuracy
