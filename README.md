# LIAR-RAW Fact-checking Pipeline (Stage A + Baseline)


## 1. 项目结构（统一后）

```text
.
├── configs/
│   ├── stage_a.yaml
│   ├── baseline_b0.yaml
│   ├── baseline_b1.yaml
│   └── deepspeed_zero3.json
├── scripts/
│   ├── run_stage_a.sh
│   ├── run_llm_baseline.py
│   ├── train_llm_baseline_sft.py
│   ├── train_llm_baseline_b0.sh
│   └── train_llm_baseline_b1.sh
├── src/fact_checking/
│   ├── retrieval/      # Stage A
│   ├── baselines/      # Baseline B0/B1
│   └── utils/
├── requirements.txt
└── README.md
```

---

## 2. 环境安装（含 accelerate）

> 推荐 Python 3.10~3.11 + CUDA 环境。

### 2.1 基础环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

> `requirements.txt` 已固定 PyTorch/torchvision/torchaudio 为 **CUDA 12.4 (`+cu124`)** 版本，
> 并通过 `https://download.pytorch.org/whl/cu124` 获取对应 wheel，以避免误装 CUDA 13.x 相关包。

### 2.2 accelerate 安装与配置

```bash
pip install -U "accelerate>=1.1" "deepspeed>=0.15"
accelerate config
```

`accelerate config` 建议：
- compute environment: `LOCAL_MACHINE`
- distributed type: 单卡选 `NO`，多卡选 `MULTI_GPU`
- mixed precision: 推荐 `bf16`（硬件支持时）

可检查是否安装成功：

```bash
accelerate env
```

### 2.3 （可选）FlashAttention2

`configs/baseline_b0.yaml` / `configs/baseline_b1.yaml` 默认 `sft_train.use_flash_attention_2: true`。  
若环境没有安装 `flash-attn`，训练脚本会自动回退到默认 attention 实现并打印 warning。

如需启用 FlashAttention2，可在匹配 CUDA/PyTorch 版本的前提下安装：

```bash
pip install flash-attn --no-build-isolation
```

若不需要 FlashAttention2，也可直接在配置中关闭：

```yaml
sft_train:
  use_flash_attention_2: false
```

### 2.4 （可选）FLA fast path（`fla` + `causal-conv1d`）

有些模型（尤其是带自定义 `trust_remote_code` 的实现）会额外尝试加载  
`flash-linear-attention`（Python 包名通常是 `fla`）和 `causal-conv1d`。  
这和 FlashAttention2（`flash-attn`）是两条独立依赖链，二者不能互相替代。

建议安装顺序（先装 `causal-conv1d`，再装 `fla`）：

```bash
# 建议先确认与当前 torch/cuda 对齐
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY

# 1) 安装 causal-conv1d
pip install -U causal-conv1d --no-build-isolation

# 2) 安装 flash-linear-attention (fla)
pip install -U fla --no-build-isolation
```

若编译失败，通常是编译工具链或 CUDA 环境不匹配，建议先检查：

```bash
nvcc --version
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)
PY
```

可用下面脚本做最小验证：

```bash
python - <<'PY'
ok = True
try:
    import causal_conv1d  # noqa: F401
    print("causal-conv1d import OK")
except Exception as e:
    ok = False
    print("causal-conv1d import FAIL:", e)

try:
    import fla  # noqa: F401
    print("fla import OK")
except Exception as e:
    ok = False
    print("fla import FAIL:", e)

print("ALL_OK =", ok)
PY
```

---

## 3. Stage A 实现逻辑

Stage A 是**冻结式检索**，不训练检索器；输入 claim + reports，输出每条 claim 的 top-k 候选证据句。

### 3.1 输入与预处理

每条样本来自原始 LIAR-RAW 的 claim 记录：
- claim 文本
- label
- reports 列表（每条 report 含 content）

Stage A 会：
1. 从 `reports[*].content` 做句子切分。
2. 汇总成该 claim 的句子候选池。

### 3.2 打分与融合

对每个候选句子，计算：
- `dense_score`：query / passage encoder 的稠密相似度；
- `lexical_score`：词法重叠分数；
- `bm25_score`：本地 BM25-like 分数。

融合公式：

```text
hybrid = alpha_dense * dense_score
       + alpha_lexical * lexical_score
       + alpha_bm25 * bm25_score
```

默认即：

```text
hybrid = 0.70 * dense + 0.20 * lexical + 0.10 * bm25
```

### 3.3 MMR 去冗余

在融合分数基础上执行 MMR（Maximum Marginal Relevance）：
- 保留与 claim 高相关句子；
- 抑制重复、近重复句子；
- 最终返回 `top_k` 句子。

### 3.4 输出

按 split 产出 JSONL（如 `stage_a_train.jsonl`），每条样本含：
- claim 元信息；
- `candidates`（每个候选含 report_id、sent_idx、text、各类分数、link/domain 等）。

运行：

```bash
bash scripts/run_stage_a.sh
# 或
PYTHONPATH=src python -m fact_checking.retrieval.build_stage_a --config configs/stage_a.yaml
```

---

## 4. Stage A / Baseline B0 / B1 配置参数说明

### 4.1 `configs/stage_a.yaml`

- `output_dir`: Stage A 输出目录。
- `data.train_path|val_path|test_path`: 原始数据 JSON 路径。
- `retrieval.embedder_model`: 句向量模型路径/名称（如 BGE）。
- `retrieval.device`: 检索设备（`cuda`/`cpu`）。
- `retrieval.max_length`: encoder 截断长度。
- `retrieval.batch_size`: 编码批大小。
- `retrieval.top_k`: 每条 claim 保留候选句数量。
- `retrieval.alpha_dense|alpha_lexical|alpha_bm25`: 三类分数融合权重（建议和为 1）。
- `retrieval.mmr_lambda`: MMR 的相关性/多样性平衡（越大越偏相关性）。

### 4.2 `configs/baseline_b0.yaml`

`baseline_b0` 为 sentence-only 提示构造：`claim + top-k sentences -> label`。

- `output_dir`: SFT 运行根目录。每次训练会自动落到 `output_dir/<experiment_name>_<timestamp>/`。
- `wandb.*`: W&B 开关与项目配置。
- `data.train_candidates|val_candidates|test_candidates`: Stage A 产出的 JSONL。
- `baseline.variant`: 实验名。SFT 优先使用该值；若缺省，则使用当前时间戳作为实验名。
- `baseline.model_name_or_path`: LLM 路径。
- `baseline.top_k`: 每条 claim 使用前 k 条候选句。
- `baseline.use_context`: B0 固定为 `false`（不拼接句子上下文）。
- `baseline.context_k`: 上下文窗口大小（B0 通常不生效）。
- `baseline.prompt_mode`: `zero_shot`/`few_shot` 等。
- `baseline.few_shot_k`: few-shot 检索示例数。
- `baseline.few_shot_mmr_lambda`: few-shot 示例去重 MMR 系数。
- `baseline.retrieval_model`: few-shot 检索 embedding 模型。
- `baseline.retrieval_batch_size|max_length`: few-shot 检索编码参数。
- `baseline.max_new_tokens|temperature|do_sample`: 生成控制参数。
- `sft_train.tokenized_cache_dir`: 预分词缓存目录（可选）。未配置时自动写入当前 run 目录下的 `tokenized_cache/`，避免不同实验互相污染。
- `sft_train.*`: SFT 阶段训练参数（batch、梯度累积、学习率、epoch、warmup、bf16、gradient checkpointing、scheduler、clip 等）。

### 4.3 `configs/baseline_b1.yaml`

`baseline_b1` 为 sentence + report context：`claim + top-k sentences(+/-k context) -> label`。

除与 B0 相同参数外，关键区别：
- `baseline.use_context: true`
- `baseline.context_k`: 控制每个命中句前后拼接的上下文句数（通常 1）。

运行（推理）：

```bash
PYTHONPATH=src python scripts/run_llm_baseline.py --config configs/baseline_b0.yaml --split test
PYTHONPATH=src python scripts/run_llm_baseline.py --config configs/baseline_b1.yaml --split test
```

运行（SFT）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_llm_baseline_sft.py --config configs/baseline_b0.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_llm_baseline_sft.py --config configs/baseline_b1.yaml
```

### 4.4 用 ZeRO-3 `ds_checkpoint` 导出的 best 模型跑 test eval（B0）

当 `deepspeed_zero3` 且未开启 `stage3_gather_16bit_weights_on_model_save` 时，`best/` 下通常只有 tokenizer 与 `ds_checkpoint/`，不能直接被 `AutoModelForCausalLM.from_pretrained()` 读取。需要先把 ZeRO 分片权重聚合为 fp32 权重，再放入一个可被 HuggingFace 读取的目录。

假设你的 best 目录是（示例）：

```text
outputs/liar-raw/llm_baseline/b0_20260422-101500/best
```

可按如下步骤：

```bash
# 1) 从 DeepSpeed 分片聚合出 fp32 权重（会生成 pytorch_model.bin）
python outputs/liar-raw/llm_baseline/b0_20260422-101500/best/ds_checkpoint/zero_to_fp32.py \
  outputs/liar-raw/llm_baseline/b0_20260422-101500/best/ds_checkpoint \
  outputs/liar-raw/llm_baseline/b0_20260422-101500/best/pytorch_model.bin

# 2) 补齐 config（从训练基座模型拷贝）
python - <<'PY'
from transformers import AutoConfig
base_model = "/data/models/Qwen2.5-7B-Instruct"  # 改成你训练时 baseline.model_name_or_path
out_dir = "outputs/liar-raw/llm_baseline/b0_20260422-101500/best"
cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
cfg.save_pretrained(out_dir)
print("saved config.json to", out_dir)
PY

# 3) 临时覆盖配置里的 baseline.model_name_or_path 指向 best 目录
cp configs/baseline_b0.yaml configs/eval/baseline_b0_eval_best.yaml
python - <<'PY'
import yaml
p = "configs/eval/baseline_b0_eval_best.yaml"
cfg = yaml.safe_load(open(p, "r", encoding="utf-8"))
cfg["baseline"]["model_name_or_path"] = "outputs/liar-raw/llm_baseline/b0_20260422-101500/best"
yaml.safe_dump(cfg, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print("patched", p)
PY

# 4) 跑 test 推理
PYTHONPATH=src python scripts/run_llm_baseline.py --config configs/eval/baseline_b0_eval_best.yaml --split test
```

输出文件默认在：

```text
outputs/liar-raw/llm_baseline/b0_test.predictions.jsonl
```

可再用你自己的评测脚本统计 accuracy / macro-F1。

---

## 5. Stage A / Baseline 快速命令

### Stage A

```bash
bash scripts/run_stage_a.sh
```

## 6. 注意事项

1. 该流水线是 oracle-free 训练思路，训练信号以 claim 级标签为主。
2. Stage A 是冻结检索，不是可训练 dense retriever。
3. SFT 训练 run 目录按 `<baseline.variant 或时间戳>_<timestamp>` 自动创建。
