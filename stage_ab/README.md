# Stage AB: Baseline B0 / B1

新增了两个基于 LLM 的 baseline：

- **B0 (Sentence-only)**: `claim + top-k sentences -> classifier`
- **B1 (Sentence + report context)**: `claim + top-k sentences(+/-k context) -> classifier`

## 配置

- `configs/baseline_b0.yaml`
- `configs/baseline_b1.yaml`
- `configs/deepspeed_zero3.json`

默认模型路径：

- 分类模型：`/data/models/Qwen3.5-9B`
- few-shot 检索 embedding：`/home/fenglin/project/models/bge-base-en-v1.5/`

## 运行 baseline 测试

```bash
cd stage_ab
PYTHONPATH=src python scripts/run_llm_baseline.py --config configs/baseline_b0.yaml --split test
PYTHONPATH=src python scripts/run_llm_baseline.py --config configs/baseline_b1.yaml --split test
```

也可直接用封装脚本：

```bash
bash scripts/predict_llm_baseline_b0.sh
bash scripts/predict_llm_baseline_b1.sh
```

## SFT 训练（L20 四卡 + DeepSpeed ZeRO-3）

```bash
cd stage_ab
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_llm_baseline_sft.py --config configs/baseline_b0.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_llm_baseline_sft.py --config configs/baseline_b1.yaml
```

或直接：

```bash
bash scripts/train_llm_baseline_b0.sh
bash scripts/train_llm_baseline_b1.sh
```

> few-shot 模式下，会先用 train claims 建索引，然后用 embedding + MMR 检索 top-10 相似案例拼进 prompt。
