# LIAR-RAW 事实核查构建-训练-推理流水线

本文档是当前项目 README 的中文版本。本仓库用于通过一套统一流程运行 LIAR-RAW 事实核查实验：

```text
build -> train -> infer
```

- `build`：从原始 LIAR-RAW 声明和报告中构建候选证据文件。
- `train`：基于构建阶段的输出微调事实核查模型。
- `infer`：通过兼容 OpenAI API 的服务运行已训练检查点，并保存评估指标。

## 项目结构

```text
.
├── configs/
│   ├── build/
│   ├── train/
│   ├── infer/
│   ├── pipeline/
│   └── experiment/
├── scripts/
│   └── pipeline/run_exp.sh
├── src/
│   ├── fact_checking/
│   │   ├── build/
│   │   ├── infer/
│   │   ├── pipeline/
│   │   ├── retrieval/
│   │   ├── data/
│   │   └── utils/
│   └── sft/
├── requirements.txt
└── pyproject.toml
```

## 环境准备

推荐使用 Python 3.10-3.11 和 CUDA 环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

依赖集合包含 PyTorch CUDA 12.4 wheels、Transformers、Accelerate、DeepSpeed、PEFT、Hydra 和 OmegaConf。

请先为训练环境配置一次 Accelerate：

```bash
accelerate config
```

## 一条命令运行

运行完整实验：

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0_2 pipeline.mode=full
```

也可以使用等价的封装脚本：

```bash
bash scripts/pipeline/run_exp.sh experiment=b0_2 pipeline.mode=full
```

只运行某一个阶段：

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=build
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=train
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=infer
```

默认情况下，流水线支持断点续跑。构建输出会按配置指纹缓存到：

```text
outputs/cache/build/<build_sha1>/
```

实验运行结果会保存到：

```text
outputs/runs/<experiment_name>/<run_sha1>/
```

每次运行都会写入一个 `manifest.json`，其中记录各阶段状态和产物路径。

## 配置

Hydra 配置组：

```text
configs/pipeline/default.yaml
configs/build/default.yaml
configs/train/default.yaml
configs/infer/vllm_api.yaml
configs/experiment/*.yaml
```

当前包含的实验变体：

```text
b0, b0_1, b0_2, b1, b1_1, b2
```

示例：

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b1 baseline.top_k=5
PYTHONPATH=src python -m fact_checking.pipeline.run -m experiment=b0,b1 baseline.top_k=5,10
```

## 阶段说明

构建阶段读取 `data/raw/LIAR-RAW/train.json`、`val.json` 和 `test.json`。对每条声明，它会将关联报告切分成候选句子，并结合稠密相似度、词汇重叠度和本地 BM25 近似结果进行打分，然后用 MMR 增加证据多样性，最终写出：

```text
build_train.jsonl
build_val.jsonl
build_test.jsonl
```

训练阶段会接收流水线生成并解析后的 SFT 配置路径，然后在独立进程中启动 Accelerate/DeepSpeed。

推理阶段使用配置中的检查点，启动或复用 vLLM 兼容 OpenAI 的服务，调用 `/v1/completions`，解析 LIAR-RAW 标签，并保存预测结果、指标和混淆矩阵。

## 常用覆盖项

强制重新运行某个阶段：

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.force.build=true
```

指定 GPU：

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b0_2 \
  train.cuda_visible_devices=0,1,2,3 \
  infer.cuda_visible_devices=4
```

连接到已经运行的推理服务：

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b0_2 \
  pipeline.mode=infer \
  infer.server.manage=false \
  infer.base_url=http://127.0.0.1:8000/v1
```
