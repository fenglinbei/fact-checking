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

### 配置参数参考

#### `pipeline` (`configs/pipeline/default.yaml`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | str | `full` | 运行模式：`full`（完整）、`build`（构建）、`train`（训练）、`infer`（推理） |
| `steps` | list | `[]` | 显式指定阶段列表（优先级高于 `mode`） |
| `resume` | bool | `true` | 断点续跑：复用已完成的阶段产物 |
| `force.build` | bool | `false` | 强制重新执行构建阶段 |
| `force.train` | bool | `false` | 强制重新执行训练阶段 |
| `force.infer` | bool | `false` | 强制重新执行推理阶段 |
| `output_root` | str | `outputs/runs` | 实验运行输出的根目录 |
| `cache_root` | str | `outputs/cache` | 缓存构建产物的根目录 |
| `run_dir` | str | `null` | 覆盖运行输出目录（用于一次性运行） |
| `runtime.env.*` | dict | NCCL/PyTorch 相关 | 注入子进程的环境变量 |

#### `build` (`configs/build/default.yaml`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data.train_path` | str | `data/raw/LIAR-RAW/train.json` | 训练集 JSON 文件路径 |
| `data.val_path` | str | `data/raw/LIAR-RAW/val.json` | 验证集 JSON 文件路径 |
| `data.test_path` | str | `data/raw/LIAR-RAW/test.json` | 测试集 JSON 文件路径 |
| `retrieval.embedder_model` | str | BGE 模型路径 | SentenceTransformer 嵌入模型路径 |
| `retrieval.device` | str | `cuda` | 嵌入计算设备：`cuda` 或 `cpu` |
| `retrieval.max_length` | int | `256` | 嵌入模型的最大 token 长度 |
| `retrieval.batch_size` | int | `64` | 嵌入编码的批次大小 |
| `retrieval.top_k` | int | `32` | 每条声明检索的候选句子数量 |
| `retrieval.alpha_dense` | float | `0.70` | 稠密（嵌入）相似度权重 |
| `retrieval.alpha_lexical` | float | `0.20` | 词汇重叠 F1 权重 |
| `retrieval.alpha_bm25` | float | `0.10` | BM25 近似评分权重 |
| `retrieval.mmr_lambda` | float | `0.70` | MMR 权衡：相关性 (λ) vs 多样性 (1-λ) |
| `retrieval.chunking.strategy` | str | `sentence` | 证据分块策略：`sentence` 或 `ctx_window` |
| `retrieval.chunking.context_k` | int | `1` | 上下文窗口的前后句子数（仅 `ctx_window` 策略） |

#### `train` (`configs/train/default.yaml`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | str | `accelerate_deepspeed` | 训练启动器：`accelerate_deepspeed` 或 `single` |
| `cuda_visible_devices` | str | `"0,1,2,3"` | 训练使用的 GPU 设备 ID |
| `nproc_per_node` | int | `4` | 每个节点的进程数 |
| `num_machines` | int | `1` | 机器数量 |
| `mixed_precision` | str | `bf16` | 混合精度模式：`bf16`、`fp16` 或 `no` |
| `deepspeed_config` | str | `configs/deepspeed_zero2.json` | DeepSpeed 配置 JSON 路径 |
| `checkpoint_for_infer` | str | `best` | 推理用的检查点选择：`best`（最佳）或 `last`（最后） |
| `run_dir` | str | `null` | 覆盖训练输出目录 |

#### `sft_train` (`configs/train/default.yaml`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_length` | int | `2048` | 最大序列长度（token 数，包含 prompt + target） |
| `per_device_train_batch_size` | int | `1` | 每个 GPU 的训练批次大小 |
| `per_device_eval_batch_size` | int | `1` | 每个 GPU 的评估批次大小 |
| `gradient_accumulation_steps` | int | `8` | 梯度累积步数 |
| `learning_rate` | float | `1.0e-5` | 峰值学习率 |
| `weight_decay` | float | `0.0` | 权重衰减系数 |
| `warmup_ratio` | float | `0.03` | 线性预热的步数比例 |
| `lr_scheduler_type` | str | `cosine` | 学习率调度：`cosine`、`linear`、`constant` |
| `num_train_epochs` | int | `2` | 训练轮数 |
| `bf16` | bool | `true` | 启用 bfloat16 混合精度 |
| `use_flash_attention_2` | bool | `true` | 启用 Flash Attention 2 |
| `gradient_checkpointing` | bool | `true` | 启用梯度检查点 |
| `padding` | str | `longest` | 填充策略：`longest` 或 `max_length` |
| `use_length_bucket` | bool | `true` | 启用 LengthGroupedSampler 高效批处理 |
| `logging_steps` | int | `2` | 每 N 步记录指标 |
| `save_steps` | int | `50` | 每 N 步保存检查点 |
| `eval_steps` | int | `50` | 每 N 步执行评估 |
| `dataloader_num_workers` | int | `4` | 数据加载的工作线程数 |
| `max_grad_norm` | float | `1.0` | 梯度裁剪的最大范数 |

**LoRA** (`sft_train.lora.*`)，在实验中按需启用：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lora.enabled` | bool | `false` | 启用 LoRA 微调 |
| `lora.r` | int | `16` | LoRA 秩 |
| `lora.alpha` | int | `32` | LoRA 缩放因子 |
| `lora.dropout` | float | `0.05` | LoRA dropout 率 |
| `lora.bias` | str | `none` | 偏置处理方式：`none`、`all`、`lora_only` |
| `lora.target_modules` | list | q/k/v/o/gate/up/down_proj | 应用 LoRA 的线性模块 |
| `lora.modules_to_save` | list | `null` | 额外全量微调的模块 |

#### `baseline` (`configs/experiment/*.yaml`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `variant` | str | `b0` | 实验变体标识 |
| `model_name_or_path` | str | Qwen2.5-7B-Instruct | HuggingFace 基础模型路径 |
| `top_k` | int | `5` | 放入 prompt 的证据条数 |
| `prompt_mode` | str | `zero_shot` | 提示模式 |
| `few_shot_k` | int | `16` | 少样本示例数量 |
| `few_shot_mmr_lambda` | float | `0.7` | 少样本选择时的 MMR λ 值 |
| `retrieval_model` | str | BGE 模型路径 | 检索用的嵌入模型 |
| `retrieval_batch_size` | int | `256` | 检索编码的批次大小 |
| `retrieval_max_length` | int | `256` | 检索编码的最大 token 长度 |
| `max_new_tokens` | int | `8` | 最大生成 token 数 |
| `temperature` | float | `0.0` | 生成温度 |
| `do_sample` | bool | `false` | 是否启用采样（否则贪心解码） |
| `output_mode` | str | `label_only` | 输出格式：`label_only` 或 `explanation_label` |
| `prompt_version` | str | `v1` | prompt 模板版本：`v1` 或 `v2` |
| `prompt_truncation.enabled` | bool | `false` | 启用证据截断以适配 max_length |
| `prompt_truncation.strategy` | str | `tail_evidence` | 截断策略（仅有 `tail_evidence`） |
| `prompt_truncation.min_evidence_to_keep` | int | `1` | 截断后保留的最少证据条数 |

#### `infer` (`configs/infer/vllm_api.yaml`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `provider` | str | `vllm_openai` | 推理后端 |
| `split` | str | `test` | 推理用的数据划分 |
| `checkpoint` | str | `best` | 加载的检查点：`best` 或 `last` |
| `served_model_name` | str | `fact-checking-sft` | API 服务器注册的模型名称 |
| `host` | str | `127.0.0.1` | vLLM 服务器主机 |
| `port` | int | `8000` | vLLM 服务器端口 |
| `base_url` | str | `null` | 完整基础 URL 覆盖（如远程服务器） |
| `wait_seconds` | int | `180` | 等待服务器就绪的最大秒数 |
| `request_timeout_seconds` | int | `120` | 每个 API 请求的超时秒数 |
| `log_predictions` | int | `5` | 记录到日志的预测样本数 |
| `cuda_visible_devices` | str | `"0"` | vLLM 服务器使用的 GPU |
| `tensor_parallel_size` | int | `1` | 张量并行的 GPU 数量 |
| `gpu_memory_utilization` | float | `0.90` | vLLM 使用的 GPU 内存比例 |
| `dtype` | str | `auto` | 模型数据类型：`auto`、`float16`、`bfloat16` |
| `max_model_len` | int | `null` | 最大模型上下文长度（`null` 使用模型默认值） |
| `max_new_tokens` | int | `null` | 覆盖 baseline 的 max_new_tokens（`null` 使用 baseline 值） |
| `temperature` | float | `null` | 覆盖 baseline 的 temperature（`null` 使用 baseline 值） |
| `server.manage` | bool | `true` | 自动管理 vLLM 服务器生命周期 |
| `server.stop_after_infer` | bool | `true` | 推理完成后关闭 vLLM 服务器 |
| `server.extra_args` | list | `[]` | 传递给 vLLM 启动命令的额外 CLI 参数 |

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
