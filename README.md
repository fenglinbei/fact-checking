# LIAR-RAW Fact-Checking Build-Train-Infer Pipeline

This repository runs LIAR-RAW fact checking experiments through one unified flow:

```text
build -> train -> infer
```

- `build`: construct candidate evidence files from raw LIAR-RAW claims and reports.
- `train`: fine-tune the fact-checking model from the build outputs.
- `infer`: run the trained checkpoint through an OpenAI-compatible API server and save metrics.

## Project Layout

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

## Environment

Recommended: Python 3.10-3.11 with CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

The dependency set includes PyTorch CUDA 12.4 wheels, Transformers, Accelerate,
DeepSpeed, PEFT, Hydra, and OmegaConf.

Configure Accelerate once for the training environment:

```bash
accelerate config
```

## One-Command Runs

Run a full experiment:

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0_2 pipeline.mode=full
```

Equivalent wrapper:

```bash
bash scripts/pipeline/run_exp.sh experiment=b0_2 pipeline.mode=full
```

Run a single phase:

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=build
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=train
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=infer
```

By default the pipeline is resumable. Build outputs are cached by configuration
fingerprint under:

```text
outputs/cache/build/<build_sha1>/
```

Experiment runs are stored under:

```text
outputs/runs/<experiment_name>/<run_sha1>/
```

Each run writes a `manifest.json` with phase status and artifact paths.

## Configuration

Hydra config groups:

```text
configs/pipeline/default.yaml
configs/build/default.yaml
configs/train/default.yaml
configs/infer/vllm_api.yaml
configs/experiment/*.yaml
```

Experiment variants currently include:

```text
b0, b0_1, b0_2, b1, b1_1, b2
```

Examples:

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b1 baseline.top_k=5
PYTHONPATH=src python -m fact_checking.pipeline.run -m experiment=b0,b1 baseline.top_k=5,10
```

### Configuration Reference

#### `pipeline` (`configs/pipeline/default.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `full` | Run mode: `full`, `build`, `train`, or `infer` |
| `steps` | list | `[]` | Explicit ordered list of phases (overrides `mode`) |
| `resume` | bool | `true` | Reuse completed phases from previous runs |
| `force.build` | bool | `false` | Force re-run the build phase |
| `force.train` | bool | `false` | Force re-run the training phase |
| `force.infer` | bool | `false` | Force re-run the inference phase |
| `output_root` | str | `outputs/runs` | Root directory for experiment run outputs |
| `cache_root` | str | `outputs/cache` | Root directory for cached build artifacts |
| `run_dir` | str | `null` | Override run output directory (one-off runs) |
| `runtime.env.*` | dict | NCCL/PyTorch env | Environment variables injected into subprocesses |

#### `build` (`configs/build/default.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data.train_path` | str | `data/raw/LIAR-RAW/train.json` | Path to training split JSON |
| `data.val_path` | str | `data/raw/LIAR-RAW/val.json` | Path to validation split JSON |
| `data.test_path` | str | `data/raw/LIAR-RAW/test.json` | Path to test split JSON |
| `retrieval.embedder_model` | str | BGE model path | Path to SentenceTransformer embedding model |
| `retrieval.device` | str | `cuda` | Device for embedding: `cuda` or `cpu` |
| `retrieval.max_length` | int | `256` | Max token length for the embedder |
| `retrieval.batch_size` | int | `64` | Batch size for embedding encoding |
| `retrieval.top_k` | int | `32` | Number of candidate sentences to retrieve per claim |
| `retrieval.alpha_dense` | float | `0.70` | Dense (embedding) similarity weight |
| `retrieval.alpha_lexical` | float | `0.20` | Lexical overlap F1 weight |
| `retrieval.alpha_bm25` | float | `0.10` | BM25-like score weight |
| `retrieval.mmr_lambda` | float | `0.70` | MMR tradeoff: relevance (λ) vs diversity (1-λ) |

#### `train` (`configs/train/default.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | `accelerate_deepspeed` | Training launcher: `accelerate_deepspeed` or `single` |
| `cuda_visible_devices` | str | `"0,1,2,3"` | GPU device IDs for training |
| `nproc_per_node` | int | `4` | Number of processes per node |
| `num_machines` | int | `1` | Number of machines |
| `mixed_precision` | str | `bf16` | Mixed precision mode: `bf16`, `fp16`, or `no` |
| `deepspeed_config` | str | `configs/deepspeed_zero2.json` | Path to DeepSpeed config JSON |
| `checkpoint_for_infer` | str | `best` | Checkpoint selection: `best` or `last` |
| `run_dir` | str | `null` | Override training output directory |

#### `sft_train` (`configs/train/default.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_length` | int | `2048` | Max sequence length in tokens (prompt + target) |
| `per_device_train_batch_size` | int | `1` | Training batch size per GPU |
| `per_device_eval_batch_size` | int | `1` | Evaluation batch size per GPU |
| `gradient_accumulation_steps` | int | `8` | Number of gradient accumulation steps |
| `learning_rate` | float | `1.0e-5` | Peak learning rate |
| `weight_decay` | float | `0.0` | Weight decay coefficient |
| `warmup_ratio` | float | `0.03` | Fraction of steps for linear warmup |
| `lr_scheduler_type` | str | `cosine` | LR schedule: `cosine`, `linear`, `constant` |
| `num_train_epochs` | int | `2` | Number of training epochs |
| `bf16` | bool | `true` | Enable bfloat16 mixed precision |
| `use_flash_attention_2` | bool | `true` | Enable Flash Attention 2 |
| `gradient_checkpointing` | bool | `true` | Enable gradient checkpointing |
| `padding` | str | `longest` | Padding strategy: `longest` or `max_length` |
| `use_length_bucket` | bool | `true` | Enable LengthGroupedSampler for efficient batching |
| `logging_steps` | int | `2` | Log metrics every N steps |
| `save_steps` | int | `50` | Save checkpoint every N steps |
| `eval_steps` | int | `50` | Evaluate every N steps |
| `dataloader_num_workers` | int | `4` | Number of dataloader workers |
| `max_grad_norm` | float | `1.0` | Max gradient norm for clipping |

**LoRA** (`sft_train.lora.*`), enabled per experiment:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora.enabled` | bool | `false` | Enable LoRA fine-tuning |
| `lora.r` | int | `16` | LoRA rank |
| `lora.alpha` | int | `32` | LoRA scaling factor |
| `lora.dropout` | float | `0.05` | LoRA dropout rate |
| `lora.bias` | str | `none` | Bias treatment: `none`, `all`, `lora_only` |
| `lora.target_modules` | list | q/k/v/o/gate/up/down_proj | Linear modules to apply LoRA |
| `lora.modules_to_save` | list | `null` | Extra modules to fully fine-tune |

#### `baseline` (`configs/experiment/*.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variant` | str | `b0` | Experiment variant identifier |
| `model_name_or_path` | str | Qwen2.5-7B-Instruct | Base HuggingFace model path |
| `top_k` | int | `5` | Number of evidence items included in the prompt |
| `use_context` | bool | `false` | Include context sentences around each match |
| `context_k` | int | `1` | Number of sentences before/after for context window |
| `prompt_mode` | str | `zero_shot` | Prompting mode |
| `few_shot_k` | int | `16` | Number of few-shot examples in prompt |
| `few_shot_mmr_lambda` | float | `0.7` | MMR lambda for few-shot example selection |
| `retrieval_model` | str | BGE model path | Embedding model for retrieval |
| `retrieval_batch_size` | int | `256` | Batch size for retrieval encoding |
| `retrieval_max_length` | int | `256` | Max token length for retrieval encoding |
| `max_new_tokens` | int | `8` | Max tokens to generate |
| `temperature` | float | `0.0` | Generation temperature |
| `do_sample` | bool | `false` | Enable sampling (vs greedy decoding) |
| `output_mode` | str | `label_only` | Output format: `label_only` or `explanation_label` |
| `prompt_version` | str | `v1` | Prompt template version: `v1` or `v2` |
| `prompt_truncation.enabled` | bool | `false` | Enable evidence truncation to fit max_length |
| `prompt_truncation.strategy` | str | `tail_evidence` | Truncation strategy (only `tail_evidence`) |
| `prompt_truncation.min_evidence_to_keep` | int | `1` | Min evidence items to preserve after truncation |

#### `infer` (`configs/infer/vllm_api.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | `vllm_openai` | Inference backend provider |
| `split` | str | `test` | Data split for inference |
| `checkpoint` | str | `best` | Checkpoint to load: `best` or `last` |
| `served_model_name` | str | `fact-checking-sft` | Model name registered with the API server |
| `host` | str | `127.0.0.1` | vLLM server host |
| `port` | int | `8000` | vLLM server port |
| `base_url` | str | `null` | Full base URL override (e.g. remote server) |
| `wait_seconds` | int | `180` | Max wait time for server to become ready |
| `request_timeout_seconds` | int | `120` | Timeout per API request |
| `log_predictions` | int | `5` | Number of predictions to log for inspection |
| `cuda_visible_devices` | str | `"0"` | GPU for the vLLM server |
| `tensor_parallel_size` | int | `1` | Number of GPUs for tensor parallelism |
| `gpu_memory_utilization` | float | `0.90` | Fraction of GPU memory used by vLLM |
| `dtype` | str | `auto` | Model dtype: `auto`, `float16`, `bfloat16` |
| `max_model_len` | int | `null` | Max model context length (`null` = model default) |
| `max_new_tokens` | int | `null` | Override baseline max_new_tokens (`null` = use baseline) |
| `temperature` | float | `null` | Override baseline temperature (`null` = use baseline) |
| `server.manage` | bool | `true` | Manage vLLM server lifecycle automatically |
| `server.stop_after_infer` | bool | `true` | Shutdown vLLM server after inference |
| `server.extra_args` | list | `[]` | Extra CLI flags passed to vLLM launch command |

## Phase Details

The build phase reads `data/raw/LIAR-RAW/train.json`, `val.json`, and
`test.json`. For each claim it splits the associated reports into candidate
sentences, scores them with dense similarity, lexical overlap, and a local BM25
approximation, applies MMR for diversity, and writes:

```text
build_train.jsonl
build_val.jsonl
build_test.jsonl
```

The train phase receives those paths through a resolved SFT config generated by
the pipeline, then launches Accelerate/DeepSpeed in a separate process.

The infer phase uses the configured checkpoint, starts or reuses a vLLM
OpenAI-compatible server, calls `/v1/completions`, parses LIAR-RAW labels, and
saves predictions, metrics, and confusion matrices.

## Useful Overrides

Force a phase to rerun:

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.force.build=true
```

Use specific GPUs:

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b0_2 \
  train.cuda_visible_devices=0,1,2,3 \
  infer.cuda_visible_devices=4
```

Connect to an already running inference server:

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b0_2 \
  pipeline.mode=infer \
  infer.server.manage=false \
  infer.base_url=http://127.0.0.1:8000/v1
```
