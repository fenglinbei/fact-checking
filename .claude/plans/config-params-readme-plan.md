# Plan: Add Config Parameter Documentation to README

## Goal
In both `README.md` and `README.zh-CN.md`, add a comprehensive reference section documenting all configurable parameters from the YAML config files.

## Changes

### 1. README.md — Add "Configuration Reference" section after existing "Configuration" section

Add subsections for each config group:

#### `pipeline` parameters (from `configs/pipeline/default.yaml`)
- `mode` — full | build | train | infer
- `steps` — explicit ordered list of phases to run
- `resume` — enable resumable runs (bool)
- `force.build/train/infer` — force re-run individual phases
- `output_root` — root directory for experiment run outputs
- `cache_root` — root directory for cached build artifacts
- `run_dir` — override run output directory
- `runtime.env.*` — environment variables injected into subprocess

#### `build` parameters (from `configs/build/default.yaml`)
- `data.train_path` / `val_path` / `test_path` — LIAR-RAW JSON split paths
- `retrieval.embedder_model` — embedding model path (BGE)
- `retrieval.device` — cuda | cpu
- `retrieval.max_length` — embedder max token length
- `retrieval.batch_size` — embedder batch size
- `retrieval.top_k` — number of candidate sentences to retrieve per claim
- `retrieval.alpha_dense` — dense similarity weight
- `retrieval.alpha_lexical` — lexical overlap F1 weight
- `retrieval.alpha_bm25` — BM25 score weight
- `retrieval.mmr_lambda` — MMR relevance vs diversity tradeoff

#### `train` parameters (from `configs/train/default.yaml`)
- `backend` — accelerate_deepspeed | single
- `cuda_visible_devices` — GPU device IDs
- `nproc_per_node` — number of processes per node
- `num_machines` — number of machines
- `mixed_precision` — bf16 | fp16 | no
- `deepspeed_config` — path to DeepSpeed JSON config
- `checkpoint_for_infer` — best | last
- `run_dir` — override training output directory

#### `sft_train` parameters (from `configs/train/default.yaml` + experiment overrides)
- `max_length` — max sequence length in tokens
- `per_device_train_batch_size` / `per_device_eval_batch_size`
- `gradient_accumulation_steps`
- `learning_rate` / `weight_decay` / `warmup_ratio` / `lr_scheduler_type`
- `num_train_epochs`
- `bf16` / `use_flash_attention_2` / `gradient_checkpointing`
- `padding` — longest | max_length
- `use_length_bucket` — enable LengthGroupedSampler
- `logging_steps` / `save_steps` / `eval_steps`
- `dataloader_num_workers`
- `lora.*` — LoRA fine-tuning params (enabled, r, alpha, dropout, bias, target_modules, modules_to_save)

#### `baseline` parameters (from experiment configs)
- `variant` — experiment variant name
- `model_name_or_path` — base model path
- `top_k` — number of evidence items in prompt
- `use_context` — include context window around matched sentences
- `context_k` — number of surrounding sentences in context window
- `prompt_mode` — zero_shot
- `few_shot_k` — number of few-shot examples
- `few_shot_mmr_lambda` — MMR lambda for few-shot selection
- `retrieval_model` / `retrieval_batch_size` / `retrieval_max_length`
- `max_new_tokens` — max tokens to generate
- `temperature` / `do_sample`
- `output_mode` — label_only | explanation_label
- `prompt_version` — v1 | v2
- `prompt_truncation.enabled` — enable evidence truncation
- `prompt_truncation.strategy` — tail_evidence
- `prompt_truncation.min_evidence_to_keep` — minimum evidence items to keep

#### `infer` parameters (from `configs/infer/vllm_api.yaml`)
- `provider` — vllm_openai
- `split` — test
- `checkpoint` — best | last
- `served_model_name` — model name for API
- `host` / `port` / `base_url`
- `wait_seconds` / `request_timeout_seconds`
- `cuda_visible_devices` / `tensor_parallel_size` / `gpu_memory_utilization`
- `dtype` — auto | float16 | bfloat16
- `max_model_len` / `max_new_tokens` / `temperature`
- `server.manage` — manage vLLM server lifecycle
- `server.stop_after_infer` — shutdown server after inference
- `server.extra_args` — extra CLI args passed to vLLM

### 2. README.zh-CN.md — Same structure, Chinese translations

Add the same "配置参数参考" section, with all parameter descriptions translated to Chinese.

### Files to change
1. `README.md` — add ~80 lines of parameter docs
2. `README.zh-CN.md` — add ~80 lines of parameter docs (Chinese)
