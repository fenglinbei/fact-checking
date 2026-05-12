# `generate_oracle_prompts.py` 执行逻辑

该脚本是 learned-lambda 流程的第 1 步：先显式读取或按签名构建 PreMMR 缓存，再复用 claim/sentence embedding，针对一组固定 `lambda` 值分别重新运行 MMR 证据选择，并为每个 `lambda` 生成一份 build JSONL prompt 文件。后续 `compute_oracle_lambda.py` 会读取这些 per-lambda prompt 文件，计算每条 claim 的 oracle `lambda`。

## 典型用法

```bash
bash scripts/learned_lambda/run_generate_oracle_prompts.sh
```

该启动脚本默认复用 `configs/experiment/b3_mmr_topk_sweep_1024.yaml` 的 build 策略，覆盖 `top_k=12`，重新构建当前 split 的 PreMMR cache，并在后续 prompt 生成阶段按签名自动读取：

```bash
PYTHONPATH=src python scripts/learned_lambda/generate_oracle_prompts.py \
  --experiment b3_mmr_topk_sweep_1024 \
  --rebuild-premmr-cache \
  --output-dir outputs/learned_lambda/prompts/ \
  --top-k 12
```

## 输入参数

- `--premmr-cache`：可选。PreMMR pickle 路径，例如 `train.pkl`。不传时会根据 `--experiment` 解析出的 build 配置计算签名，并自动定位 `outputs/cache/pre_mmr/<fingerprint>/<split>.pkl`。
- `--premmr-cache-root`：可选。签名化 PreMMR cache 根目录，默认 `outputs/cache/pre_mmr`。
- `--rebuild-premmr-cache`：可选。生成 prompt 前重新构建当前 `--split-name` 对应的签名化 PreMMR cache。
- `--output-dir`：必填。per-lambda JSONL 输出目录，不存在时会自动创建。
- `--experiment`：可选。Hydra experiment 名称。提供后会从 `pipeline/default` 合成配置，并复用其中的 `build.retrieval` 与 `build.prompt`。
- `--config-overrides`：可选。与 `--experiment` 一起使用的额外 Hydra override，例如 `build.retrieval.chunking.theta=0.6`。
- `--model-name-or-path`：可选。用于构造 chat prompt 和统计 token 数的 tokenizer 路径；会覆盖 `build.prompt.model_name_or_path`。如果没有提供 `--experiment`，则必须显式传入。
- `--top-k`：MMR 后保留的证据数量；会覆盖 `build.retrieval.top_k`，无配置时默认 `16`。
- `--alpha-dense`、`--alpha-lexical`、`--alpha-bm25`：dense、lexical overlap、BM25-like 三类检索分数的混合权重；会覆盖 `build.retrieval` 中的同名配置，无配置时默认分别为 `0.70`、`0.20`、`0.10`。
- `--cpu-workers`：CPU 后处理线程数；会覆盖 `build.retrieval.cpu_workers`，无配置时默认 `1`。
- `--lambda-grid`：需要枚举的 MMR `lambda` 网格，默认 `0.00,0.05,...,1.00`。
- `--prompt-max-length`：prompt 与 target 的总长度预算；会覆盖 `build.prompt.max_length`，无配置时默认 `1024`。
- `--prompt-output-mode`：prompt 输出模式；会覆盖 `build.prompt.output_mode`，无配置时默认 `label_only`。
- `--prompt-label-format`：标签格式；会覆盖 `build.prompt.label_format`，无配置时默认 `letter`，即将六分类标签映射为 `A-F`。
- `--split-name`：输出文件名中的 split 后缀，默认 `train`，可选 `train`、`val`、`test`。
- `--no-progress`：可选。关闭 lambda 网格和样本级 MMR/prompt 生成进度条。

## 主流程

1. 解析命令行参数；如果提供 `--experiment`，通过 Hydra compose 加载完整 pipeline 配置，并取出 `build` 配置。
2. 将 `--lambda-grid` 按逗号拆分成浮点数列表。
3. 创建 `--output-dir`。
4. 合并 build 策略：
   - `top_k`、`alpha_dense`、`alpha_lexical`、`alpha_bm25`、`cpu_workers` 来自 `build.retrieval`，命令行参数优先。
   - prompt 配置来自 `build.prompt`，命令行参数优先。
   - chunking 策略通过 `build_chunking_strategy(build.retrieval.chunking, build.retrieval)` 构造。
5. 如果 chunking 策略的 embedder 配置与 PreMMR retrieval 配置兼容，则在 chunking 阶段复用 PreMMR 中的 sentence embedding。
6. 如果没有显式传入 `--premmr-cache`，或传入了 `--rebuild-premmr-cache`，则根据 build 配置计算 PreMMR fingerprint，并确保 `outputs/cache/pre_mmr/<fingerprint>/<split>.pkl` 存在；`--rebuild-premmr-cache` 会先删除该 split 的旧缓存再重建。
7. 通过 `_load_pickle()` 读取最终确定的 PreMMR cache，得到 `pre_samples`。
8. 通过 `_load_prompt_tokenizer()` 加载 tokenizer；如果 tokenizer 没有 `pad_token`，会将其设置为 `eos_token`。
9. 遍历每个 `lambda`，显示 lambda 网格进度条，并调用 `_mmr_phase_from_premmr()` 生成对应 JSONL；每个 `lambda` 内部会显示样本级 MMR/prompt 生成进度条：

```text
{output_dir}/lambda_{lambda:.2f}_{split_name}.jsonl
```

例如：

```text
outputs/learned_lambda/prompts/lambda_0.70_train.jsonl
```

## 单个 lambda 的处理逻辑

`_mmr_phase_from_premmr()` 会逐条处理 `PreMMRSample`：

1. 将缓存中的 sentence dict 恢复为 `SentenceRecord`。
2. 从缓存读取 `sent_emb` 和 `claim_emb`，不重新运行 embedding 模型。
3. 计算 dense score：

```python
dense_scores = sent_emb @ claim_emb
```

4. 对 claim 与每个 sentence 计算 lexical overlap 和 BM25-like score。
5. 对 dense、lexical、BM25-like 三组分数分别做 min-max scaling。
6. 按权重计算混合分数：

```python
hybrid_scores = (
    alpha_dense * dense_scaled
    + alpha_lexical * lexical_scaled
    + alpha_bm25 * bm25_scaled
)
```

7. 调用 `maximal_marginal_relevance()`，使用当前 `lambda` 和 `hybrid_scores` 选择最多 `top_k` 条候选证据。
8. 对每个候选句子使用配置中的 chunking 策略生成证据文本；无 `--experiment` 时默认等价于 `SentenceChunking`。
9. 按 canonicalized evidence text 去重；如果重复，保留 `hybrid_score` 更高的候选。
10. 将候选按 `hybrid_score` 降序排序，并截断到 `top_k`。
11. 调用 `_build_training_row()` 将检索结果转成最终 JSONL 行。

## Prompt 构造逻辑

默认配置为 `label_only + letter`：

- user prompt 要求模型将 claim 分到一个 LIAR-RAW 标签。
- 标签以 `A-F` 显示，同时保留对应原始标签名和定义。
- 输出格式要求为一行：

```text
Label: <a single letter from A-F>
```

target 会按金标标签构造，例如：

```text
Label: B
```

prompt 使用 tokenizer 的 `apply_chat_template(..., add_generation_prompt=True)` 生成，与主 build 流程保持一致。

## 自动截断逻辑

因为 `auto_length=True`，脚本会控制 prompt 长度：

1. 先计算 target token 数。
2. 将 `prompt_max_length - target_token_count` 作为 prompt budget。
3. 如果 prompt 超过 budget，优先从证据列表尾部删除证据；尾部通常是排序后分数较低的证据。
4. 如果只剩一条证据仍超预算，则对这条证据做 token 级二分截断。
5. 如果证据完全放不下，则退化为 no-evidence prompt。

输出行中会记录 `prompt_token_count`、`target_token_count`、`evidence_count`、`evidence_count_before`、`was_truncated`、`evidence_text_truncated` 等字段。

## 输出 JSONL 字段

每行对应一条 claim，在每个 lambda 文件中保持一行一条样本。主要字段包括：

- `event_id`
- `claim`
- `label`
- `explain`
- `candidates`
- `prompt`
- `target`
- `gold_label`
- `gold_id`
- `gold_explain`
- `prompt_add_special_tokens`
- `preserve_prompt_prefix`
- `prompt_token_count`
- `target_token_count`
- `evidence_count`
- `evidence_count_before`
- `was_truncated`
- `evidence_text_truncated`

## 当前实现注意点

- 脚本可以消费显式传入的 PreMMR cache，也可以通过 `--rebuild-premmr-cache` 复用 build 配置重新生成签名化 PreMMR cache。
- 显式传入 `--premmr-cache` 时，`split-name` 只影响输出文件名，不会校验该 cache 实际来自哪个 split；自动签名缓存模式下，`split-name` 同时用于选择和构建对应 split 的 PreMMR cache。
- 输出文件同名时会被覆盖。
- `lambda` 文件名使用 `{lambda:.2f}`，可以覆盖默认 0.05 搜索粒度而不发生一位小数四舍五入冲突。
- `--config-overrides` 只能与 `--experiment` 一起使用。
- 如果复用 semantic chunking，建议确保 `--premmr-cache` 与该 experiment 的 `build.retrieval.embedder_model`、`build.retrieval.max_length` 一致，否则 chunking 阶段可能无法复用 PreMMR embedding。
