# Experiment Results Index

日期：2026-06-12  
工作目录：`/data/liaozijie/fact-checking`

本文是当前 checkout 下已运行结果的索引，不重新解释所有实验结论。目标是回答三个问题：

1. 每类结果从哪里来。
2. 主要设置是什么。
3. 当前应该把哪些目录当作可复用结果、哪些只是缓存、半成品或归档。

## 索引口径

本次扫描范围：

- `outputs/`
- `data/processed/coverage/source_coverage_v2_flash/`
- 现有说明文档 `docs/`

判定规则：

- `complete`：有 `manifest.json` 且对应 phase 完成，或有 `training_complete.json`，或有完整 split summary。
- `partial`：有 resolved config / checkpoint / eval step 指标，但缺少完成标记或最终 best/test 指标。
- `cache`：构建候选、MMR、pre-MMR、selector 中间表等输入产物，不单独当作实验结论。
- `archive`：历史迁移或已归档结果，只在追溯时使用。

指标说明：

- 表中 `best` 多数取当前可见 infer/test 或 infer/val 的最高 `macro_f1`。这只是索引定位信号，不等于采用规则。
- `sentence_trace_method` 的 `test_best_seen` 是 test split 上可见的最高 selection / macro-F1 版本；真实采用仍以对应文档中的 val selection 规则为准。
- 没有 `metrics.json` 的目录不代表没跑过，可能只是 build-only、cache-only 或 LLM teacher 数据。

## 顶层分组

| 分组 | 主要目录 | 状态 | 来源/入口 | 用途 |
|---|---|---|---|---|
| Hydra pipeline / SFT 主线 | `outputs/runs/` | 混合：完整、build-only、infer-only 都有 | `src/fact_checking/pipeline/run.py`，`configs/experiment/*.yaml`，`scripts/pipeline/run_exp.sh` | 早期 b0/b3/b4、MMR sweep、selector trace full pipeline、RawFC downstream |
| Sentence trace LoRA 主线 | `outputs/sentence_trace_method/` | 多数关键 LoRA 已完成，若干 fullFT/新矩阵为 partial | `scripts/sentence_trace_method/*`，`src/sft/*` | Llama/Qwen + LIAR-RAW/RAWFC 的 label-token LoRA、logit adjust、coverage subset retrain |
| Source coverage 数据质量 | `outputs/data_quality/`，`data/processed/coverage/source_coverage_v2_flash/` | `source_coverage_v2_flash` complete；旧 `source_coverage` 不建议复用 | `scripts/phase11_data_quality/rerun_coverage_flash.sh`，`tag_source_coverage.py` | 生成 coverage sidecar 和物化训练子集 |
| Selector / evidence selection | `outputs/selectors/`，`outputs/selector_trace_verifier/` | 大量 completed eval 和 step-level 指标；`outputs/selectors/selectors/` 中已清理确认相同的重复副本，剩余为根目录缺失或内容不同的文件 | `scripts/phase5_selectors/*`，`scripts/selector_trace_verifier/*` | evidence map、chain graph、cross/list/sequential selector、LLM action selector |
| Oracle evidence / oracle verifier | `outputs/oracle_evidence/`，`outputs/oracle_pointwise/`，`outputs/oracle_direct_verifier/` | completed/partial 混合 | `docs/D-oracle-evidence/*` 对应脚本 | oracle evidence set、pointwise selector、oracle direct verifier 上界 |
| MMR / learned lambda / RL MMR | `outputs/learned_lambda*`，`outputs/rl_mmr/` | 多数是模型/策略训练产物或缓存 | `docs/C-mmr-learned-lambda/*` | learned-lambda、sensitivity-gated、DPO step-wise lambda |
| Phase10 chunking ablation | `outputs/phase10_chunking_ablation/` | RawFC 多个 completed；LIAR-RAW 有 partial | `docs/A-chunking-infra/*`，phase10 scripts | raw/sentence/semantic chunking 粒度对照 |
| Build/cache 输入 | `outputs/cache/` | cache-only | pipeline build cache / chunk MMR cache | 供训练、infer、selector 复用，不直接作为结论 |
| 可视化/临时分析 | `outputs/analysis/` | analysis artifacts | `docs/E-selectors/evidence-map-selector-comparison-web.md` | evidence map HTML、translation sidecars、retrieval signal ablation |
| 历史归档 | `outputs/archive/` | archive | backbone migration run | 仅追溯使用 |

## 当前可复用结果

优先复用：

- `outputs/data_quality/source_coverage_flash/{liar_raw,rawfc}/`
- `data/processed/coverage/source_coverage_v2_flash/{liar_raw,rawfc}/{all,covered,covered_weak}/`
- `outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw`
- `outputs/sentence_trace_method/rawfc__llama31_8b_lora_halfbatch_ep8_eval100_pat8_rawfc`
- `outputs/runs/heuristic_lambda_mmr_fullft/484dc7dca0ad`
- `outputs/runs/b3_selector_trace_full_pipeline/*`
- `outputs/selectors/evidence_map_selector/` 与 `outputs/selectors/evidence_chain_graph/` 中 v0.6/v0.7 系列，用于 selector 对比和 Web UI。

不建议直接复用：

- `outputs/data_quality/source_coverage/{liar_raw,rawfc}/`：旧 DeepSeek/API 配置错误版本，LLM review 不可靠，保留作历史对照。
- `outputs/sentence_trace_method/*` 中只有 `train.resolved.yaml`、没有 `training_complete.json` 且无 best eval 的目录：这些是配置草稿、fullFT partial 或未完成矩阵。
- `outputs/cache/build/<hash>/`：只是 build candidate cache，需要通过 run manifest 的 `build_id` 追溯。
- `outputs/selectors/selectors/`：同步错误留下的嵌套输出根。本次已删除与 `outputs/selectors/<same-relative-path>` 完全一致的 1476 个副本；剩余 128 个文件根目录缺失对应文件或同名但内容不同，暂时保留。

## Hydra Pipeline Results (`outputs/runs`)

| group | runs | completed full | source/settings | best visible infer metric |
|---|---:|---:|---|---|
| `b0` | 2 | 0 | `experiment=b0`，Qwen2.5-7B，top_k=5，zero-shot；build-only | 无 infer 指标 |
| `b3_label_token_ce_1024` | 1 | 1 | semantic chunking，Qwen2.5-7B LoRA，max_length=1024，label-token CE，无 logit adjust | val acc `0.3006` / macro-F1 `0.3015` |
| `b3_mmr_topk_sweep_1024` | 9 | 9 | semantic，MMR `lambda=0.7`，top_k `0..8`，Qwen2.5 LoRA，logit adjust | best visible val acc `0.2967` / macro-F1 `0.3003` |
| `mmr_lambda_sweep` | 5 | 4 | 2048 context，lambda sweep，Qwen2.5 LoRA | best visible val acc `0.2991` / macro-F1 `0.2961` |
| `mmr_lambda_sweep_1024` | 6 | 6 | 1024 context，lambda sweep，Qwen2.5 LoRA | best visible test acc `0.2470` / macro-F1 `0.2279` |
| `mmr_topk_sweep_infer` | 17 | 17 | infer-only top_k sweep，复用 best checkpoint | best visible test acc `0.2766` / macro-F1 `0.2729` |
| `heuristic_lambda_mmr` | 2 | 2 | semantic，heuristic lambda，Qwen2.5 LoRA | test acc `0.2766` / macro-F1 `0.2799` |
| `heuristic_lambda_mmr_fullft` | 1 | 1 | semantic，heuristic lambda，Qwen2.5 fullFT | test acc `0.3110` / macro-F1 `0.3215` |
| `mmr_sensitivity_gated` | 1 | 1 | sensitivity gated lambda，Qwen2.5 LoRA | test acc `0.2742` / macro-F1 `0.2795` |
| `reranker_only` | 1 | 1 | semantic reranker-only 对照，Qwen2.5 LoRA | test acc `0.2694` / macro-F1 `0.2745` |
| `b4_3class` | 1 | 1 | ModernBERT-large fullFT，3-class collapse 诊断 | test acc `0.4365` / macro-F1 `0.4021` |
| `b4_mmr_lambda_sweep` | 11 | 0 | ModernBERT/fullFT lambda sweep，resolved configs 和 train eval 为主 | 无完整 infer 指标 |
| `b3_pointwise_oracle_selector_1024` | 3 | 2 | pointwise oracle selector downstream，semantic，Qwen2.5 LoRA | best val acc `0.2582` / macro-F1 `0.2582` |
| `b3_pointwise_oracle_selector_v1b_1024` | 1 | 1 | pointwise V1b true-side anchor | val acc `0.2630` / macro-F1 `0.2632` |
| `b3_pointwise_stage2_sentence_1024` | 1 | 1 | sentence-level pointwise stage2 | test acc `0.2614` / macro-F1 `0.2515` |
| `b3_raw_top_evidence_hybrid_1024_2gpu` | 1 | 1 | raw top-evidence hybrid order，Qwen2.5 LoRA | test acc `0.2574` / macro-F1 `0.2351` |
| `b3_raw_top_evidence_original_order_1024_2gpu` | 1 | 1 | raw top-evidence original order | test acc `0.2542` / macro-F1 `0.2271` |
| `b3_oracle_direct_order_sensitivity` | 8 | 8 | oracle direct verifier evidence order sensitivity | best val acc `0.6327` / macro-F1 `0.6430` |
| `b3_oracle_direct_verifier_val_evidence_checks` | 4 | 4 | oracle direct verifier 的 fixed-MMR / pointwise evidence check | best val acc `0.2716` / macro-F1 `0.2663` |
| `b3_selector_trace_full_pipeline` | 11 | 11 | selector trace full pipeline，hybrid/v0.6/v0.6d/v0.6e/fullFT 等 | best val acc `0.3454` / macro-F1 `0.3546` |
| `rawfc_v0_6c_selector_trace_full_pipeline` | 1 | 1 | RAWFC v0.6c closed evidence downstream | test acc `0.5800` / macro-F1 `0.5510` |
| `v0_6c_rawfc3_rule_step_adaptive5_10*` | 2 | 0 | RAWFC closed evidence build/config roots | manifest only / no infer |

来源文档：

- `docs/Z-cross-cutting/202605201437_experiment_progress_timeline.md`
- `docs/C-mmr-learned-lambda/*`
- `docs/D-oracle-evidence/*`
- `docs/E-selectors/*`
- `docs/Z-cross-cutting/v0_7_lora_group1_vs_baseline.md`

## Sentence Trace Method Results

公共设置：

- 目录：`outputs/sentence_trace_method/`
- 主要入口：`scripts/sentence_trace_method/run_lora_matrix.sh`、`run_lora_label_token_logit_adjust_eval_only.sh`、`label_token_infer.py`
- 主要模型：`/data/models/Meta-Llama-3.1-8B-Instruct` 与 `/data/models/Qwen3-4B-Instruct-2507`
- 训练范式：label-token CE，LoRA `r=16/alpha=32/dropout=0.05`，ZeRO-2，bf16，FlashAttention。
- LIAR-RAW tuned LoRA 默认采用 val 选择的 `tau=0.5`；test 上更高的 tau 只作 post-hoc 参考。

| run | status | schema/model | key settings | visible test result |
|---|---|---|---|---|
| `liar_raw__llama31_8b_lora` | complete | LIAR6 / Llama-3.1-8B | LoRA，GA=8，5 epochs，eval=50，uniform class weights | `label_token` test acc `0.3030` / macro-F1 `0.2985` / selection `0.6742` |
| `liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw` | complete; current LIAR baseline | LIAR6 / Llama-3.1-8B | LoRA，GA=4，8 epochs，eval=100，LIAR class weights；eval-only tau sweep | adopted default: val-selected `tau=0.5`; visible test best `tau0p75` acc `0.3125` / macro-F1 `0.3241` / selection `0.7224` |
| `liar_raw__llama31_8b__v0_7_bm_lora_halfbatch_ep8_eval100_pat8` | complete | LIAR6 / Llama-3.1-8B | v0.7 budgeted marginal source，LoRA halfbatch | test acc `0.3022` / macro-F1 `0.3115` / selection `0.6949` |
| `liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw` | complete; coverage subset first round | LIAR6 / Llama-3.1-8B | train on `covered_weak` rows, eval on full val/test, eval=50, tau sweep | visible post-hoc test `tau0p5` acc `0.3157` / macro-F1 `0.3251` / selection `0.7217`; pre-registered selected `tau0` did not beat old baseline |
| `liar_raw__qwen3_4b_2507_lora` | complete | LIAR6 / Qwen3-4B-2507 | LoRA，GA=8，5 epochs | test acc `0.2190` / macro-F1 `0.1964` / selection `0.5131` |
| `liar_raw__qwen3_4b_2507__v0_7_bm_lora_halfbatch_ep8_eval100_pat8` | complete | LIAR6 / Qwen3-4B-2507 | v0.7 budgeted marginal，LoRA halfbatch | test acc `0.2174` / macro-F1 `0.1867` / selection `0.4908` |
| `rawfc__llama31_8b_lora` | complete | RAWFC3 / Llama-3.1-8B | LoRA，GA=8，5 epochs | test acc `0.5850` / macro-F1 `0.5872` / selection `0.9664` |
| `rawfc__llama31_8b_lora_halfbatch_ep8_eval100_pat8_rawfc` | complete; current RAWFC Llama LoRA | RAWFC3 / Llama-3.1-8B | LoRA，GA=4，8 epochs，eval=100，uniform RAWFC weights | test acc `0.6250` / macro-F1 `0.6258` / selection `1.0126` |
| `rawfc__llama31_8b__v0_7_bm_lora_halfbatch_ep8_eval100_pat8` | complete | RAWFC3 / Llama-3.1-8B | v0.7 budgeted marginal，LoRA halfbatch | test acc `0.6000` / macro-F1 `0.6001` / selection `0.9870` |
| `rawfc__qwen3_4b_2507_lora` | complete | RAWFC3 / Qwen3-4B-2507 | LoRA，GA=8，5 epochs | test acc `0.5600` / macro-F1 `0.5650` / selection `0.9319` |
| `rawfc__qwen3_4b_2507__v0_7_bm_lora_halfbatch_ep8_eval100_pat8` | complete | RAWFC3 / Qwen3-4B-2507 | v0.7 budgeted marginal，LoRA halfbatch | test acc `0.5950` / macro-F1 `0.5950` / selection `0.9731` |
| `*_fullFT` 或无 `_lora` 的 sentence_trace dirs | partial/config roots | LIAR6/RAWFC3 | 多数只有 `train.resolved.yaml`，无 `training_complete.json` | 只作配置来源，不作为完成结果 |

来源文档：

- `docs/B-classifier-collapse/202606101830_sentence_trace_lora_logit_adjust_tuning.md`
- `docs/B-classifier-collapse/20260611_coverage_v2_flash_coveredweak_first_round.md`

## Source Coverage Results

可用版本是 `source_coverage_v2_flash`。

| root | dataset | split | rows | covered | weak_covered | uncovered | status |
|---|---|---:|---:|---:|---:|---:|---|
| `outputs/data_quality/source_coverage_flash/liar_raw` | LIAR-RAW | train | 10065 | 2037 | 3126 | 4902 | complete |
| `outputs/data_quality/source_coverage_flash/liar_raw` | LIAR-RAW | val | 1274 | 320 | 431 | 523 | complete |
| `outputs/data_quality/source_coverage_flash/liar_raw` | LIAR-RAW | test | 1251 | 321 | 432 | 498 | complete |
| `outputs/data_quality/source_coverage_flash/rawfc` | RAWFC | train | 1612 | 109 | 340 | 1163 | complete |
| `outputs/data_quality/source_coverage_flash/rawfc` | RAWFC | val | 200 | 18 | 45 | 137 | complete |
| `outputs/data_quality/source_coverage_flash/rawfc` | RAWFC | test | 200 | 19 | 39 | 142 | complete |

物化数据：

- `data/processed/coverage/source_coverage_v2_flash/liar_raw/{all,covered,covered_weak}/{train,val,test}.json`
- `data/processed/coverage/source_coverage_v2_flash/rawfc/{all,covered,covered_weak}/{train,val,test}.json`
- `materialization_summary.json` 保留每个数据集的行数与分布校验。

旧版本保留：

- `outputs/data_quality/source_coverage/{liar_raw,rawfc}/`
- `outputs/data_quality/source_coverage/liar_raw_bge/`
- `outputs/data_quality/source_coverage/liar_raw_debug/`

这些旧版本可用于方法对比，但不要作为后续 retraining 默认输入。

来源文档：

- `docs/Z-cross-cutting/202606110040_source_coverage_v2_flash_quality_report.md`

## Selector / Evidence Selection Results

`outputs/selectors/` 是最大混乱源之一。当前结构同时包含正式输出、smoke/debug 输出、step-level eval，以及同步错误留下的 `outputs/selectors/selectors/`。已清理可证明完全相同的嵌套副本；剩余嵌套文件不是确认重复项，建议只读追溯，不直接清理。

| family | output root | metric files | representative visible result | source docs |
|---|---|---:|---|---|
| aspect coverage | `outputs/selectors/aspect_coverage` | 4 summaries | LLM decomposition / rule aspect coverage 诊断；已在文档中标为 No-Go | `docs/F-feature-diagnostics/*` |
| count amplified stance bucket | `outputs/selectors/count_amplified_stance_bucket_selector` | 10 | smoke/full val selector comparisons；代表 recall@5 约 `0.37` | `docs/E-selectors/202605171708_oracle_observation_count_amplified_bucket_selector.md` |
| direct evidence cross encoder | `outputs/selectors/direct_evidence_cross_encoder` | 11 | v0.4a/v0.4d smoke 与 full eval；小样本最高 recall@5 约 `0.48`，不可当 full-val 结论 | `docs/E-selectors/202605272342_direct_evidence_cross_encoder_v04_execution_and_v04a_plan.md` |
| evidence map selector | `outputs/selectors/evidence_map_selector` | 14 | v0.5/v0.6 evidence-map teacher 和 selector metrics；full val recall@5 约 `0.3694` | `docs/E-selectors/202605281545_evidence_map_verifier_v05b_checkpoint_diagnostic.md` |
| evidence chain graph | `outputs/selectors/evidence_chain_graph` | 28 diagnostics | v0.6b/v0.6c/v0.6d/v0.7 chain graph traces，LIAR/RAWFC train/val/test | `docs/E-selectors/202605291359_v05c_prompt_evidence_diagnostic_analysis.md` |
| question decomp retrieval | `outputs/selectors/question_decomp_retrieval` | 12 | qwen/dense/deepseek question-decomposition retrieval traces | `docs/F-feature-diagnostics/202605212359_llm_decomp_plus_aspect_coverage.md` |
| LLM action selector | `outputs/selectors/llm_action_selector` | 207 | Qwen2.5-3B action selector step eval；大量 during-train metrics | `docs/E-selectors/202605270134_llm_action_selector_utility_pairwise_report.md` |
| stage2 cross/list/sequential | `outputs/selectors/stage2_sentence_*` | 26 | cross-encoder/listwise/sequential selector；train eval recall@5 可到 `0.4214`，val 仍需按文档判断 | `docs/E-selectors/202605202145_sequential_pointer_selector_step4.md` |
| utility listwise / VIG | `outputs/selectors/utility_listwise`，`outputs/selectors/vig_utility` | 10 | utility-listwise 和 saved-score VIG-lite ranker | `docs/E-selectors/202605271419_utility_listwise_v0_v01_archive.md` |

`outputs/selector_trace_verifier/`：

- 约 670 个 `metrics.json`，主要是 method-upgrade / RAWFC / backbone 对比的 verifier step eval。
- 代表性最高可见结果来自 `rawfc_v0_6c_eval25_backbone/v0_6c_rawfc3_rule_step_adaptive5_10_eval25_ministral3_8b_lora/eval/step-225/metrics.json`，acc `0.6850` / macro-F1 `0.6881`。
- 这些指标多为 step-level 训练曲线，不应与 full pipeline infer 指标直接混比。

## Oracle Results

| family | root | status | representative result | source |
|---|---|---|---|---|
| Oracle evidence | `outputs/oracle_evidence/` | complete/partial 混合 | `rawfc_qwen3_4b_2507_fullpool_margin` 的 RAWFC oracle metrics，val acc `0.9050` / macro-F1 `0.9055`；train shard 更高但不可和 val/test 混用 | `docs/D-oracle-evidence/*` |
| Oracle pointwise | `outputs/oracle_pointwise/` | completed logreg / eval metrics | `v1`、`v1b`、`stage2_margin_sentence` pointwise selector/logreg | `docs/E-selectors/202605171203_oracle_pointwise_supervision_v1.md` |
| Oracle direct verifier | `outputs/oracle_direct_verifier/` | train eval complete，order sensitivity configs complete | stage2 sentence direct verifier train eval step-600 acc `0.7125` / macro-F1 `0.7183`；downstream order sensitivity 在 `outputs/runs/b3_oracle_direct_order_sensitivity/` | `docs/D-oracle-evidence/202605192010_oracle_sentence_direct_verifier.md` |

## Learned Lambda / RL MMR Results

| group | root | setting | visible result / role | source |
|---|---|---|---|---|
| learned lambda baseline | `outputs/learned_lambda/` | chunk-embedding regression, candidate pool full, prompt_top_k=5 | comparison summary: fixed `0.70` val macro-F1 `0.3060` vs oracle `0.3384`, delta `+0.0324` | `docs/C-mmr-learned-lambda/202605141045_Improving_Learned_Lambda.md` |
| learned lambda variants | `outputs/learned_lambda_cls`、`*_coarse`、`*_high_margin`、`*_soft`、`*_v2` | classifier/regression/soft variants; each has `training_meta.json` and `predictor.pt` | predictor artifacts only; not downstream wins by themselves | `docs/C-mmr-learned-lambda/*` |
| RL MMR DPO | `outputs/rl_mmr/dpo_stepwise/` | step-wise lambda trajectories, preference pairs, DPO checkpoints | cache/checkpoints for DPO policy; scalar lambda route later marked weak | `docs/C-mmr-learned-lambda/202605151936_dpo_step_wise_lambda.md` |
| sensitivity / soft label | `outputs/rl_mmr/sensitivity_search`，`outputs/rl_mmr/soft_label` | sensitivity-gated and soft-label lambda predictors | supports `outputs/runs/mmr_sensitivity_gated` and related analyses | `docs/C-mmr-learned-lambda/*` |

## Phase10 Chunking Ablation

| group | root | status | setting | visible result |
|---|---|---|---|---|
| LIAR-RAW raw chunking | `outputs/phase10_chunking_ablation/runs/chunking_granularity_liar_raw` | partial | raw chunking，Qwen3-4B-2507 LoRA，max1024 | train/eval step metrics only |
| RAWFC raw chunking | `outputs/phase10_chunking_ablation/runs/chunking_granularity_rawfc` | 6 complete configs / many eval steps | raw chunking，Qwen3-4B-2507 LoRA，budget adaptive5_10，pool32，min5/max20，max1024 | 66 step metrics；use run-specific `training_complete.json` and `configs/train.resolved.yaml` |
| cache | `outputs/phase10_chunking_ablation/cache/` | cache-only | build/pre-MMR/chunk-MMR cache | input reuse only |

## Build And Intermediate Caches

| cache root | size from scan | role | reuse rule |
|---|---:|---|---|
| `outputs/cache/build` | 101 dirs / 82 manifests | pipeline build outputs: `build_{train,val,test}.jsonl` | Locate by run manifest `build_id`; do not infer experiment from hash alone |
| `outputs/cache/pre_mmr` | 12 dirs | pre-MMR candidate pools | Input cache for MMR/learned-lambda work |
| `outputs/cache/chunk_mmr` | 5 dirs | chunk-MMR pickles | Used by learned-lambda and RL MMR |
| `outputs/cache/dense_only` | 3 dirs | dense-only / phase9 configs | historical comparison inputs |
| `outputs/cache/method_upgrade` | 3 dirs | method-upgrade configs/cache | supports later selector/verifier work |
| `outputs/cache/backbone_migration` | 2 dirs | backbone migration configs and text backbones | archived/historical |

## Analysis And Web Artifacts

| root | contents | status |
|---|---|---|
| `outputs/analysis/map/v0.7/` | per-case evidence map comparison HTML and `.zh.json` translations | active analysis/Web UI source data |
| `outputs/analysis/map/evidence-map-web.log` | local Web app log | runtime log |
| `outputs/analysis/retrieval_signal_ablation/` | Qwen3 val/test MMR CSV/JSON signal ablations | analysis artifact |
| `outputs/logs/vllm/` | oracle direct vLLM server log and pid json | runtime log only |
| `outputs/liar-raw/` | no files found in current scan | empty / placeholder |

## Known Confusions To Avoid

- `source_coverage_v2_flash` 是当前可用 coverage 版本；不要把旧 `outputs/data_quality/source_coverage/*` 当成最终数据。
- `outputs/cache/build/<hash>` 是 build cache，不是 run。真实 run 信息在 `outputs/runs/<experiment>/<run>/manifest.json`。
- `sentence_trace_method` 中 `training_complete.json` 才是 completion marker；只有 `train/best` 或 checkpoint 不足以证明 run 完整。
- LIAR-RAW coverage retrain的 post-hoc test `tau0p5` 不能替代 val-selected adoption rule。
- `outputs/selectors/selectors/` 已清理确认相同的重复副本，但仍剩余 128 个文件；这些文件根目录缺失对应路径或同名但内容不同，不能继续按重复项删除。
- selector 的 recall@5 / top1_match 与 verifier 的 macro-F1 是不同层级指标，不要直接横向排名。
- oracle train shard 指标通常高于 val/test，不应作为部署或下游比较结论。

## 下一步整理建议

1. 给 `outputs/runs/overall_run_analysis/` 增补一个机器可读 index，记录每个 run 的 `run_id`、`build_id`、config path、best infer metric。
2. 为 `outputs/sentence_trace_method/` 生成 `runs_index.csv`，字段包括 `run_name`、`schema`、`model`、`lora`、`training_complete`、`best_val_setting`、`best_test_setting`、`selected_setting`。
3. 对 `outputs/selectors/selectors/` 剩余 128 个文件做来源判定：根目录缺失的文件可考虑迁移回 `outputs/selectors/<same-relative-path>`，2 个同名不同内容文件需人工比较后再定。
4. 对所有 future run 统一写入 `experiment_source` 字段：脚本路径、commit hash、doc path、dataset path、eval split、adoption rule。
