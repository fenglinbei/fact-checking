# Scripts 目录索引

按实验推进阶段组织。`phaseN_` 前缀反映研究推进顺序。

## phase1_pipeline — Phase 1: 基线流水线 (b0, b1, b2)

MMR lambda 扫描、top-k 扫描，确定 `fixed-lambda=0.7` 为强 baseline。

| 入口脚本 | 作用 |
|----------|------|
| `run_exp.sh` | 通用流水线入口，转发参数到 `fact_checking.pipeline.run` |
| `run_mmr_lambda_sweep.sh` | b0 配置下 MMR lambda 网格扫描 (build→train→infer) |
| `run_mmr_topk_sweep_infer.sh` | top_k 扫描推理，复用已有训练 checkpoint |
| `summarize_infer_metrics.py` | 聚合推理指标为 CSV/JSON |
| `summarize_prompt_stats.py` | 聚合 prompt 统计信息 |

## phase2_learned_lambda — Phase 3: 学习 Lambda (b3)

Oracle lambda 计算 → 预测器训练。结论：scalar lambda 不可从文本特征预测。

**依赖顺序**：`generate_oracle_prompts.py` → `compute_oracle_lambda.py` → `train_predictor.py` → `evaluate_predictor.py`

## phase3_oracle_evidence — Phase 4: Oracle 证据搜索

搜索最优证据子集，建立 +18.76pp accuracy 理论上界。

**入口**：`run_search.sh` → `merge_shards.py` → `build_oracle_direct_verifier_data.py`

## phase4_verifier — Phase 5: 验证器训练与评估 (b3)

Label-token CE verifier → calibration reoracle → oracle direct → order sensitivity。

| 入口脚本 | 作用 |
|----------|------|
| `run_label_token_ce_stage1.sh` | Stage1 标签分词 CE 验证器训练 |
| `run_oracle_sentence_direct_verifier.sh` | 句子级 oracle 直接验证器 |
| `run_oracle_direct_vllm_server.sh` | vLLM OpenAI 兼容 API 服务 |

## phase5_selectors — Phase 6: 证据选择器 (b3)

多步选择器实验。Step4 Sequential Pointer 改善 order consistency。

```
phase5_selectors/
├── build/   # 数据构建 (build_*.py, generate_*.py)
├── train/   # 模型训练 (train_*.py)
├── eval/    # 评估分析 (eval_*.py, analyze_*.py, score_*.py)
└── run/     # Shell 启动脚本 (run_*.sh)
```

## phase6_rl_mmr — Phase 7: RL-MMR 实验

Trajectory → Utility → DPO → Sensitivity-gated / Soft-label 策略。

**依赖顺序**：`generate_trajectories.py` → `compute_trajectory_utility.py` → `build_preference_pairs.py` → `train_dpo_step_lambda.py` → `evaluate_dpo_step_lambda.py`

## phase9_utils — 横切工具

| 脚本 | 作用 |
|------|------|
| `diag_vllm.py` | vLLM 推理启动诊断 |
| `visualize_chunking_evidence.py` | 证据分块可视化 |
| `sync.sh` | 远程 rsync 输出目录 |
