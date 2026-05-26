# Oracle Aspect Coverage Diagnostic Implementation

## 目的

在把 `deep + claim aspect coverage` 接入 sequential selector 前，先验证轻量 claim aspect 是否能解释 Stage2 oracle selection。

本实现只做：

```text
rule-based claim aspect extraction
candidate-aspect encoder alignment
oracle-vs-hybrid coverage diagnostic
```

暂不修改 selector 训练架构。

## 新增文件

```text
src/fact_checking/selectors/aspects.py
src/fact_checking/selectors/test_aspects.py
scripts/selectors/analyze_oracle_aspect_coverage.py
```

## Aspect 契约

每个 local aspect 需要尽量满足：

```text
atomic: 单一核查点，避免多个并列事实混在一起
retrievable: 包含实体 / 数字时间 / 动作谓词 / 政策主题等检索 anchor
decontextualized: 不以 otherwise / instead / where 等上下文 cue 开头，不保留未解析代词
```

输出字段：

```text
aspect_id
type
text
raw_span
added_context
is_atomic
is_decontextualized
retrievability_score
quality
drop_reason
features
```

`full_claim_anchor` 单独保存，不参与 local aspect 去重，也不默认代表 claim aspect coverage。若某条 claim 没有合格 local aspect，诊断脚本默认可回退到 full claim anchor，避免整条样本在 embedding 诊断中缺失。

## 规则能力边界

当前 `rule_aspect_v1` 可以处理：

```text
数字、金额、百分比、年份、日期窗口
并列政策动作展开
but instead 对比补全
otherwise 条件补全
negation / comparison / policy_action / causal_condition cue window
保守实体上下文窗口
```

当前不尝试处理：

```text
复杂隐含 subclaim
跨句去上下文化
不确定代词消解
LLM-style yes/no subquestion generation
```

不确定片段会进入 `dropped_aspects`，用于 debug，不进入 diagnostic 主缓存。

## 推荐命令

先做 extract-only smoke：

```bash
PYTHONPATH=src python scripts/selectors/analyze_oracle_aspect_coverage.py \
  --oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --output-dir outputs/selectors/aspect_coverage/rule_v1_val_extract_smoke \
  --sample-limit 100 \
  --extract-only
```

完整 val 诊断：

```bash
PYTHONPATH=src python scripts/selectors/analyze_oracle_aspect_coverage.py \
  --oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --split val \
  --output-dir outputs/selectors/aspect_coverage/rule_v1_deberta_val \
  --model-name microsoft/deberta-v3-base \
  --batch-size 128 \
  --max-length 128 \
  --device cuda
```

若模型只在本地路径可用，改 `--model-name` 为对应路径。若要避免联网，增加：

```text
--local-files-only
```

`microsoft/deberta-v3-base` 的 tokenizer 依赖 `sentencepiece`。仓库 `pyproject.toml` 与 `requirements.txt` 已声明该依赖；若当前环境缺失，先执行：

```bash
python3 -m pip install sentencepiece
```

## 输出

```text
claim_aspects.jsonl
aspect_extraction_summary.json
candidate_aspect_alignment.jsonl
oracle_aspect_coverage_analysis.json
analysis_summary.json
analysis.md
manifest.json
```

## Stop/Go

默认进入 selector ablation 的条件：

```text
uncovered_gain AUROC >= 0.57
或 oracle top5 coverage 比 hybrid top5 高 >= 3pp
```

若输出 `decision=go_selector_ablation`，再实现 `targeted_feature_profile=aspect`。

若输出 `decision=stop_or_refine_aspects`，先审计 `claim_aspects.jsonl` 与 `dropped_aspects`，调整 aspect 规则或考虑更强 claim decomposition，再接 selector。

## Val 结论

完整 val 已跑：

```text
outputs/selectors/aspect_coverage/rule_v1_deberta_val/
```

结论为 `stop_or_refine_aspects`，不应把当前 `rule_aspect_v1` 直接接入 `deberta_sequential_aspect`。

关键结果：

```text
n_events = 1274
n_local_aspects = 2096
claims_with_no_local_aspects = 147
uncovered_gain AUROC = 0.4820
oracle_vs_hybrid_coverage_lift_pp = -0.07
oracle coverage mean = 0.949202
hybrid top5 coverage mean = 0.949909
```

更细的 selected-vs-nonselected probe 也接近随机：

```text
max_aspect_score AUROC = 0.5051
mean_aspect_score AUROC = 0.5030
aspect_score_entropy separability_auc = 0.5012
```

该结果不是“略低于阈值”，而是说明当前规则 aspect + encoder alignment 基本没有选择信号。`entity_context` 占 1167 / 2096 个 local aspects，很多 aspect 仍是实体窗口或接近整句的片段；同时 `microsoft/deberta-v3-base` mean-pooling alignment 出现高分饱和，selected 与 nonselected 的 `mean_aspect_score` 均值只差约 `0.00054`。

用 `cppo` 环境对本地缓存的 `BAAI/bge-base-en-v1.5` 做了 100 条 sample probe：

```text
outputs/selectors/aspect_coverage/rule_v1_bge_val_sample100_cppo/
uncovered_gain AUROC = 0.4670
oracle_vs_hybrid_coverage_lift_pp = -1.77
```

这说明问题不只是不合适的裸 DeBERTa encoder；至少在当前规则 aspect 质量下，换一个句向量模型也没有立刻打开信号。

下一步建议不是直接上完整 LLM selector，而是先做 LLM-aspect 上界诊断：

```text
LLM 把 claim 分解为 2-5 个 self-contained check questions / atomic aspects
用同一套 oracle coverage diagnostic 评估
仅当 uncovered_gain AUROC >= 0.57 或 oracle_vs_hybrid_lift >= 3pp 时，才接入 selector
```

如果 LLM-aspect 仍不过线，应停止 claim-aspect coverage 主线，转向更直接的 pairwise utility / verifier-aware evidence contribution / oracle-prefix imitation 方向。

2026-05-21 后续已实现该 LLM-aspect 上界诊断入口，见：

```text
docs/implementation/202605212359_llm_decomp_plus_aspect_coverage.md
scripts/selectors/generate_llm_claim_decomp_aspects.py
scripts/selectors/run_llm_decomp_plus_aspect_coverage.sh
```

2026-05-22 已核查 Qwen decomp+ plain full-val 结果，产物位于：

```text
outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_val_plain/
outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_val_plain_coverage/
```

这次生成缓存质量已足以作为 gate 判定依据：`parse_failures=1/1274`，`claims_with_no_local_aspects=1/1274`，`n_local_aspects=2962`，valid subclaims 均值 `2.33`。但 coverage 仍 no-go：`uncovered_gain AUROC=0.4730`，`oracle_vs_hybrid_coverage_lift=-1.51pp`，oracle coverage mean `0.8743` 低于 hybrid top5 `0.8893`。

因此 LLM-aspect 上界诊断也未支持 claim-aspect coverage 主线。除非做一个很小的 cross-encoder / NLI scorer sanity check，否则不应继续通过更强 claim decomposition、闭源 API 或规则增强投入该方向；下一步应转向 verifier-aware utility、prefix-level evidence contribution 或 oracle-margin distillation。

## 已验证

轻量验证已通过：

```text
PYTHONPATH=src python -m unittest src/fact_checking/selectors/test_aspects.py
python -m compileall src/fact_checking/selectors/aspects.py src/fact_checking/selectors/test_aspects.py scripts/selectors/analyze_oracle_aspect_coverage.py
PYTHONPATH=src python scripts/selectors/analyze_oracle_aspect_coverage.py --help
PYTHONPATH=src python scripts/selectors/analyze_oracle_aspect_coverage.py ... --sample-limit 20 --extract-only
```

`--sample-limit 20 --extract-only` smoke 中，20 条 claim 保留 36 个 diagnostic local aspects，drop 82 个 debug-only aspects，说明当前门槛偏保守，适合作为第一版诊断入口。

安装 `sentencepiece` 后，完整 encoder alignment smoke 已用本地缓存的 `microsoft/deberta-v3-base --local-files-only --device cpu` 跑通 2 条样本，并写出 `candidate_aspect_alignment.jsonl`、`oracle_aspect_coverage_analysis.json`、`analysis_summary.json` 与 `analysis.md`。该 smoke 的 `uncovered_gain AUROC=0.5379`、`oracle_vs_hybrid_lift=0.12pp` 仅用于链路验证，不能作为 val 结论。
