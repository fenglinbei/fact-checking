# Atom-Union 候选池消融实验计划（LIAR-RAW）

## 1. 实验问题

在固定 ABC chunks、claim atoms、Evidence Map 方法、learned-marginal selector 权重、prompt policy 和 verifier checkpoint 的条件下，仅改变 selector 可见的候选池来源，回答三个问题：

1. atom-route 相比整 claim 检索是否带来增量；
2. claim baseline 是否能补充 atom-route；
3. 两路融合后的最终 MMR 去冗余是否有效。

本实验不训练新的 selector，也不训练新的 verifier。主指标为 LIAR-RAW test Accuracy 和 Macro-F1，val 仅用于完整性检查，不用于重新选择 checkpoint 或调参。

## 2. 四个受控变体

所有变体的最终候选池上限统一为 `n=20`。

| 变体 | 候选池定义 | 最终处理 | 对应问题 |
|---|---|---|---|
| `baseline_only` | 整 claim 检索，经 claim-MMR 得到 top-20 | 截断至 20 | atom-route 是否有增量 |
| `atom_only` | 每个 atom 分别检索 top-20，按 RRF 聚合 | 截断至 20 | baseline 是否有补充价值 |
| `union_no_mmr` | baseline top-20 与 atom RRF top-20 合并、文本去重、按 Union relevance 排序 | 截断至 20，不做最终 MMR | 融合本身是否有效 |
| `union_full` | 与 `union_no_mmr` 相同 | 在完整去重池上执行 MMR，取 top-20，`lambda=0.70` | 最终去冗余是否有效 |

### 当前主产物的口径校正

当前 `outputs/selectors/atom_anchor/liar_raw_abc_v0_1/03_atom_union/` 产物没有 `union_mmr_applied`，其 manifest 也没有 `final_pool_size`；它使用 baseline top-5 与 atom top-20 合并，更接近旧版 `union_no_mmr`，不能直接当作本计划中受控的 `union_full`。因此四个候选池都从同一份新建的 baseline top-20 / atom top-20 retrieval 产物生成；只复用已有 chunks、claim atoms、selector 权重和 verifier checkpoint。

## 3. 固定项

| 层 | 固定设置 |
|---|---|
| Dataset | LIAR-RAW，官方 train/val/test split |
| Chunks | ABC claim-aware cache，fingerprint `d4cbf7c18126` |
| Claim atoms | `outputs/selectors/atom_anchor/liar_raw_abc_v0_1/01_claim_atoms/` |
| Retrieval encoder | `bge-base-en-v1.5`，fp32 |
| Candidate pool size | 20 |
| Evidence Map | `deepseek-v4-flash`，`atom_evidence_map_v0_1`，temperature=0，top_p=1.0，thinking disabled |
| Selector | `learned_marginal_proxy`，复用主方法 `weights.json` |
| Trace policy | fullpool，`target_resolved_rate=1.0`，minmax5_10 |
| Prompt | `mrec_min`，full evidence text |
| Verifier | 复用主方法 Ministral-3-8B LoRA `train/best` |
| Eval | val/test，checkpoint 固定为 best；同时报告原始 label-token 与 tau=0.75 |

候选池改变会改变 Evidence Map teacher 的输入列表，因此四个变体需要各自生成 map 标注；否则 candidate ID、fingerprint 与 map 对齐关系不成立。这里保持 teacher 配置完全一致，并依赖 P0-1 中 R-B/R-D 修复后的缓存键和重试逻辑。正式全量运行前必须先确认这些修复已经落地。

## 4. 实现入口

- Pool builder：`scripts/phase5_selectors/build/build_atom_retrieval_union.py --pool-mode ...`
- 矩阵 wrapper：`scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh`
- 结果汇总：`scripts/sentence_trace_method/summarize_atom_union_pool_ablation.py`

主要输出：

```text
outputs/selectors/atom_union_pool_ablation/liar_raw_abc_n20/
├── 02_atom_retrieval/
├── baseline_only/{03_atom_union,04_evidence_map,05_mrec_*}/
├── atom_only/{03_atom_union,04_evidence_map,05_mrec_*}/
├── union_no_mmr/{03_atom_union,04_evidence_map,05_mrec_*}/
├── union_full/{03_atom_union,04_evidence_map,05_mrec_*}/
└── summary/{summary.json,summary.csv,summary.md}
```

Verifier 输入和评估结果写入：

```text
outputs/sentence_trace_method/
  liar_raw__ministral3_8b__atom_union_pool_ablation_<pool_mode>_reuse_main_ckpt/
```

## 5. 执行阶段

### 阶段 A：命令与结构检查（无 GPU/API）

```bash
DRY_RUN=true \
bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh
```

验收：四种 `--pool-mode` 均展开；没有 `sft.label_token_trainer`；eval 的 `run-dir` 指向同一主方法 checkpoint。

### 阶段 B：32 条 mock smoke test

```bash
SAMPLE_LIMIT=32 \
MOCK_EVIDENCE_MAPS=true \
MODE=build \
bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh
```

验收：

- 每个 split、每种模式均生成 union manifest、map features、MREC diagnostics 和 verifier build report；
- `baseline_only` 所有候选均有 `from_baseline=true`；
- `atom_only` 所有候选均有 `from_atom_route=true`；
- 只有 `union_full` 的 `union_mmr_applied=true`；
- 候选数上限均为 20，无空池；
- 三个 split 的 event ID 顺序与 raw split 一致。

### 阶段 C：32 条真实 teacher smoke test

使用独立输出根，避免 mock annotation 被 `--resume` 复用：

```bash
ABLATION_ROOT=outputs/selectors/atom_union_pool_ablation/liar_raw_abc_n20_real_smoke32 \
CASE_OUTPUT_ROOT=outputs/sentence_trace_method/atom_union_pool_ablation_real_smoke32 \
SAMPLE_LIMIT=32 \
MODE=build \
bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh
```

验收：四种模式的 `fallback_missing_annotation=0`；schema fallback 比例记录但不得静默忽略；同一 teacher 参数写入四组 manifest。

### 阶段 D：全量 build

```bash
MODE=build \
bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh
```

此阶段包括 baseline20/atom20 retrieval、四组 pool、四组 Evidence Map、固定权重的四组 MREC trace 和 verifier-data build。不会启动 verifier 训练。

### 阶段 E：复用主 checkpoint 推理并汇总

```bash
MODE=eval \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh

MODE=summarize \
bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh
```

## 6. 报告口径

主表使用未校准的 `label_token` test Accuracy / Macro-F1；tau=0.75 作为补充列，不用每个变体重新选择 tau。建议同时报告：

- val/test Accuracy、Macro-F1；
- 平均候选池大小及 min/max，证明容量受控；
- MMR 实际应用率，`union_full` 应为 1，其余应为 0；
- mean selected steps（平均 K*）与 resolved-atom rate；
- map parse/fallback 分布。

核心比较为：

1. `atom_only - baseline_only`：atom-route 的价值；
2. `union_no_mmr - atom_only`：baseline 补充价值；
3. `union_full - union_no_mmr`：最终 MMR 的价值；
4. `union_full` 与当前已报告主方法指标只作背景参照，不把旧产物误写成完全同口径复现。

## 7. 完成判据

只有满足以下条件后，才能在 `TODO.md` 将 P1-1 标为完成：

- 四种模式均有 LIAR-RAW full test metrics；
- 四组使用同一个 selector weight SHA256 和 verifier checkpoint；
- 候选池来源与 MMR 审计通过；
- map 标注无缺失，fallback 率已报告；
- `summary.csv` 与论文组件消融表中的数值一致；
- 论文中明确写出旧主产物与本次受控 `union_full` 的口径差异。
