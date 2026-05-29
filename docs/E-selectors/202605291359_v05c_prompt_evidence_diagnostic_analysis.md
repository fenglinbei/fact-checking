# v0.5c Prompt × Evidence Diagnostic Analysis

日期：2026-05-29

结果目录：`outputs/selectors/evidence_map_selector/v0_5c_val_prompt_evidence_diagnostic`

状态：full val diagnostic completed；20 个 eval job 均有结果。

## 1. 结论

v0.5c 支持一个组合结论：

```text
v0.5b 的低分主要来自 prompt OOD + selector evidence gap；
不是 token budget / truncation gap，也不是 checkpoint sensitivity。
```

其中 prompt OOD 是最强信号：在 evidence 完全使用 oracle_top5 的条件下，`plain_original` 可以复现旧 oracle-direct verifier 的上界表现，而 `map_full` 会把 macro-F1 直接压低约 0.35。

selector evidence gap 也是真实存在的：即使改回 `plain_original`，当前 selector evidence 的 macro-F1 仍只有 0.27-0.29，距离 `oracle_top5 + plain_original` 约 0.42-0.45。

因此当前不建议继续用旧 verifier 直接评估 full map prompt，也不建议仅靠继续手调 v0.5a greedy map 权重来解决分类链路。

## 2. Prompt OOD

最关键 paired test 是同一 oracle evidence 下只替换 prompt rendering。

| checkpoint | oracle_top5 plain_original | oracle_top5 map_full | delta |
| --- | ---: | ---: | ---: |
| checkpoint-600 | 0.7211 | 0.3685 | -0.3525 |
| checkpoint-500 | 0.7067 | 0.3593 | -0.3473 |

这说明旧 oracle-direct verifier 对 `map_full` prompt 严重 OOD。该下降不来自 evidence source，因为 evidence source 固定为 `oracle_top5`；也不来自 evidence order，因为 paired delta 检查中 selected texts before truncation 全部一致。

原 oracle-direct eval prior 与本次 plain oracle anchor 基本一致：

| checkpoint | original oracle-direct macro-F1 | v0.5c oracle_top5 plain_original |
| --- | ---: | ---: |
| checkpoint-600 | 0.7183 | 0.7211 |
| checkpoint-500 | 0.7099 | 0.7067 |

因此 `plain_original` 是有效 anchor，`map_full` 的下降可以解释为 prompt distribution shift。

## 3. Evidence Gap

在 `plain_original` 下，当前 selector evidence 与 oracle evidence 的差距很大。

| evidence_source | checkpoint-600 plain macro-F1 | gap to oracle |
| --- | ---: | ---: |
| oracle_top5 | 0.7211 | 0.0000 |
| v0_5a_base_only_top5 | 0.2858 | -0.4353 |
| v0_5a_evidence_map_top5 | 0.2793 | -0.4417 |
| original_pool_order_top5 | 0.2705 | -0.4505 |
| fusion_refit_all_features_plus_direct_ce_top5 | 0.2679 | -0.4532 |

checkpoint-500 下结论一致，selector 与 oracle 的 gap 仍在 0.42 以上。

这说明 prompt 改回 plain 只能移除 map rendering 伤害，不能解决 selector evidence 与 oracle utility set 的差距。

## 4. Selector 对比

当前 selector 之间没有稳定、足够大的分类差异。

| prompt/checkpoint | best selector | macro-F1 |
| --- | --- | ---: |
| plain_original / checkpoint-600 | v0_5a_base_only_top5 | 0.2858 |
| plain_original / checkpoint-500 | v0_5a_evidence_map_top5 | 0.2858 |
| map_full / checkpoint-600 | v0_5a_base_only_top5 | 0.2842 |
| map_full / checkpoint-500 | v0_5a_base_only_top5 | 0.2775 |

`v0_5a_evidence_map_top5` 的 explainability 指标最好，但分类指标没有稳定转化：

| selector | recall@5 | jaccard@5 | ndcg@5 | weighted_atom_coverage@5 | direct_or_partial_rate@5 | background_rate@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0_5a_evidence_map_top5 | 0.3438 | 0.2297 | 0.3109 | 0.7205 | 0.6173 | 0.3852 |
| v0_5a_base_only_top5 | 0.3557 | 0.2377 | 0.3053 | 0.6655 | 0.4427 | 0.5604 |
| fusion_refit_all_features_plus_direct_ce_top5 | 0.3694 | 0.2485 | 0.2977 | 0.5528 | 0.3274 | 0.6750 |
| original_pool_order_top5 | 0.3435 | 0.2294 | 0.2872 | 0.6780 | 0.4892 | 0.5133 |

解释层特征提升了 atom/directness/background 指标，但同时没有改善 oracle overlap；旧 verifier 的最终分类也没有稳定收益。

## 5. Truncation

`map_full` 的 token overhead 明显，但没有形成 evidence loss。

| evidence_source | token_delta map-full vs plain | evidence_count_delta | truncation_rate map_full |
| --- | ---: | ---: | ---: |
| oracle_top5 | +183.20 | 0.00 | 0.0000 |
| original_pool_order_top5 | +197.78 | 0.00 | 0.0000 |
| fusion_refit_all_features_plus_direct_ce_top5 | +181.20 | -0.00 | 0.0008 |
| v0_5a_base_only_top5 | +189.33 | 0.00 | 0.0008 |
| v0_5a_evidence_map_top5 | +197.41 | 0.00 | 0.0000 |

因此本轮不能把下降解释为 token budget 截断。`map_full` 伤害更像格式 / 分布偏移，而不是 tail-pop evidence 丢失。

## 6. Checkpoint Sensitivity

checkpoint-600 与 checkpoint-500 的差异均小于 0.03。

最大差异是 `oracle_top5 + plain_original`：`0.7211 - 0.7067 = 0.0144`。

因此本轮不触发 checkpoint sensitivity 判定。后续主比较可固定 `checkpoint-600`，因为它在 oracle anchor 上更强且更接近原 oracle-direct prior。

## 7. Case Signals

`analysis/case_studies.md` 中几个固定 case 支持上述判断。

- `4855.json`：oracle evidence 下 `plain_original` 正确为 `barely-true`，`map_full` 变为 `mostly-true`；说明 map rendering 会把旧 verifier 往 true-side 推。
- `10443.json`：多数组合 plain 正确为 `half-true`，map_full 更容易预测到 `mostly-true` / `true`；同样体现 true-side bias。
- `11447.json`：大多数 evidence/prompt 都能判为 `false`，但 `v0_5a_evidence_map_top5 + map_full` 反而变为 `barely-true`；说明 explanation-oriented selection 不是稳定分类改进。

label shift 也支持这一点：`oracle_top5 + checkpoint-600` 从 plain 到 map 后，`true` 预测比例从 0.15 升到 0.27，`mostly-true + true` 从 0.36 升到 0.53。

## 8. Route Decision

本轮路线标签：

```text
PROMPT_OOD: yes
EVIDENCE_GAP: yes
TRUNCATION_GAP: no
CHECKPOINT_SENSITIVITY: no
```

推荐下一步：

1. 先跑 `map_minimal` second-pass ablation，只回答一个问题：压缩 map metadata 后，oracle_top5 是否还能接近 plain anchor。
2. 如果 `oracle_top5 + map_minimal` 仍显著低于 plain anchor，则停止用旧 verifier 评估 map prompt，转向 map-aware verifier training。
3. 对分类链路，短期默认使用 `plain_original`；map 信息保留为 selector feature / explanation output，不直接塞给旧 verifier。
4. selector 方向不要继续只调 v0.5a greedy weights。更合理的下一步是 set-level verifier utility distillation，或 learned map-feature fusion，但评价目标必须对齐 verifier utility 而不是只看 atom coverage。

一句话结论：

```text
v0.5b 的低分主要来自旧 verifier 对 full map prompt 的强 OOD，以及当前 selector evidence 与 oracle utility set 的大差距；
下一步应先做 map_minimal / compressed prompt ablation，并同步准备 map-aware verifier training 或 set-level utility distillation。
```
