# Structure-only reservation audit — 2026-07-17

本文件记录 2026-07-17 GPU 窗口内的实验终态与论文使用边界。它是内部 provenance 文档，不是投稿正文。

## 可进入当前主稿的完整结果

- LIAR-RAW canonical test（seed 42）：$n=1{,}251$，Accuracy 35.17，Macro-F1 35.40。
- RAWFC canonical test（seed 42）：$n=200$，Accuracy 65.00，Macro-F1 65.08，parse failure 为 0；训练正常早停于 step 950，validation-best 为 step 550。
- SciFact official development：sentence Selection-only 39.95、Selection+Label 38.65、abstract Label-only 71.98、Label+Rationale 64.78。
- LIAR-RAW seed-42/43 fixed-$K$ one-shot/state-conditioned crossover：两组均使用 checkpoint 800、$K=5$、$n=1{,}234$；按现有主稿仅作方向敏感性诊断，不支持显著性、因果、普遍顺序优势或稳健 co-adaptation 主张。

## RAWFC seed-43 checkpoint sensitivity

seed 43 在完成 step 650 validation、best 与 checkpoint-650 adapter 保存后，因旧 `latest_state/pytorch_model` 的 NFS 清理竞态终止。best/checkpoint-650 adapter 完全一致，SHA256 为 `beb673526d7ddf339ffba18b132fa7222ed64e3c0fec3edfaa51de298e80fe49`，含 812 个 LoRA tensors。该 run 没有 `training_complete.json`，因此不是完整训练 replication。

从冻结 best adapter 直接导出的结果如下：

| Split | Accuracy | Macro-P | Macro-R | Macro-F1 | $n$ | Parse failures |
|---|---:|---:|---:|---:|---:|---:|
| val | 67.00 | 66.95 | 67.04 | 66.93 | 200 | 0 |
| test | 62.00 | 63.03 | 62.03 | 62.37 | 200 | 0 |

相对 seed-42 canonical，seed 43 checkpoint-650 的 validation Macro-F1 高 2.31 pp，而 test Macro-F1 低 2.71 pp。该方向分裂只支持“RAWFC 对 seed/checkpoint 选择存在敏感性”的内部判断；它不能进入主结果均值、方差或完整多种子复现实验。导出 manifest SHA256 为 `f7bd7034992ca379c3dc177122f8c2b8d2446466c45e9f8107fada8abde0d4aa`。

## No-map checkpoint-800 mechanism tail

no-map verifier 使用 seed 42，从零训练并在稳定 checkpoint 800 后精确 SIGINT。它没有 `training_complete.json`，属于预设 fixed-step diagnostic artifact，不是完整训练 run。

- checkpoint-800 adapter SHA256：`86847100d511613a7929ff6f520b745e2a152472d4bf6d8062aac15e0ecf4c91`。
- cap manifest SHA256：`cfedc72e9a0c072bcae569c4416fe8cd35f8a96fc9a730b940559cc36032528b`。
- natural minmax$(5,10)$ step-800 validation：Accuracy 33.75，Macro-F1 34.08。该指标与 fixed-$K$ mechanism cell 不同口径，不作 map-effect 结论。

fixed-$K$ 输入矩阵使用 $K=5$、共同 support $n=1{,}234$。与 structure cell 相比，no-map cell 的 visible sequence 在 83.06% 样本上不同、visible set 在 49.51% 样本上不同、top-1 在 34.04% 样本上不同；输入审计 SHA256 为 `3bb83997e4d8837c2b21570028968a2c8d6068bfcc03d81bb824a25bc61cfe55`。

硬截止前只完整得到 no-map-trained verifier $V_N$ 的一行：

| Evaluation verifier | No-map prompt $N$ | Structure prompt $S$ | $S-N$ |
|---|---:|---:|---:|
| $V_N$ | 34.89 | 35.59 | +0.70 pp |

$V_S$ inference 在 11:40 GPU 硬截止时被中断，没有 raw-logits complete manifest；因此不存在完整 2×2、matched-minus-crossed 或 difference-in-differences summary。上述单行只能保留为 partial artifact，不能写入投稿正文或用于声明 Evidence Map 的独立效果。

## 工程修复

`label_token_trainer.py` 的 latest-state 发布已改为固定退役槽协议：旧状态先原子退役，新状态再发布；对 `ENOTEMPTY`、`EBUSY`、`ESTALE` 有界退避。退役槽持续无法回收时，各 rank 同步跳过本次 latest-state 更新并保留上一份可恢复状态，避免训练退出或多个约 34 GB 目录无界累积。

## 当前论文边界

当前主稿继续使用 seed-42 canonical RAWFC 数字，不把 seed-43 checkpoint-650 写成完整 replication。No-map 结果在补齐 $V_S(N)$ 与 $V_S(S)$、生成严格四格 summary 之前也不进入正文。下一 GPU 窗口只需续跑缺失的 $V_S$ matrix inference，不应重跑已完成的 no-map training 或 $V_N$ raw logits。

严格续跑入口为：

```bash
bash scripts/phase5_selectors/eval/run_no_map_structure_fixed5_resume_vs_only_step800.sh
```

该入口会先核验冻结矩阵、$V_N/V_S$ checkpoint、no-map cap manifest、两份 prepared input，以及已完成的 $V_N$ raw-logits/materialized manifests 的固定 SHA256；通过后只允许执行 $V_S$ checkpoint-800 的 validation raw-logit inference，随后在 CPU 上 fanout 并生成完整 2×2 summary。可用 `DRY_RUN=true` 做无 GPU 预检。
