# Oracle Calibration + Re-Oracle 四阶段实施计划

## 背景

当前 v1a/v1b pointwise selector 没有稳定超过 fixed-MMR，主要风险不是 selector 是否能模仿 oracle set，而是 oracle set 本身继承了旧 verifier 的 false-side bias。下一轮不应继续直接扩展 selector，而应先把 verifier 的 label-token 决策面校准，再用新 verifier 重做 oracle 搜索。

Stage 1 的训练输入采用 fixed-MMR 标准 build prompt：`b3_mmr_topk_sweep_1024`、`mmr_lambda=0.70`、`top_k=5`、`max_length=1024`。不使用 oracle prompt 训练 verifier，避免把旧 oracle 搜索结果的偏差直接写进 verifier。

---

## Stage 1: Label-token Weighted CE Verifier

目标：把 verifier 从“生成完整 label 文本”的 SFT loss，改成只在 `Label:` 后的 A-F label token 上做加权 CE，优先修复 `true` / `mostly-true` recall。

训练信号：

```text
input  = build_prompt + "Label:"
target = correct label letter token in {A, B, C, D, E, F}
loss   = WeightedCrossEntropy(logits[A:F], target)
```

初始类别权重：

| label | weight |
|---|---:|
| pants-fire | 1.0 |
| false | 1.0 |
| barely-true | 1.2 |
| half-true | 1.2 |
| mostly-true | 2.0 |
| true | 3.0 |

checkpoint 选择指标：

$$\mathrm{score}=\mathrm{MacroF1}+0.5\times\mathrm{MacroF1}_{\{\mathrm{mostly\text{-}true},\mathrm{true}\}}$$

验收标准：

1. `train/best` 能被现有 vLLM infer pipeline 直接加载。
2. val split 的 `true` / `mostly-true` recall 明显高于旧 verifier。
3. Macro F1 不应大幅低于 fixed-MMR verifier；若 true-side 提升但 macro F1 下降，需要进入 Stage 2 时显式记录 tradeoff。

---

## Stage 2: Calibration-aware Re-Oracle

目标：用 Stage 1 verifier 重跑 oracle evidence search，使 oracle set 的搜索目标更接近最终 argmax correctness，而不是旧 verifier 下的正确标签 logprob 单点最大化。

推荐 oracle objective：

$$
u(S)=\log P(y^*|c,S)-\max_{y\neq y^*}\log P(y|c,S)
$$

同时保存：

1. `candidate_pool` 完整候选池。
2. 每个候选 set 的 `candidate_scores`。
3. `candidate_pool_fingerprint`。
4. `gold_logprob`、`best_wrong_logprob`、`margin`、`pred_label`。

验收标准：

1. oracle set 在 `true` / `mostly-true` 上不再系统性低于 fixed-MMR。
2. oracle-only-correct 样本仍保留足够规模。
3. margin-positive set 与 argmax-correct 的一致性高于旧 logprob-only oracle。

---

## Stage 3: Filtered Preference / Utility Supervision

目标：只把校准后确实有帮助的 oracle set 作为监督，不再无条件 hard imitation。

样本保留优先级：

| 条件 | 用法 |
|---|---|
| oracle correct 且 MMR wrong | 强正例 |
| oracle correct 且 MMR correct | 弱正例或低权重 |
| oracle wrong 且 MMR correct | 不做正例，作为 objective failure 分析 |
| oracle wrong 且 MMR wrong | 暂不进入 selector 训练 |

偏好对：

```text
S+ = margin-positive oracle set
S- = fixed-MMR / reranker-only / random / low-margin set
```

权重：

$$
w_i=\sigma((\mathrm{margin}_{oracle}-\mathrm{margin}_{baseline})/T)
$$

验收标准：

1. 训练池中每个 label 的样本数、权重均值和正负 pair 数可审计。
2. `true` / `mostly-true` 不再被简单删除，而是用 margin 与 correctness 控制权重。
3. 监督数据能追溯到 `candidate_pool_fingerprint`，避免候选池漂移。

---

## Stage 4: Selector Training + Full Pipeline Evaluation

目标：重新训练 selector，并用统一 verifier evaluation pipeline 检验是否真正超过 fixed-MMR。

路线：

1. 先训练 pointwise utility selector，验证校准后 oracle supervision 是否可被吸收。
2. 若 pointwise overlap 高但指标仍低，转 sequential selector 或 set-level reranker。
3. 所有结果必须跑同一套 build/infer 评估：`val` 先判定方向，`test` 只用于最终确认。

必须对比：

| 方法 | 目的 |
|---|---|
| fixed-MMR lambda=0.7 top_k=5 | 主基线 |
| Stage 1 verifier + fixed-MMR | 分离 verifier 改动收益 |
| Stage 1 verifier + re-oracle | 检查 oracle 上限 |
| Stage 1 verifier + selector | 检查 selector 是否吸收 oracle supervision |

停止条件：

1. 如果 re-oracle 上限仍低，继续修 verifier 或候选池，不训练 selector。
2. 如果 re-oracle 上限高但 selector 低，问题在 selector 表达能力或监督过滤。
3. 如果 selector overlap 高但 verifier 指标低，优先检查 evidence order、prompt truncation、candidate fingerprint 和 infer-time decoding 一致性。
