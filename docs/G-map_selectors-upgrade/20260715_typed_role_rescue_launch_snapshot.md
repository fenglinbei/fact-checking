# Typed Role-Rescue 启动快照（2026-07-15）

## 已完成门禁

- train/val/test = 10065/1274/1251，8 个 role-rescue cell 全量构建完成。
- 七个固定容量 cell 在每个 event 上具有相同的 verifier-visible evidence count。
- 正式 build 的 skipped/truncation rate 均为 0。
- 选择不读取 gold label 或 verifier output。
- 冻结 matrix 的 native reference equivalence gate 已通过。
- 11,466 个 cell-prompt 引用去重为 7,362 个唯一 prompt，复用率 35.79%。

## Validation 结构统计

- Resolving core：均值 2.345；909/1274 在 K=5 前 `target_resolved`，365/1274 到达 K cap。
- Prompt-feasible R-only：均值 2.343。
- 七个固定容量组：共同均值 4.939；K=5 为 1234/1274，其余 40 个 event 由源池容量或共同 prompt-feasible 投影降为 K=1--4。
- Role available：Cor 630/1274，Opp 549/1274，Ctx 738/1274。
- Full 实际提升：Cor 584、Opp 528、Ctx 610 个 event；Full 与 Random 的集合在 803/1274 个 event 上不同。
- Validation 平均 evidence token cost：Learned-fixed5 248.09、Retrieval 264.50、Full 273.88、Ctx 282.04、Cor 283.38、Opp 286.53、Random 288.60、R-only 124.37。

## 冻结 verifier 结果

所有 delta 方向均为 new - old，主指标为 Macro-F1。

| 对比 | Old | New | Delta | paired randomization p | Holm p |
|---|---:|---:|---:|---:|---:|
| Full - Random（主检验） | 0.3460 | 0.3607 | +0.0147 | 0.0805 | 不适用 |
| Cor - Random | 0.3460 | 0.3516 | +0.0056 | 0.3977 | 1.0000 |
| Opp - Random | 0.3460 | 0.3609 | +0.0148 | 0.0215 | 0.1077 |
| Ctx - Random | 0.3460 | 0.3421 | -0.0039 | 0.5256 | 1.0000 |
| Full - Retrieval | 0.3606 | 0.3607 | +0.0001 | 0.9917 | 1.0000 |
| Full - Learned-fixed5 | 0.3644 | 0.3607 | -0.0037 | 0.6275 | 1.0000 |
| R-only - Random（容量诊断） | 0.3460 | 0.3260 | -0.0200 | 0.0600 | 排除 |

Opp 的 exact McNemar p=0.0248（42 个 wrong-to-correct，23 个 correct-to-wrong）；但它是次要检验，Holm 后未达到 0.05。Full 主检验也未达到 0.05。

## 当前解释

1. 只在 `target_resolved` 停止会丢失后续有效信息，R-only 的负差异支持“简单饱和过早”的担忧。
2. Opp/qualifying evidence 是当前最明确的正向结构信号；Cor 较弱，Ctx 的无条件提升为负。
3. 等权 Full portfolio 会稀释 Opp，尚不能作为已验证的新主方法；下一版应学习或校准 role-conditioned marginal gain，而不是把所有非零角色统一加分。
4. 相同 evidence count 不等于相同 token cost。Full 相对 Random 的 token 更少，因此其正差异不是由更多 token 直接造成；Full 相对 Learned-fixed5 仍需 matched-token 实验。

## 训练队列

- tmux：`typed_role_rescue_20260715`
- 顺序：`full random retr cor opp ctx r_only`
- 配方：4×L20、LoRA r16/alpha32/dropout0.1、EBS16、LR 2e-5、12 epochs、eval/save 100、patience 8。
- 截止：13:45 graceful stop，14:00 hard stop；每个 cell 保存并恢复 `latest_state`。
- 01:36 已进入 Full 训练；配置/数据 gate 通过，正式 prompt truncation rate=0。
