# Typed Role-Rescue 实验计划（2026-07-15）

## 1. 实验问题

本轮不直接证明某个新效用函数已经正确，而先回答三个可检验问题：

1. `target_resolved` 后继续加入同 atom 的独立佐证、反向/限定证据或有效上下文，是否会改变冻结 verifier 的判断？
2. 哪一种角色在相同 evidence-count 约束下提供稳定的正边际，而不是仅由更多 token 或重新训练 verifier 造成表面增益？
3. 将这些角色显式写入选择策略后，经过 policy-matched LoRA 训练是否仍有端到端收益？

主因果结论来自同一个冻结 verifier；各策略独立训练只作为端到端适配结果，不能用于角色归因。

## 2. 固定输入与共同边界

- 数据：LIAR-RAW，train/val/test = 10065/1274/1251。
- 候选池：现有 Atom-Union full pool，最多 20 条，不重跑 retrieval。
- Evidence Map：复用 `04_evidence_map/candidate_evidence_map_features_{split}.jsonl`，不重新调用 LLM。
- 基础顺序：复用 `05_mrec_v0_2_learned_marginal_proxy_fullpool/selection_trace_{split}.jsonl`。
- Resolving core：从 learned-marginal 顺序起点取到首次 `target_resolved`；若前 5 条仍未 resolved，则在第 5 条停止。
- 主容量：除 `R-only` 外使用共同上限 `K_max=5`。去重或候选池结构可能使少数样本低于 5；这些固定容量组必须逐样本保持相同的 selector 层条数。`R-only` 保留 1--5 条的真实 core 长度，用作压缩诊断，不与固定容量组直接宣称容量匹配。
- Prompt：`mrec_min`、full evidence text、1024 tokens；记录实际 evidence/prompt token 与截断率。
- Prompt-feasible 配对：七个固定容量组在每个样本上共同取其 diagnostic 可见条数的最小值，再做无截断重建，保证 verifier 实际看到的 evidence count 也逐样本相同。
- 随机性：所有随机补位由 `(seed, event_id, candidate_uid)` 的 SHA-256 顺序确定。
- 禁止信号：选择过程不读取 gold label、verifier 输出或 verifier reward。

## 3. 角色定义

角色证据只从 resolving core 之后的 learned-marginal suffix 中选择，并保持原 learned 顺序作为角色内优先级。

| 角色 | 条件 | 去重/封顶 |
|---|---|---|
| Cor | 同 atom、同 support/refute 方向、direct/partial，且相对 core 是新的非空 `source_group` | 每个 atom-direction 槽最多一条；这是跨 source-group 佐证代理，不等价于已验证的来源独立性 |
| Opp | 同 atom 的 support/refute 反方向，或 qualify/mixed/有效 CONTRAST | 每个 atom 槽最多一条 |
| Ctx | 同 atom 的 background/context，或非 irrelevant 且带 background-context/partial/context/span 信号的 insufficient | 每个 atom 槽最多一条；confidence 仅表示标注置信度，不单独把 `insufficient-none-irrelevant` 变成有效上下文 |

同文本或同 duplicate group 的候选不能作为新增角色证据。

## 4. 实验单元

| Cell | 选择规则 | 作用 |
|---|---|---|
| `r_only` | 仅 resolving core，长度 1--5 | 过早饱和/容量压缩诊断 |
| `random` | core + 稳定随机补到 `K_max=5` | 固定上限随机容量控制 |
| `retr` | core + retrieval 顺序补到 `K_max=5` | 固定上限常规检索控制 |
| `cor` | core + Cor 槽优先，再从全部未选且非重复候选按共同随机顺序补到 5 | 独立佐证边际 |
| `opp` | core + Opp 槽优先，再从全部未选且非重复候选按共同随机顺序补到 5 | 反向/限定证据边际 |
| `ctx` | core + Ctx 槽优先，再从全部未选且非重复候选按共同随机顺序补到 5 | 背景/桥接边际 |
| `full` | core + 所有可用 atom-role 槽按 learned rank选择，再随机补到 5 | typed portfolio 初版 |
| `learned_fixed5` | 原 learned-marginal 顺序前 5 条，再进入共同 prompt-feasible 容量投影 | 同上限的当前顺序基线；仅用于冻结 verifier 比较，本轮不单独训练 |
| `native_gate_anchor` | 冻结 reference contract 中的 BACES prompt-feasible cell | 仅验证 matrix inference 与 native inference 等价，不进入方法比较或新训练 |

每个 trace 必须记录：角色可用率、显式提升率、最终角色实际出现率、与 random/retr 的同集率、原始 source index/UID、evidence count 与 token cost。

## 5. 两层评估

### A. 冻结 verifier 角色归因（主实验）

复用已完成的 `learned_marginal_proxy_fullpool_minmax5_10` best LoRA checkpoint，对 val 的全部 cell 做一次去重 matrix inference。该层固定模型权重，因此用于回答证据集合本身的作用。

主检验预注册为 `full - random`。其余角色比较作为次要/诊断检验，并对
Macro-F1 paired-randomization p-value 报告 Holm 校正：

- `cor - random`
- `opp - random`
- `ctx - random`
- `full - random`
- `full - retr`
- `full - learned_fixed5`
- `r_only - random`（仅作容量差异诊断）

报告 Macro-F1/accuracy/NLL、paired bootstrap 95% CI、paired randomization、exact McNemar、wrong-to-correct、correct-to-wrong，并同时报告角色 availability 与 realized selection rate。

逐样本匹配 evidence count 只消除条数差异，不保证 evidence token 完全相同；在 matched-token 实验完成前，不把冻结层差异表述为已经完全排除了 token 容量混杂。单一 random seed 用于本轮主队列，最终稳定性结论需补充多个稳定随机 seed 的冻结层复算。

### B. Policy-matched LoRA（次实验）

对七个新 cell 使用完全相同训练配方分别训练：Ministral-3-8B、LoRA r16/alpha32/dropout0.1、4 GPU x GA4 = EBS16、LR 2e-5、12 epochs、eval/save 100、patience 8、相同 LIAR class weights。

训练顺序按信息价值排列：

1. `full`
2. `random`
3. `retr`
4. `cor`
5. `opp`
6. `ctx`
7. `r_only`

该顺序保证队列即使在本轮 GPU 截止点中断，最先完成的三个单元也能形成
`Full / Random / Retrieval` 的完整端到端容量控制；后续角色单元按边际归因依次补齐。

分别训练的结果只说明策略经过 verifier adaptation 后的端到端上限，不用于声称某种角色具有因果贡献。

## 6. 容量后续实验

固定 K=5 完成角色筛选后，仅对通过筛选的 `full` 目标增加：

- matched-token：与 minmax5_10 相同平均 prompt token；
- adaptive：选择达到 pool-relative typed utility `(1-delta)` 的最小集合；
- same-set order：集合固定，仅比较 typed acquisition order 与 learned order。

本轮不把 adaptive 与角色归因混在同一主表中。

## 7. Go / No-Go

将 typed portfolio 升为论文主方法至少满足：

1. 在冻结 verifier 下，Full 或某个结构角色相对 random/retr 有稳定 paired 改善，且不是仅由可用样本占比造成；
2. Full 的 human sufficiency 后续标注优于 R-only；
3. policy-matched Full 在 val 上不劣于 minmax5_10，并在 matched-token 下保持收益或节省容量；
4. richer objective 不再让绝大多数样本在一条证据后完全饱和；
5. exact/typed 方案相对“OPEN -> Opp -> Cor -> Ctx”的手工策略在非平凡比例样本上改变集合，否则 exact solver 降为 reference oracle。

如果只有独立训练有效、冻结 verifier 无角色边际，则优先解释为 verifier adaptation，而不能据此验证 typed evidence utility。
