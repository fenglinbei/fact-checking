# Exp1 Claim Atomization 人工可靠性分析（仲裁完成）

- 生成时间：`2026-07-18T18:07:20.520020+00:00`
- 写作锚点：`writing_outline_v0.4.2_structure_only.md`。
- 样本：LIAR-RAW / RAWFC validation 各 100 claims，共 200 claims、257 atoms。
- Gold 规则：A/B 完全一致则保留共识；否则由第三位标注者在看不到 A/B 标签时独立仲裁。
- 完成状态：47/47 仲裁任务、50/50 字段决策均通过结构审计并成功消费。
- 结论边界：Exp1 只审计 claim atomization，不验证 Evidence Map 或 downstream causality。

## Final gold 质量率

| 维度 | 单位/N | Gold 通过 | Gold 错误 | 95% claim-cluster CI |
|---|---:|---:|---:|---:|
| 忠实性 | atom/257 | 255/257 (99.22%) | 2/257 (0.78%) | 98.02%--100.00% |
| 原子性 | atom/257 | 246/257 (95.72%) | 11/257 (4.28%) | 92.80%--98.12% |
| 完整覆盖（missed=0） | claim/200 | 198/200 (99.00%) | 2/200 (1.00%) | 97.50%--100.00% |
| 严格三维全通过（次要） | claim/200 | 187/200 (93.50%) | 13/200 (6.50%) | 90.00%--96.50% |

Strict pass 定义为该 claim 的 `completeness_missed=0`，且其所有 atoms 均同时满足 faithfulness=yes 和 atomicity=yes。它是对 atom 数敏感的逻辑合取，只作次要诊断。

## 双标 IAA 与最终 gold 的分工

| 维度 | IAA N | Pre-adj Exact | Cohen κ | Gwet AC1 | Final gold pass |
|---|---:|---:|---:|---:|---:|
| 忠实性 | 257 | 95.72% | 0.133 | 0.955 | 99.22% |
| 原子性 | 257 | 88.72% | 0.306 | 0.866 | 95.72% |
| 完整覆盖 | 199 | 95.48% | -0.023 | 0.953 | 99.00% |

完整覆盖的 pre-adjudication IAA 分母为 199，因为一个 claim 存在标注者内部重复字段冲突；该 claim 已进入第三人队列，因此 final gold 分母恢复为 200。IAA 始终只由原两位标注者计算，第三人结果仅用于形成唯一 gold。多数通过类明显偏斜，故 Exact、κ 与 AC1 需并列解释。

## 数据集分层

| 数据集 | Faithfulness | Atomicity | Complete coverage | Strict pass |
|---|---:|---:|---:|---:|
| liar_raw | 98.54% | 94.89% | 98.00% | 91.00% |
| rawfc | 100.00% | 96.67% | 100.00% | 96.00% |

## 仲裁质量控制

- Atom 项目：37/37；claim completeness 项目：10/10。
- 所有任务恰好一条 active annotation，均由指定仲裁者提交，且全部 `review_complete=confirmed`；0 cancelled、0 drafts、0 notes。
- Atom 决策：atomicity 29（yes=26/no=3），faithfulness 11（yes=10/no=1）；completeness 10（0=8/1=2）。
- Live task data、prepared queues、blind queue、XML config 和 formal A/B source snapshot 的哈希/语义指纹均一致。
- 三个双字段任务严格通过 `questions[i].field` 映射 `decision_i`；未使用顺序不同的 `fields_to_adjudicate` 解释结果。
- `project.result_count=0` 是未刷新的聚合字段；逐任务与 completion 表完整。一个跨日挂页产生 76,919.796 秒 lead-time 离群，不影响标签结构。

## 协议披露

第三人始终看不到 A/B 原始标签，但 pilot project 18 与本次 resolution queue 有 2 个语义任务重合（`liar_raw/8322.json/A1` atomicity, `liar_raw/1082.json` completeness_missed）。因此仲裁对 A/B 标签是盲化的，但这些重合项并非对任务内容的首次接触；该事实保留在 metrics 与最终 gold 产物中，正文不据此作更强的独立性主张。

## 解释

最终 human gold 显示，faithfulness、atomicity 与 complete coverage 的通过率分别为 99.22%、95.72% 和 99.00%，说明当前 LLM atomization 在该审计样本上总体高度符合人工质量判断，可作为后续结构构建的可靠上游输入。Atomicity 错误率为 4.28%，仍是主要残余风险。Pre-adjudication IAA 则表明人工判断过程总体稳定；它与 final gold 的 artifact-quality 结论分属不同证据层。该结论不能外推到 Evidence Map 的 relation/directness/confidence，也不能证明 claim decomposition 因果性地改善 downstream F1。

## 生成文件

- `metrics.json`：仲裁审计、pre-adj IAA、final gold 指标及 bootstrap 区间。
- `adjudication_annotations.jsonl`：按稳定 ID 导出的第三人结果。
- `gold_atom_annotations.jsonl`：257 个 atom 的唯一 gold 与逐字段 resolution provenance。
- `gold_claim_annotations.jsonl`：200 个 claim 的 completeness gold 与严格诊断。
- `paper_insert_v0.4.2.md`：Exp1 正文、Exp2 占位和 Limitations 替换稿。
- `manifest.json`：最终文件哈希；最后发布。
