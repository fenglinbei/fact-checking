# Exp1 Claim Atomization 人工可靠性分析（预仲裁）

- 生成时间：`2026-07-17T14:26:18.734861+00:00`
- 写作锚点：`writing_outline_v0.4.2_structure_only.md`。
- 正式标注者：**Yulin**、**Zhiqiang**
- 样本：**257 atoms / 200 claims**
- 抽样设计：LIAR-RAW 与 RAWFC 各 100 claims，70% 随机、30% 困难优先；这是设计样本，不是候选总体的自然分布估计。
- 状态：两位标注者的正式双标已齐；双人分歧和一个 claim 内部冲突保持未决，因此本报告不是最终 gold error rate。
- 结论边界：仅审计 claim atomization；不外推到 Evidence Map 的 relation/directness/confidence，也不建立与下游 F1 的因果关系。

## 主要结果

| 维度 | 单位/N | Yulin 通过率 | Zhiqiang 通过率 | Exact | Cohen κ | Gwet AC1 | 未仲裁通过率界 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 忠实性 | atom/257 | 98.05% | 96.89% | 95.72% | 0.133 | 0.955 | 95.33%–99.61% |
| 原子性 | atom/257 | 94.94% | 87.55% | 88.72% | 0.306 | 0.866 | 85.60%–96.89% |
| 完整覆盖（`missed=0`） | claim/199 | 97.99% | 97.49% | 95.48% | -0.023 | 0.953 | 95.48%–100.00% |

完整覆盖的分母为 199：一个 claim 存在同一标注者在不同 atoms 上填写不一致，已暂时排除。未仲裁通过率界不是置信区间；它只表示在逐项独立仲裁后，当前标签允许的通过率范围。

三个维度均由多数通过类主导。类别偏斜会影响 Cohen κ，但少数失败类的一致性也确实有限，因此 Exact、κ、AC1 与少数类 agreement 必须共同解释，不能用 AC1 抵消低 κ。

### 少数失败类一致性

| 维度 / 少数类 | 少数类 agreement | 两人共同判失败 | 二值分歧 |
|---|---:|---:|---:|
| 忠实性 / no | 15.38% | 1 | 11 |
| 原子性 / no | 35.56% | 8 | 29 |
| 完整覆盖 / incomplete | 0.00% | 0 | 9 |

### 95% claim-cluster bootstrap 区间

| 维度 | Yulin 通过率 CI | Zhiqiang 通过率 CI | Exact CI | κ CI | AC1 CI |
|---|---:|---:|---:|---:|---:|
| 忠实性 | 96.36%–99.60% | 94.38%–98.86% | 92.97%–98.08% | -0.029–0.427 | 0.924–0.980 |
| 原子性 | 91.76%–97.67% | 83.20%–91.50% | 84.65%–92.40% | 0.103–0.494 | 0.810–0.912 |
| 完整覆盖 | 95.98%–99.50% | 94.97%–99.50% | 92.46%–97.99% | -0.037–-0.005 | 0.919–0.979 |

区间使用 percentile bootstrap（5000 次）；在 LIAR-RAW 与 RAWFC 内分别按 claim 有放回抽样，并携带该 claim 的全部 atoms。各指标使用 snapshot 中记录的派生 seed。

## 序数完整性与辅助字段

- 原始 `completeness_missed`（0/1/2/3+）Exact：**190/199（95.48%）**。
- Within-1：**199/199（100.00%）**。
- 线性加权 κ：**-0.023**；四类别 nominal AC1=0.954。正文的“完整覆盖”使用二值 `0` vs `>0` AC1，不混用这一四类别 AC1。
- Claim complexity Exact：**174/200（87.00%）**；κ=0.677。该字段仅作辅助分层，不是 LLM 质量维度。

## 数据集分层

| 数据集 | 忠实性 Exact / AC1 | 原子性 Exact / AC1 | 完整覆盖 Exact / AC1 |
|---|---:|---:|---:|
| liar_raw | 94.16% / 0.937 | 88.32% / 0.859 | 92.00% / 0.913 |
| rawfc | 97.50% / 0.974 | 89.17% / 0.873 | 98.99% / 0.990 |

总体值是该等量 claim 设计样本上的 atom-micro / claim rate，不是按原始候选池规模自然加权的总体估计。LIAR-RAW 含 137 atoms、RAWFC 含 120 atoms，因此 atom-level micro 指标仍受每个 claim 的 atom 数影响。

## 次要派生分析：严格三维全通过

严格 claim pass 定义为 `completeness=0` 且该 claim 的每个 atom 均同时满足 faithfulness=yes、atomicity=yes。Yulin 为 91.46%，Zhiqiang 为 81.41%；Exact=81.91%，κ=0.245，AC1=0.764。
逐组件仲裁允许的预仲裁界为 **77.39%–95.98%**（154–191 / 199）。该逻辑合取对 atom 数敏感，只作附录诊断，不作为正文综合质量分。

## 质量控制与待仲裁项

- 标注者内部 claim 冲突：**1**。
- 未提交草稿：**1**。
- Atom 分歧：**37 条，涉及 36 claims**。
- 完整性 Exact 分歧：**9**；按现指导书 `差值 >= 2` 的完整性仲裁项为 **0**。
- 按现指导书的协议仲裁量：**37 条 atom 记录 / 36 claims**。
- 为形成唯一 gold 的扩展 resolution 队列：**47 条记录 / 39 claims**。
- Gold resolution 协议版本：`exp1-exact-gold-resolution-v1-20260717`。现指导书仅要求仲裁完整性 `差值 >= 2`；扩展协议为形成唯一 gold，另纳入所有主维度 Exact mismatch 与内部冲突。
- 未提交草稿不改变 257/257 正式完成数；当前草稿中的未提交修改字段为 `claim_complexity`；不改变三项主要结果。

## 流程证据边界

本次分析输入不含 20 条 calibration 或每 50 条 running-checkpoint 的独立产物，因此不能追溯验证指导书中的过程门槛是否按时执行。这里的 full-sample κ 只描述最终双标结果，不能替代那些过程检查，也不据此声称门槛已通过或未通过。

## 结论

在该预仲裁设计样本上，两位标注者均给出较高的多数类通过率；原子性的通过率和一致性相对最低，三个维度的失败类判定稳定性仍有限，最终 gold 质量率待独立仲裁后确定。该结果缩小了 v0.4.2 对 atomization 风险“尚未量化”的空白，但不支持“claim decomposition 已被普遍验证可靠”或“能改善下游事实核查”的更强结论。

与 v0.4.2 对齐的正文候选段落和 Limitations 替换稿见 `paper_insert_v0.4.2.md`；在完成独立仲裁前，不建议把该诊断加入 Abstract 或贡献列表。

## 生成文件

- `metrics.json`：完整指标与 bootstrap 区间。
- `atom_annotations_a.jsonl`、`atom_annotations_b.jsonl`：两位正式标注者导出。
- `claim_annotations.jsonl`：按 claim 折叠后的标签。
- `disagreements.jsonl`：全部 Exact 分歧。
- `adjudication_queue.jsonl`：含 A/B 标签的仲裁审计队列。
- `adjudication_tasks_blind.jsonl`：不含 A/B 标签、可交给独立仲裁者的任务。
- `data_issues.jsonl`：claim 内部冲突与未提交草稿。
- `gold_resolution_protocol.md`：原指导书与扩展 exact-gold 协议的边界。
- `paper_insert_v0.4.2.md`：正文候选小节与 Limitations 替换稿。
- `manifest.json`：输入指纹、文件哈希与完成标记；最后发布。
