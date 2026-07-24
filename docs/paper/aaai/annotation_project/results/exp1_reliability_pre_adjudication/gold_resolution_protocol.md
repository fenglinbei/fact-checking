# Exp1 Gold Resolution Protocol

Version: `exp1-exact-gold-resolution-v1-20260717`

本文件区分原标注指导书中的仲裁条件与为形成唯一 exact gold 所需的扩展规则。扩展规则不回写两位标注者的原始标签，也不改变 pre-adjudication IAA。

## 原指导书协议

- faithfulness：两位标注者不一致时，由独立第三人仲裁。
- atomicity：两位标注者不一致时，由独立第三人仲裁。
- completeness_missed：两位标注者的等级差值至少为 2 时仲裁。
- 第三人不查看 A/B 原始标签，先独立作答。

当前对应 **37 条记录 / 36 claims**。其中满足 `completeness_missed` 差值至少 2 的记录为 0。

## Exact-gold 扩展规则

为给后续 gold-based 分析提供唯一标签，扩展队列还纳入：

- 所有 completeness_missed exact mismatches，包括 0 与 1 的分歧；
- 同一标注者在同一 claim 的重复 claim-level 字段中产生的内部冲突；
- 若将来出现辅助 claim_complexity 的内部冲突，可单独校正，但标注者间 complexity 分歧不属于三项质量主指标。

当前扩展队列为 **47 条记录 / 39 claims**。盲化任务使用语义键与待仲裁字段生成稳定 ID，不包含 A/B 标签。

## 输出使用约束

- `metrics.json` 中的 IAA 始终来自原双标，不混入仲裁者。
- 仲裁完成前只报告 pre-adjudication rates/bounds，不称为最终 gold error rate。
- Exp1 只覆盖 claim atomization，不能用来验证 Evidence Map relation/directness/confidence。
