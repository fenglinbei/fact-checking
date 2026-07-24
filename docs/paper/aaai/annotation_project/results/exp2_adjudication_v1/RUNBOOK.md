# Exp2 仲裁操作说明

## 项目入口

- Evidence Map 仲裁（125 条）：https://fc.fenglin.pro/projects/22/data
- 项目名：[ZIJIE ONLY] Exp2-Evidence-Map-Adjudication-v1
- 仲裁账号：1349410043@qq.com（Zijie）

这是独立盲仲裁项目。页面不包含 Yulin/Zhiqiang 的原始标签，不包含 LLM 的 relation/directness/confidence，也不显示谁对谁错。

## 仲裁口径

本项目使用 exact-gold 扩展口径：

- 39 条只仲裁 Relation；
- 4 条只仲裁 Directness；
- 82 条同时仲裁 Relation 与 Directness；
- 共 125 个 pair、121 个 Relation 判断、86 个 Directness 判断，合计 207 个字段判断。

Confidence 是标注者对自己判断的自信程度，不属于待裁决 gold 字段，因此本项目不标 Confidence。

## 标注要求

1. 每页只填写实际显示的待仲裁字段；显示的问题均为必填。
2. 只依据当前页面的英文 evidence 与 atom 判断，不结合其他证据，不上网查证。
3. 中文仅用于辅助理解；英文原文是最终判定依据。
4. 不打开 Yulin/Zhiqiang 的 Exp2 项目，也不查看含 A/B 标签的详细仲裁队列。
5. 不尝试猜测 LLM 原标签。
6. 对 refute/qualify、insufficient/background 等边界案例，建议在 notes 中简要写明理由。
7. 提交前勾选独立盲化确认项。

## 源数据提示

- 两条任务缺少 evidence 中文翻译，页面已显示“暂无中文翻译，请以英文为准”，英文完整。
- 一条英文 evidence 继承了源数据中的控制字符，但正文可读，不影响仲裁。
- 这些问题均已记录为非阻塞数据告警，没有修改英文判定内容。

## 完成条件

- Project 22 达到 125/125 tasks submitted。
- 每条任务只有一个有效提交，提交者必须是 Zijie。
- 共应得到 121 个 relation_decision_0、86 个 directness_decision_0 和 125 个 review_complete=confirmed。
- 无 cancelled completion、无遗留 draft、无缺失的动态必填字段。

完成后不要修改 Yulin/Zhiqiang 的原始双标。原双标继续用于 IAA；仲裁结果只在 gold 汇总阶段覆盖当前任务列出的分歧字段。

## 审计信息

- 协议：exp2-exact-gold-resolution-v1-20260719
- 创建标记：exp2-exact-gold-adjudication-v1-20260719
- 盲化源队列 SHA-256：efc15417cad9bdb82dc16b571d7f6b6e7412c054199512b118d7a38eb5852ef0
- Label Studio 任务数据 SHA-256：6bdcc587d7403587a4980958f1742464317361f88f979ef42a7058222bf71469
- 标注配置 SHA-256：cf9b3531bc5c1c1572ab6297fbe5e4102ca7f3e4e7de54ab10392993c173dd29
- 启动前在线备份：label_studio_data/backups/pre_exp2_adjudication_20260719_231804.sqlite3
- 备份 SHA-256：26d49df2aa2e50fdf2cc8c15b693d3d3d2c9dafa828e87ab7e3eaa881bb220ec
- 启动前计划：live_dry_run.json
- 正式启动报告：launch_report.json
- 启动后复核：live_postcheck.json
