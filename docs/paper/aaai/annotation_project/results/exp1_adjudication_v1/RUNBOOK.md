# Exp1 仲裁操作说明

## 项目入口

- Atom 仲裁（37 条）：https://fc.fenglin.pro/projects/20/data
- Completeness 仲裁（10 条）：https://fc.fenglin.pro/projects/21/data
- 仲裁账号：`1349410043@qq.com`（Zijie）

两个项目均为独立盲仲裁项目。页面不包含 Yulin/Zhiqiang 的原始标签，也不包含谁对谁错的信息。

## 标注顺序

1. 先完成 Atom 项目，再完成 Completeness 项目。
2. Atom 页面只会动态显示当前样本实际需要仲裁的 1–2 个问题；显示的问题均须作答。
3. Completeness 页面只判断 `completeness_missed`，不要判断 claim complexity。
4. 英文 claim/atom 是判定依据，中文仅作辅助。
5. 对 `no`、非零 completeness 或边界案例，建议在 notes 中简要写明理由。
6. 不要打开其他人的 Exp1 项目，也不要查看含 A/B 标签的 `adjudication_queue.jsonl`。

## Prior exposure 记录

Zijie 的旧 pilot 曾包含 `liar_raw/8322.json/A1` 和 `liar_raw/1082.json`。其中 `1082.json` 的旧 completeness 填写在不同 atoms 上不一致。两条均应在本次专用项目中从头独立判断，不复用旧 pilot 标签。

## 完成条件

- Project 20：37/37 tasks submitted；对应 40 个必填判断（29 atomicity、11 faithfulness）。
- Project 21：10/10 tasks submitted；对应 10 个 completeness 判断。
- 每条任务只有一个有效提交，提交者必须是 Zijie。

完成后不要手工改写 Yulin/Zhiqiang 的原标注。仲裁结果只在 gold 汇总阶段覆盖指定分歧字段；原双标 IAA 保持不变。

## 审计信息

- 协议：`exp1-exact-gold-resolution-v1-20260717`
- 创建标记：`exp1-exact-gold-adjudication-v1-20260717`
- 启动报告：`launch_report.json`
- 任务清单：`task_manifest.json`
- 启动前在线备份：`label_studio_data/backups/pre_exp1_adjudication_20260717_223713.sqlite3`
- 备份 SHA-256：`d84acb3f3c5faf509032efcc62b40618725ad4345b35255d56ee89ad62d57954`
