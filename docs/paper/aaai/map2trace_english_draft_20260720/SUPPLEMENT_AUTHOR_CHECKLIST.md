# 技术补充材料：作者待确认清单

主入口为 `supplementary_material_draft.tex`。红色 `TODO--AUTHOR` 表示无法仅凭当前论文与代码冻结的事实，提交前必须由作者确认或从最终归档产物回填。

## AAAI 模板状态

- 主文件已复用终稿源码中的 `article` + `aaai2027`、Letter 纸张、字体与双栏设置；没有再加载 `geometry`、`hyperref` 等 AAAI 禁用或冲突包。
- 当前保留终稿源码的匿名投稿选项 `\usepackage[submission]{aaai2027}`。若后续文件属于 camera-ready 版本，应按当年 author kit 去掉 `submission`，并同步处理作者信息和版权选项。
- 版式按参考附录 PDF 设置：第一页仅以 “Appendix” 起始，不另设论文标题、作者、摘要或目录；随后使用 A、A.1 的章节编号及 A.1 形式的公式、图、表编号。

## 优先回填

1. **人工标注**
   - annotator 与 adjudicator 的教育/职业背景及相关经验；
   - 是否为论文作者、是否接触过抽样条目；
   - 培训与 calibration 流程、guideline 版本、练习集规模和通过标准；
   - mutual blinding、模型标签/置信度/veracity label 的可见性，以及 adjudicator 可见信息；
   - 招募方式、补偿标准、计费方式、工作量和完成时长；
   - 知情同意，以及伦理审批、豁免或非人类受试研究认定。

2. **冻结版本与完整性**
   - 三个数据集的 release/date、下载地址、许可与原始文件 SHA-256；
   - atomizer、Evidence Mapper、embedding model、verifier backbones 的不可变 revision；
   - prompt/schema、tokenizer/chat template、LoRA adapter、预测文件和缓存的哈希。

3. **实验台账**
   - 主表每个 cell 是单次运行还是多 seed 汇总；
   - RQ2 random/shuffle seeds；
   - RQ3 主设置的 validation 选择准则，以及各容量设置的 realized-\(K\)、token、截断和 underfill 分布；
   - baseline 每一行的输入边界、backbone、原论文表号、split、metric 和 copied/reproduced 状态。

4. **案例与发布**
   - 案例选择规则、event/report/chunk ID、source-link 状态及错误/冲突案例；
   - artifact repository、归档 DOI、release date、release license；
   - 软件锁文件、硬件拓扑、wall time、GPU-hours、API token/费用。

5. **伦理与数据治理**
   - 数据、模型、adapter、API cache 与人工标注导出的再分发许可；
   - 外部 API provider/region、日志与保留策略、是否用于服务商训练、删除机制、访问控制、加密及机构审批；
   - 敏感内容提示、退出机制和 annotator support procedure。

## 已按编辑原则留在正文之外

- 不讨论 SciFact validation 协议差异；
- verifier adaptation 只描述 LoRA；
- 不讨论 gold/teacher ordering 声明；
- BM25-like 仅报告实际实现，不解释参数差异；
- 不复述 SciFact 主结果，也不展开 RAWFC scorer 细节；
- 不解释 Human Audit 样本与数值差异。

## 构建

在本目录执行：

```bash
./build_supplement.sh
```

输出位置为 `dist/supplementary_material_draft.pdf`。当前工作环境需要先提供 `latexmk` 与可用的 PDFLaTeX 发行版。
