## Claim Atomization Reliability Study

为审计 claim atomization 这一上游输入，我们从 LIAR-RAW 与 RAWFC validation data 各抽取 100 条 claims（70% 随机、30% 困难优先），得到 257 个 atoms，并由两位标注者独立评估 faithfulness、atomicity 与 completeness。表中的结果均为 pre-adjudication IAA；一个 claim 存在同一标注者的 claim-level 内部冲突，因此 completeness 暂按 199 条 clean claims 统计。这些人工标签仅用于事后可靠性审计，不参与 atomization 生成、Evidence Map 构建、selector preference 构造、verifier 训练或 checkpoint selection。

| Dimension | Unit / N | Annotator A pass | Annotator B pass | Exact | Cohen's $\kappa$ | Gwet AC1 |
|---|---:|---:|---:|---:|---:|---:|
| Faithfulness | atom / 257 | 98.05% | 96.89% | 95.72% | 0.133 | 0.955 |
| Atomicity | atom / 257 | 94.94% | 87.55% | 88.72% | 0.306 | 0.866 |
| Complete coverage | claim / 199 | 97.99% | 97.49% | 95.48% | -0.023 | 0.953 |

两位标注者在 faithfulness 与 complete coverage 上均给出较高的多数类通过率和 raw exact agreement；atomicity 的通过率差异与分歧更集中。三个维度的类别分布均明显偏斜，且少数失败类别的一致性有限，因此我们并列报告 Exact、Cohen's $\kappa$ 与 Gwet AC1，而不以任一单项系数替代其他证据。该结果只刻画这一等量数据集、困难样本过采样设计下的 pre-adjudication reliability；双人分歧仍待独立仲裁，因而不能视为最终 gold error rate，也不外推到 Evidence Map 标注或 downstream performance。

表注：Faithfulness 与 atomicity 为 atom-level micro statistics；complete coverage 将 `completeness_missed=0` 视为 complete、`>0` 视为 incomplete。完整的 bootstrap intervals、dataset-stratified results、minority-class agreement、strict all-criteria diagnostic 与仲裁队列放入附录。

### Limitations replacement (v0.4.2 paragraphs 1--2)

首先，claim decomposition 并不总能稳定提高事实核查表现，错误拆分、遗漏限定条件或过度细分仍可能向 retrieval 与 Evidence Map 传播 \citep{Hu2025DecompositionDilemmas}。为审计并初步量化这一风险，我们在 200 条 claims、257 个 atoms 上开展了两位标注者的独立双标，分别评估 faithfulness、atomicity 与 completeness。预仲裁结果显示多数通过类别上的 raw agreement 较高，但类别分布明显偏斜，少数失败类别的一致性有限，其中 atomicity 是分歧最集中的维度。由于 claim-level 冲突与双人分歧仍待独立仲裁，这些结果只刻画本研究设计样本上的 pre-adjudication reliability，不能作为最终 gold error rate，也不能证明 claim decomposition 普遍改善 downstream verification。

第二，Evidence Map 仍依赖 LLM API，且上述人工审计只覆盖 claim atomization，不能外推为对 relation、directness 或 confidence 标注的验证。Evidence Map 的人工双标与仲裁尚未完成，self-reported confidence 也未经校准；冻结缓存、prompt/schema hash、调用日期与调用元数据提高了可复现性和 artifact-level 可审计性，但不能将这些结构标注等同于人工 gold supervision。
