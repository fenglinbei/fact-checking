## Claim Atomization Reliability Study (Exp1)

为审计 claim atomization 这一上游输入，我们从 LIAR-RAW 与 RAWFC validation data 各抽取 100 条 claims（70% 随机、30% 困难优先），得到 257 个 atoms。两位标注者独立评估 faithfulness、atomicity 与 completeness；所有主维度 exact mismatches 以及一个 claim-level 内部冲突均由第三位标注者在看不到 A/B 标签时独立仲裁。

| Dimension | Unit / Gold N | Final gold pass | Pre-adj. Exact | Cohen's $\kappa$ | Gwet AC1 |
|---|---:|---:|---:|---:|---:|
| Faithfulness | atom / 257 | 99.22% | 95.72% | 0.133 | 0.955 |
| Atomicity | atom / 257 | 95.72% | 88.72% | 0.306 | 0.866 |
| Complete coverage | claim / 200 | 99.00% | 95.48% | -0.023 | 0.953 |

**结果表述。** 表X表明，自动 claim atomization 在本次 LIAR-RAW/RAWFC 审计样本上高度符合独立人工质量判断。最终 human gold 中，99.22% 的 atoms 被判定为 faithful，95.72% 满足单一、可独立核验的 atomicity 要求，且 99.00% 的 claims 获得 complete atom coverage；对应的 dataset-stratified claim-cluster bootstrap 95% 区间分别为 98.02%--100.00%、92.80%--98.12% 和 97.50%--100.00%。即使要求同一 claim 的所有 atoms 同时通过 faithfulness 与 atomicity，并完整覆盖原 claim，仍有 187/200（93.50%）达到 strict pass。因此，Exp1 不仅记录了人工审计过程，也支持在本研究覆盖的数据与抽样范围内，将当前 atomization 视为质量较高且可用于后续 retrieval 与 Evidence Map 构建的可靠上游结构输入。残余错误主要集中在 atomicity（4.28%），说明复合命题的拆分粒度仍是后续质量控制最需要关注的环节。

**证据解释。** 上述质量结论来自 final human gold，而人工审计本身的稳定性由 pre-adjudication IAA 单独衡量。两位标注者在三个维度上的 exact agreement 为 88.72%--95.72%，Gwet AC1 为 0.866--0.955，说明独立人工判断的总体模式具有较高可重复性；同时，较低的 Cohen's $\kappa$ 与少数失败类 agreement 表明错误边界仍弱于多数通过类，不能只凭 AC1 消解这一不确定性。换言之，final gold 支持“模型生成的 atoms 基本符合人工质量判断”，IAA 支持“这一人工判断过程总体稳定”，二者不能混为同一个指标。Complete coverage 的 pre-adjudication IAA 基于 199 条 internally consistent claims，冲突项经仲裁后 final gold 分母恢复为 200。该结论不进一步证明 Evidence Map 标注可靠、atomization 带来 downstream performance 提升，或 audit trace 构成模型预测的忠实解释。

## Evidence Map Annotation Reliability Study (Exp2; Placeholder)

**占位说明。** Exp2 将独立审计 Evidence Map 中 candidate--atom pair 的 `relation`、`directness` 与 `confidence` 标注。当前版本只冻结结果位置与报告口径；在正式双标、分歧仲裁和 artifact audit 完成前不填入数值，也不据此扩展本文的结果或贡献表述。

| Field | Human reliability | Planned LLM comparison | Diagnostic |
|---|---|---|---|
| Relation | Cohen's $\kappa$ | Overall and per-relation accuracy | Confusion matrix |
| Directness | Spearman $\rho$ | Ordinal agreement (TBD) | Ordinal error analysis |
| Confidence | TBD after target definition | TBD after target definition | Calibration / ECE target TBD |

表注：Exp2 衡量结构标注本身的可靠性；RQ4 的 component ablation 衡量 map signals 对下游结果的敏感性，二者不是同一个问题。`gold_confidence` 记录的是人工标注者自信度，填入结果前需另行冻结其与 LLM confidence 的比较及校准目标，不能预先把它等同于事实 gold。

### Limitations replacement (v0.4.2 paragraphs 1--2)

首先，claim decomposition 并不总能稳定提高事实核查表现，错误拆分、遗漏限定条件或过度细分仍可能向 retrieval 与 Evidence Map 传播 \citep{Hu2025DecompositionDilemmas}。为审计并量化这一风险，我们在 200 条 claims、257 个 atoms 上完成两位标注者的独立双标与第三人盲化仲裁。最终 gold 的 faithfulness、atomicity 与 complete coverage 通过率分别为 99.22%、95.72% 和 99.00%。这些结果支持本研究审计样本内的 atomization artifact 高度符合人工质量判断，并将 atomicity 定位为主要残余风险。由于样本来自两个 validation set 的等量抽样且过采样困难样本，该结论不能外推为 claim decomposition 普遍可靠，也不能证明它会因果性地改善 downstream verification。

第二，Evidence Map 仍依赖 LLM API，且 Exp1 只覆盖 claim atomization，不能外推为对 relation、directness 或 confidence 标注的验证。Exp2 的独立双标、仲裁与校准分析仍待完成；冻结缓存、prompt/schema hash、调用日期与调用元数据提高了可复现性和 artifact-level 可审计性，但不能将这些结构标注等同于人工 gold supervision。
