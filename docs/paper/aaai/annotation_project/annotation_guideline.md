# LLM 标注可信度人工评测：标注指导书

> **用途**：用于论文"API 标注可信度实验"的人工 gold 标注。
> **适用实验**：可信度实验 1（Atom 忠实性/完整性/原子性）+ 可信度实验 2（Evidence Map 标注准确率）。
> **标注工具**：Label Studio。
> **标注者**：2 位 NLP 研究背景标注者独立双标，分歧由项目作者裁决产生 gold。
> **数据语言**：英文（LIAR-RAW / RAWFC）。
> **指导书语言**：中文说明 + 英文示例。

---

## 0. 背景与目标

本项目用 LLM（DeepSeek-V4-Flash）自动完成两件事：
1. **Claim Atomization**：把一条声明（claim）拆成若干可独立验证的原子命题（atom）。
2. **Evidence Map**：对每条候选证据（evidence）与每个 atom，标注它们之间的关系。

这套自动标注驱动了后续的证据选择与判别。本评测的目的是**人工核查这些自动标注是否可信**——不是去做事实核查本身，而是判断"LLM 拆得对不对、标得对不对"。

**关键区分**：你不是在做事实核查（不需要上网查证 claim 是否为真），而是在评估 LLM 产出的结构化标注的质量。

---

## 1. 核心概念

### 1.1 Claim 与 Atom

- **Claim（声明）**：一条待核查的事实断言，如：
  > "We have less Americans working now than in the 70s."

- **Atom（原子命题）**：从 claim 中拆出的、最小可独立验证的事实断言。一条 claim 可能拆出 1–6 个 atom。例如上面 claim 可能拆出：
  > A1: The number of Americans working now is less than in the 1970s.

### 1.2 Evidence（证据）

- 与 claim 相关的原始报道中的片段（chunk），已由系统切分好。每条证据有编号 E01, E02, …。

### 1.3 Relation（关系）

LLM 为每个 (evidence, atom) 对标注一个 relation，取值固定为以下 7 类之一：

| Relation | 含义 | 判断要点 |
|---|---|---|
| **support** | 证据支持该 atom 为真 | 证据提供了该 atom 为真的依据 |
| **refute** | 证据反驳该 atom（证明其为假） | 证据提供了该 atom 为假的依据 |
| **qualify** | 证据部分支持/部分反驳，或给出限定条件 | 证据既非完全支持也非完全反驳，添加了 nuance |
| **mixed** | 证据对该 atom 同时含有支持和反驳的成分 | 证据内部存在矛盾立场 |
| **insufficient** | 证据与该 atom 有关，但信息不足以判断 support/refute | 证据提到了相关实体但未给出可验证的立场 |
| **background** | 证据仅提供背景上下文，不直接涉及该 atom 的真值 | 证据提供 context 但不判断 atom 对错 |
| **irrelevant** | 证据与该 atom 无关 | 证据与该 atom 没有语义关联 |

### 1.4 Directness（直接性）

LLM 标注证据对该 atom 的直接程度，4 级：

| Directness | 含义 |
|---|---|
| **direct** | 证据直接陈述了该 atom 的真值（可直接判断对错） |
| **partial** | 证据间接涉及该 atom，需推理才能联系 |
| **context** | 证据仅提供背景上下文 |
| **none** | 证据与该 atom 无直接关联（通常配 irrelevant） |

### 1.5 Confidence（置信度）

LLM 给出的 $c \in [0,1]$，表示它对这一标注的自信程度。本评测中你也需要给出你的 confidence（0–1），用于后续校准分析。

---

## 2. 可信度实验 1：Atom 质量评测

### 2.1 任务说明

对每条 claim，系统给出了 LLM 拆出的 atoms。你需要在 **3 个维度** 上评估每个 atom，**不需要改写 atom**。

### 2.2 评测维度

#### 维度 A：忠实性（Faithfulness）—— 二值（yes/no）

> 该 atom 的语义是否能 **完全从 claim 本身推出**，且 **没有引入 claim 中不存在的信息**？

- **yes**：atom 是 claim 的忠实改写/拆分，没有添加外部信息。
- **no**：atom 引入了 claim 中没有的事实、实体、因果或立场（即"幻觉"）。

**判定要点**：
- 同义改写算 yes（如 "cant do squat" → "cannot do anything"）。
- 引入 claim 没说的数量、时间、主体、因果 = 幻觉 = no。
- 不需要判断 atom 本身是真是假，只判断它是否"忠于 claim"。

**示例**：
- Claim: "The insurance commissioner cant do squat about health care."
- Atom: "The insurance commissioner cannot do anything about health care." → **yes**（同义改写）
- Atom: "The insurance commissioner lacks authority over health care regulation." → **no**（引入了"regulation"概念，claim 没说）

#### 维度 B：完整性（Completeness）—— **Claim 级**标注，单选（漏检断言数分档）

> claim 中有多少个 **可独立验证的事实断言** 没有被 **该 claim 的全部 atoms** 覆盖？

**⚠ 重要：这是 claim 级标注，不是 atom 级。** 判定时必须对照界面上方展示的"该 claim 的全部 atoms 列表"（A1/A2/…），统计所有 atoms 合起来覆盖了 claim 的几个断言、漏掉了几个。多 atom claim 只需在全景区填一次。

从以下选项中选择一个：
- **0**：完整覆盖，claim 中所有可独立验证断言都有对应 atom。
- **1**：漏掉 1 个可独立验证断言。
- **2**：漏掉 2 个可独立验证断言。
- **3+**：漏掉 3 个或更多可独立验证断言。

**判定要点**：
- "可独立验证" = 这个断言可以单独被证据支持/反驳。
- 形容词/程度副词如果是可验证的（如"most"、"highest"），算独立断言。
- 纯语法成分（时态、冠词）不算断言。
- 如果 claim 只有一个断言且有一个 atom 覆盖，选 **0**。
- **覆盖是合起来的**：A1 覆盖断言 1、A2 覆盖断言 2，则两个断言都已覆盖，漏检数=0；不要因为"A1 没覆盖断言 2"就填 1。

**示例**：
- Claim: "Tim Kaine urged $500 billion in Medicare cuts."
- Atoms: [A1: "Tim Kaine urged $500 billion in Medicare cuts."]
- 漏检数 = **0**（单一断言，单一 atom，完整覆盖）
- Claim: "Tom said X and Jerry said Y." ｜ Atoms: [A1: "Tom said X."]
- 漏检数 = **1**（Jerry said Y 这个断言没有任何 atom 覆盖）
- 若 atoms 为空或缺少"urged"这个动作断言 → 选 **1** 或更高。

#### 维度 C：原子性（Atomicity）—— 二值（yes/no）

> 该 atom 是否已经是 **最小可独立验证** 的命题？（即：能否再拆出更小的可独立验证子断言？）

- **yes**：atom 是最小粒度，无法再拆出可独立验证的子断言。
- **no**：atom 含有 ≥2 个可独立验证断言（黏在一起），应该进一步拆分。

**判定要点**：
- "Tom said X and Jerry said Y" 应拆成两个 atom → 黏在一起 = no。
- "X happened in 2019" 不需要再拆出"2019"为单独 atom（时间是断言的一部分，不是独立断言）→ yes。
- 含并列主语/谓语且各自可验证 = no。

**示例**：
- Atom: "There are four combat-ready brigades and 40 brigades total in the U.S. Army." → **no**（两个可独立验证断言黏在一起）
- Atom: "There are four combat-ready brigades in the U.S. Army." → **yes**（最小可验证）

### 2.3 标注界面（Label Studio）

每条样本展示：claim 文本 + LLM 拆出的 atoms 全景列表。对每个 atom 标注：

| 字段 | 类型 | 取值 |
|---|---|---|
| `atom_id` | 只读 | A1, A2, … |
| `faithfulness` | 单选 | yes / no |
| `atomicity` | 单选 | yes / no |
| `notes` | 文本 | 可选备注 |

对整条 claim 标注（claim 级，每个 claim 填一次）：
| 字段 | 类型 | 取值 |
|---|---|---|
| `claim_complexity` | 单选 | simple（单一断言）/ compound（多断言） |
| `completeness_missed` | 单选 | 0 / 1 / 2 / 3+（对照全部 atoms 合起来的覆盖判定） |

### 2.4 IAA 计算

- faithfulness / atomicity：两位标注者的 Cohen's κ。
- completeness_missed：两位标注者的 exact-match 率 + 若差异 ≤1 视为一致的宽松一致率。

---

## 3. 可信度实验 2：Evidence Map 标注准确率评测

### 3.1 任务说明

对每个 (evidence, atom) 对，系统给出了 LLM 标注的 (relation, directness, confidence)。你需要 **独立给出你的 gold 标注**，后续与 LLM 标注比对。

### 3.2 标注内容

每个 (evidence, atom) 对，你需要标注：

| 字段 | 类型 | 取值 |
|---|---|---|
| `gold_relation` | 单选 | support / refute / qualify / mixed / insufficient / background / irrelevant |
| `gold_directness` | 单选 | direct / partial / context / none |
| `gold_confidence` | 滑块 | 0.0–1.0（你对这个标注的自信程度） |
| `notes` | 文本 | 可选，尤其当 relation 难判断时说明理由 |

### 3.3 Relation 判定流程（务必按顺序）

遇到一个 (evidence, atom) 对，按以下顺序判断：

```
1. 证据与 atom 是否有语义关联？
   ├─ 否 → irrelevant (directness=none)
   └─ 是 → 继续

2. 证据是否直接涉及 atom 的真值（对错）？
   ├─ 否，仅提供背景 → background (directness=context)
   └─ 是 → 继续

3. 证据信息是否足以判断 support/refute？
   ├─ 否，信息不完整 → insufficient (directness=partial)
   └─ 是 → 继续

4. 证据对 atom 的立场是？
   ├─ 完全支持 → support (directness=direct)
   ├─ 完全反驳 → refute (directness=direct)
   ├─ 部分支持/部分反驳或加限定 → qualify (directness=partial)
   └─ 同时支持和反驳（证据内部矛盾） → mixed (directness=partial)
```

### 3.4 Relation 判定示例

**示例 1：support / direct**
- Atom: "Tim Kaine urged \$500 billion in Medicare cuts."
- Evidence: "Tim Kaine urged \$500 billion in Medicare cuts."
- → **support / direct**（证据直接陈述了 atom 的内容）

**示例 2：refute / direct**
- Atom: "In October 2019, Barnes & Noble sold a satire as a children's book."
- Evidence: "The official statement by Barnes and Noble was that the book was an adult parody that's to be displayed in the adult section."
- → **refute / direct**（证据直接否定 atom：不是儿童书区，是成人区）

**示例 3：background / context**
- Atom: "There are four combat-ready brigades in the U.S. Army."
- Evidence: "Only two under-strength Marine and four skeletonized Army divisions remained."
- → **background / context**（证据提到了军队编制，但说的是 division 不是 brigade，且未直接判断"四个战斗准备旅"是否为真）

**示例 4：qualify / partial**
- Atom: "We have less Americans working now than in the 70s."
- Evidence: "Since the 1980s, Americans have quit less, and many have clung to crappy jobs."
- → **qualify / partial**（证据部分相关——暗示就业稳定性变化，但未直接验证"工作人数少于 70 年代"这一数量断言）

**示例 5：irrelevant / none**
- Atom: "Tim Kaine urged \$500 billion in Medicare cuts."
- Evidence: "May 1, 2001 — Americans work more than anyone in the industrialized world."
- → **irrelevant / none**（证据讨论的是美国人工作量，与 Tim Kaine 的 Medicare 削减无关）

### 3.5 判定原则

1. **只看 evidence 文本本身**，不结合其他证据、不上网查证。
2. **gold_relation 是你对证据-atom 关系的独立判断**，不要被 LLM 给出的标注影响（标注界面会隐藏 LLM 的 relation，只在最后比对时揭示）。
3. **refute 与 qualify 的区分**：refute = 证据明确证明 atom 为假；qualify = 证据既非完全支持也非完全反驳，添加了 nuance 或条件。如果你拿不准，倾向 qualify。
4. **support 与 insufficient 的区分**：support = 证据足以支撑 atom 为真；insufficient = 证据相关但信息不足以做判断。
5. **confidence 含义**：0 = 纯猜，0.5 = 一半把握，1 = 完全确定。用于后续校准分析，请诚实填写。

### 3.6 采样说明

(evidence, atom) 对按 **自然分布** 从候选池中采样（不做类别均衡）。因此你可能遇到某些 relation 类型（如 refute、mixed）样本偏少，这是正常的——反映真实分布。

### 3.7 IAA 计算

- gold_relation：Cohen's κ + 整体准确率 + per-relation 准确率 + 混淆矩阵。
- gold_directness：Spearman ρ。
- gold_confidence：与 LLM confidence 的相关性 + 后续 ECE 校准分析。

---

## 4. 仲裁机制

### 4.1 分歧识别

两位标注者独立标注完成后，逐样本比对：

**实验 1（Atom）分歧条件**：
- faithfulness 不一致（一人 yes 一人 no）
- atomicity 不一致
- completeness_missed 差异 ≥ 2

**实验 2（Map）分歧条件**：
- gold_relation 不一致
- gold_directness 差异 ≥ 2 级（如 direct vs none）

### 4.2 仲裁流程

1. 项目作者作为第 3 人，对分歧样本重新标注（不看两位标注者的原始标注，独立判断）。
2. 仲裁结果作为 gold label。
3. 仲裁时若发现标注指导书有歧义，更新指导书并通知标注者，已标样本不回溯（记录版本号）。

### 4.3 IAA 与 Gold 的关系

- **IAA**：用两位标注者的双标结果计算（不含仲裁者），反映标注任务的清晰度。
- **Gold**：分歧样本用仲裁结果，非分歧样本用两人一致结果。Gold 用于后续可信度实验 3（校准）和实验 4（噪声注入/gold 上界）。

---

## 5. 标注流程与质量控制

### 5.1 流程

```
阶段 0  标注者培训（30 min）
         ├─ 阅读本指导书
         ├─ 共同标注 20 条 calibration 样本
         └─ 讨论分歧，对齐标准
         ↓
阶段 1  正式标注（独立双标）
         ├─ 实验 1: 200 claim 的 atom 评测
         └─ 实验 2: 200–300 (evidence, atom) pair 的 map 评测
         ↓
阶段 2  分歧仲裁（项目作者）
         ↓
阶段 3  IAA 计算 + gold 汇总
```

### 5.2 质量控制

- **Calibration**：正式标注前 20 条 calibration 样本，要求两位标注者 κ ≥ 0.6 方可进入正式标注；否则再讨论一轮。
- **进度检查**：每完成 50 条，计算一次 running κ，若低于 0.5 暂停讨论。
- **时间预估**：实验 1 约 3–4 秒/atom，实验 2 约 10–15 秒/pair。总计约 15–20 工时/人。

### 5.3 Claim 抽样策略（实验 1）

- **70% 随机**：从 LIAR-RAW（100 条）和 RAWFC（100 条）验证集均匀随机抽样。
- **30% 困难优先**：优先选 claim 文本长度 > P75、含否定/比较/数量/日期等易拆错特征的 claim。困难样本从同一验证集中按特征过滤抽取。

### 5.4 Pair 抽样策略（实验 2）

- 从实验 1 的 200 条 claim 对应的候选池中，**按 relation 自然分布** 采样 (evidence, atom) 对。
- 目标 200–300 pair，确保每条 claim 贡献 1–2 个 pair。
- 若某 relation 类型（如 mixed）自然占比 <3%，不做人为补足，如实反映分布。

---

## 6. 数据格式

### 6.1 实验 1 输入样本（Label Studio 导入）

```json
{
  "event_id": "2020.json",
  "dataset": "liar_raw",
  "claim": "The insurance commissioner cant do squat about health care.",
  "atoms": [
    {"atom_id": "A1", "proposition": "The insurance commissioner cannot do anything about health care.", "type": "attribution"}
  ]
}
```

### 6.2 实验 1 输出标注

```json
{
  "event_id": "2020.json",
  "annotator": "A",
  "claim_complexity": "simple",
  "atom_annotations": [
    {"atom_id": "A1", "faithfulness": "yes", "completeness_missed": 0, "atomicity": "yes", "notes": ""}
  ]
}
```

### 6.3 实验 2 输入样本

```json
{
  "event_id": "6117.json",
  "dataset": "liar_raw",
  "claim": "Tim Kaine urged $500 billion in Medicare cuts.",
  "atom_id": "A1",
  "atom_proposition": "Tim Kaine urged $500 billion in Medicare cuts.",
  "evidence_id": "E01",
  "evidence_text": "Tim Kaine urged $500 billion in Medicare cuts."
}
```

### 6.4 实验 2 输出标注

```json
{
  "event_id": "6117.json",
  "evidence_id": "E01",
  "atom_id": "A1",
  "annotator": "A",
  "gold_relation": "support",
  "gold_directness": "direct",
  "gold_confidence": 0.95,
  "notes": ""
}
```

---

## 7. 常见问题（FAQ）

**Q1: atom 的 proposition 和 claim 完全一样（没拆分），算忠实性 yes 吗？**
A: 算 yes。单一断言的 claim 拆出 1 个与原文相同的 atom 是合理的，忠实性只看有没有引入幻觉。

**Q2: evidence 提到了 atom 中的实体，但讨论的是另一件事，算什么 relation？**
A: 算 irrelevant。只提实体不构成对 atom 真值的判断。

**Q3: evidence 来自与 atom 立场相反的报道，但说的是另一件事，算 refute 吗？**
A: 不算。refute 必须是证据 **直接否定 atom 的真值**。报道立场相反但内容不直接对应 = irrelevant 或 background。

**Q4: claim 含讽刺/反讽，atom 字面化了，算幻觉吗？**
A: 如果 atom 字面化了反讽且 claim 本意非字面，算 no（引入了 claim 没有的"字面为真"假设）。在 notes 里说明。

**Q5: evidence 里有多个句子，有的支持有的反对，算 mixed 还是 qualify？**
A: 如果 evidence 内部同时含 support 和 refute 立场 = mixed；如果只是加了条件/nuance 但不矛盾 = qualify。

**Q6: 我对 relation 判断不确定，confidence 该填多少？**
A: 诚实填写。0.3 = 倾向某个但很不确定，0.6 = 较有把握，0.9 = 很确定。这些值用于后续校准分析，不追求"高分"。

**Q7: 同一条 evidence 对同一 atom 可能既有 direct 又有 context 成分，directness 怎么选？**
A: 取最直接的成分。只要有任何一句能 direct 判断 atom 真值，就选 direct。

**Q8: completeness_missed 如果 claim 有 3 个断言但 atoms 只覆盖 2 个，填 1 还是填漏掉的内容？**
A: 填数字 1（漏掉 1 个断言）。在 notes 里可补充漏掉的断言内容，但字段只填数字。**注意 completeness 是 claim 级标注**：对照界面上方"全部 atoms 列表"合起来覆盖了几个断言来判定，不要只看当前 atom。多 atom claim 只需在全景区填一次。

---

## 8. 附录：Relation 与 Directness 的合法组合参考

并非所有组合都常见，下表给出参考（不强制，按实际情况标注）：

| Relation | 典型 Directness | 说明 |
|---|---|---|
| support | direct / partial | 直接支持多为 direct，间接推理支持多为 partial |
| refute | direct / partial | 同上 |
| qualify | partial | 限定条件通常是 partial |
| mixed | partial | 内部矛盾通常需要 partial 推理 |
| insufficient | partial | 信息不足通常仍有部分关联 |
| background | context | 背景上下文 |
| irrelevant | none | 无关联 |

---

*指导书版本：v1.0。若标注过程中发现歧义需更新，在此记录版本号与变更内容。*

---

## 9. Exp3/Exp4：证据组织与状态转移双标

本节只适用于新增的 LIAR-RAW trace-alignment 小规模实验。它与前述 Exp1/Exp2
审计并行存在，不替代 Atomization 或 Evidence Map 审计。与第 4 节不同，本实验由
两位非作者标注者独立双标，**不做作者仲裁**；所有主观分歧原样保留，不产生 gold。
这里的 reasoning trace 仅指外显的证据选择顺序和 atom-state transition，不是模型
latent chain-of-thought。

### 9.1 通用规则

1. 英文是权威文本，中文只用于辅助理解；若二者冲突，以英文为准并勾选
   `data_issue=translation`（仅 preference 页面提供该字段）。
2. 只依据页面展示的 claim 和 evidence，不上网、不调用外部知识，也不要猜测
   A/B 或状态转移由哪个系统产生。
3. 不与另一位标注者讨论具体题目，不查看对方项目或私有 key。
4. 页面没有 confidence，也不要求找“最早已经充分”的证据前缀。
5. pilot 只用于检查界面和说明，不进入论文统计；不得根据 pilot 的效果方向改变是否
   开展正式实验。
6. 必须等待管理员通知后再进入下一阶段：pilot preference → formal preference →
   pilot transition → formal transition。Preference 全部完成前不得打开会显示状态的
   Transition 项目。
7. 页面标题中的 revision（如 `v1`）必须与管理员通知一致。若 pilot 后修改界面、
   翻译或说明，管理员会递增 revision 并重新开放该 revision 的 pilot；旧 revision
   只作审计，不得继续用于新 revision 的 formal 标注。

### 9.2 Exp3 Preference：有序证据序列比较

固定问题是：

> 哪一个有序证据序列更能帮助独立事实核查者形成准确且有依据的判断？

请比较完整的 Sequence A 与 Sequence B，综合考虑：

- 证据是否与 claim 的关键核查点相关；
- 是否包含形成有依据判断所需的支持、反驳、限定或必要背景；
- 证据顺序是否帮助逐步建立、修正或核对判断；
- 无关、重复或过早出现的背景是否妨碍核查。

**不得按文风、流畅度、篇章美感或“说服力”选择。** 两侧使用同一中性展示格式；
来源编号和 domain 只用于区分证据，不代表来源可信度排名。五级选项含义如下：

| 提交值 | 含义 |
|---|---|
| `strongly_prefer_a` | A 在核查用途上明显优于 B |
| `prefer_a` | A 略优于 B，但差距有限 |
| `tie` | 两者对独立事实核查的帮助基本相当 |
| `prefer_b` | B 略优于 A，但差距有限 |
| `strongly_prefer_b` | B 在核查用途上明显优于 A |

`data_issue` 与 `notes` 均为可选。仅对真实的数据/显示问题勾选
`translation`、`missing_or_malformed_evidence`、`duplicate_evidence`、
`source_or_format` 或 `other`；不要把“我更偏好另一侧”当作数据问题。若页面仍可判断，
应照常提交偏好并在 notes 说明。后续排除规则会在解盲前统一冻结。

### 9.3 Exp4 Transition：状态转移与边际贡献

Transition 页面依次展示 focal atom、状态图例、此前同一 atom 的证据前缀、当前证据
和 proposed before→after state。状态只表示当前外显证据组织下的摘要：

| 状态 | 含义 |
|---|---|
| U | Unresolved：尚未形成支持、反驳或限定判断 |
| S | Supported：当前证据总体支持 focal atom |
| R | Refuted：当前证据总体反驳 focal atom |
| Q | Qualified/mixed：存在限定、部分支持/反驳或混合信息 |
| C | Conflict：支持与反驳证据发生冲突 |

状态映射的固定口径为 support→S、refute→R、qualify/mixed→Q；insufficient、
background、irrelevant 本身不改变 U。已有 S 与 R 相遇时产生 C。请分别回答两个
互不替代的问题：

**Transition validity**

- `valid`：结合此前前缀，当前证据足以支撑 proposed 状态更新；对于 self-transition，
  保持原状态也恰当。
- `partially_valid`：证据与更新方向有关，但只能支持较弱/更有限的更新，或关键依据仍
  不完整。
- `invalid`：当前证据不能支撑 proposed 更新，或更合理的 after-state 明显不同。

**Marginal contribution**

- `clear`：相对此前前缀，当前证据增加了明确且影响判断的新信息；
- `limited`：增加了一些相关信息、佐证或限定，但增量较小；
- `none`：没有新增与判断相关的信息，主要是重复、无关或纯背景。

Validity 与 marginal contribution 必须独立判断。例如，一个维持 S→S 的重复佐证可能
是 `valid + limited`，也可能在完全重复时是 `valid + none`；不要因为状态未变化就自动
判 invalid，也不要因为证据相关就自动判 clear。每个 claim 在正式 transition 样本中
最多出现一个 focal step。

---

*指导书版本补充：v1.1（2026-07-24），新增 Exp3/Exp4 随机化、system-blind、
double-annotated protocol；不改变既有 Exp1/Exp2 指导与其历史仲裁流程。*
