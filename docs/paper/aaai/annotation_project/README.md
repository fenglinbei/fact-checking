# 标注项目：LLM 标注可信度人工评测

本目录包含"API 标注可信度实验"的全部文件——指导书、Label Studio 配置、抽样脚本、待标注数据、标注结果。所有文件集中在此目录内，便于管理与开源发布。

## 在线标注平台

- **访问地址**：https://fc.fenglin.pro
- **登录账号**：`admin@annotation.local` / `annotation2026`
- **当前独立标注项目**（2026-07-11 修复后）：
  - `[YULIN ONLY] Exp1-Atom-Quality`（ID=14）：257 个任务，已迁入 38 条
  - `[ZHIQIANG ONLY] Exp1-Atom-Quality`（ID=15）：257 个任务，已迁入 50 条
  - `[YULIN ONLY] Exp2-Evidence-Map`（ID=16）：250 个任务，已迁入 48 条
  - `[ZHIQIANG ONLY] Exp2-Evidence-Map`（ID=17）：250 个任务，暂未开始
  - `[ZIJIE ONLY] Exp1-Atom-Quality`（ID=18）：257 个任务，已迁入 16 条
  - `[ZIJIE ONLY] Exp2-Evidence-Map`（ID=19）：250 个任务，已迁入 12 条
- **账号对应**：Yulin=`1849812973@qq.com`，Zhiqiang=`3180643570@qq.com`，Zijie=`1349410043@qq.com`。每人只进入带自己名字的 `ONLY` 项目。
- 旧混合项目 ID=12/13 已软归档，数据仍完整保留，仅用于审计。

### 部署架构

```
标注者浏览器 → HTTPS → fc.fenglin.pro (公网 nginx)
                         ↓ proxy_pass :18080
                     公网 sshd 反向隧道 (:18080)
                         ↓ SSH -R
                     本地 Label Studio (:8090)
```

- **本地 Label Studio**：conda 环境 `fc-annotation`，数据目录 `label_studio_data/`，由 `scripts/keepalive.sh` 保活
- **SSH 反向隧道**：autossh 本地 8090 → 公网 18080，同一保活脚本负责检查和重建
- **nginx**：根路径 `/` 反代 18080（Label Studio），`/evidence-map/` 保留给 case study

### 运维命令

```bash
# 查看本地 Label Studio、保活和隧道状态
curl -sI http://127.0.0.1:8090/
tail -f label_studio_data/keepalive.log
tail -f label_studio_data/tunnel.log

# 重启 Label Studio（如需）
kill "$(cat /tmp/ls_keepalive.pid)" 2>/dev/null || true
kill "$(cat /tmp/ls_labelstudio.pid)" 2>/dev/null || true
cd /data/liaozijie/fact-checking/docs/paper/aaai/annotation_project
nohup bash scripts/keepalive.sh >> label_studio_data/keepalive.log 2>&1 &

# 重启隧道（如需）
kill "$(cat /tmp/ls_tunnel.pid)" 2>/dev/null || true
# keepalive 会在下一轮自动重建隧道

# 重载 nginx（修改配置后）
ssh dig "nginx -t && nginx -s reload"
```

## 目录结构

```
annotation_project/
├── README.md                      # 本文件
├── annotation_guideline.md        # 标注指导书（中文说明 + 英文示例）
├── config/
│   ├── exp1_atom_quality.xml      # Label Studio 配置：实验1 Atom 质量评测
│   └── exp2_evidence_map.xml      # Label Studio 配置：实验2 Evidence Map 评测
├── scripts/
│   ├── export_tasks.py            # 抽样脚本：从已有产物导出待标注任务
│   ├── keepalive.sh               # Label Studio 与反向隧道保活
│   ├── add_zijie_independent_projects.py  # 补建 Zijie 独立项目
│   └── migrate_to_independent_projects.py  # 独立项目迁移与指纹校验
├── data/
│   ├── exp1_tasks.jsonl           # 实验1 待标注任务（200 claim）
│   ├── exp2_tasks.jsonl           # 实验2 待标注任务（250 pair，不含 LLM 标注）
│   ├── exp2_llm_labels.jsonl      # 实验2 LLM 原始标注（供后续比对，不在标注界面展示）
│   └── sampling_stats.json        # 抽样统计
├── results/                       # 标注完成后存放导出结果
│   ├── migration_20260711/        # 拆分前逐 task 审计导出与迁移报告
│   ├── exp1_annotations_A.json    # 标注者 A 的实验1标注
│   ├── exp1_annotations_B.json    # 标注者 B 的实验1标注
│   ├── exp2_annotations_A.json    # 标注者 A 的实验2标注
│   └── exp2_annotations_B.json    # 标注者 B 的实验2标注
└── label_studio_data/
    └── backups/                   # 切换前 SQLite 备份（git ignored）
```

## 数据来源

抽样脚本从以下已有产物导出（均为主方法 `atom_anchor` 流水线的输出）：

| 实验 | 数据源 | 字段 |
|---|---|---|
| 实验 1 | `claim_atom_cache_val_*.jsonl` | event_id, claim, claim_atoms(atom_id/proposition/type) |
| 实验 2 | `evidence_map_candidate_pool_val.jsonl` + `deepseek_evidence_map_annotations_val.jsonl` | claim, atoms, candidates(evidence_id/text), LLM 标注(relation/directness/confidence) |

## 抽样策略

### 实验 1（Atom 质量）：70% 随机 + 30% 困难优先
- 每数据集抽 100 条 claim（LIAR-RAW 100 + RAWFC 100 = 200 条）
- 困难判定：claim 长度 > 120 字符，或含否定/比较/数量/日期/极值特征词
- 困难样本不足时从随机池补足

### 实验 2（Evidence Map）：自然分布采样
- 每数据集抽 125 个 (evidence, atom) pair（共 250 pair）
- 按 LLM 标注的 relation 自然分布采样，不做类别均衡
- **task 文件不含 LLM 标注**（避免锚定偏差），LLM 标注单独存于 `exp2_llm_labels.jsonl`

### 当前抽样分布

**实验 1**：
| 数据集 | 池大小 | 抽样 | easy | hard |
|---|---|---|---|---|
| liar_raw | 1274 | 100 | 70 | 30 |
| rawfc | 200 | 100 | 70 | 30 |

**实验 2 relation 分布**（自然分布，反映真实频率）：
| relation | liar_raw | rawfc | 合计 |
|---|---|---|---|
| support | 21 | 94 | 115 |
| refute | 3 | 16 | 19 |
| qualify | 4 | 1 | 5 |
| insufficient | 29 | 7 | 36 |
| background | 14 | 3 | 17 |
| irrelevant | 54 | 4 | 58 |
| **合计** | **125** | **125** | **250** |

## 使用流程

### 1. 重新生成待标注数据（可选）

如果需要调整抽样数量或种子：
```bash
cd scripts
python export_tasks.py --n-claim 100 --n-pair 125 --seed 42
# 自定义输出目录
python export_tasks.py --output-dir ../data --n-claim 150 --n-pair 200
```

### 2. 部署 Label Studio

```bash
pip install label-studio
label-studio start  # http://localhost:8080
```

### 3. 创建项目

**实验 1**：
1. Create Project → Name: `Exp1-Atom-Quality-A`
2. Import Data → 上传 `data/exp1_tasks.jsonl`
3. Labeling Setup → Custom → 粘贴 `config/exp1_atom_quality.xml`
4. 为标注者 B 创建副本项目 `Exp1-Atom-Quality-B`（同数据同配置）

**实验 2**：
1. Create Project → Name: `Exp2-Evidence-Map-A`
2. Import Data → 上传 `data/exp2_tasks.jsonl`（注意：不要导入 exp2_llm_labels.jsonl）
3. Labeling Setup → Custom → 粘贴 `config/exp2_evidence_map.xml`
4. 为标注者 B 创建副本项目

### 4. 标注

- 两位标注者各自在独立项目中标注，互不可见
- 标注前先做 20 条 calibration（见指导书 5.1 节），κ ≥ 0.6 方可进入正式标注

### 5. 导出与比对

```bash
# 从每个项目 Export → JSON，存入 results/
# 后续用 IAA 脚本计算一致率 + 仲裁分歧（脚本待编写）
```

## 关键设计

1. **双盲**：两位标注者用独立项目，互不可见，避免互相影响
2. **隐藏 LLM 标注**：实验 2 的 task 文件不含 relation/directness/confidence，避免锚定偏差
3. **可复现抽样**：固定 seed=42，重新运行可得到完全相同的待标注样本
4. **LLM 标注可追溯**：`exp2_llm_labels.jsonl` 保存 LLM 原始标注，标注完成后按 (event_id, atom_id, evidence_id) join 比对

> Label Studio Community 版不能强制限制同组织用户只能访问指定项目，因此项目标题和说明均带有 `YULIN ONLY` / `ZHIQIANG ONLY`。标注者必须只进入自己的两个项目；管理员应定期检查是否出现跨账号提交。

## 2026-07-11 独立项目迁移

旧 ID=12/13 曾错误配置为共享顺序队列、每 task 仅 1 次标注，导致不同标注者拿到互不重叠的任务。本次修复：

- 为每位正式标注者创建 Exp1/Exp2 完整任务集；
- 既有提交按原 task、原作者迁入对应项目，结果内容通过 SHA-256 指纹核对；
- `1349410043@qq.com` 的 28 条 pilot 结果保留在旧软归档项目与审计导出中，不冒充迁入正式标注者；
- 标注者打开自己的项目后，从自己尚未标注的最早 task 继续，因此会先回补原共享队列中的前段样本。
- Zijie 的 ID=18/19 项目随后按相同规则补建，原有 Exp1 16 条、Exp2 12 条均按本人身份迁入。
