# DVC 接入计划 — fact-checking 项目

## Context

当前 fact-checking 项目跨多台机器（笔记本 / GPU 服务器）协作时存在以下痛点：

- **原始数据集** `data/raw/LIAR-RAW/`（479 MB）被 `.gitignore` 排除，换机器需要手工拷贝。
- **检索中间缓存** `outputs/cache/build/`（1.9 GB）和 `outputs/cache/pre_mmr/`（3.2 GB）按 fingerprint(sha1) 组织，重算成本极高（dense + lexical + MMR），但目前完全本地化、无法同步。
- **重型推理产物**（如 `outputs/runs/*/infer/test_predictions.jsonl`）已被 git 排除，只在本机存在，难以归档与异机复现。

引入 DVC 的目的：**在不改动现有 Hydra+fingerprint 流水线的前提下**，给上述资产做版本管理与跨机同步。明确**不**做：

- ✘ 不写 `dvc.yaml`，不接管 build/train/infer 流水线（已有 fingerprint 缓存 + `runner.py`，再叠 DVC pipeline 是冗余）。
- ✘ 不跟踪模型 checkpoint（用户决定）。
- ✘ 不替换 git 中已有的小体积评估指标 allowlist（`outputs/runs/**/{train/eval,infer}/*.{json,jsonl,png}` 保留 git 跟踪，方便 diff）。

存储后端：**SSH/SFTP** 远程服务器。

---

## 范围与决策

| 资产 | 路径 | 大小 | DVC 跟踪粒度 |
|------|------|------|--------------|
| 原始数据集 | `data/raw/LIAR-RAW/` | 479 MB | 整目录 `dvc add` 一次 |
| build 缓存 | `outputs/cache/build/<sha1>/` | 各 fingerprint 独立 | **按 fingerprint 子目录**逐个 add |
| pre_mmr 缓存 | `outputs/cache/pre_mmr/<sha1>/` | 各 fingerprint 独立 | **按 fingerprint 子目录**逐个 add |
| 重型推理产物 | `outputs/runs/<run>/infer/test_predictions.jsonl` 等 | 每次实验 ~MB-GB | 按文件 `dvc add`，仅对值得归档的运行执行 |

**为何 fingerprint 子目录逐个 add，而不是 add 父目录**：父目录是 append-only 的内容寻址目录，一旦 add 父目录，每次新 fingerprint 出现都会让 DVC 重新对全部 GB 级内容做 md5 校验，并使父级 `.dvc` 失效；按子目录 add 后每个 `.dvc` 一旦写入就不再变动，符合 fingerprint 的不可变语义。

---

## 实施步骤

### 1. 安装 DVC（含 SSH 支持）

修改 `/home/fenglin/project/fact-checking/requirements.txt`，追加：

```
dvc[ssh]>=3.50
```

随后 `pip install -r requirements.txt`。

**验证**：`dvc --version` 显示 3.50+；`dvc doctor` 中 supported remotes 列表包含 `ssh`。

### 2. 初始化 DVC 仓库

在项目根目录：

```bash
dvc init
```

会自动创建 `.dvc/config`、`.dvc/.gitignore` 及空的 `.dvcignore`。

**验证**：`git status` 显示新增三件套；`dvc status` 无报错输出。

### 3. 配置 SSH 远程

```bash
dvc remote add -d origin ssh://<user>@<host>/srv/dvc-storage/fact-checking
dvc remote modify origin keyfile ~/.ssh/id_ed25519
dvc remote modify origin port 22
```

要点（来自 Plan 校验）：
- paramiko 默认**不读 `~/.ssh/config` 别名**，host 字段必须填真实主机名/IP。
- 优先 SSH key，**不要**用 `password` / `ask_password`；若必须配置敏感信息，写入 `.dvc/config.local`（已被 DVC 自动 gitignore）。
- **登录前在远端预先创建目录**：`ssh <user>@<host> 'mkdir -p /srv/dvc-storage/fact-checking'`，DVC 不会激进地 mkdir。

**验证**：`dvc remote list` 显示 `origin ssh://… *`（带 default 标记）；尝试 `dvc push` 一个小文件不再提示口令。

### 4. 写入 `.dvcignore`

文件路径：`/home/fenglin/project/fact-checking/.dvcignore`，内容：

```
__pycache__/
*.pyc
*.log
run.log
swanlog/
multirun/
wandb/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
# 防止与 git allowlist 冲突的双重跟踪
outputs/runs/**/train/eval/
outputs/runs/**/infer/*.json
outputs/runs/**/infer/*.jsonl
outputs/runs/**/infer/*.png
```

### 5. 跟踪原始数据集

```bash
dvc add data/raw/LIAR-RAW
```

DVC 会：
1. 生成 `data/raw/LIAR-RAW.dvc`（元数据指针）
2. 在 `data/raw/.gitignore` 中追加 `LIAR-RAW`

**同步修改** `/home/fenglin/project/fact-checking/.gitignore`：删除第 213 行 `data/raw/LIAR-RAW/`（DVC 已在子目录精准接管，根 `.gitignore` 的粗粒度规则反而会让 `.dvc` 文件无法被 git 跟踪）。

提交：

```bash
git add data/raw/LIAR-RAW.dvc data/raw/.gitignore .gitignore
git commit -m "track LIAR-RAW dataset with DVC"
dvc push
```

**验证**：`dvc status` 输出 `Data and pipelines are up to date.`；`dvc push -v` 显示文件已传至 SSH 远端。

### 6. 跟踪 build / pre_mmr 缓存（按需）

对于需要在异机复现的 fingerprint，执行（示例）：

```bash
dvc add outputs/cache/build/1396eafb4f9d
dvc add outputs/cache/pre_mmr/53a3588e485d
git add outputs/cache/build/1396eafb4f9d.dvc outputs/cache/pre_mmr/53a3588e485d.dvc \
        outputs/cache/build/.gitignore outputs/cache/pre_mmr/.gitignore
git commit -m "track build/pre_mmr cache for <experiment-name>"
dvc push
```

**注意**：仅对**已结束的运行**执行 `dvc add`，运行中的 fingerprint 目录不要 add（DVC 会对正在写入的文件计算哈希，结果不可靠）。

### 7. 跟踪重型推理产物（按需）

只对值得归档的 run 执行：

```bash
dvc add outputs/runs/<experiment_name>/<run_hash>/infer/test_predictions.jsonl
git add outputs/runs/.../test_predictions.jsonl.dvc outputs/runs/<...>/infer/.gitignore
git commit -m "archive test_predictions for <run>"
dvc push
```

不要批量 `dvc add outputs/runs/`——会与 git allowlist 冲突，且无谓地把所有运行入库。

### 8. 异机首拉取流程

```bash
git clone <repo> && cd fact-checking
pip install -r requirements.txt
dvc pull   # 自动拉取所有 .dvc 文件指向的数据
```

`dvc pull` 通过硬链接（默认 reflink → hardlink → symlink → copy）把数据放回原路径，磁盘占用最低。

---

## 日常工作流（3 行命令）

**产出新内容后同步到远端**：
```bash
dvc add <new-asset-path>
git add <path>.dvc <相应的 .gitignore> && git commit -m "..."
dvc push
```

**异机拉取最新**：
```bash
git pull && dvc pull
```

**检查是否需要 push**：`dvc status -c` 对比本地缓存与远端。

---

## 关键文件改动清单

需要新建：
- `/home/fenglin/project/fact-checking/.dvc/config`（`dvc init` 与 `dvc remote add` 自动生成）
- `/home/fenglin/project/fact-checking/.dvc/.gitignore`（自动生成）
- `/home/fenglin/project/fact-checking/.dvcignore`（手写，见第 4 步）
- `data/raw/LIAR-RAW.dvc`、`outputs/cache/{build,pre_mmr}/<sha1>.dvc`（按需生成）

需要编辑：
- `/home/fenglin/project/fact-checking/requirements.txt`：追加 `dvc[ssh]>=3.50`
- `/home/fenglin/project/fact-checking/.gitignore`：移除第 213 行 `data/raw/LIAR-RAW/`（DVC 在子目录接管）

**无需改动**：
- `src/fact_checking/pipeline/runner.py`（已确认无 chmod / 缓存重写，硬链接安全）
- 所有 Hydra 配置文件、训练脚本

---

## Hydra / fingerprint 缓存相容性注意

1. **运行中目录不要 add**：`runner.py` 在写入 `outputs/cache/build/<sha1>/` 期间，若并发 `dvc add` 会哈希到半成品文件。改用「运行结束后才 add」的纪律即可。
2. **DVC 默认硬链接**：源文件会变为指向 `.dvc/cache/` 的硬链接，节省磁盘但**只读**。已确认 `runner.py` 不会原地改写已有缓存文件（manifest.json 不通过 DVC 跟踪），安全。若日后出现 `Permission denied` 错误，可降级为 `dvc config cache.type copy`。
3. **`.dvc/cache` 体积管理**：GPU 服务器侧 cache 会增长，周期性执行 `du -sh .dvc/cache` 监控；不再需要的旧 fingerprint 用 `dvc gc -w -c` 清理（`-w` 仅保留 workspace 引用，`-c` 同步清远端）。
4. **`hydra.run.dir`**：当前已配置为 `outputs/runs/<name>`，与 DVC 跟踪目标对齐，无需调整。

---

## 端到端验证

1. `dvc doctor` —— 无错误，输出中 `Remote backends supported` 含 `ssh`。
2. `dvc remote list` —— 显示 `origin ssh://… *`。
3. `dvc add data/raw/LIAR-RAW && dvc push -v` —— 上传过程无口令提示（确认 key auth）。
4. **第二台机器**：`git clone … && dvc pull` 后，`ls data/raw/LIAR-RAW/` 应可见 `train.json`/`val.json`/`test.json` 且大小一致；`find outputs/cache/build -type f | head` 能列出真实内容。
5. `dvc status -c` —— 输出 `Cache and remote 'origin' are in sync.`。
6. `git status` —— `dvc add` 之后无未跟踪的脏文件。
7. 跑一次 build 阶段 `python -m fact_checking.pipeline.run experiment=<x> pipeline.mode=build`，确认新 fingerprint 目录正常产生且**未**被自动跟踪（必须人工 `dvc add`）。
8. 检查 `.gitignore` 没有意外把 `*.dvc` 也排除。

---

## 后续可选扩展（非本次范围）

- 若日后想做实验指标对比，可在不改动 runner 的前提下试用 `dvc exp` 单独跟踪 metric 文件。
- 若团队规模扩大，可把 SSH remote 升级为 S3/MinIO（`dvc remote modify origin url …` 即可，已上传数据需重传）。
