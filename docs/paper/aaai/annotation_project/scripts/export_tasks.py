#!/usr/bin/env python3
"""导出 Label Studio 待标注任务。

可信度实验 1（Atom 质量）：从 claim atom cache 产物抽样 claim，输出 exp1_tasks.jsonl
可信度实验 2（Evidence Map）：从 evidence_map_candidate_pool 产物展开 per-pair，输出 exp2_tasks.jsonl

抽样策略：
  实验 1：70% 随机 + 30% 困难优先（claim 长度 > P75 或含否定/比较/数量/日期特征）
  实验 2：按 LLM 标注的 relation 自然分布采样（不做类别均衡）

用法：
  python export_tasks.py --output-dir ../data
  python export_tasks.py --output-dir ../data --n-claim 200 --n-pair 250 --seed 42

输出文件：
  data/exp1_tasks.jsonl   实验1 待标注任务
  data/exp2_tasks.jsonl   实验2 待标注任务
  data/exp2_llm_labels.jsonl  实验2 的 LLM 原始标注（供后续比对，不在标注界面展示）
  data/sampling_stats.json    抽样统计（各数据集/各 relation 分布）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ============================================================================
# 数据源路径（相对项目根目录）
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]

# claim atom cache：实验 1 的数据源
# 两个数据集的 cache 位置略有差异（liar 在 cache/claim_atoms/，rawfc 在 01_claim_atoms/cache/）
# 用 (搜索根目录, 文件名 glob) 表达
ATOM_CACHE_SEARCH = {
    "liar_raw": (
        PROJECT_ROOT / "outputs/selectors/atom_anchor/liar_raw_abc_v0_1",
        "claim_atom_cache_val_*.jsonl",
    ),
    "rawfc": (
        PROJECT_ROOT / "outputs/selectors/atom_anchor/rawfc_abc_v0_1_baseline20",
        "claim_atom_cache_val_*.jsonl",
    ),
}

# evidence_map_candidate_pool：实验 2 的数据源（含 claim + atoms + candidates + evidence_id + text）
EM_POOL_DIRS = {
    "liar_raw": PROJECT_ROOT / "outputs/selectors/atom_anchor/liar_raw_abc_v0_1/04_evidence_map",
    "rawfc": PROJECT_ROOT / "outputs/selectors/atom_anchor/rawfc_abc_v0_1_baseline20/04_evidence_map",
}

# evidence_map annotations：实验 2 的 LLM 原始标注（供后续比对）
EM_ANNOT_DIRS = {
    "liar_raw": PROJECT_ROOT / "outputs/selectors/atom_anchor/liar_raw_abc_v0_1/04_evidence_map",
    "rawfc": PROJECT_ROOT / "outputs/selectors/atom_anchor/rawfc_abc_v0_1_baseline20/04_evidence_map",
}

# 困难 claim 特征关键词（实验 1 的 30% 困难采样用）
HARD_KEYWORDS = [
    "not", "no", "never", "none", "n't",  # 否定
    "more than", "less than", "most", "least", "fewer", "greater",  # 比较
    "percent", "billion", "million", "thousand", "$",  # 数量
    "since", "before", "after", "during", "in 20",  # 日期
    "first", "last", "only", "highest", "lowest",  # 极值
]


# ============================================================================
# 工具函数
# ============================================================================
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"⚠ 文件不存在: {path}", file=sys.stderr)
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ 写入 {len(rows)} 条 → {path}")


def is_hard_claim(claim: str) -> bool:
    """判断是否为困难 claim：长度 > 120 或含否定/比较/数量/日期特征。"""
    claim_lower = claim.lower()
    if len(claim) > 120:
        return True
    return any(kw in claim_lower for kw in HARD_KEYWORDS)


# ============================================================================
# 实验 1：Atom 质量评测任务导出
# ============================================================================
def find_atom_cache_file(search_root: Path, pattern: str) -> Path | None:
    """在 search_root 下递归搜索匹配 pattern 的 atom cache 文件。"""
    candidates = sorted(search_root.rglob(pattern))
    return candidates[0] if candidates else None


def export_exp1_tasks(
    n_per_dataset: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """导出实验 1 任务：每个 task = 一条 claim + 其 atoms。

    抽样：70% 随机 + 30% 困难优先。
    """
    rng = random.Random(seed)
    all_tasks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    for dataset, (search_root, pattern) in ATOM_CACHE_SEARCH.items():
        cache_file = find_atom_cache_file(search_root, pattern)
        if cache_file is None:
            print(f"⚠ {dataset} 无 val atom cache，跳过", file=sys.stderr)
            continue

        rows = load_jsonl(cache_file)
        # 只保留解析成功的样本
        rows = [r for r in rows if r.get("parse_status") == "ok" and r.get("claim_atoms")]

        n_easy_target = int(n_per_dataset * 0.7)
        n_hard_target = n_per_dataset - n_easy_target

        # 分离 easy / hard
        easy_pool = [r for r in rows if not is_hard_claim(r.get("claim", ""))]
        hard_pool = [r for r in rows if is_hard_claim(r.get("claim", ""))]

        rng.shuffle(easy_pool)
        rng.shuffle(hard_pool)

        sampled_easy = easy_pool[:n_easy_target]
        sampled_hard = hard_pool[:n_hard_target]

        # 若困难样本不足，从 easy 补
        if len(sampled_hard) < n_hard_target:
            deficit = n_hard_target - len(sampled_hard)
            sampled_hard.extend(easy_pool[n_easy_target : n_easy_target + deficit])

        sampled = sampled_easy + sampled_hard
        rng.shuffle(sampled)

        for r in sampled:
            atoms = []
            for a in r.get("claim_atoms", []):
                atoms.append(
                    {
                        "atom_id": a.get("atom_id", ""),
                        "proposition": a.get("proposition", a.get("text", "")),
                        "type": a.get("type", ""),
                    }
                )
            all_tasks.append(
                {
                    "event_id": r["event_id"],
                    "dataset": dataset,
                    "claim": r["claim"],
                    "gold_label": r.get("gold_label", ""),
                    "complexity": r.get("complexity", ""),
                    "atoms": atoms,
                }
            )

        stats[dataset] = {
            "total_pool": len(rows),
            "sampled": len(sampled),
            "easy": len(sampled_easy),
            "hard": len(sampled_hard),
            "atom_count_dist": dict(Counter(len(r.get("claim_atoms", [])) for r in sampled)),
        }

    print(f"实验 1：共导出 {len(all_tasks)} 条 claim 任务")
    return all_tasks, stats


# ============================================================================
# 实验 2：Evidence Map 标注任务导出
# ============================================================================
def export_exp2_tasks(
    n_per_dataset: int = 125,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """导出实验 2 任务：每个 task = 一个 (evidence, atom) pair。

    抽样：按 LLM 标注的 relation 自然分布采样。
    同时输出 LLM 原始标注（供后续比对，不在标注界面展示）。

    返回：(tasks, llm_labels, stats)
    """
    rng = random.Random(seed)
    all_tasks: list[dict[str, Any]] = []
    all_llm_labels: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    for dataset, pool_dir in EM_POOL_DIRS.items():
        pool_file = pool_dir / "evidence_map_candidate_pool_val.jsonl"
        annot_file = EM_ANNOT_DIRS[dataset] / "deepseek_evidence_map_annotations_val.jsonl"

        if not pool_file.exists():
            print(f"⚠ {dataset} 无 candidate_pool val 文件: {pool_file}", file=sys.stderr)
            continue

        pool_rows = load_jsonl(pool_file)
        annot_rows = load_jsonl(annot_file) if annot_file.exists() else []

        # 构建 LLM 标注索引：event_id -> {(evidence_id, atom_id): pair_annotation}
        llm_index: dict[str, dict[tuple[str, str], dict]] = {}
        for ar in annot_rows:
            eid = ar["event_id"]
            em = ar.get("evidence_map", {})
            pairs = em.get("candidate_atom_alignments", [])
            llm_index[eid] = {}
            for p in pairs:
                key = (p.get("evidence_id", ""), p.get("atom_id", ""))
                llm_index[eid][key] = p

        # 展开所有 (evidence, atom) pair
        candidate_pairs: list[dict[str, Any]] = []
        for pr in pool_rows:
            eid = pr["event_id"]
            claim = pr.get("claim", "")
            atoms_dict = {
                a.get("atom_id", ""): a.get("proposition", a.get("text", ""))
                for a in pr.get("claim_atoms", [])
            }
            for cand in pr.get("candidates", []):
                evidence_id = cand.get("evidence_id", "")
                evidence_text = cand.get("text", "")
                for atom_id, atom_prop in atoms_dict.items():
                    # 只保留有 LLM 标注的 pair（即 em 产物里有的）
                    llm_pair = llm_index.get(eid, {}).get((evidence_id, atom_id))
                    if llm_pair is None:
                        continue
                    candidate_pairs.append(
                        {
                            "event_id": eid,
                            "dataset": dataset,
                            "claim": claim,
                            "atom_id": atom_id,
                            "atom_proposition": atom_prop,
                            "evidence_id": evidence_id,
                            "evidence_text": evidence_text,
                            "_llm_label": llm_pair,  # 临时字段，不进 task
                        }
                    )

        # 自然分布采样（不均衡）
        rng.shuffle(candidate_pairs)
        sampled = candidate_pairs[:n_per_dataset]

        for s in sampled:
            llm = s.pop("_llm_label")
            # task（不含 LLM 标注，避免锚定）
            all_tasks.append(
                {
                    "event_id": s["event_id"],
                    "dataset": s["dataset"],
                    "claim": s["claim"],
                    "atom_id": s["atom_id"],
                    "atom_proposition": s["atom_proposition"],
                    "evidence_id": s["evidence_id"],
                    "evidence_text": s["evidence_text"],
                }
            )
            # LLM 原始标注（单独文件，供后续比对）
            all_llm_labels.append(
                {
                    "event_id": s["event_id"],
                    "dataset": s["dataset"],
                    "atom_id": s["atom_id"],
                    "evidence_id": s["evidence_id"],
                    "llm_relation": llm.get("relation", ""),
                    "llm_directness": llm.get("directness", ""),
                    "llm_confidence": llm.get("confidence", 0.0),
                    "llm_evidence_role": llm.get("evidence_role", ""),
                }
            )

        # 统计 relation 分布
        rel_dist = Counter(s["llm_relation"] for s in all_llm_labels if s["dataset"] == dataset)
        stats[dataset] = {
            "total_pairs_pool": len(candidate_pairs),
            "sampled": len([t for t in all_tasks if t["dataset"] == dataset]),
            "relation_distribution": dict(rel_dist),
        }

    print(f"实验 2：共导出 {len(all_tasks)} 个 (evidence, atom) pair 任务")
    return all_tasks, all_llm_labels, stats


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="导出 Label Studio 待标注任务")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="输出目录（默认 ../data）",
    )
    parser.add_argument("--n-claim", type=int, default=100, help="实验1: 每数据集抽样 claim 数（默认100）")
    parser.add_argument("--n-pair", type=int, default=125, help="实验2: 每数据集抽样 pair 数（默认125）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认42）")
    args = parser.parse_args()

    print("=" * 60)
    print("导出 Label Studio 待标注任务")
    print(f"  输出目录: {args.output_dir}")
    print(f"  随机种子: {args.seed}")
    print("=" * 60)

    all_stats: dict[str, Any] = {"seed": args.seed}

    # 实验 1
    print("\n[实验 1] Atom 质量评测任务导出")
    print(f"  每数据集抽样 {args.n_claim} 条 claim（70% 随机 + 30% 困难）")
    exp1_tasks, exp1_stats = export_exp1_tasks(n_per_dataset=args.n_claim, seed=args.seed)
    save_jsonl(exp1_tasks, args.output_dir / "exp1_tasks.jsonl")
    all_stats["exp1"] = exp1_stats

    # 实验 2
    print("\n[实验 2] Evidence Map 标注任务导出")
    print(f"  每数据集抽样 {args.n_pair} 个 pair（自然分布）")
    exp2_tasks, exp2_llm_labels, exp2_stats = export_exp2_tasks(
        n_per_dataset=args.n_pair, seed=args.seed
    )
    save_jsonl(exp2_tasks, args.output_dir / "exp2_tasks.jsonl")
    save_jsonl(exp2_llm_labels, args.output_dir / "exp2_llm_labels.jsonl")
    all_stats["exp2"] = exp2_stats

    # 汇总统计
    stats_path = args.output_dir / "sampling_stats.json"
    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✓ 抽样统计 → {stats_path}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("抽样汇总")
    print("=" * 60)
    print(f"实验 1: {len(exp1_tasks)} 条 claim 任务")
    for ds, s in exp1_stats.items():
        print(f"  {ds}: 池 {s['total_pool']} → 抽 {s['sampled']} (easy {s['easy']} / hard {s['hard']})")
    print(f"实验 2: {len(exp2_tasks)} 个 pair 任务")
    for ds, s in exp2_stats.items():
        print(f"  {ds}: 池 {s['total_pairs_pool']} → 抽 {s['sampled']}")
        print(f"    relation 分布: {s['relation_distribution']}")


if __name__ == "__main__":
    main()
