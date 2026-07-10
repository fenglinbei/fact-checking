#!/usr/bin/env python3
"""把 exp1_tasks.jsonl（nested）展平为 exp1_tasks_flat.jsonl，并给每条任务
追加 all_atoms_text 字段（该 claim 全部 atoms 的全景文本）。

all_atoms_text 用于 Label Studio 的"该 Claim 的全部 Atoms 全景"展示区，
解决 completeness 标注时标注者只看到单个 atom、无法判断 claim 整体覆盖的问题。

用法:
    python flatten_exp1_with_panorama.py
        [--input ../data/exp1_tasks.jsonl]
        [--output ../data/exp1_tasks_flat.jsonl]

输出 flat 版每行字段:
    event_id, dataset, claim, atom_id, proposition, type, all_atoms_text
其中 all_atoms_text 形如:
    A1 | attribution
      Hillary Clinton is the one that labeled ...
    A2 | causal
      The 'global justice' initiative is new ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def render_panorama(atoms: list[dict]) -> str:
    """渲染该 claim 全部 atoms 的全景文本（多行，带 atom_id / type / proposition）。"""
    lines: list[str] = []
    for a in atoms:
        aid = a.get("atom_id", "?")
        atype = a.get("type", "")
        prop = a.get("proposition", a.get("text", ""))
        header = f"{aid}" + (f" | {atype}" if atype else "")
        lines.append(header)
        lines.append(f"  {prop}")
    return "\n".join(lines)


def flatten(nested_rows: list[dict]) -> list[dict]:
    flat: list[dict] = []
    for r in nested_rows:
        atoms = r.get("atoms", []) or []
        panorama = render_panorama(atoms)
        # 兜底：万一某条没有 atoms，仍产出一条空 atom 行，避免漏标
        if not atoms:
            atoms = [{"atom_id": "-", "proposition": "", "type": ""}]
        for a in atoms:
            flat.append(
                {
                    "event_id": r.get("event_id", ""),
                    "dataset": r.get("dataset", ""),
                    "claim": r.get("claim", ""),
                    "atom_id": a.get("atom_id", ""),
                    "proposition": a.get("proposition", a.get("text", "")),
                    "type": a.get("type", ""),
                    # claim 级：同一 claim 下每条 atom 任务都带相同的全景
                    "all_atoms_text": panorama,
                }
            )
    return flat


def main() -> None:
    here = Path(__file__).resolve().parent
    default_in = here.parent / "data" / "exp1_tasks.jsonl"
    default_out = here.parent / "data" / "exp1_tasks_flat.jsonl"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=default_in)
    p.add_argument("--output", type=Path, default=default_out)
    args = p.parse_args()

    nested = load_jsonl(args.input)
    flat = flatten(nested)
    save_jsonl(flat, args.output)

    # 简单统计
    n_claims = len({r["event_id"] for r in flat})
    print(f"输入 nested: {len(nested)} claims  |  输出 flat: {len(flat)} atom 任务")
    print(f"唯一 claim 数: {n_claims}")
    print(f"已写入: {args.output}")
    # 抽样校验
    if flat:
        sample = flat[0]
        print("\n样例 all_atoms_text:")
        print(sample["all_atoms_text"][:300])


if __name__ == "__main__":
    main()
