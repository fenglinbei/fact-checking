#!/usr/bin/env python3
"""为 exp1 flat_zh 任务补上 all_atoms_text 的中文翻译字段 all_atoms_text_zh。

all_atoms_text 是多行全景文本，形如:
    A1 | attribution
      Hillary Clinton is the one that labeled ...
    A2 | causal
      The 'global justice' initiative is new ...

直接整段翻译会破坏 atom_id/type 结构。本脚本只翻译其中的 proposition 行
（缩进 2 空格的行），atom_id/type 行保持原样，复用已有 proposition 翻译缓存，
不调用 API。

用法:
    python add_panorama_translation.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    flat_zh_path = DATA / "exp1_tasks_flat_zh.jsonl"
    cache_path = DATA / "exp1_tasks_flat_zh.translation_cache.json"

    rows = [json.loads(l) for l in flat_zh_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    prop_zh: dict[str, str] = cache.get("proposition", {})

    missing = 0
    for r in rows:
        panorama = r.get("all_atoms_text", "")
        if not panorama:
            r["all_atoms_text_zh"] = ""
            continue
        out_lines: list[str] = []
        for line in panorama.split("\n"):
            # proposition 行: 以 2 空格缩进开头
            if line.startswith("  "):
                prop = line[2:]
                zh = prop_zh.get(prop)
                if zh is None:
                    missing += 1
                    zh = prop  # 兜底: 没有翻译则保留英文
                out_lines.append("  " + zh)
            else:
                # atom_id | type 行保持原样
                out_lines.append(line)
        r["all_atoms_text_zh"] = "\n".join(out_lines)

    flat_zh_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"处理 {len(rows)} 条, 未命中翻译 proposition {missing} 条")
    # 样例
    if rows:
        s = next((r for r in rows if "A2" in r.get("all_atoms_text", "")), rows[0])
        print("\n样例 all_atoms_text_zh:")
        print(s["all_atoms_text_zh"][:300])


if __name__ == "__main__":
    main()
