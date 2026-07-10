#!/usr/bin/env python3
"""用 DeepSeek API 为标注任务添加中文翻译。

为 Label Studio 的待标注任务（实验1 + 实验2）添加中文翻译字段：
  - claim_zh: claim 的中文翻译
  - proposition_zh: atom proposition 的中文翻译（实验1）
  - evidence_text_zh: evidence text 的中文翻译（实验2）

翻译定位：辅助参考（以英文原文为准），用于帮助标注者快速理解内容。

用法：
  # 从 .env 加载 API key
  source /data/liaozijie/fact-checking/.env
  python translate_tasks.py

  # 或直接传 key
  DEEPSEEK_API_KEY=sk-xxx python translate_tasks.py

  # 自定义输入输出
  python translate_tasks.py --input ../data/exp1_tasks_flat.jsonl --output ../data/exp1_tasks_flat_zh.jsonl

输出：在原 JSONL 每行追加翻译字段，保持原字段不变。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ============================================================================
# 配置
# ============================================================================
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
API_KEY_ENV = "DEEPSEEK_API_KEY"
TIMEOUT = 60
MAX_RETRIES = 4
CONCURRENCY = 16

PROJECT_ROOT = Path("/data/liaozijie/fact-checking")
DATA_DIR = PROJECT_ROOT / "docs/paper/aaai/annotation_project/data"

SYSTEM_PROMPT = "You are a professional translator. Translate the given English text to Chinese. Output ONLY the Chinese translation, no explanation, no quotes."

USER_PROMPT_TEMPLATE = """Translate the following English text to Chinese. This is a fact-checking dataset for annotation assistance.

Rules:
1. Output ONLY the Chinese translation.
2. Keep entity names (people, organizations, place names) in English if they are well-known, otherwise transliterate.
3. Preserve numbers, dates, and quantities exactly.
4. Keep the original meaning faithful; do not add or omit information.
5. If the text contains quotes, keep the Chinese translation of the quote.

Text to translate:
{text}"""


# ============================================================================
# API 调用
# ============================================================================
def call_deepseek(text: str, api_key: str) -> str:
    """调用 DeepSeek API 翻译单条文本，带重试。"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code in (408, 409, 429) or e.code >= 500:
                delay = min(2**attempt, 30)
                time.sleep(delay)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            delay = min(2**attempt, 30)
            time.sleep(delay)
            continue
    raise RuntimeError(f"翻译失败（重试 {MAX_RETRIES} 次）: {text[:80]}")


# ============================================================================
# 批量翻译
# ============================================================================
def translate_field(items: list[dict], field: str, api_key: str, target_field: str, concurrency: int = 16) -> dict[str, str]:
    """批量翻译某个字段，返回 {item_key: translation} 映射。"""
    # 去重文本，避免重复翻译
    unique_texts = set()
    for item in items:
        val = item.get(field, "")
        if val:
            unique_texts.add(val)

    print(f"  翻译 {len(unique_texts)} 条唯一文本（字段: {field}）...")

    translations: dict[str, str] = {}
    texts_list = list(unique_texts)

    def _translate(text: str) -> tuple[str, str]:
        return text, call_deepseek(text, api_key)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_translate, t): t for t in texts_list}
        done = 0
        for future in as_completed(futures):
            try:
                original, translated = future.result()
                translations[original] = translated
                done += 1
                if done % 20 == 0:
                    print(f"    进度: {done}/{len(texts_list)}")
            except Exception as e:
                original = futures[future]
                print(f"    ⚠ 翻译失败: {original[:50]}... -> {e}", file=sys.stderr)
                translations[original] = ""  # 失败留空

    return translations


def process_file(input_path: Path, output_path: Path, fields: list[str], api_key: str, concurrency: int = 16) -> None:
    """处理单个 JSONL 文件：读取、翻译、写入。"""
    items = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))

    print(f"\n处理: {input_path.name} ({len(items)} 条)")

    # 缓存文件（避免重跑）
    cache_path = output_path.with_suffix(".translation_cache.json")

    # 加载已有缓存
    cache: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  已加载缓存: {sum(len(v) for v in cache.values())} 条翻译")

    for field in fields:
        target_field = f"{field}_zh"
        if field not in cache:
            cache[field] = {}

        # 找出需要翻译的（缓存里没有的）
        needed = set()
        for item in items:
            val = item.get(field, "")
            if val and val not in cache[field]:
                needed.add(val)

        if needed:
            new_translations = translate_field(items, field, api_key, target_field, concurrency)
            cache[field].update(new_translations)
            # 实时保存缓存
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(f"  字段 {field}: 全部命中缓存，跳过翻译")

    # 写入输出文件
    output_lines = []
    for item in items:
        for field in fields:
            val = item.get(field, "")
            if val and val in cache.get(field, {}):
                item[f"{field}_zh"] = cache[field][val]
            else:
                item[f"{field}_zh"] = ""
        output_lines.append(json.dumps(item, ensure_ascii=False))

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"  ✓ 写入 {len(items)} 条 → {output_path}")


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="为标注任务添加中文翻译")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key（默认从 DEEPSEEK_API_KEY 环境变量读取）")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="数据目录")
    parser.add_argument("--concurrency", type=int, default=16, help="并发数")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"错误: 未找到 API key。请设置 ${API_KEY_ENV} 环境变量或用 --api-key 传入。", file=sys.stderr)
        sys.exit(1)

    _concurrency = args.concurrency

    print("=" * 60)
    print("为标注任务添加中文翻译")
    print(f"  翻译引擎: DeepSeek {MODEL}")
    print(f"  并发数: {_concurrency}")
    print(f"  数据目录: {args.data_dir}")
    print("=" * 60)

    # 实验1：claim + proposition
    exp1_input = args.data_dir / "exp1_tasks_flat.jsonl"
    exp1_output = args.data_dir / "exp1_tasks_flat_zh.jsonl"
    if exp1_input.exists():
        process_file(exp1_input, exp1_output, ["claim", "proposition"], api_key, _concurrency)
    else:
        print(f"\n⚠ 实验1输入文件不存在: {exp1_input}")

    # 实验2：claim + evidence_text
    exp2_input = args.data_dir / "exp2_tasks.jsonl"
    exp2_output = args.data_dir / "exp2_tasks_zh.jsonl"
    if exp2_input.exists():
        process_file(exp2_input, exp2_output, ["claim", "evidence_text", "atom_proposition"], api_key, _concurrency)
    else:
        print(f"\n⚠ 实验2输入文件不存在: {exp2_input}")

    print("\n" + "=" * 60)
    print("翻译完成！")
    print(f"  实验1: {exp1_output}")
    print(f"  实验2: {exp2_output}")
    print("\n下一步：更新 Label Studio XML 配置添加中文翻译字段，然后重新导入数据。")
    print("=" * 60)


if __name__ == "__main__":
    main()
