# LIAR-RAW + Ministral3-8B Chunking 实验总体指标

**实验数量**: 10 | **模型**: Ministral3-8B (LoRA fine-tuned) | **测试集样本数**: 1251

## 一、总体指标表（Test Set）

| Chunking | Selector | Acc ↑ | Macro F1 ↑ |
|----------|----------|------:|----------:|
| abc | budget | 0.3301 | 0.3344 |
| abc | top5 | 0.3341 | 0.3364 |
| report | budget | 0.3333 | 0.3424 |
| report | top5 | 0.3341 | 0.3366 |
| semantic07 | budget | 0.3413 | 0.3504 |
| semantic07 | top5 | 0.3421 | 0.3488 |
| sentence | budget | 0.3317 | 0.3331 |
| sentence | top5 | 0.3437 | 0.3507 |
| sentwin1 | budget | 0.3341 | 0.3422 |
| sentwin1 | top5 | 0.3421 | 0.3498 |

## 二、Prompt 长度统计（Val Set, Token Count）

| Chunking | Selector | Mean | P50 | P90 | P95 | Max |
|----------|----------|-----:|----:|----:|----:|----:|
| abc | budget | 546.0 | 554.5 | 604.0 | 619.0 | 688 |
| abc | top5 | 536.8 | 529.5 | 679.4 | 731.3 | 975 |
| report | budget | 546.0 | 561.0 | 645.0 | 664.3 | 968 |
| report | top5 | 818.0 | 852.5 | 980.0 | 999.0 | 1020 |
| semantic07 | budget | 544.5 | 556.0 | 613.0 | 631.0 | 761 |
| semantic07 | top5 | 528.0 | 505.0 | 708.0 | 792.3 | 1014 |
| sentence | budget | 544.2 | 551.0 | 597.0 | 612.3 | 715 |
| sentence | top5 | 414.3 | 402.0 | 509.0 | 547.0 | 977 |
| sentwin1 | budget | 541.8 | 547.0 | 599.0 | 615.0 | 797 |
| sentwin1 | top5 | 631.9 | 620.0 | 798.0 | 844.3 | 1020 |

## 三、Evidence 单元数量统计（Val Set）

| Chunking | Selector | Mean | Min | Max | Truncation Rate |
|----------|----------|-----:|----:|----:|----------------:|
| abc | budget | 5.34 | 1 | 14 | 0.0000 |
| abc | top5 | 4.95 | 1 | 5 | 0.0008 |
| report | budget | 2.54 | 1 | 11 | 0.0000 |
| report | top5 | 3.93 | 1 | 5 | 0.5094 |
| semantic07 | budget | 5.78 | 1 | 15 | 0.0000 |
| semantic07 | top5 | 4.95 | 1 | 5 | 0.0047 |
| sentence | budget | 8.38 | 1 | 20 | 0.0000 |
| sentence | top5 | 4.97 | 1 | 5 | 0.0000 |
| sentwin1 | budget | 4.26 | 1 | 10 | 0.0000 |
| sentwin1 | top5 | 4.95 | 1 | 5 | 0.0078 |

## 四、综合对比表

| Chunking | Selector | Acc ↑ | F1 ↑ | Prompt Mean | Prompt P95 | Ev Mean | Ev Max | Trunc Rate |
|----------|----------|------:|-----:|------------:|-----------:|--------:|-------:|-----------:|
| abc | budget | 0.3301 | 0.3344 | 546.0 | 619.0 | 5.34 | 14 | 0.0000 |
| abc | top5 | 0.3341 | 0.3364 | 536.8 | 731.3 | 4.95 | 5 | 0.0008 |
| report | budget | 0.3333 | 0.3424 | 546.0 | 664.3 | 2.54 | 11 | 0.0000 |
| report | top5 | 0.3341 | 0.3366 | 818.0 | 999.0 | 3.93 | 5 | 0.5094 |
| semantic07 | budget | 0.3413 | 0.3504 | 544.5 | 631.0 | 5.78 | 15 | 0.0000 |
| semantic07 | top5 | 0.3421 | 0.3488 | 528.0 | 792.3 | 4.95 | 5 | 0.0047 |
| sentence | budget | 0.3317 | 0.3331 | 544.2 | 612.3 | 8.38 | 20 | 0.0000 |
| sentence | top5 | 0.3437 | 0.3507 | 414.3 | 547.0 | 4.97 | 5 | 0.0000 |
| sentwin1 | budget | 0.3341 | 0.3422 | 541.8 | 615.0 | 4.26 | 10 | 0.0000 |
| sentwin1 | top5 | 0.3421 | 0.3498 | 631.9 | 844.3 | 4.95 | 5 | 0.0078 |

## 五、各 Chunking 策略 Best Results（按 F1）

| Chunking | Best Selector | Acc | F1 | Prompt Mean | Ev Mean |
|----------|--------------|----:|----:|------------:|--------:|
| abc | top5 | 0.3341 | 0.3364 | 536.8 | 4.95 |
| report | budget | 0.3333 | 0.3424 | 546.0 | 2.54 |
| semantic07 | budget | 0.3413 | 0.3504 | 544.5 | 5.78 |
| sentence | top5 | 0.3437 | 0.3507 | 414.3 | 4.97 |
| sentwin1 | top5 | 0.3421 | 0.3498 | 631.9 | 4.95 |

## 六、Selector 类型对比（平均值）

| Selector | Avg Acc | Avg F1 | Avg Prompt Mean | Avg Ev Mean | Avg Trunc Rate |
|----------|--------:|-------:|----------------:|------------:|---------------:|
| budget | 0.3341 | 0.3405 | 544.5 | 5.26 | 0.0000 |
| top5 | 0.3392 | 0.3445 | 585.8 | 4.75 | 0.1046 |

---

**说明**:
- `budget` = s4_union_budget_promptmatched, `top5` = s4_union_top5
- Acc / F1 来自 **test** split 的 best checkpoint 评估
- Prompt 长度和 Evidence 统计来自 **val** split（prompt_stats 仅覆盖 train/val）
- Truncation Rate 表示 prompt 超过 max_length (1024 tokens) 被截断的比例