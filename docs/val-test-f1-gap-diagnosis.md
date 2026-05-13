# Val/Test F1 差距诊断与处理方案

## Context

两组实验的 val F1 与 test F1 存在明显差距：
- **b3_mmr_topk_sweep_1024**: val 最高 0.321 (top_k=6), test 最高 0.277 (top_k=3/5), gap 3-5 pp
- **mmr_lambda_sweep_1024**: gap 5-14 pp，低 lambda 值时差距尤其大
- **mmr_lambda_sweep** (2048ctx): lambda=0.8 时 val=0.300, test=0.245, gap 5.5 pp

用户想知道是否需要重新用测试集跑所有 checkpoint。

## 诊断结论

### 1. 代码路径差异贡献很小（~0.4 pp）

验证集和测试集评估使用了不同的代码路径：
- **Val eval** (`eval.py`): 单次前向传播，"Label:" token 在 token 空间拼接
- **Test eval** (`vllm_infer.py`): vLLM 自回归生成，"Label:" 作为字符串拼接

但实测数据表明这个差异很小。`mmr_lambda_sweep` 的 lambda=0.8 同时有两种评估结果：
- 训练时 val eval (eval.py): **0.2997**
- 推理管线 val eval (api.py): **0.2961**
- 差距仅 **0.0036** (0.36 pp)

### 2. 差距主要是真实的泛化差距

从 `b3_mmr_topk_sweep_1024` 数据看：

| top_k | best val F1 | test F1 | gap |
|-------|-------------|---------|-----|
| 0     | 0.261       | 0.207   | 5.3 pp |
| 3     | 0.314       | 0.277   | 3.7 pp |
| 5     | 0.308       | 0.277   | 3.1 pp |
| 6     | 0.321       | 0.275   | 4.6 pp |

3-5 pp 的 val/test gap 在多分类任务（6 类）中是正常范围。重要的是：**超参数排名在 val 和 test 之间基本一致**——top_k=3-6 在两组中都表现最好。

### 3. mmr_lambda_sweep_1024 的大 gap 需要关注

低 lambda (0.0-0.4) 的 gap 高达 10-14 pp，远超正常范围。这可能是因为：
- 低 lambda = 低多样性检索 → 训练集与 val 的检索文档重叠更高 → val 偏高
- 需要确认这些 run 的训练是否稳定（检查 loss 曲线）

## 建议：不需要重新测所有 checkpoint

### 理由
1. 代码路径差异仅贡献 ~0.4 pp，不影响结论
2. Val 上的超参数排名与 test 基本一致，说明 val 选出的 best checkpoint 是合理的
3. 重新跑所有 checkpoint 计算成本高，收益小

### 建议做的事情

#### A. 修复 tokenization 不一致（可选，为了正确性）

4 个文件需要对 prompt 做 `.rstrip()` 以匹配训练时的 tokenization：

1. `src/sft/dataset/loaders.py:53` — eval collator 的 prompt 没有 rstrip
2. `src/sft/vllm_infer.py:101` — vLLM 推理的 prompt 没有 rstrip
3. `src/sft/vllm_online_eval.py:385` — online eval 的 prompt 没有 rstrip
4. `src/fact_checking/infer/api.py:203` — API 推理的 prompt 没有 rstrip

每个改动都是在 `sample.prompt` 后加 `.rstrip()`。

#### B. 验证性抽查（可选，如果想确认）

用 `infer.py`（与 val eval 相同代码路径）跑 test 集，确认 test 指标与 vLLM 推理结果接近：

```bash
python -m sft.infer \
  --run-dir outputs/runs/b3_mmr_topk_sweep_1024/<某个run>/train \
  --checkpoint best \
  --split test
```

如果结果与 vLLM test F1 相差 < 1 pp，确认差距是泛化问题。

#### C. 关注 mmr_lambda_sweep_1024 的异常 gap

低 lambda 值的 gap 异常大（10-14 pp），建议：
- 检查这些 run 的训练 loss 曲线是否正常
- 考虑这些配置是否本身就不稳定，在结论中适当讨论

## 验证方式

修复 tokenization 后：
1. 选 1-2 个已完成的 run，重新跑 test inference
2. 确认 test F1 变化 < 0.5 pp（说明修复影响很小，原结论不变）

## 关键文件

- `src/sft/dataset/loaders.py:53` — eval collator prompt 处理
- `src/sft/dataset/tokenization.py:18` — 训练时的 rstrip 参考
- `src/sft/eval.py:111-148` — val eval 逻辑
- `src/sft/vllm_infer.py:100-107` — test eval 逻辑
- `src/fact_checking/infer/api.py:203` — API 推理逻辑
