# 当前情况

我当前的Stage已经提供了以下数据：

输入:

1. 原始的claim
2. 多条reports
3. 每条report对应多个sentences

输出:

1. 经检索以及去重得到的 top-k 个句子
2. 每个句子所在的原始 report（随候选一起输出 source_report）

其中每个claim的输出文件如下：

{
  "event_id": "11972.json",
  "claim": "Building a wall on the U.S.-Mexico border will take literally years.",
  "label": "true",
  "explain": "Perry said: \"Building a wall\" on the U.S.-Mexico border \"will take literally years.\" If Trump has a fast-track plan to plan the wall, purchase required land, complete needed studies and erect the wall in a year or less, it’s not public. Meantime, engineering experts agree the wall would most likely take years to complete. Keep in mind, too, it took more than six years to build roughly 700 miles of fence and barriers along the roughly 2,000-mile U.S.-Mexico border. Click here formore on the six PolitiFact ratings and how we select facts to check.",
  "candidates": [
    {
      "report_id": 3683124,
      "sent_idx": 4,
      "text": "Once work on President Donald Trump ’ s border wall begin , construction be rapid .",
      "dense_score": 0.7152296304702759,
      "lexical_score": 0.31578946113586426,
      "bm25_score": 3.2303080558776855,
      "hybrid_score": 0.8949738144874573,
      "source_report": {
        "report_id": 3683124,
        "link": "https://www.theguardian.com/us-news/2021/jan/16/my-neighbourhood-is-being-destroyed-to-pacify-his-supporters-the-race-to-complete-trumps-wall",
        "domain": "https://www.theguardian.com/us-news/2021/jan/16/my-neighbourhood-is-being-destroyed-to-pacify-his-supporters-the-race-to-complete-trumps-wall",
        "content": "..."
      }
    },
    {
      "report_id": 7119951,
      "sent_idx": 2,
      "text": "The brick in Trump ’ s border wall take several form .",
      "dense_score": 0.6512283086776733,
      "lexical_score": 0.5,
      "bm25_score": 4.687113285064697,
      "hybrid_score": 0.8859743475914001,
      "source_report": {
        "report_id": 7119951,
        "link": "https://www.aljazeera.com/opinions/2019/10/21/donald-trump-found-a-different-way-to-build-his-wall",
        "domain": "https://www.aljazeera.com/opinions/2019/10/21/donald-trump-found-a-different-way-to-build-his-wall",
        "content": "..."
      }
    },
    ...
  ]
}

# 需要补充的代码

现在，需要为该项目添加以下baseline测试：

## B0：Sentence-only baseline

```text
claim + top-k sentences -> classifier
```

## B1：Sentence + source report context

```text
claim + top-k sentences + ±k context sentence -> classifier
```

# 细节要求

## classifier

这版classifier使用LLM实现，暂定使用Qwen/Qwen3.5-9B，模型文件已本地下载至"/data/models/Qwen3.5-9B"。

需要编写一个对应的few shot prompt / zero shot prompt，其中few shot例子用 embedding（模型："/home/fenglin/project/models/bge-base-en-v1.5/"） + MMR 检索top-10 claim最相近案例

## 训练

LLM需要经过train set的sft。

训练时使用L20四卡完成，deepSpeed Zero3策略训练

## ±k context sentence

使用该句evidence所在report的上下K句作为上下文，原始report已在source_report字段给出
