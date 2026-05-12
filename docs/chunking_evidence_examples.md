# Chunking Evidence Visualization

- experiment: `b4`
- split: `test`
- strategies: `sentence, ctx_window, raw, semantic, ctx_semantic`
- chunking config: `{"strategy": "sentence", "context_k": 1, "theta": 0.7, "embedder_model": null, "device": "cpu", "max_length": 256, "batch_size": 64, "precision": "fp32"}`

## Strategy notes

- `sentence`: single selected sentence
- `ctx_window`: selected sentence +/- context_k=1
- `raw`: full source report
- `semantic`: merge adjacent sentences when cosine similarity > theta=0.7
- `ctx_semantic`: partition into context_k=1 windows, then merge adjacent windows when cosine similarity > theta=0.7

## Example 1: `11269.json`

- label: `pants-fire`
- claim: Denali is the Kenyan word for black power.
- anchor: report_id=`2587609`, sent_idx=`0`, score=`1.0144`, source_sentences=`5`
- source: https://api.politifact.com/factchecks/2015/sep/03/viral-image/no-denali-not-kenyan-word-black-power/ https://api.politifact.com/factchecks/2015/sep/03/viral-image/no-denali-not-kenyan-word-black-power/

### Source context

- S00 **anchor** McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? .
- S01 One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power .
- S02 A search of multiple Swahili dictionary turn up no result for the word `` Denali .

### Shape summary

| strategy | status | span | sentences | words | chars |
|---|---|---:|---:|---:|---:|
| `sentence` | ok | S00 (anchor included) | 1 | 11 | 77 |
| `ctx_window` | ok | S00-S01 (anchor included) | 2 | 38 | 245 |
| `raw` | ok | S00-S04 (anchor included) | 5 | 91 | 562 |
| `semantic` | ok | S00-S01 (anchor included) | 2 | 38 | 245 |
| `ctx_semantic` | ok | S00-S01 (anchor included) | 2 | 38 | 245 |

### sentence

English evidence:

> McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? .

Chinese translation:

> 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？

### ctx_window

English evidence:

> McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power .

Chinese translation:

> 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？一张在 Facebook 上流传的图片指责总统从其肯尼亚血统中寻找灵感，并声称“Denali”是肯尼亚语中“黑色权力”的意思。

### raw

English evidence:

> McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power . A search of multiple Swahili dictionary turn up no result for the word `` Denali . As our fact-checker friend at Snopes find , the Swahili word for `` black `` be `` mweusi , `` and the Swahili word for power be `` nguvu . The word `` Denali `` doesn ’ t ... [truncated, 62 chars omitted]

Chinese translation:

> 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？一张在 Facebook 上流传的图片指责总统从其肯尼亚血统中寻找灵感，并声称“Denali”是肯尼亚语中“黑色权力”的意思。查询多部斯瓦希里语词典都找不到“Denali”这个词。正如 Snopes 的事实核查者发现的那样，斯瓦希里语中“黑色”是“mweusi”，“权力”是“nguvu”。“Denali”并不出现在斯瓦希里语中，而斯瓦希里语是肯尼亚的两种国家语言之一。

### semantic

English evidence:

> McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power .

Chinese translation:

> 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？一张在 Facebook 上流传的图片指责总统从其肯尼亚血统中寻找灵感，并声称“Denali”是肯尼亚语中“黑色权力”的意思。

### ctx_semantic

English evidence:

> McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power .

Chinese translation:

> 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？一张在 Facebook 上流传的图片指责总统从其肯尼亚血统中寻找灵感，并声称“Denali”是肯尼亚语中“黑色权力”的意思。

## Example 2: `11972.json`

- label: `true`
- claim: Building a wall on the U.S.-Mexico border will take literally years.
- anchor: report_id=`7119951`, sent_idx=`2`, score=`0.6406`, source_sentences=`4`
- source: https://www.aljazeera.com/opinions/2019/10/21/donald-trump-found-a-different-way-to-build-his-wall https://www.aljazeera.com/opinions/2019/10/21/donald-trump-found-a-different-way-to-build-his-wall

### Source context

- S00 President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator .
- S01 This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , the president find a different way to “ build the wall ” .
- S02 **anchor** The brick in Trump ’ s border wall take several form .
- S03 While they may not have be able to build a complete wall on the US ’ s southern border , the Trump administration have make it virtually impossible for most asylum seeker to request protection in the US .

### Shape summary

| strategy | status | span | sentences | words | chars |
|---|---|---:|---:|---:|---:|
| `sentence` | ok | S02 (anchor included) | 1 | 10 | 54 |
| `ctx_window` | ok | S01-S03 (anchor included) | 3 | 80 | 452 |
| `raw` | ok | S00-S03 (anchor included) | 4 | 117 | 666 |
| `semantic` | ok | S00-S03 (anchor included) | 4 | 117 | 666 |
| `ctx_semantic` | ok | S00-S03 (anchor included) | 4 | 117 | 666 |

### sentence

English evidence:

> The brick in Trump ’ s border wall take several form .

Chinese translation:

> 特朗普边境墙中的“砖块”有几种形式。

### ctx_window

English evidence:

> This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , the president find a different way to “ build the wall ” . The brick in Trump ’ s border wall take several form . While they may not have be able to build a complete wall on the US ’ s southern border , the Trump administration have make it virtually impossible for most asylum seeker to request protection in the US .

Chinese translation:

> 因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。

### raw

English evidence:

> President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , the president find a different way to “ build the wall ” . The brick in Trump ’ s border wall take several form . While they may not have be able to bui ... [truncated, 166 chars omitted]

Chinese translation:

> 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。

### semantic

English evidence:

> President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , the president find a different way to “ build the wall ” . The brick in Trump ’ s border wall take several form . While they may not have be able to bui ... [truncated, 166 chars omitted]

Chinese translation:

> 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。

### ctx_semantic

English evidence:

> President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , the president find a different way to “ build the wall ” . The brick in Trump ’ s border wall take several form . While they may not have be able to bui ... [truncated, 166 chars omitted]

Chinese translation:

> 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。

## Example 3: `11096.json`

- label: `false`
- claim: Says John McCain has done nothing to help the vets.
- anchor: report_id=`8805969`, sent_idx=`3`, score=`0.4101`, source_sentences=`4`
- source: https://abcnews.go.com/Politics/donald-trump-owe-mccain-apology/story?id=32547286 https://abcnews.go.com/Politics/donald-trump-owe-mccain-apology/story?id=32547286

### Source context

- S01 Trump say he leave to a `` standing ovation `` after speak at the Family Leadership Council summit . `` When I leave the room , it be a total standing ovation , `` say Trump .
- S02 When speak about McCain on Saturday , Trump say he like `` people who be n't capture .
- S03 **anchor** Trump say veteran be treat like `` third-class citizen , `` add that McCain have `` do nothing to help the vet .

### Shape summary

| strategy | status | span | sentences | words | chars |
|---|---|---:|---:|---:|---:|
| `sentence` | ok | S03 (anchor included) | 1 | 19 | 112 |
| `ctx_window` | ok | S02-S03 (anchor included) | 2 | 35 | 199 |
| `raw` | ok | S00-S03 (anchor included) | 4 | 92 | 541 |
| `semantic` | ok | S03 (anchor included) | 1 | 19 | 112 |
| `ctx_semantic` | ok | S03 (anchor included) | 1 | 19 | 112 |

### sentence

English evidence:

> Trump say veteran be treat like `` third-class citizen , `` add that McCain have `` do nothing to help the vet .

Chinese translation:

> 特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。

### ctx_window

English evidence:

> When speak about McCain on Saturday , Trump say he like `` people who be n't capture . Trump say veteran be treat like `` third-class citizen , `` add that McCain have `` do nothing to help the vet .

Chinese translation:

> 周六谈到麦凯恩时，特朗普说他喜欢“没有被俘的人”。特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。

### raw

English evidence:

>  -- Republican presidential candidate Donald Trump say he do not owe John McCain an apology for say the Arizona senator be only a war hero “ because he be capture . Trump say he leave to a `` standing ovation `` after speak at the Family Leadership Council summit . `` When I leave the room , it be a total standing ovation , `` say Trump . When speak about McCain on Saturday , Trump say he like `` people who be n't capture . Trump say veteran be treat like `` third-class citizen , `` add that Mc ... [truncated, 41 chars omitted]

Chinese translation:

> 共和党总统候选人唐纳德·特朗普表示，他不欠约翰·麦凯恩道歉；此前他说这位亚利桑那州参议员只是因为“被俘”才成为战争英雄。特朗普说，他在家庭领导委员会峰会发言后是在“全场起立鼓掌”中离开的。他说：“当我离开房间时，那是全场起立鼓掌。”周六谈到麦凯恩时，特朗普说他喜欢“没有被俘的人”。特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。

### semantic

English evidence:

> Trump say veteran be treat like `` third-class citizen , `` add that McCain have `` do nothing to help the vet .

Chinese translation:

> 特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。

### ctx_semantic

English evidence:

> Trump say veteran be treat like `` third-class citizen , `` add that McCain have `` do nothing to help the vet .

Chinese translation:

> 特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。
