# Chunking Evidence Visualization

- experiment: `b4`
- split: `test`
- strategies: `sentence, ctx_window, raw, semantic, ctx_semantic`
- theta sweep: `0.3, 0.5, 0.7`
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

> McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power . A search of multiple Swahili dictionary turn up no result for the word `` Denali . As our fact-checker f ... [truncated, 212 chars omitted]

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

### Theta sweep

#### semantic

| field | theta=0.3 | theta=0.5 | theta=0.7 |
|---|---|---|---|
| span | S00-S04 (anchor included) | S00-S04 (anchor included) | S00-S01 (anchor included) |
| sentences | 5 | 5 | 2 |
| words | 91 | 91 | 38 |
| English evidence | McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power . A search of multiple Swahili dictionary turn up no result for the word `` Denali . As our fact-checker f ... [truncated, 212 chars omitted] | McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power . A search of multiple Swahili dictionary turn up no result for the word `` Denali . As our fact-checker f ... [truncated, 212 chars omitted] | McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power . |
| Chinese translation | 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？一张在 Facebook 上流传的图片指责总统从其肯尼亚血统中寻找灵感，并声称“Denali”是肯尼亚语中“黑色权力”的意思。查询多部斯瓦希里语词典都找不到“Denali”这个词。正如 Snopes 的事实核查者发现的那样，斯瓦希里语中“黑色”是“mweusi”，“权力”是“nguvu”。“Denali”并不出现在斯瓦希里语中，而斯瓦希里语是肯尼亚的两种国家语言之一。 | 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？一张在 Facebook 上流传的图片指责总统从其肯尼亚血统中寻找灵感，并声称“Denali”是肯尼亚语中“黑色权力”的意思。查询多部斯瓦希里语词典都找不到“Denali”这个词。正如 Snopes 的事实核查者发现的那样，斯瓦希里语中“黑色”是“mweusi”，“权力”是“nguvu”。“Denali”并不出现在斯瓦希里语中，而斯瓦希里语是肯尼亚的两种国家语言之一。 | 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？一张在 Facebook 上流传的图片指责总统从其肯尼亚血统中寻找灵感，并声称“Denali”是肯尼亚语中“黑色权力”的意思。 |

#### ctx_semantic

| field | theta=0.3 | theta=0.5 | theta=0.7 |
|---|---|---|---|
| span | S00-S04 (anchor included) | S00-S04 (anchor included) | S00-S01 (anchor included) |
| sentences | 5 | 5 | 2 |
| words | 91 | 91 | 38 |
| English evidence | McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power . A search of multiple Swahili dictionary turn up no result for the word `` Denali . As our fact-checker f ... [truncated, 212 chars omitted] | McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power . A search of multiple Swahili dictionary turn up no result for the word `` Denali . As our fact-checker f ... [truncated, 212 chars omitted] | McKinley to `` Denali `` because it the Kenyan word for `` black power `` ? . One image circulate on Facebook accuse the president of reach back to his Kenyan root for inspiration , claim that `` ‘ Denali ’ be the Kenyan word for ‘ black power . |
| Chinese translation | 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？一张在 Facebook 上流传的图片指责总统从其肯尼亚血统中寻找灵感，并声称“Denali”是肯尼亚语中“黑色权力”的意思。查询多部斯瓦希里语词典都找不到“Denali”这个词。正如 Snopes 的事实核查者发现的那样，斯瓦希里语中“黑色”是“mweusi”，“权力”是“nguvu”。“Denali”并不出现在斯瓦希里语中，而斯瓦希里语是肯尼亚的两种国家语言之一。 | 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？一张在 Facebook 上流传的图片指责总统从其肯尼亚血统中寻找灵感，并声称“Denali”是肯尼亚语中“黑色权力”的意思。查询多部斯瓦希里语词典都找不到“Denali”这个词。正如 Snopes 的事实核查者发现的那样，斯瓦希里语中“黑色”是“mweusi”，“权力”是“nguvu”。“Denali”并不出现在斯瓦希里语中，而斯瓦希里语是肯尼亚的两种国家语言之一。 | 麦金利山改名为“德纳里”，是因为它在肯尼亚语中表示“黑色权力”吗？一张在 Facebook 上流传的图片指责总统从其肯尼亚血统中寻找灵感，并声称“Denali”是肯尼亚语中“黑色权力”的意思。 |

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

> This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , the president find a different way to “ build the wall ” . The brick in Trump ’ s border wall take several form . While they may not have be able to build a complete wall on the US ’ s southern border , the Trump adm ... [truncated, 102 chars omitted]

Chinese translation:

> 因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。

### raw

English evidence:

> President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted]

Chinese translation:

> 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。

### semantic

English evidence:

> President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted]

Chinese translation:

> 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。

### ctx_semantic

English evidence:

> President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted]

Chinese translation:

> 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。

### Theta sweep

#### semantic

| field | theta=0.3 | theta=0.5 | theta=0.7 |
|---|---|---|---|
| span | S00-S03 (anchor included) | S00-S03 (anchor included) | S00-S03 (anchor included) |
| sentences | 4 | 4 | 4 |
| words | 117 | 117 | 117 |
| English evidence | President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted] | President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted] | President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted] |
| Chinese translation | 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。 | 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。 | 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。 |

#### ctx_semantic

| field | theta=0.3 | theta=0.5 | theta=0.7 |
|---|---|---|---|
| span | S00-S03 (anchor included) | S00-S03 (anchor included) | S00-S03 (anchor included) |
| sentences | 4 | 4 | 4 |
| words | 117 | 117 | 117 |
| English evidence | President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted] | President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted] | President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted] |
| Chinese translation | 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。 | 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。 | 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。 |

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

>  -- Republican presidential candidate Donald Trump say he do not owe John McCain an apology for say the Arizona senator be only a war hero “ because he be capture . Trump say he leave to a `` standing ovation `` after speak at the Family Leadership Council summit . `` When I leave the room , it be a total standing ovation , `` say Trump . When spe ... [truncated, 191 chars omitted]

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

### Theta sweep

#### semantic

| field | theta=0.3 | theta=0.5 | theta=0.7 |
|---|---|---|---|
| span | S00-S03 (anchor included) | S00-S03 (anchor included) | S03 (anchor included) |
| sentences | 4 | 4 | 1 |
| words | 92 | 92 | 19 |
| English evidence |  -- Republican presidential candidate Donald Trump say he do not owe John McCain an apology for say the Arizona senator be only a war hero “ because he be capture . Trump say he leave to a `` standing ovation `` after speak at the Family Leadership Council summit . `` When I leave the room , it be a total standing ovation , `` say Trump . When spe ... [truncated, 191 chars omitted] |  -- Republican presidential candidate Donald Trump say he do not owe John McCain an apology for say the Arizona senator be only a war hero “ because he be capture . Trump say he leave to a `` standing ovation `` after speak at the Family Leadership Council summit . `` When I leave the room , it be a total standing ovation , `` say Trump . When spe ... [truncated, 191 chars omitted] | Trump say veteran be treat like `` third-class citizen , `` add that McCain have `` do nothing to help the vet . |
| Chinese translation | 共和党总统候选人唐纳德·特朗普表示，他不欠约翰·麦凯恩道歉；此前他说这位亚利桑那州参议员只是因为“被俘”才成为战争英雄。特朗普说，他在家庭领导委员会峰会发言后是在“全场起立鼓掌”中离开的。他说：“当我离开房间时，那是全场起立鼓掌。”周六谈到麦凯恩时，特朗普说他喜欢“没有被俘的人”。特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。 | 共和党总统候选人唐纳德·特朗普表示，他不欠约翰·麦凯恩道歉；此前他说这位亚利桑那州参议员只是因为“被俘”才成为战争英雄。特朗普说，他在家庭领导委员会峰会发言后是在“全场起立鼓掌”中离开的。他说：“当我离开房间时，那是全场起立鼓掌。”周六谈到麦凯恩时，特朗普说他喜欢“没有被俘的人”。特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。 | 特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。 |

#### ctx_semantic

| field | theta=0.3 | theta=0.5 | theta=0.7 |
|---|---|---|---|
| span | S00-S03 (anchor included) | S00-S03 (anchor included) | S03 (anchor included) |
| sentences | 4 | 4 | 1 |
| words | 92 | 92 | 19 |
| English evidence |  -- Republican presidential candidate Donald Trump say he do not owe John McCain an apology for say the Arizona senator be only a war hero “ because he be capture . Trump say he leave to a `` standing ovation `` after speak at the Family Leadership Council summit . `` When I leave the room , it be a total standing ovation , `` say Trump . When spe ... [truncated, 191 chars omitted] |  -- Republican presidential candidate Donald Trump say he do not owe John McCain an apology for say the Arizona senator be only a war hero “ because he be capture . Trump say he leave to a `` standing ovation `` after speak at the Family Leadership Council summit . `` When I leave the room , it be a total standing ovation , `` say Trump . When spe ... [truncated, 191 chars omitted] | Trump say veteran be treat like `` third-class citizen , `` add that McCain have `` do nothing to help the vet . |
| Chinese translation | 共和党总统候选人唐纳德·特朗普表示，他不欠约翰·麦凯恩道歉；此前他说这位亚利桑那州参议员只是因为“被俘”才成为战争英雄。特朗普说，他在家庭领导委员会峰会发言后是在“全场起立鼓掌”中离开的。他说：“当我离开房间时，那是全场起立鼓掌。”周六谈到麦凯恩时，特朗普说他喜欢“没有被俘的人”。特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。 | 共和党总统候选人唐纳德·特朗普表示，他不欠约翰·麦凯恩道歉；此前他说这位亚利桑那州参议员只是因为“被俘”才成为战争英雄。特朗普说，他在家庭领导委员会峰会发言后是在“全场起立鼓掌”中离开的。他说：“当我离开房间时，那是全场起立鼓掌。”周六谈到麦凯恩时，特朗普说他喜欢“没有被俘的人”。特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。 | 特朗普说退伍军人被当作“三等公民”对待，并补充说麦凯恩“没有为退伍军人做任何事”。 |

## Example 4: `7562.json`

- label: `mostly-true`
- claim: Already, a prototype driverless car has traveled more than 300,000 miles in the crowded maze of California streets without a single accident.
- anchor: report_id=`9015460`, sent_idx=`1`, score=`1.1824`, source_sentences=`4`
- source: https://www.mynews13.com/fl/orlando/news/2013/3/29/politifact_weekly_wr https://www.mynews13.com/fl/orlando/news/2013/3/29/politifact_weekly_wr

### Source context

- S00 He think innovation here be second to none and that technology such a Google 's driverless car can lead to a new economic boom , not to mention safe road .
- S01 **anchor** Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident .
- S02 It turn out the former Governor 's claim that driverless car be accident-free be true .
- S03 One driverless car be rear-ended , and the other be be control by a human at the time it crash .

### Shape summary

| strategy | status | span | sentences | words | chars |
|---|---|---:|---:|---:|---:|
| `sentence` | ok | S01 (anchor included) | 1 | 23 | 140 |
| `ctx_window` | ok | S00-S02 (anchor included) | 3 | 68 | 384 |
| `raw` | ok | S00-S03 (anchor included) | 4 | 88 | 481 |
| `semantic` | ok | S01 (anchor included) | 1 | 23 | 140 |
| `ctx_semantic` | ok | S01 (anchor included) | 1 | 23 | 140 |

### sentence

English evidence:

> Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident .

Chinese translation: *(not provided; use --translations or --translation-template)*

### ctx_window

English evidence:

> He think innovation here be second to none and that technology such a Google 's driverless car can lead to a new economic boom , not to mention safe road . Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident . It turn out the former Governor 's claim that driverl ... [truncated, 34 chars omitted]

Chinese translation: *(not provided; use --translations or --translation-template)*

### raw

English evidence:

> He think innovation here be second to none and that technology such a Google 's driverless car can lead to a new economic boom , not to mention safe road . Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident . It turn out the former Governor 's claim that driverl ... [truncated, 131 chars omitted]

Chinese translation: *(not provided; use --translations or --translation-template)*

### semantic

English evidence:

> Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident .

Chinese translation: *(not provided; use --translations or --translation-template)*

### ctx_semantic

English evidence:

> Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident .

Chinese translation: *(not provided; use --translations or --translation-template)*

### Theta sweep

#### semantic

| field | theta=0.3 | theta=0.5 | theta=0.7 |
|---|---|---|---|
| span | S00-S03 (anchor included) | S00-S03 (anchor included) | S01 (anchor included) |
| sentences | 4 | 4 | 1 |
| words | 88 | 88 | 23 |
| English evidence | He think innovation here be second to none and that technology such a Google &#x27;s driverless car can lead to a new economic boom , not to mention safe road . Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident . It turn out the former Governor &#x27;s claim that driverl ... [truncated, 131 chars omitted] | He think innovation here be second to none and that technology such a Google &#x27;s driverless car can lead to a new economic boom , not to mention safe road . Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident . It turn out the former Governor &#x27;s claim that driverl ... [truncated, 131 chars omitted] | Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident . |
| Chinese translation |  |  |  |

#### ctx_semantic

| field | theta=0.3 | theta=0.5 | theta=0.7 |
|---|---|---|---|
| span | S00-S03 (anchor included) | S00-S03 (anchor included) | S01 (anchor included) |
| sentences | 4 | 4 | 1 |
| words | 88 | 88 | 23 |
| English evidence | He think innovation here be second to none and that technology such a Google &#x27;s driverless car can lead to a new economic boom , not to mention safe road . Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident . It turn out the former Governor &#x27;s claim that driverl ... [truncated, 131 chars omitted] | He think innovation here be second to none and that technology such a Google &#x27;s driverless car can lead to a new economic boom , not to mention safe road . Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident . It turn out the former Governor &#x27;s claim that driverl ... [truncated, 131 chars omitted] | Already , a prototype driverless car have travel more than 300,000 mile in the crowded maze of California street without a single accident . |
| Chinese translation |  |  |  |

## Example 5: `12381.json`

- label: `true`
- claim: The United States has a low voter turnout rate.
- anchor: report_id=`5866083`, sent_idx=`0`, score=`1.0049`, source_sentences=`5`
- source: https://theconversation.com/why-so-few-young-americans-vote-132649 https://theconversation.com/why-so-few-young-americans-vote-132649

### Source context

- S00 **anchor** The United States have one of the low rate of youth voter turnout in the world .
- S01 Youth turnout in the first state to hold primary and caucus have range from 10 % in Alabama to 24 % in Iowa .
- S02 Fewer than 1 in 5 young people cast ballot in all Super Tuesday state .

### Shape summary

| strategy | status | span | sentences | words | chars |
|---|---|---:|---:|---:|---:|
| `sentence` | ok | S00 (anchor included) | 1 | 16 | 80 |
| `ctx_window` | ok | S00-S01 (anchor included) | 2 | 39 | 190 |
| `raw` | ok | S00-S04 (anchor included) | 5 | 87 | 440 |
| `semantic` | ok | S00 (anchor included) | 1 | 16 | 80 |
| `ctx_semantic` | ok | S00 (anchor included) | 1 | 16 | 80 |

### sentence

English evidence:

> The United States have one of the low rate of youth voter turnout in the world .

Chinese translation: *(not provided; use --translations or --translation-template)*

### ctx_window

English evidence:

> The United States have one of the low rate of youth voter turnout in the world . Youth turnout in the first state to hold primary and caucus have range from 10 % in Alabama to 24 % in Iowa .

Chinese translation: *(not provided; use --translations or --translation-template)*

### raw

English evidence:

> The United States have one of the low rate of youth voter turnout in the world . Youth turnout in the first state to hold primary and caucus have range from 10 % in Alabama to 24 % in Iowa . Fewer than 1 in 5 young people cast ballot in all Super Tuesday state . Compared to primary and caucus in the past , few young people be vote in 2020 , while o ... [truncated, 90 chars omitted]

Chinese translation: *(not provided; use --translations or --translation-template)*

### semantic

English evidence:

> The United States have one of the low rate of youth voter turnout in the world .

Chinese translation: *(not provided; use --translations or --translation-template)*

### ctx_semantic

English evidence:

> The United States have one of the low rate of youth voter turnout in the world .

Chinese translation: *(not provided; use --translations or --translation-template)*

### Theta sweep

#### semantic

| field | theta=0.3 | theta=0.5 | theta=0.7 |
|---|---|---|---|
| span | S00-S04 (anchor included) | S00-S04 (anchor included) | S00 (anchor included) |
| sentences | 5 | 5 | 1 |
| words | 87 | 87 | 16 |
| English evidence | The United States have one of the low rate of youth voter turnout in the world . Youth turnout in the first state to hold primary and caucus have range from 10 % in Alabama to 24 % in Iowa . Fewer than 1 in 5 young people cast ballot in all Super Tuesday state . Compared to primary and caucus in the past , few young people be vote in 2020 , while o ... [truncated, 90 chars omitted] | The United States have one of the low rate of youth voter turnout in the world . Youth turnout in the first state to hold primary and caucus have range from 10 % in Alabama to 24 % in Iowa . Fewer than 1 in 5 young people cast ballot in all Super Tuesday state . Compared to primary and caucus in the past , few young people be vote in 2020 , while o ... [truncated, 90 chars omitted] | The United States have one of the low rate of youth voter turnout in the world . |
| Chinese translation |  |  |  |

#### ctx_semantic

| field | theta=0.3 | theta=0.5 | theta=0.7 |
|---|---|---|---|
| span | S00-S04 (anchor included) | S00-S04 (anchor included) | S00 (anchor included) |
| sentences | 5 | 5 | 1 |
| words | 87 | 87 | 16 |
| English evidence | The United States have one of the low rate of youth voter turnout in the world . Youth turnout in the first state to hold primary and caucus have range from 10 % in Alabama to 24 % in Iowa . Fewer than 1 in 5 young people cast ballot in all Super Tuesday state . Compared to primary and caucus in the past , few young people be vote in 2020 , while o ... [truncated, 90 chars omitted] | The United States have one of the low rate of youth voter turnout in the world . Youth turnout in the first state to hold primary and caucus have range from 10 % in Alabama to 24 % in Iowa . Fewer than 1 in 5 young people cast ballot in all Super Tuesday state . Compared to primary and caucus in the past , few young people be vote in 2020 , while o ... [truncated, 90 chars omitted] | The United States have one of the low rate of youth voter turnout in the world . |
| Chinese translation |  |  |  |

## 512-Token Context Packing Case Study

Evidence chunks are greedily packed under the same evidence-only token budget.

- event_id: `11972.json`
- label: `true`
- claim: Building a wall on the U.S.-Mexico border will take literally years.
- budget: `512` evidence tokens
- tokenizer: `/home/fenglin/project/hateSpeechDetection/models/base/Qwen2.5-7B-Instruct`
- ranked candidate anchors considered: `64`

| strategy | status | packed items | used tokens | utilization | skipped duplicate chunks | skipped over budget |
|---|---|---:|---:|---:|---:|---:|
| `sentence` | ok | 18 | 510/512 | 99.6% | 0 | 46 |
| `ctx_window` | ok | 7 | 502/512 | 98.0% | 0 | 57 |
| `raw` | ok | 4 | 489/512 | 95.5% | 47 | 13 |
| `semantic` | ok | 10 | 503/512 | 98.2% | 15 | 39 |
| `ctx_semantic` | ok | 10 | 503/512 | 98.2% | 15 | 39 |

### Context pack: sentence

| # | tokens | span | source | score | evidence preview | Chinese translation |
|---:|---:|---|---|---:|---|---|
| 1 | 15 | S02 (anchor included) | report=7119951, sent=2 | 0.6406 | The brick in Trump ’ s border wall take several form . | 特朗普边境墙中的“砖块”有几种形式。 |
| 2 | 43 | S00 (anchor included) | report=2904313, sent=0 | 0.5258 | WASHINGTON ( Reuters ) - President Donald Trump ’ s “ wall ” along the U . -Mexico border would be a series of fence and wall that would cost a much a $ 21 . |  |
| 3 | 35 | S02 (anchor included) | report=4646302, sent=2 | 0.5258 | Donald Trump ’ s promise to build a wall across the U . -Mexico border be one of the most grandiose and unnecessary policy ever pursue in U . |  |
| 4 | 26 | S00 (anchor included) | report=4646302, sent=0 | 0.4832 | Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . |  |
| 5 | 20 | S02 (anchor included) | report=7257499, sent=2 | 0.4805 | ADVERTISEMENT “ In support of CBP ’ s border infrastructure program , the U . |  |
| 6 | 19 | S04 (anchor included) | report=3683124, sent=4 | 0.4127 | Once work on President Donald Trump ’ s border wall begin , construction be rapid . |  |
| 7 | 45 | S03 (anchor included) | report=4531902, sent=3 | 0.3719 | For all the talk of secure the border and building a wall , there be surprisingly little visual material that convey just how vast this stretch of space be . -Mexico border span 1,954 mile . |  |
| 8 | 28 | S02 (anchor included) | report=4681261, sent=2 | 0.3624 | Segments of the first border wall in Texas built after President Trump take office , near Donna in 2019 . |  |
| 9 | 39 | S01 (anchor included) | report=4646302, sent=1 | 0.3456 | Terms of Service Privacy Policy Cookie Policy Do Not Sell My Personal Information Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . |  |
| 10 | 17 | S04 (anchor included) | report=8783470, sent=4 | 0.3391 | The following interview take place in Tucson , AZ in the U . |  |
| 11 | 48 | S01 (anchor included) | report=7257499, sent=1 | 0.3188 | The Department of Homeland Security ( DHS ) announce last month that it be take step to repair the Rio Grande Valley ’ s flood barrier system after the Trump administration make hole in the structure to make way for the border wall . |  |
| 12 | 22 | S01 (anchor included) | report=2904313, sent=1 | 0.3036 | 6 billion , and take more than three year to construct , base on a U . |  |
| 13 | 24 | S01 (anchor included) | report=5864045, sent=1 | 0.3036 | Washington be literally at a standstill over the funding for a wall on the United States southern border . |  |
| 14 | 42 | S06 (anchor included) | report=4681261, sent=6 | 0.3019 | Ultimately , Trump ’ s administration erect about 80 mile of new barrier before he leave office , include 21 at the Texas-Mexico border , pay for by the U . |  |
| 15 | 28 | S00 (anchor included) | report=7512619, sent=0 | 0.2886 | Greg Abbott want to build a border wall , but do Texas have the ability — or money — to do so ? . |  |
| 16 | 19 | S00 (anchor included) | report=4681261, sent=0 | 0.2886 | Greg Abbott will confront same hurdle a Bush , Trump to build border wall . |  |
| 17 | 29 | S00 (anchor included) | report=3683124, sent=0 | 0.2629 | Kevin Cooley/Redux/EyevineConstruction along the border wall at Signal Mountain outside of Mexicali , California . |  |
| 18 | 11 | S00 (anchor included) | report=4249538, sent=0 | 0.2427 | Can it be take down ? . |  |

### Context pack: ctx_window

| # | tokens | span | source | score | evidence preview | Chinese translation |
|---:|---:|---|---|---:|---|---|
| 1 | 92 | S01-S03 (anchor included) | report=7119951, sent=2 | 0.6406 | This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , the president find a different way to “ build the wall ” . The brick in Trump ’ s border wall take several form . While they may not have be able to build a complete wall on the US ’ s southern border , the Trump adm ... [truncated, 102 chars omitted] | 因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。 |
| 2 | 61 | S00-S01 (anchor included) | report=2904313, sent=0 | 0.5258 | WASHINGTON ( Reuters ) - President Donald Trump ’ s “ wall ” along the U . -Mexico border would be a series of fence and wall that would cost a much a $ 21 . 6 billion , and take more than three year to construct , base on a U . |  |
| 3 | 71 | S01-S02 (anchor included) | report=4646302, sent=2 | 0.5258 | Terms of Service Privacy Policy Cookie Policy Do Not Sell My Personal Information Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . Donald Trump ’ s promise to build a wall across the U . -Mexico border be one of the most grandiose and unnecessary policy ever pursue in U . |  |
| 4 | 62 | S00-S01 (anchor included) | report=4646302, sent=0 | 0.4832 | Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . Terms of Service Privacy Policy Cookie Policy Do Not Sell My Personal Information Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . |  |
| 5 | 64 | S01-S02 (anchor included) | report=7257499, sent=2 | 0.4805 | The Department of Homeland Security ( DHS ) announce last month that it be take step to repair the Rio Grande Valley ’ s flood barrier system after the Trump administration make hole in the structure to make way for the border wall . ADVERTISEMENT “ In support of CBP ’ s border infrastructure program , the U . |  |
| 6 | 109 | S03-S05 (anchor included) | report=3683124, sent=4 | 0.4127 | McDaniel , tall and slim in a tan jumpsuit , begin take fly lesson in the 80 , and have since log 2,000 hour in the air . Once work on President Donald Trump ’ s border wall begin , construction be rapid . Sasabe , a sleepy border town , locate over an hour from the near city of Tucson , be transform into a construction site . “ I don ’ t think you ... [truncated, 82 chars omitted] |  |
| 7 | 43 | S00-S01 (anchor included) | report=7512619, sent=0 | 0.2886 | Greg Abbott want to build a border wall , but do Texas have the ability — or money — to do so ? . But the two-term Republican governor say it &#x27;s time to secure the border . |  |

### Context pack: raw

| # | tokens | span | source | score | evidence preview | Chinese translation |
|---:|---:|---|---|---:|---|---|
| 1 | 136 | S00-S03 (anchor included) | report=7119951, sent=2 | 0.6406 | President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted] | 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。 |
| 2 | 133 | S00-S03 (anchor included) | report=2904313, sent=0 | 0.5258 | WASHINGTON ( Reuters ) - President Donald Trump ’ s “ wall ” along the U . -Mexico border would be a series of fence and wall that would cost a much a $ 21 . 6 billion , and take more than three year to construct , base on a U . Twitter number paint grim profitability picture Trump change tack , back &#x27;one China &#x27; policy in call with Xi The report b ... [truncated, 278 chars omitted] |  |
| 3 | 94 | S00-S02 (anchor included) | report=4646302, sent=2 | 0.5258 | Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . Terms of Service Privacy Policy Cookie Policy Do Not Sell My Personal Information Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . Donald Trump ’ s promise to build a wall across ... [truncated, 93 chars omitted] |  |
| 4 | 126 | S00-S04 (anchor included) | report=4531902, sent=3 | 0.3719 | TheIntercept_videoDonateBecome a memberBest of Luck With the WallPresident Trump just announce that he intend to fulfill his campaign pledge to build a wall along the U . This be what that border look like . In partnership withIn partnership withWhat do the southern border of the United States look like ? . For all the talk of secure the border and ... [truncated, 221 chars omitted] |  |

### Context pack: semantic

| # | tokens | span | source | score | evidence preview | Chinese translation |
|---:|---:|---|---|---:|---|---|
| 1 | 136 | S00-S03 (anchor included) | report=7119951, sent=2 | 0.6406 | President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted] | 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。 |
| 2 | 43 | S00 (anchor included) | report=2904313, sent=0 | 0.5258 | WASHINGTON ( Reuters ) - President Donald Trump ’ s “ wall ” along the U . -Mexico border would be a series of fence and wall that would cost a much a $ 21 . |  |
| 3 | 35 | S02 (anchor included) | report=4646302, sent=2 | 0.5258 | Donald Trump ’ s promise to build a wall across the U . -Mexico border be one of the most grandiose and unnecessary policy ever pursue in U . |  |
| 4 | 62 | S00-S01 (anchor included) | report=4646302, sent=0 | 0.4832 | Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . Terms of Service Privacy Policy Cookie Policy Do Not Sell My Personal Information Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . |  |
| 5 | 20 | S02 (anchor included) | report=7257499, sent=2 | 0.4805 | ADVERTISEMENT “ In support of CBP ’ s border infrastructure program , the U . |  |
| 6 | 19 | S04 (anchor included) | report=3683124, sent=4 | 0.4127 | Once work on President Donald Trump ’ s border wall begin , construction be rapid . |  |
| 7 | 65 | S03-S04 (anchor included) | report=4531902, sent=3 | 0.3719 | For all the talk of secure the border and building a wall , there be surprisingly little visual material that convey just how vast this stretch of space be . -Mexico border span 1,954 mile . In place , there already be a border fence — more than 650 mile of it . |  |
| 8 | 85 | S00-S02 (anchor included) | report=4681261, sent=2 | 0.3624 | Greg Abbott will confront same hurdle a Bush , Trump to build border wall . The Texas Tribune The governor want to finish the wall that Donald Trump begin on the Texas-Mexico border , but he &#x27;ll have to overcome the same hurdle that impede the ex-president &#x27;s effort . Segments of the first border wall in Texas built after President Trump take offic ... [truncated, 24 chars omitted] |  |
| 9 | 16 | S04 (anchor included) | report=8783470, sent=4 | 0.3391 | The following interview take place in Tucson , AZ in the U . |  |
| 10 | 22 | S01 (anchor included) | report=2904313, sent=1 | 0.3036 | 6 billion , and take more than three year to construct , base on a U . |  |

### Context pack: ctx_semantic

| # | tokens | span | source | score | evidence preview | Chinese translation |
|---:|---:|---|---|---:|---|---|
| 1 | 136 | S00-S03 (anchor included) | report=7119951, sent=2 | 0.6406 | President Donald Trump ’ s obsession with build a wall on the United States-Mexico border know few limit . he reportedly even entertain extreme idea such a build a moat over the border and fill it with alligator . This be why when it become apparent that his administration be not go to succeed in erect a complete physical barrier on the border , th ... [truncated, 316 chars omitted] | 唐纳德·特朗普总统对在美墨边境建墙的执念几乎没有边界；据报道，他甚至考虑过在边境修护城河并放入鳄鱼等极端想法。因此，当他的政府显然无法在边境建成完整的实体屏障时，总统找到了另一种“建墙”的方式。特朗普边境墙中的“砖块”有几种形式。虽然他们可能没能在美国南部边境建成完整的墙，但特朗普政府几乎让大多数寻求庇护者无法在美国申请保护。 |
| 2 | 43 | S00 (anchor included) | report=2904313, sent=0 | 0.5258 | WASHINGTON ( Reuters ) - President Donald Trump ’ s “ wall ” along the U . -Mexico border would be a series of fence and wall that would cost a much a $ 21 . |  |
| 3 | 35 | S02 (anchor included) | report=4646302, sent=2 | 0.5258 | Donald Trump ’ s promise to build a wall across the U . -Mexico border be one of the most grandiose and unnecessary policy ever pursue in U . |  |
| 4 | 62 | S00-S01 (anchor included) | report=4646302, sent=0 | 0.4832 | Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . Terms of Service Privacy Policy Cookie Policy Do Not Sell My Personal Information Over two dozen scientist have propose a wall on the U . -Mexico border that we should start build right now . |  |
| 5 | 20 | S02 (anchor included) | report=7257499, sent=2 | 0.4805 | ADVERTISEMENT “ In support of CBP ’ s border infrastructure program , the U . |  |
| 6 | 19 | S04 (anchor included) | report=3683124, sent=4 | 0.4127 | Once work on President Donald Trump ’ s border wall begin , construction be rapid . |  |
| 7 | 65 | S03-S04 (anchor included) | report=4531902, sent=3 | 0.3719 | For all the talk of secure the border and building a wall , there be surprisingly little visual material that convey just how vast this stretch of space be . -Mexico border span 1,954 mile . In place , there already be a border fence — more than 650 mile of it . |  |
| 8 | 85 | S00-S02 (anchor included) | report=4681261, sent=2 | 0.3624 | Greg Abbott will confront same hurdle a Bush , Trump to build border wall . The Texas Tribune The governor want to finish the wall that Donald Trump begin on the Texas-Mexico border , but he &#x27;ll have to overcome the same hurdle that impede the ex-president &#x27;s effort . Segments of the first border wall in Texas built after President Trump take offic ... [truncated, 24 chars omitted] |  |
| 9 | 16 | S04 (anchor included) | report=8783470, sent=4 | 0.3391 | The following interview take place in Tucson , AZ in the U . |  |
| 10 | 22 | S01 (anchor included) | report=2904313, sent=1 | 0.3036 | 6 billion , and take more than three year to construct , base on a U . |  |
