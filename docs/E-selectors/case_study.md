下面是这三个案例的 **Oracle evidence set**。顺序就是 oracle 的 `selected_indices` 顺序。

**1. `4855.json`**
Claim：Says Mitt Romney once supported President Obamas health care plan but now opposes it.  
中文：称米特·罗姆尼曾支持奥巴马总统的医保计划，但现在反对它。  
Gold label：`barely-true`  
Oracle selected indices：`[13, 8, 1, 9, 2]`

- E1 / C14  
  原文：Still, Romney also never withdraw his support from the program. After a backlash Thursday, Romney try to walk that line again, post on Facebook that he still oppose Obamacare because it have fail, drive up premium and take insurance away from people.  
  中文：不过，罗姆尼也从未撤回对该项目的支持。周四遭到反弹后，罗姆尼再次试图维持这种平衡，并在 Facebook 上表示，他仍然反对奥巴马医保，因为它“失败了”“推高了保费”并“让人们失去保险”。

- E2 / C9  
  原文：He argue that without Romneycare, the universal health care plan he sign into law a Massachusetts governor, Obamacare would never have become law.  
  中文：他认为，如果没有他担任马萨诸塞州长时签署成为法律的全民医保计划“罗姆尼医保”，奥巴马医保就不可能成为法律。

- E3 / C2  
  原文：Romney say if elect president, he would allow state to opt out of the health care law.  
  中文：罗姆尼称，如果当选总统，他会允许各州退出医保法。

- E4 / C10  
  原文：In the speech, Romney won't spend much time talk about Massachusetts, and the plan he sign that now require the state's citizen to buy health insurance — an individual mandate that be include in the federal law and drive Republican fury.git  
  中文：在演讲中，罗姆尼不会花太多时间谈马萨诸塞，以及他签署的、要求该州居民购买医保的计划；这一“个人强制参保”后来被纳入联邦法律，并激起共和党人的愤怒。

- E5 / C3  
  原文：Mitt Romney spoke in Washington on Thursday after the Supreme Court rule on President Obama's health care law.  
  中文：最高法院就奥巴马总统的医保法作出裁决后，米特·罗姆尼周四在华盛顿发表讲话。

**2. `11447.json`**
Claim：We have the highest tax rate anywhere in the world.  
中文：我们的税率是全世界最高的。  
Gold label：`false`  
Oracle selected indices：`[1, 13, 10, 2, 7]`

- E1 / C2  
  原文：Our verdict Incorrect, a number of European country have high income tax rate than Scotland. “They (Scotland) have the high tax anywhere in Europe” Boris Johnson, 4 September 2019.  
  中文：我们的结论：不正确。若干欧洲国家的所得税率高于苏格兰。鲍里斯·约翰逊曾在 2019 年 9 月 4 日称：“他们（苏格兰）拥有欧洲最高的税。”

- E2 / C14  
  原文：At his first Prime Minister's Questions, Boris Johnson say that Scotland have the high tax anywhere in Europe.  
  中文：在他首次参加首相问答时，鲍里斯·约翰逊称苏格兰拥有欧洲最高税率。

- E3 / C11  
  原文：The tax rate for 2021 range from 14.  
  中文：2021 年的税率从 14% 起。  
  注：这条原始 evidence 是截断片段，本身语义不完整。

- E4 / C3  
  原文：The tax we love to hate today.  
  中文：当今人们最讨厌的税。  
  注：这条也偏背景/噪声，不是直接判定 claim 的强证据。

- E5 / C8  
  原文：19 percent, more than anywhere else in the world.  
  中文：19%，高于世界其他任何地方。  
  注：这条是片段化证据，需要上下文才能判断具体税种和对象。

**3. `10443.json`**
Claim：In Iraq and Syria, American leadership, including our military power, is stopping the Islamic State's advance.  
中文：在伊拉克和叙利亚，包括军事力量在内的美国领导力正在阻止“伊斯兰国”的推进。  
Gold label：`half-true`  
Oracle selected indices：`[13, 2, 11, 3, 9]`

- E1 / C14  
  原文：Little suggest these group can be defeat by military mean alone, yet they espouse goal hard to accommodate in negotiated settlement.  
  中文：几乎没有迹象表明这些组织能仅靠军事手段被击败，但它们所主张的目标又很难通过谈判解决来容纳。

- E2 / C3  
  原文：These militant, know a the Islamic State in Iraq and the Levant (ISIL, or the Islamic State), declare an Islamic state or caliphate in this captured territory and claim political and theological authority over the world's Muslims.  
  中文：这些武装分子，即“伊拉克和黎凡特伊斯兰国”（ISIL，或“伊斯兰国”），在占领地区宣布建立伊斯兰国家或哈里发国，并声称对全世界穆斯林拥有政治和神学权威。

- E3 / C12  
  原文：And official have say they believe Iran be behind the October drone attack at the military outpost in southern Syria where American troop be base.  
  中文：官员们表示，他们认为伊朗是 10 月袭击叙利亚南部美军驻扎军事哨所的无人机事件幕后方。

- E4 / C4  
  原文：Its leadership be mostly Iraqi but the movement be protean.  
  中文：其领导层大多是伊拉克人，但这个运动形态多变。

- E5 / C10  
  原文：People seek to travel to engage in terrorist activity in Syria or Iraq should be in no doubt that the UK will take the strong possible action to protect our national security, include prosecute those who break the law.  
  中文：试图前往叙利亚或伊拉克参与恐怖活动的人应该清楚，英国将采取尽可能强硬的行动保护国家安全，包括起诉违法者。