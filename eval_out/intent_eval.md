## 意图识别评测

### 一、同源评测（n=105，**分数偏高，只看机制不看绝对值**）

> golden query 是围绕 intent exemplar 的同一批业务话术写的，三条通道里有两条直接在 exemplar 上建库/训练，因此这里的分数必然接近满分 —— 这是泄漏，不是能力。可信结论看下面第二张表。

- Accuracy: **0.7905**
- Macro-F1: **0.7992**
- 弃权率(need_clarify): 12.38%
- 作答样本准确率: 0.8804

| 意图 | support | Precision | Recall | F1 |
|---|---|---|---|---|
| order_status | 11 | 0.467 | 0.636 | 0.538 |
| refund | 12 | 0.714 | 0.833 | 0.769 |
| return_exchange | 11 | 0.818 | 0.818 | 0.818 |
| logistics | 11 | 0.643 | 0.818 | 0.720 |
| complaint | 11 | 1.000 | 0.909 | 0.952 |
| product_consult | 10 | 1.000 | 0.600 | 0.750 |
| invoice | 10 | 1.000 | 0.700 | 0.824 |
| promo | 10 | 0.889 | 0.800 | 0.842 |
| account | 10 | 1.000 | 1.000 | 1.000 |
| other | 9 | 0.778 | 0.778 | 0.778 |

### 二、留出实验（exemplar 5 折交叉，可信版本）

前 3 折作 dev 标定权重，后 2 折作 test 只报一次。样本：dev 30 / test 20。

| 通道 | dev 准确率 | test 准确率 |
|---|---|---|
| llm_only | 0.2667 | 0.3000 |
| dense_only | 0.2000 | 0.3500 |
| rule_only | 0.8000 | 0.8500 |
| 融合·初始假设权重（未标定）{'llm': 0.55, 'dense': 0.25, 'rule': 0.2} | 0.3667 | 0.3500 |
| 融合·dev 标定权重（已设为默认）{'llm': 0.25, 'dense': 0.1, 'rule': 0.65} | 0.8000 | 0.8500 |

> test 上最强单路 0.8500；默认权重融合 0.3500，dev 标定后 0.8500。
>
> 默认权重把最弱的一路给了最高话语权（llm 0.55），融合因此明显低于最强单路。在 dev 上标定后权重变为 {'llm': 0.25, 'dense': 0.1, 'rule': 0.65}，说明**权重是超参，必须由数据决定，不能拍脑袋**。
>
> 注意：每折留出后每类只剩 4 条训练样本，属于极端小样本。这里的差距主要反映「小样本下哪条路更稳」，不能外推成「向量路无用」。

### 三、通道消融（同源数据，只看互补结构）

| 通道组合 | Accuracy | Macro-F1 | 弃权率 | 作答准确率 |
|---|---|---|---|---|
| rule_only | 0.6190 | 0.6502 | 34.29% | 0.8696 |
| dense_only | 0.7714 | 0.7753 | 7.62% | 0.8041 |
| llm_only | 0.7905 | 0.7992 | 27.62% | 0.9342 |
| rule+dense | 0.8000 | 0.8043 | 5.71% | 0.8283 |
| fusion(三路) | 0.7905 | 0.7992 | 12.38% | 0.8804 |

#### 通道互补性（谁救回了谁）

| 通道对 | 两者都对 | 仅前者对 | 仅后者对 | 都错 | 并集正确 | 互补率 |
|---|---|---|---|---|---|---|
| dense_only vs rule_only | 59 | 22 | 6 | 18 | 87 | 32.18% |
| dense_only vs llm_only | 76 | 5 | 7 | 17 | 88 | 13.64% |
| rule_only vs llm_only | 57 | 8 | 26 | 14 | 91 | 37.36% |

> 互补率非零说明该路不是冗余的。但要注意：在 exemplar 上训练的通道会同时满分，互补率因此被压到 0 —— 这本身就是泄漏的又一个证据。

### 错分样本（同源评测）
- `帮我查下订单到哪了` → 预测 logistics（gold order_status，conf=0.47，margin=0.15）
- `取消订单以后钱退哪去了` → 预测 order_status（gold refund，conf=0.44，margin=0.12）
- `订单怎么查物流到哪了` → 预测 logistics（gold order_status，conf=0.53，margin=0.39）
- `刚付款什么时候能发货` → 预测 logistics（gold order_status，conf=0.62，margin=0.62）
- `显示待收货但是我还没收到` → 预测 logistics（gold order_status，conf=0.65，margin=0.65）
- `取消订单后钱多久回来` → 预测 order_status（gold refund，conf=0.44，margin=0.13）
- `包装拆了还能退吗` → 预测 refund（gold return_exchange，conf=0.74，margin=0.74）
- `要退回去怎么寄` → 预测 refund（gold return_exchange，conf=0.36，margin=0.25）
- `包裹在分拨中心停了很久` → 预测 order_status（gold logistics，conf=0.30，margin=0.30）
- `同城一般几天到` → 预测 order_status（gold logistics，conf=0.30，margin=0.30）
- `客服一直敷衍我` → 预测 other（gold complaint，conf=0.63，margin=0.63）
- `这个和另一款差别大吗` → 预测 other（gold product_consult，conf=0.38，margin=0.27）
