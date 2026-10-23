## 端到端对话评测（n=24，LLM-as-Judge 1-5 分）

- 平均分: **3.25**，中位数 3.0
- 通过率（≥4 分）: **45.83%**
- 关键词命中率（确定性下界）: 66.67%
- Judge 与关键词冲突条目: ['d01']
- 降级率: 0.00%，P95 延迟 21 ms

| # | 问题 | 分 | 意图 | 工具 | 关键词 |
|---|---|---|---|---|---|
| d01 | SO20260512001 这个单子到哪了 | 4 | logistics | logistics.track | ✗ |
| d02 | SO20261018006 我不想要了，能退款吗 | 2 | refund | order.query,refund.apply,kb.search | ✓ |
| d03 | SO20260518002 我想无理由退货 | 3 | return_exchange | order.query,return.create,kb.search | ✓ |
| d04 | 退款一般多久能到账 | 3 | refund | kb.search | ✓ |
| d05 | 你们的快递一直不更新，三天没动静了 | 3 | logistics | kb.search | ✓ |
| d06 | 发票抬头写错了，能重开吗 | 5 | invoice | kb.search | ✓ |
| d07 | 刚买完就降价了，能退差价吗 | 5 | promo | kb.search | ✓ |
| d08 | 手机进水了还能保修吗 | 5 | product_consult | kb.search | ✓ |
| d09 | 再不给我解决我就去12315投诉 | 4 | complaint | - | ✓ |
| d10 | 优惠券为什么下单用不了 | 3 | promo | kb.search | ✓ |
| d11 | 怎么注销账号 | 5 | account | kb.search | ✓ |
| d12 | 尺码买小了想换大一号 | 5 | return_exchange | kb.search | ✓ |
| d13 | SO20260520003 我后悔了不想要了 | 2 | refund | order.query,refund.apply,kb.search | ✗ |
| d14 | SO20260522004 这个单子怎么付不了款 | 1 | order_status | order.query | ✗ |
| d15 | SO20260510005 收到的绘本少了两本 | 2 | return_exchange | order.query,return.create,kb.search | ✗ |
| d16 | 退款要扣运费吗 | 5 | refund | kb.search | ✓ |
| d17 | 想开专票要准备什么 | 1 | promo | - | ✗ |
| d18 | 秒杀抢到了但支付超时了 | 5 | order_status | kb.search | ✓ |
| d19 | 我的账号被限制下单了 | 3 | account | kb.search | ✓ |
| d20 | 快递显示异常件是什么情况 | 2 | logistics | kb.search | ✗ |
| d21 | 退货运费险能赔多少 | 4 | return_exchange | kb.search | ✓ |
| d22 | 发票能不能只开商品不含运费 | 1 | logistics | - | ✗ |
| d23 | 积分退款之后会退回吗 | 4 | refund | kb.search | ✓ |
| d24 | 我要投诉并且要求赔偿 | 1 | complaint | - | ✗ |
