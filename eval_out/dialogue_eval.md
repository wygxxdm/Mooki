## 端到端对话评测（n=12，LLM-as-Judge 1-5 分）

- 平均分: **4.083**，中位数 4.0
- 通过率（≥4 分）: **66.67%**
- 关键词命中率（确定性下界）: 91.67%
- Judge 与关键词冲突条目: ['d01']
- 降级率: 0.00%，P95 延迟 13 ms

| # | 问题 | 分 | 意图 | 工具 | 关键词 |
|---|---|---|---|---|---|
| d01 | SO20260512001 这个单子到哪了 | 4 | logistics | logistics.track | ✗ |
| d02 | SO20261018006 我不想要了，能退款吗 | 3 | refund | order.query,refund.apply,kb.search | ✓ |
| d03 | SO20260518002 我想无理由退货 | 3 | return_exchange | order.query,return.create,kb.search | ✓ |
| d04 | 退款一般多久能到账 | 3 | refund | kb.search | ✓ |
| d05 | 你们的快递一直不更新，三天没动静了 | 4 | logistics | kb.search | ✓ |
| d06 | 发票抬头写错了，能重开吗 | 5 | invoice | kb.search | ✓ |
| d07 | 刚买完就降价了，能退差价吗 | 5 | promo | kb.search | ✓ |
| d08 | 手机进水了还能保修吗 | 5 | product_consult | kb.search | ✓ |
| d09 | 再不给我解决我就去12315投诉 | 4 | complaint | - | ✓ |
| d10 | 优惠券为什么下单用不了 | 3 | promo | kb.search | ✓ |
| d11 | 怎么注销账号 | 5 | account | kb.search | ✓ |
| d12 | 尺码买小了想换大一号 | 5 | return_exchange | kb.search | ✓ |
