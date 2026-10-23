## 在线模式评测（deepseek-chat，n=24）

> 与离线报告不可比：温度不为 0，同一句问两次答复不同；这里看的是绝对观感与成本。

- 平均分 **4.417**，通过率 87.5%
- 关键词命中率 87.5%
- 延迟 P50 **3824.3 ms** / P95 **6113.5 ms**
- LLM 调用 142 次，prompt 46832 / completion 3766 tokens
- 本轮总成本约 0.0544 元，单轮约 0.0023 元

| # | 问题 | 分 | 意图 | 工具 | 延迟 | 关键词 |
|---|---|---|---|---|---|---|
| d01 | SO20260512001 这个单子到哪了 | 5 | logistics | logistics.track | 3935 ms | ✗ |
| d02 | SO20261018006 我不想要了，能退 | 5 | refund | refund.apply,order.query,kb.search | 5008 ms | ✓ |
| d03 | SO20260518002 我想无理由退货 | 5 | return_exchange | return.create,order.query,kb.search | 6530 ms | ✓ |
| d04 | 退款一般多久能到账 | 5 | refund | kb.search | 3683 ms | ✓ |
| d05 | 你们的快递一直不更新，三天没动静了 | 5 | logistics | kb.search | 3552 ms | ✓ |
| d06 | 发票抬头写错了，能重开吗 | 5 | invoice | kb.search | 3824 ms | ✓ |
| d07 | 刚买完就降价了，能退差价吗 | 5 | refund | kb.search | 3831 ms | ✓ |
| d08 | 手机进水了还能保修吗 | 5 | product_consult | kb.search | 4323 ms | ✓ |
| d09 | 再不给我解决我就去12315投诉 | 4 | complaint | - | 516 ms | ✓ |
| d10 | 优惠券为什么下单用不了 | 4 | promo | kb.search | 4258 ms | ✓ |
| d11 | 怎么注销账号 | 5 | account | kb.search | 4206 ms | ✓ |
| d12 | 尺码买小了想换大一号 | 5 | return_exchange | kb.search | 4040 ms | ✓ |
| d13 | SO20260520003 我后悔了不想要了 | 1 | refund | refund.apply,order.query,kb.search | 6114 ms | ✗ |
| d14 | SO20260522004 这个单子怎么付不 | 3 | order_status | order.query | 3481 ms | ✓ |
| d15 | SO20260510005 收到的绘本少了两 | 4 | order_status | order.query | 3634 ms | ✓ |
| d16 | 退款要扣运费吗 | 4 | refund | kb.search | 4202 ms | ✓ |
| d17 | 想开专票要准备什么 | 5 | invoice | kb.search | 3592 ms | ✓ |
| d18 | 秒杀抢到了但支付超时了 | 5 | order_status | order.query | 3021 ms | ✓ |
| d19 | 我的账号被限制下单了 | 5 | account | kb.search | 4199 ms | ✓ |
| d20 | 快递显示异常件是什么情况 | 5 | logistics | kb.search | 3619 ms | ✓ |
| d21 | 退货运费险能赔多少 | 4 | return_exchange | kb.search | 3806 ms | ✓ |
| d22 | 发票能不能只开商品不含运费 | 5 | invoice | kb.search | 3802 ms | ✓ |
| d23 | 积分退款之后会退回吗 | 4 | refund | kb.search | 3749 ms | ✓ |
| d24 | 我要投诉并且要求赔偿 | 3 | complaint | - | 914 ms | ✗ |
