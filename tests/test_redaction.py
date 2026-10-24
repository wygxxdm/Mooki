"""PII 脱敏。

trace 会被落到内存、推到前端、导出做客诉复盘。用户随口说的一句
"我电话 138xxxx，你们打这个" 如果原样进 trace，等于把个人信息复制到
所有下游（面板、日志、导出文件、离线评测样本）。

所以脱敏放在 trace 的唯一写入口，而不是各个出口各做一次。
"""

from __future__ import annotations

import unittest

from mooki.core.observability import TraceStore
from mooki.core.text import redact_deep, redact_pii


class TestRedactText(unittest.TestCase):
    def test_phone(self):
        self.assertEqual(redact_pii("电话 13812345678"), "电话 138****5678")

    def test_id_card(self):
        self.assertEqual(redact_pii("身份证 320102199001011234"), "身份证 320102********1234")

    def test_bank_card(self):
        self.assertEqual(redact_pii("卡号 6222021234567890123"), "卡号 6222********0123")

    def test_email(self):
        self.assertEqual(redact_pii("邮箱 zhangsan@example.com"), "邮箱 z***@example.com")

    def test_order_id_is_not_touched(self):
        """订单号是排查必须的定位信息，不是个人信息，不能一起打掉。"""
        self.assertEqual(redact_pii("订单 SO20261018006 已签收"), "订单 SO20261018006 已签收")

    def test_short_numbers_untouched(self):
        self.assertEqual(redact_pii("退款 139 元，等 3 天"), "退款 139 元，等 3 天")

    def test_deep_handles_nested(self):
        payload = {"a": ["call 13900001111", {"b": "x@y.com"}], "n": 3}
        out = redact_deep(payload)
        self.assertEqual(out["a"][0], "call 139****1111")
        self.assertEqual(out["a"][1]["b"], "x***@y.com")
        self.assertEqual(out["n"], 3)


class TestTraceRedaction(unittest.TestCase):
    def test_emit_redacts_payload(self):
        store = TraceStore()
        tid = store.start(meta={"message": "我电话 13812345678"})
        store.emit(tid, "tool_call", "act", {"args": {"phone": "13900001111"}})
        events = store.get(tid)
        self.assertEqual(events[0].payload["message"], "我电话 138****5678")
        self.assertEqual(events[-1].payload["args"]["phone"], "139****1111")

    def test_order_id_survives_in_trace(self):
        store = TraceStore()
        tid = store.start()
        store.emit(tid, "tool_call", "act", {"order_id": "SO20261018006"})
        self.assertEqual(store.get(tid)[-1].payload["order_id"], "SO20261018006")


if __name__ == "__main__":
    unittest.main()
