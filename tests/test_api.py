"""接入层契约测试。

重点锁 SSE 消息的信封结构：这个 bug 很隐蔽 —— 后端看起来"发了事件"，
curl 也能看到 data 行，但前端一条都收不到，因为外层 type 被内层覆盖了。
"""

from __future__ import annotations

import json
import unittest

from mooki.api.main import _sse, event_envelope
from mooki.core.schema import TraceEvent


class TestSSEEnvelope(unittest.TestCase):
    def test_envelope_keeps_outer_type(self):
        event = TraceEvent(trace_id="t1", type="run_end", payload={"reply": "ok"})
        env = event_envelope(event)
        # 外层必须是 trace_event，否则前端的分流逻辑失效
        self.assertEqual(env["type"], "trace_event")
        # 内层保留真实事件类型
        self.assertEqual(env["event"]["type"], "run_end")
        self.assertEqual(env["event"]["payload"]["reply"], "ok")

    def test_envelope_is_json_serialisable_with_chinese(self):
        event = TraceEvent(trace_id="t1", type="intent", payload={"label": "退款"})
        raw = _sse(event_envelope(event))
        self.assertTrue(raw.startswith("data: "))
        self.assertTrue(raw.endswith("\n\n"))
        parsed = json.loads(raw[len("data: ") :].strip())
        self.assertEqual(parsed["event"]["payload"]["label"], "退款")
        self.assertIn("退款", raw)  # ensure_ascii=False，不能变成 \u 转义

    def test_naive_spread_would_break(self):
        """回归：如果谁再写成 {**model_dump()}，这个断言会挡住。"""
        event = TraceEvent(trace_id="t1", type="tool_call")
        naive = {"type": "trace_event", **event.model_dump()}
        self.assertEqual(naive["type"], "tool_call")  # 就是错的
        self.assertNotEqual(event_envelope(event)["type"], naive["type"])


if __name__ == "__main__":
    unittest.main()
