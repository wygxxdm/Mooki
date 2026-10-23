"""规划层副作用幂等测试。

多轮会话里最容易被忽略的一类错误：用户追问"钱多久到账"，意图仍是 refund，
规划节点会照着重放一遍 refund.apply。账本幂等键能兜住"不重复扣钱"，
但工具白跑一次、答复也退化成重复播报上一条结论。
这一层拦截必须在规划阶段做，而不是等工具层去幂等。
"""

from __future__ import annotations

import unittest

from mooki.core.storage import MemoryKV
from mooki.graph.runtime import AgentRuntime


class _StubLedger:
    def __init__(self, state: str | None):
        self._state = state

    def status(self, order_id: str) -> str | None:  # noqa: ARG002 - 接口签名对齐
        return self._state


class _StubServices:
    def __init__(self, state: str | None):
        self.ledger = _StubLedger(state)


def runtime_with_ledger(state: str | None, kv: MemoryKV | None = None) -> AgentRuntime:
    """只注入账本与会话 KV，绕开 __init__ 里的索引与模型加载。"""
    rt = AgentRuntime.__new__(AgentRuntime)
    rt.services = _StubServices(state)  # type: ignore[assignment]
    rt.kv = kv or MemoryKV()
    return rt


class TestSideEffectDone(unittest.TestCase):
    def test_refund_pending_blocks_replan(self):
        rt = runtime_with_ledger("refund_pending")
        reason = rt._side_effect_done("refund.apply", {"order_id": "SO20261018006"}, "那钱多久能到账")
        self.assertEqual(reason, "ledger_state=refund_pending")

    def test_refunded_also_blocks(self):
        rt = runtime_with_ledger("refunded")
        self.assertTrue(rt._side_effect_done("refund.apply", {"order_id": "SO1"}, "钱到哪了"))

    def test_fresh_order_still_plans(self):
        rt = runtime_with_ledger(None)
        self.assertEqual(rt._side_effect_done("refund.apply", {"order_id": "SO1"}, "我不想要了"), "")

    def test_explicit_reapply_is_not_blocked(self):
        """用户明确要求再退一次时不能拦，否则连幂等键的答复都拿不到。"""
        rt = runtime_with_ledger("refund_pending")
        self.assertEqual(rt._side_effect_done("refund.apply", {"order_id": "SO1"}, "能不能再退一次"), "")

    def test_cancel_requires_pending(self):
        rt = runtime_with_ledger("refunded")
        self.assertTrue(rt._side_effect_done("refund.cancel", {"order_id": "SO1"}, "撤销退款"))
        rt2 = runtime_with_ledger("refund_pending")
        self.assertEqual(rt2._side_effect_done("refund.cancel", {"order_id": "SO1"}, "撤销退款"), "")

    def test_no_order_id_never_blocks(self):
        rt = runtime_with_ledger("refund_pending")
        self.assertEqual(rt._side_effect_done("refund.apply", {}, "我要退款"), "")

    def test_session_registry_blocks_return_create(self):
        """return.create 没有订单状态机，只能靠会话登记表兜住重复调用。"""
        kv = MemoryKV()
        rt = runtime_with_ledger(None, kv)
        args = {"order_id": "SO20260518002"}
        self.assertEqual(rt._side_effect_done("return.create", args, "我要换货"), "")
        kv.set(rt._applied_key(args, "return.create"), "1", ttl=600)
        self.assertEqual(rt._side_effect_done("return.create", args, "我要换货"), "session_applied")

    def test_registry_is_per_tool_and_order(self):
        kv = MemoryKV()
        rt = runtime_with_ledger(None, kv)
        kv.set(rt._applied_key({"order_id": "SO1"}, "return.create"), "1", ttl=600)
        self.assertEqual(rt._side_effect_done("return.create", {"order_id": "SO2"}, "换货"), "")
        self.assertEqual(rt._side_effect_done("invoice.issue", {"order_id": "SO1"}, "开发票"), "")


if __name__ == "__main__":
    unittest.main()
