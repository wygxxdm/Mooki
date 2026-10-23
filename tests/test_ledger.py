"""账本 / 幂等 / 补偿测试。

这三件事都是"看起来简单、错了要赔钱"的地方：
- 幂等错了：用户重试一次，退两次钱
- 状态机错了：已退款的单子被再退一次
- 补偿错了：本地显示退款中、网关没收到，账对不上
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mooki.core.cache import TTLCache
from mooki.core.config import Settings
from mooki.tools.builtin.service_tools import ServiceTools, build_registry
from mooki.tools.ledger import OrderLedger, derive_idem
from mooki.tools.registry import ToolContext

DATA = Path(__file__).resolve().parent.parent / "data"


class TestOrderLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "ledger.jsonl"
        self.ledger = OrderLedger(self.path)

    def test_request_then_approve(self):
        rec, err, dup = self.ledger.apply(
            "refund.request", "SO1", "k1", {"refund_id": "RF1", "amount": 100}, base_status="delivered"
        )
        self.assertIsNone(err)
        self.assertFalse(dup)
        self.assertEqual(self.ledger.status("SO1"), "refund_pending")

        rec2, err2, _ = self.ledger.apply("refund.approve", "SO1", "k2", base_status="refund_pending")
        self.assertIsNone(err2)
        self.assertEqual(self.ledger.status("SO1"), "refunded")
        self.assertTrue(self.ledger.snapshot("SO1")["refunded"])

    def test_idempotency_returns_first_result(self):
        first, _, dup1 = self.ledger.apply(
            "refund.request", "SO1", "same-key", {"refund_id": "RF1"}, base_status="delivered"
        )
        second, err, dup2 = self.ledger.apply(
            "refund.request", "SO1", "same-key", {"refund_id": "RF2"}, base_status="delivered"
        )
        self.assertFalse(dup1)
        self.assertTrue(dup2)
        self.assertIsNone(err)
        # 第二次不应产生新记录，退款单号仍是第一次的
        self.assertEqual(first.seq, second.seq)
        self.assertEqual(second.payload["refund_id"], "RF1")
        self.assertEqual(len(self.ledger.history("SO1")), 1)

    def test_illegal_transition_rejected(self):
        _, err, _ = self.ledger.apply("refund.approve", "SO1", "k", base_status="delivered")
        self.assertEqual(err, "illegal_transition:delivered->refund.approve")
        self.assertIsNone(self.ledger.status("SO1"))

    def test_unpaid_order_cannot_request(self):
        _, err, _ = self.ledger.apply("refund.request", "SO1", "k", base_status="unpaid")
        self.assertEqual(err, "illegal_transition:unpaid->refund.request")

    def test_cancel_rolls_back_to_previous_status(self):
        self.ledger.apply("refund.request", "SO1", "k1", {"refund_id": "RF1"}, base_status="shipped")
        self.assertEqual(self.ledger.status("SO1"), "refund_pending")
        self.ledger.apply("refund.cancel", "SO1", "k2", {}, base_status="refund_pending")
        self.assertEqual(self.ledger.status("SO1"), "shipped")

    def test_replay_from_disk_is_consistent(self):
        self.ledger.apply("refund.request", "SO1", "k1", {"refund_id": "RF1"}, base_status="delivered")
        self.ledger.apply("refund.approve", "SO1", "k2", base_status="refund_pending")
        reloaded = OrderLedger(self.path)
        self.assertEqual(reloaded.status("SO1"), "refunded")
        self.assertIsNotNone(reloaded.find("k1"))


class TestRefundToolWithLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.service = ServiceTools(
            DATA, ticket_path=self.tmp / "tickets.jsonl", ledger_path=self.tmp / "ledger.jsonl"
        )
        self.registry = build_registry(Settings(data_dir=DATA), TTLCache(64, 60), self.service)
        self.ctx = ToolContext(user_id="u_1001", session_id="s", trace_id="t")

    def _apply(self, order_id="SO20261018006", **extra):
        return self.registry.call(
            "refund.apply", {"order_id": order_id, **extra}, self.ctx, whitelist=["refund.apply"]
        )

    def test_refund_creates_ledger_record(self):
        result = self._apply(idempotency_key="fixed-1")
        self.assertTrue(result.data["eligible"])
        self.assertTrue(result.data["refund_id"].startswith("RF"))
        self.assertEqual(result.data["state"], "refund_pending")
        self.assertEqual(self.service.ledger.status("SO20261018006"), "refund_pending")

    def test_duplicate_submit_is_idempotent(self):
        first = self._apply(idempotency_key="fixed-2")
        second = self._apply(idempotency_key="fixed-2")
        self.assertTrue(second.data["duplicate"])
        self.assertEqual(first.data["refund_id"], second.data["refund_id"])

    def test_gateway_failure_triggers_compensation(self):
        self.service._gateway_fail_next = True
        result = self._apply(idempotency_key="fixed-3")
        self.assertFalse(result.data["eligible"])
        self.assertTrue(result.data["compensated"])
        self.assertEqual(result.data["state"], "rolled_back")
        # 补偿后订单必须回到原状态，不能停在退款处理中
        self.assertEqual(self.service.ledger.status("SO20261018006", "delivered"), "delivered")

    def test_order_query_reflects_ledger_state(self):
        self._apply(idempotency_key="fixed-4")
        query = self.registry.call(
            "order.query", {"order_id": "SO20261018006"}, self.ctx, whitelist=["order.query"]
        )
        self.assertEqual(query.data["status"], "refund_pending")
        self.assertEqual(query.data["status_zh"], "退款处理中")

    def test_cancel_restores_status(self):
        self._apply(idempotency_key="fixed-5")
        cancel = self.registry.call(
            "refund.cancel", {"order_id": "SO20261018006", "reason": "user_cancel"},
            self.ctx, whitelist=["refund.cancel"],
        )
        self.assertTrue(cancel.data["cancelled"])
        self.assertEqual(self.service.ledger.status("SO20261018006"), "delivered")


class TestIdemDerivation(unittest.TestCase):
    def test_same_inputs_same_key_within_bucket(self):
        a = derive_idem("u", "SO1", "refund.request")
        b = derive_idem("u", "SO1", "refund.request")
        self.assertEqual(a, b)

    def test_different_action_different_key(self):
        self.assertNotEqual(derive_idem("u", "SO1", "a"), derive_idem("u", "SO1", "b"))


if __name__ == "__main__":
    unittest.main()
