"""Agent 与工具层测试：白名单、数据隔离、路由分支。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mooki.core.cache import TTLCache
from mooki.core.config import Settings
from mooki.tools.builtin.service_tools import ServiceTools, build_registry
from mooki.tools.registry import ToolContext

DATA = Path(__file__).resolve().parent.parent / "data"


class TestToolRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = Path(tempfile.mkdtemp())
        cls.settings = Settings(data_dir=DATA)
        cls.service = ServiceTools(
            DATA, ticket_path=tmp / "tickets.jsonl", ledger_path=tmp / "ledger.jsonl"
        )
        cls.registry = build_registry(cls.settings, TTLCache(64, 60), cls.service)
        cls.ctx = ToolContext(user_id="u_1001", session_id="s", trace_id="t")

    def test_whitelist_blocks_cross_call(self):
        result = self.registry.call(
            "refund.apply", {"order_id": "SO20260512001"}, self.ctx, whitelist=["order.query"]
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "tool_not_in_whitelist")

    def test_other_user_order_hidden(self):
        # SO20260522004 属于 u_1002，查不到时不能泄露"存在但不属于你"
        result = self.registry.call(
            "order.query", {"order_id": "SO20260522004"}, self.ctx, whitelist=["order.query"]
        )
        self.assertTrue(result.ok)
        self.assertFalse(result.data["found"])

    def test_missing_args_rejected(self):
        result = self.registry.call("logistics.track", {}, self.ctx, whitelist=["logistics.track"])
        self.assertFalse(result.ok)
        self.assertTrue(result.error.startswith("missing_args"))

    def test_read_cache_hit(self):
        self.registry.call("order.query", {"order_id": "SO20260512001"}, self.ctx, whitelist=["order.query"])
        again = self.registry.call(
            "order.query", {"order_id": "SO20260512001"}, self.ctx, whitelist=["order.query"]
        )
        self.assertTrue(again.from_cache)

    def test_refund_window_blocks_expired_order(self):
        result = self.registry.call(
            "refund.apply",
            {"order_id": "SO20260518002", "reason": "no_reason"},
            self.ctx,
            whitelist=["refund.apply"],
        )
        self.assertFalse(result.data["eligible"])
        self.assertTrue(any(b.startswith("exceed_no_reason_window") for b in result.data["blockers"]))

    def test_manifest_is_mcp_shaped(self):
        manifest = self.registry.manifest()
        self.assertTrue(manifest)
        for item in manifest:
            self.assertEqual(set(item), {"name", "description", "inputSchema"})


if __name__ == "__main__":
    unittest.main()
