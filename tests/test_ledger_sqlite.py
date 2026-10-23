"""SQLite 账本：幂等键唯一性由存储层保证。

jsonl 版的互斥是 OrderLedger 实例上的 `threading.Lock`。一旦每个请求各建一个
ServiceTools（多 worker 部署、或每请求 new 一个），锁就退化成各管各的，
同一笔退款重试会落下多条事实。这里的断言就是锁住这个差别。
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from mooki.tools.ledger import OrderLedger, SqliteLedger, build_ledger


class TestSqliteLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mooki-ledger-"))

    def test_build_ledger_picks_backend_by_suffix(self):
        self.assertIsInstance(build_ledger(self.tmp / "a.db"), SqliteLedger)
        self.assertIsInstance(build_ledger(self.tmp / "a.jsonl"), OrderLedger)

    def test_transitions_and_replay(self):
        path = self.tmp / "l.db"
        ledger = SqliteLedger(path)
        self.addCleanup(ledger.close)
        rec, err, dup = ledger.apply("refund.request", "SO1", "k1", {"refund_id": "RF1"}, base_status="delivered")
        self.assertIsNone(err)
        self.assertFalse(dup)
        self.assertEqual(ledger.status("SO1"), "refund_pending")

        ledger.apply("refund.approve", "SO1", "k2", base_status="refund_pending")
        self.assertEqual(ledger.status("SO1"), "refunded")

        # 重新打开：状态必须从事件重放出来
        reopened = SqliteLedger(path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.status("SO1"), "refunded")
        self.assertIsNotNone(reopened.find("k1"))

    def test_duplicate_key_returns_first_result(self):
        ledger = SqliteLedger(self.tmp / "l.db")
        self.addCleanup(ledger.close)
        first, _, dup1 = ledger.apply("refund.request", "SO1", "k1", {"refund_id": "RF1"}, base_status="delivered")
        second, _, dup2 = ledger.apply("refund.request", "SO1", "k1", {"refund_id": "RF2"}, base_status="delivered")
        self.assertFalse(dup1)
        self.assertTrue(dup2)
        self.assertEqual(first.payload["refund_id"], second.payload["refund_id"])
        self.assertEqual(ledger.snapshot("SO1")["refund_id"], "RF1")

    def test_illegal_transition_rejected(self):
        ledger = SqliteLedger(self.tmp / "l.db")
        self.addCleanup(ledger.close)
        _, err, _ = ledger.apply("refund.approve", "SO1", "k", base_status="delivered")
        self.assertIn("illegal_transition", err or "")
        self.assertIsNone(ledger.status("SO1"))

    def test_concurrent_same_key_only_one_wins(self):
        """多线程 + 同一幂等键：只能有一条事实。"""
        ledger = SqliteLedger(self.tmp / "l.db")
        self.addCleanup(ledger.close)
        created, duplicated = [], []
        lock = threading.Lock()

        def run():
            _, _, dup = ledger.apply(
                "refund.request", "SO1", "same-key", {"refund_id": "RF1"}, base_status="delivered"
            )
            with lock:
                (duplicated if dup else created).append(1)

        threads = [threading.Thread(target=run) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(created), 1)
        self.assertEqual(len(duplicated), 11)
        self.assertEqual(len(ledger.history("SO1")), 1)

    def test_separate_instances_share_uniqueness(self):
        """两个独立实例指向同一个库文件：跨实例也必须互斥（多 worker 的本质）。"""
        path = self.tmp / "l.db"
        a, b = SqliteLedger(path), SqliteLedger(path)
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        _, _, dup_a = a.apply("refund.request", "SO1", "k1", {"refund_id": "RF1"}, base_status="delivered")
        _, _, dup_b = b.apply("refund.request", "SO1", "k1", {"refund_id": "RF2"}, base_status="delivered")
        self.assertFalse(dup_a)
        self.assertTrue(dup_b)
        third = SqliteLedger(path)
        self.addCleanup(third.close)
        self.assertEqual(len(third.history("SO1")), 1)


if __name__ == "__main__":
    unittest.main()
