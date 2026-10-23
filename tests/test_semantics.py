"""LSA 向量化、记忆折叠、意图通道消融的回归测试。

这三个模块出过真 bug（OOV 静默退化、窗口静默丢历史、通道同源导致重复计票），
所以必须有测试锁住行为，不能只靠"跑起来看着对"。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mooki.core.embedder import HashingEmbedder, LSAEmbedder, clone_unfitted
from mooki.core.storage import MemoryKV
from mooki.core.config import Settings
from mooki.memory.short_term import ShortTermMemory


class _EchoLLM:
    """摘要器桩：把订单号留住，方便验证"压缩不该吃掉实体"。"""

    def summarize(self, turns: list[str]) -> str:
        return "用户咨询订单 SO20260512001 的物流，诉求为查询进度"


class TestLSA(unittest.TestCase):
    def test_fit_requires_corpus(self):
        emb = LSAEmbedder(dim=16)
        with self.assertRaises(RuntimeError):
            emb.embed(["未训练"])

    def test_semantic_ordering(self):
        corpus = [
            "退款到账时效说明，微信一个工作日到账",
            "退款失败常见原因，账户注销或冻结",
            "发票开具流程，电子发票二十四小时开出",
            "发票抬头错误需要红冲后重新开具",
            "物流配送时效，同城一到两天",
        ]
        emb = LSAEmbedder(dim=4, min_df=1)
        emb.fit(corpus)
        vecs = emb.embed(["退款多久到账", "发票怎么开"])
        self.assertEqual(vecs.shape[0], 2)
        # 同域相似度应高于跨域
        self.assertGreater(float(vecs[0] @ vecs[1]), -1.0)
        sim_refund = float(vecs[0] @ emb.embed([corpus[0]])[0])
        sim_invoice = float(vecs[0] @ emb.embed([corpus[2]])[0])
        self.assertGreater(sim_refund, sim_invoice)

    def test_persist_roundtrip_is_stable(self):
        corpus = ["退款到账时效", "退货运费承担规则", "发票开具流程", "物流配送时效", "账号安全与找回"]
        emb = LSAEmbedder(dim=3, min_df=1)
        emb.fit(corpus)
        original = emb.embed(["退款多久到账"])
        with tempfile.TemporaryDirectory() as tmp:
            emb.save(Path(tmp))
            restored = LSAEmbedder(dim=3)
            self.assertTrue(restored.restore(Path(tmp)))
            again = restored.embed(["退款多久到账"])
        # 符号确定化之后，跨进程恢复必须逐位一致，否则查询向量和库里向量不在一个空间
        np.testing.assert_allclose(original, again, rtol=1e-6, atol=1e-6)

    def test_clone_is_unfitted_and_shares_type(self):
        emb = LSAEmbedder(dim=32)
        emb.fit(["退款到账", "发票开具", "物流配送"])
        clone = clone_unfitted(emb)
        self.assertIsInstance(clone, LSAEmbedder)
        self.assertFalse(clone.fitted)
        self.assertEqual(clone.dim, 32)

    def test_clone_hash_embedder(self):
        clone = clone_unfitted(HashingEmbedder(64))
        self.assertIsInstance(clone, HashingEmbedder)
        self.assertEqual(clone.dim, 64)


class TestMemoryFolding(unittest.TestCase):
    def setUp(self):
        # window=2 -> 上限 4 条，超过就折叠
        self.mem = ShortTermMemory(MemoryKV(), window=2, compress_at=10_000)
        self.mem.append("s", "user", "我的订单 SO20260512001 还没收到，实付 269 元")
        for i in range(6):
            self.mem.append("s", "assistant", f"第 {i} 条较长的历史回复，用于把窗口撑满")

    def test_no_silent_drop_before_fold(self):
        # 未调用 maybe_compress 时，append 不应自行裁剪 —— 否则历史会凭空消失
        self.assertEqual(len(self.mem.load("s").turns), 7)

    def test_fold_keeps_window_and_summary(self):
        self.assertTrue(self.mem.maybe_compress("s", _EchoLLM()))
        memory = self.mem.load("s")
        self.assertLessEqual(len(memory.turns), 4)
        self.assertTrue(memory.summary)
        # 关键：订单号在槽位里，压缩后依然可用
        self.assertEqual(memory.slots.get("order_id"), "SO20260512001")

    def test_fold_is_idempotent_within_window(self):
        self.mem.maybe_compress("s", _EchoLLM())
        self.assertFalse(self.mem.maybe_compress("s", _EchoLLM()))


class _StubClassifier:
    """只用于验证 route 参数能关掉其它通道，不依赖真实 LLM。"""

    def __init__(self):
        from mooki.core.cache import TTLCache
        from mooki.core.embedder import HashingEmbedder
        from mooki.core.llm import HeuristicChatClient, LLMService
        from mooki.intent.classifier import IntentClassifier

        settings = Settings()
        self.inner = IntentClassifier(
            LLMService(HeuristicChatClient(), settings), HashingEmbedder(256), TTLCache(16, 60)
        )

    def classify(self, text, routes=None):
        return self.inner.classify(text, routes=routes)


class TestRouteAblation(unittest.TestCase):
    def test_route_filter_changes_vote_count(self):
        from mooki.core.schema import VoteSource

        clf = _StubClassifier()
        full = clf.classify("我要退款，东西不想要了")
        only_dense = clf.classify("我要退款，东西不想要了", routes={VoteSource.DENSE})
        self.assertGreaterEqual(len(full.votes), len(only_dense.votes))
        self.assertTrue(all(v.source is VoteSource.DENSE for v in only_dense.votes))

    def test_complementarity_counts(self):
        from mooki.eval.intent_eval import complementarity

        hits = {
            "a": {"q1": True, "q2": False, "q3": True},
            "b": {"q1": True, "q2": True, "q3": False},
        }
        out = complementarity({"a": hits["a"], "b": hits["b"]})
        self.assertEqual(len(out), 0)  # 键名不匹配（要求 *_only），返回空
        out = complementarity({"dense_only": hits["a"], "rule_only": hits["b"]})
        matrix = out["dense_only vs rule_only"]
        self.assertEqual(matrix["两者都对"], 1)
        self.assertEqual(matrix["仅dense_only对"], 1)
        self.assertEqual(matrix["仅rule_only对"], 1)
        self.assertEqual(matrix["并集正确数"], 3)


if __name__ == "__main__":
    unittest.main()
