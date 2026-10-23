"""核心模块单测。全部离线、确定性，可直接进 CI。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import datetime as dt
import unittest

from mooki.core.cache import TTLCache
from mooki.core.embedder import HashingEmbedder
from mooki.core.resilience import CircuitBreaker, call_guarded
from mooki.core.schema import Intent, RetrievedChunk
from mooki.memory.assembler import ContextAssembler
from mooki.memory.short_term import ShortTermMemory
from mooki.core.storage import MemoryKV
from mooki.rag.bm25 import BM25Index, tokenize
from mooki.rag.chunker import chunk_document
from mooki.rag.hybrid import dedupe, rrf_fuse
from mooki.rag.validity import filter_valid


class TestChunker(unittest.TestCase):
    def test_front_matter_and_sections(self):
        raw = (
            "---\ntitle: 退款规则\ndomain: refund\nvalid_until: 2027-01-01\n---\n"
            "## 到账时效\n微信 1 个工作日。\n\n## 失败原因\n账户注销。\n"
        )
        chunks = chunk_document("refund", raw, max_chars=400)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].title, "退款规则 · 到账时效")
        self.assertEqual(chunks[0].domain, "refund")
        self.assertEqual(chunks[0].valid_until, "2027-01-01")
        self.assertIn("微信", chunks[0].text)


class TestBM25(unittest.TestCase):
    def test_tokenize_bigram(self):
        toks = tokenize("退款到账")
        self.assertIn("退款", toks)
        self.assertIn("到账", toks)

    def test_search_hits_right_doc(self):
        idx = BM25Index()
        idx.fit([("a", "退款到账时效说明"), ("b", "退货运费承担规则"), ("c", "发票开具流程")])
        hits = idx.search("退款多久到账", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "a")


class TestRRF(unittest.TestCase):
    def test_fusion_prefers_agreement(self):
        fused = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k=60)
        top = [doc for doc, _ in fused[:2]]
        self.assertIn("a", top)
        self.assertIn("c", top)

    def test_dedupe_removes_near_duplicate(self):
        def chunk(cid, text):
            return RetrievedChunk(doc_id="d", chunk_id=cid, title="t", domain="refund", text=text)

        kept = dedupe([chunk("1", "退款一般在 1 到 3 个工作日到账。"), chunk("2", "退款一般在 1 到 3 个工作日到账")])
        self.assertEqual(len(kept), 1)


class TestValidity(unittest.TestCase):
    def test_expired_dropped(self):
        def chunk(cid, valid_until):
            return RetrievedChunk(
                doc_id="d", chunk_id=cid, title="t", domain="refund", text="政策", valid_until=valid_until
            )

        kept, report = filter_valid(
            [chunk("1", "2026-06-30"), chunk("2", "2027-12-31")], today=dt.date(2026, 10, 23)
        )
        self.assertEqual([c.chunk_id for c in kept], ["2"])
        self.assertEqual(report.dropped_count, 1)


class TestCacheAndBreaker(unittest.TestCase):
    def test_ttl_cache_stats(self):
        cache = TTLCache(maxsize=2, ttl=60)
        cache.set("k", 1)
        self.assertEqual(cache.get("k"), 1)
        cache.get("missing")
        self.assertEqual(cache.stats.hits, 1)
        self.assertEqual(cache.stats.misses, 1)

    def test_breaker_opens_then_falls_back(self):
        breaker = CircuitBreaker("t", fail_threshold=2, reset_s=999)

        def boom():
            raise RuntimeError("down")

        for _ in range(2):
            call_guarded(boom, timeout_s=1, breaker=breaker, fallback=lambda r: "fallback")
        self.assertEqual(breaker.state, "open")
        result = call_guarded(boom, timeout_s=1, breaker=breaker, fallback=lambda r: "cached")
        self.assertTrue(result.degraded)
        self.assertEqual(result.value, "cached")


class _StubLLM:
    def summarize(self, turns):
        return "用户询问退款，提到订单 SO20260512001，诉求为退款"


class TestShortTermMemory(unittest.TestCase):
    def test_slots_survive_compression(self):
        mem = ShortTermMemory(MemoryKV(), window=4, compress_at=5)
        mem.append("s1", "user", "我要退款，订单 SO20260512001，金额 269 元")
        for i in range(8):
            mem.append("s1", "assistant", f"这是第 {i} 轮很长的回复内容用于触发压缩" * 3)
        before = mem.load("s1")
        self.assertEqual(before.slots.get("order_id"), "SO20260512001")
        compressed = mem.maybe_compress("s1", _StubLLM())
        self.assertTrue(compressed)
        after = mem.load("s1")
        self.assertEqual(after.slots.get("order_id"), "SO20260512001")  # 槽位不参与压缩
        self.assertTrue(after.summary)


class TestAssembler(unittest.TestCase):
    def test_budget_respected_and_report(self):
        asm = ContextAssembler(budget=300)
        passages = [f"第{i}段：退款政策说明" + "细则" * 40 for i in range(6)]
        messages, report, blocks = asm.assemble(
            system="你是客服",
            profile={"vip_level": "gold"},
            summary="历史摘要",
            history=["user: 你好", "assistant: 在的"],
            retrieval=passages,
            query="退款多久到账",
        )
        self.assertLessEqual(report.used, 320)
        self.assertTrue(report.sections["retrieval"].truncated)
        self.assertEqual(messages[-1]["content"], "退款多久到账")


class TestEmbedder(unittest.TestCase):
    def test_normalized_and_similar(self):
        emb = HashingEmbedder(dim=256)
        vecs = emb.embed(["退款多久到账", "退款到账要几天", "发票怎么开"])
        norms = (vecs ** 2).sum(axis=1)
        self.assertTrue(all(abs(n - 1.0) < 1e-4 for n in norms))
        self.assertGreater(float(vecs[0] @ vecs[1]), float(vecs[0] @ vecs[2]))


if __name__ == "__main__":
    unittest.main()
