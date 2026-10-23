"""检索流水线：改写 → 并行召回 → RRF → 有效期过滤 → 去重 → 重排 → 截断。

设计要点：
1. 每一档能力都是可配置的（RetrievalConfig），这样同一份代码既能跑线上全链路，
   也能跑消融实验 —— 消融不是另写一份脚本，而是换个 config。
2. 降级是分层的，不是全有全无：
      重排失败 → 退回 RRF 顺序
      向量路熔断 → 只用 BM25
      BM25 也挂 → 只用向量
      两路都挂 → 返回空 + degraded 标记，由 Agent 走兜底话术
3. 缓存 key 含 config，避免消融实验之间互相污染。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from mooki.core.cache import TTLCache
from mooki.core.config import Settings
from mooki.core.embedder import Embedder
from mooki.core.resilience import CircuitBreaker, call_guarded
from mooki.core.schema import RetrievedChunk
from mooki.core.text import normalize_text
from mooki.rag.hybrid import dedupe, rrf_fuse
from mooki.rag.ingest import IndexBundle
from mooki.rag.rewrite import QueryRewriter
from mooki.rag.validity import filter_valid


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    use_rewrite: bool = False
    use_bm25: bool = True
    use_dense: bool = True
    use_rrf: bool = True
    use_rerank: bool = False
    use_domain_prior: bool = False
    bm25_weight: float = 1.0
    dense_weight: float = 1.0


CONFIGS: dict[str, RetrievalConfig] = {
    # 基线：只走向量，不改写、不融合、不重排
    "direct": RetrievalConfig("direct", use_bm25=False, use_rrf=False),
    "bm25": RetrievalConfig("bm25", use_bm25=True, use_dense=False, use_rrf=False),
    "hybrid": RetrievalConfig("hybrid"),
    "hybrid_rewrite": RetrievalConfig("hybrid_rewrite", use_rewrite=True),
    "full": RetrievalConfig("full", use_rewrite=True, use_rerank=True),
    # + 意图域先验：用分类结果把候选往对应知识域拉，等价于一个软 filter
    "full_prior": RetrievalConfig("full_prior", use_rewrite=True, use_rerank=True, use_domain_prior=True),
}

DEFAULT_CONFIG = "full"
DOMAIN_BOOST = 1.3


@dataclass
class RetrievalOutput:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""
    effective_query: str = ""
    config: str = DEFAULT_CONFIG
    degraded: list[str] = field(default_factory=list)
    stages: dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False
    dropped_invalid: int = 0
    candidates: dict[str, int] = field(default_factory=dict)

    def as_context(self) -> list[str]:
        return [f"{c.title}：{c.text}" for c in self.chunks]

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        llm,
        index: IndexBundle,
        cache: TTLCache,
        rewriter: QueryRewriter,
    ):
        self.settings = settings
        self.embedder = embedder
        self.llm = llm
        self.index = index
        self.cache = cache
        self.rewriter = rewriter
        self.bm25_breaker = CircuitBreaker("retrieval.bm25", settings.breaker_fail_threshold, settings.breaker_reset_s)
        self.dense_breaker = CircuitBreaker("retrieval.dense", settings.breaker_fail_threshold, settings.breaker_reset_s)
        self.rerank_breaker = CircuitBreaker("retrieval.rerank", settings.breaker_fail_threshold, settings.breaker_reset_s)

    # ---------- 各阶段 ----------

    def _bm25_candidates(self, query: str, filters: dict | None) -> list[tuple[str, float]]:
        hits = self.index.bm25.search(query, self.settings.bm25_top_k)
        if filters:
            hits = [
                (cid, score)
                for cid, score in hits
                if all(getattr(self.index.chunk_map[cid], k, None) == v for k, v in filters.items())
            ]
        return hits

    def _dense_candidates(self, query: str, filters: dict | None) -> list[tuple[str, float]]:
        vec = self.embedder.embed([query])[0]
        hits = self.index.store.search(vec, self.settings.dense_top_k, where=filters or None)
        return [(h.id, h.score) for h in hits]

    def _rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> tuple[list[RetrievedChunk], bool]:
        pool = chunks[: max(top_k * 3, 10)]
        result = call_guarded(
            lambda: self.llm.rerank(query, [f"{c.title}\n{c.text}" for c in pool]),
            timeout_s=self.settings.tool_timeout_s,
            breaker=self.rerank_breaker,
            fallback=lambda _reason: None,
        )
        order = result.value if result.ok else None
        if not order:
            return chunks, True
        reranked = []
        for rank, idx in enumerate(order):
            if 0 <= idx < len(pool):
                chunk = pool[idx]
                chunk.rerank_score = float(len(pool) - rank)
                reranked.append(chunk)
        tail = [c for c in chunks if c.chunk_id not in {r.chunk_id for r in reranked}]
        return reranked + tail, False

    # ---------- 主流程 ----------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        config_name: str = DEFAULT_CONFIG,
        filters: dict | None = None,
        history: list[str] | None = None,
        domain_prior: str | None = None,
    ) -> RetrievalOutput:
        config = CONFIGS.get(config_name, CONFIGS[DEFAULT_CONFIG])
        top_k = top_k or self.settings.rerank_top_k
        cache_key = (
            f"{config.name}|{top_k}|{sorted((filters or {}).items())}|{domain_prior}|{normalize_text(query)}"
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return replace(cached, cache_hit=True)

        stages: dict[str, float] = {}
        degraded: list[str] = []
        out = RetrievalOutput(query=query, effective_query=query, config=config.name)

        # 1. 改写
        t0 = time.time()
        effective = query
        if config.use_rewrite:
            effective = self.rewriter.rewrite(query, history)
        out.effective_query = effective
        stages["rewrite"] = round((time.time() - t0) * 1000, 1)

        # 2. 并行召回（两路互不阻塞；用两次 guarded 调用，任一失败不拖垮整体）
        t0 = time.time()
        bm25_hits: list[tuple[str, float]] = []
        dense_hits: list[tuple[str, float]] = []
        if config.use_bm25:
            res = call_guarded(
                lambda: self._bm25_candidates(effective, filters),
                timeout_s=self.settings.tool_timeout_s,
                breaker=self.bm25_breaker,
                fallback=lambda _r: [],
            )
            bm25_hits = res.value or []
            if res.degraded:
                degraded.append(f"bm25:{res.error}")
        if config.use_dense:
            res = call_guarded(
                lambda: self._dense_candidates(effective, filters),
                timeout_s=self.settings.tool_timeout_s,
                breaker=self.dense_breaker,
                fallback=lambda _r: [],
            )
            dense_hits = res.value or []
            if res.degraded:
                degraded.append(f"dense:{res.error}")
        stages["recall"] = round((time.time() - t0) * 1000, 1)
        out.candidates = {"bm25": len(bm25_hits), "dense": len(dense_hits)}

        # 3. 融合
        t0 = time.time()
        bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25_hits)}
        dense_rank = {cid: i for i, (cid, _) in enumerate(dense_hits)}
        if config.use_rrf and bm25_hits and dense_hits:
            fused = rrf_fuse(
                [[cid for cid, _ in bm25_hits], [cid for cid, _ in dense_hits]],
                k=self.settings.rrf_k,
                weights=[config.bm25_weight, config.dense_weight],
            )
        elif bm25_hits and dense_hits:
            # 不融合：按分数各自归一后取并集（direct 只有一路，这里不会进）
            fused = [(cid, s) for cid, s in bm25_hits] + [
                (cid, s) for cid, s in dense_hits if cid not in bm25_rank
            ]
        else:
            fused = [(cid, s) for cid, s in (bm25_hits or dense_hits)]

        # 域先验不在这里做：重排会整体重排序，把先验覆盖掉。
        # 统一放到重排之后做最终重打分（见下方）。
        stages["fuse"] = round((time.time() - t0) * 1000, 1)

        # 4. 落成领域对象
        chunks: list[RetrievedChunk] = []
        for cid, score in fused:
            src = self.index.chunk_map.get(cid)
            if src is None:
                continue
            chunks.append(
                RetrievedChunk(
                    doc_id=src.doc_id,
                    chunk_id=src.chunk_id,
                    title=src.title,
                    domain=src.domain,
                    text=src.text,
                    score=score,
                    bm25_rank=bm25_rank.get(cid),
                    dense_rank=dense_rank.get(cid),
                    rrf_score=score,
                    valid_until=src.valid_until,
                )
            )

        # 5. 有效期过滤 + 去重
        t0 = time.time()
        chunks, report = filter_valid(chunks)
        out.dropped_invalid = report.dropped_count
        if report.dropped_count:
            degraded.append(f"validity:dropped={report.dropped_count}")
        chunks = dedupe(chunks)
        stages["postprocess"] = round((time.time() - t0) * 1000, 1)

        # 6. 重排
        t0 = time.time()
        if config.use_rerank and len(chunks) > 1:
            chunks, fallback = self._rerank(effective, chunks, top_k)
            if fallback:
                degraded.append("rerank:fallback")
        stages["rerank"] = round((time.time() - t0) * 1000, 1)

        # 7. 域先验重打分（在重排之后）
        # 用软加权而不是硬 filter：跨域问题（退款问物流时效）硬过滤会直接漏答案。
        if config.use_domain_prior and domain_prior:
            chunks.sort(
                key=lambda c: (c.rerank_score if c.rerank_score is not None else c.rrf_score)
                * (DOMAIN_BOOST if c.domain == domain_prior else 1.0),
                reverse=True,
            )

        out.chunks = chunks[:top_k]
        out.degraded = degraded
        out.stages = stages
        if not out.chunks:
            degraded.append("retrieval:empty")
        self.cache.set(cache_key, out)
        return out
