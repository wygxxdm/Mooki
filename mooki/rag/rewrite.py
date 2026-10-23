"""Query Rewrite。

改写要解决的问题：多轮会话里的指代（"那我这个能退吗"）、口语化（"钱咋还没到"）、
以及售后领域同义词（退钱/退款/原路退回）。

改写结果带 TTL 缓存 —— 同一句话在会话里反复出现时不必重复推理，
这也是把 P95 延迟压下来的主要手段之一。
"""

from __future__ import annotations

from mooki.core.cache import TTLCache
from mooki.core.text import normalize_text


class QueryRewriter:
    def __init__(self, llm, cache: TTLCache):
        self.llm = llm
        self.cache = cache
        self.stats = {"calls": 0, "cache_hits": 0}

    def rewrite(self, query: str, history: list[str] | None = None) -> str:
        if not query.strip():
            return query
        key = normalize_text(query)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        self.stats["calls"] += 1
        try:
            rewritten = self.llm.rewrite_query(query, history)
        except Exception:  # noqa: BLE001 - 改写失败必须退回原句，不能影响主链路
            rewritten = query
        rewritten = (rewritten or query).strip() or query
        # 改写结果异常（过长/空）时保守回退
        if len(rewritten) > 200:
            rewritten = query
        self.cache.set(key, rewritten)
        return rewritten
