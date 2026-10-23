"""融合与去重。

RRF（Reciprocal Rank Fusion）只依赖排名、不依赖分数，所以在 BM25 分数和余弦相似度
量纲完全不同的情况下也能直接融合 —— 这是它比"归一化加权和"更常用的原因。
公式：score(d) = Σ_i w_i / (k + rank_i(d))，k 默认 60，作用是压低头部差距。
"""

from __future__ import annotations

import re

from mooki.core.text import normalize_text

_PUNCT = re.compile(r"[\s,.;:!?，。；：！？、\-—_（）()【】\[\]\"'“”‘’]+")


def rrf_fuse(
    ranked_lists: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """ranked_lists: 每个召回通道的 id 有序列表（最相关在前）。"""
    if not ranked_lists:
        return []
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for ids, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _fingerprint(text: str) -> str:
    return _PUNCT.sub("", normalize_text(text))


def dedupe(chunks: list, sim_threshold: int = 40) -> list:
    """去掉正文重复或高包含关系的 chunk。

    长文档切分时重叠窗口会产出近似重复块；它们会同时挤进 top-k，
    把真正不同的证据挤出预算，所以必须在重排之前去重。
    """
    kept: list = []
    seen: list[str] = []
    for chunk in chunks:
        fp = _fingerprint(chunk.text)
        duplicate = False
        for prev in seen:
            if fp == prev:
                duplicate = True
                break
            shorter, longer = (fp, prev) if len(fp) <= len(prev) else (prev, fp)
            if len(shorter) >= sim_threshold and shorter in longer:
                duplicate = True
                break
        if not duplicate:
            kept.append(chunk)
            seen.append(fp)
    return kept
