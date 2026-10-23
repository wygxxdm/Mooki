from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_ASCII = re.compile(r"[a-z0-9]+")
_NON_CJK = re.compile(r"[^\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """中英混合切分：英文/数字整词 + 中文连续段切字符 bigram。

    BM25 与 LSA 共用同一套切分，保证两路的"词表"口径一致，
    否则 RRF 融合时两路看到的是不同的特征空间，融合就失去意义。
    """
    text = text.lower()
    tokens = _ASCII.findall(text)
    for seg in _NON_CJK.sub(" ", text).split():
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return tokens



def normalize_text(text: str) -> str:
    """归一化：去空白、全角转半角标点、小写。用于缓存 key 与规则匹配。"""
    text = text.strip().lower()
    table = str.maketrans("，。！？；：（）【】“”‘’", ",.!?;:()[]\"\"''")
    return _WS.sub(" ", text.translate(table))


def estimate_tokens(text: str) -> int:
    """粗估 token：中文约 1 token / 1.5 字，其它约 1 token / 4 字符。

    只用于上下文预算切分，不追求与具体 tokenizer 对齐。
    """
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = max(0, len(text) - cjk)
    return max(1, int(cjk / 1.5 + other / 4))


def truncate_tokens(text: str, budget: int) -> str:
    if estimate_tokens(text) <= budget:
        return text
    # 按比例截断再回退，避免逐字循环
    ratio = budget / max(estimate_tokens(text), 1)
    cut = max(1, int(len(text) * ratio) - 2)
    return text[:cut].rstrip() + "…"
