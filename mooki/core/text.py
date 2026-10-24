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


# ---------- PII 脱敏 ----------
# trace 会被存下来、推到前端、导出给客诉复盘。手机号和证件号不该跟着跑一路。
_PHONE = re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)")
_ID_CARD = re.compile(r"(?<!\d)(\d{6})\d{8}(\d{3}[\dXx])(?!\d)")
_BANK_CARD = re.compile(r"(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)")
_EMAIL = re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def redact_pii(text: str) -> str:
    """把手机号、身份证、银行卡、邮箱打码，保留首尾便于人工核对。

    订单号（SO 开头）不动 —— 它是排查必须的定位信息，不属于个人信息。
    地址不在这里处理：中文地址没有稳定模式，硬匹配会把"XX 路 3 号"这类
    正常描述一起打掉，宁可在采集侧就不收整段地址。
    """
    if not text:
        return text
    text = _PHONE.sub(r"\1****\2", text)
    text = _ID_CARD.sub(r"\1********\2", text)
    text = _BANK_CARD.sub(r"\1********\2", text)
    text = _EMAIL.sub(r"\1***@\2", text)
    return text


def redact_deep(value):
    """递归脱敏：trace payload 是嵌套 dict/list，只在最外层替换不够。"""
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {k: redact_deep(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_deep(v) for v in value]
    return value
