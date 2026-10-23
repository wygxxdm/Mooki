"""知识有效期与业务一致性校验。

这是整个 RAG 里最"生产"的一环：知识库会过期，但向量库不会自己知道。
一条去年写的"退款 3 个工作日到账"如果被召回，答案就是错的，而且是那种
看起来特别有底气的错。所以必须在召回后、重排前做硬过滤。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass
class ValidityReport:
    dropped: list[dict]
    kept: int

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def filter_valid(chunks: list, today: dt.date | None = None) -> tuple[list, ValidityReport]:
    today = today or dt.date.today()
    kept, dropped = [], []
    for chunk in chunks:
        valid_until = _parse_date(getattr(chunk, "valid_until", None))
        if valid_until is not None and valid_until < today:
            dropped.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "valid_until": chunk.valid_until,
                    "reason": "knowledge_expired",
                }
            )
            continue
        if not getattr(chunk, "text", "").strip():
            dropped.append({"chunk_id": chunk.chunk_id, "reason": "empty"})
            continue
        kept.append(chunk)
    return kept, ValidityReport(dropped=dropped, kept=len(kept))


def consistency_flags(chunk, order: dict | None) -> list[str]:
    """知识条目与单据数据的一致性检查，用于生成答案时提示冲突。"""
    flags: list[str] = []
    if not order:
        return flags
    text = getattr(chunk, "text", "")
    meta = getattr(chunk, "domain", "")
    if meta == "refund" and order.get("status") == "unpaid" and "已支付" in text:
        flags.append("knowledge_mentions_paid_but_order_unpaid")
    return flags
