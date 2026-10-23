from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ms() -> int:
    return int(time.time() * 1000)


class Intent(str, Enum):
    """10 类售后意图。"""

    ORDER_STATUS = "order_status"
    REFUND = "refund"
    RETURN_EXCHANGE = "return_exchange"
    LOGISTICS = "logistics"
    COMPLAINT = "complaint"
    PRODUCT_CONSULT = "product_consult"
    INVOICE = "invoice"
    PROMO = "promo"
    ACCOUNT = "account"
    OTHER = "other"


class Urgency(str, Enum):
    """4 档紧急度，与意图正交，独立判定。"""

    P0 = "P0"  # 资损 / 强情绪 / 监管风险
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"

    @property
    def rank(self) -> int:
        return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}[self.value]


class TaskType(str, Enum):
    QUERY = "query"
    TRANSACTION = "transaction"
    SEARCH = "search"
    ESCALATE = "escalate"


class VoteSource(str, Enum):
    LLM = "llm"
    DENSE = "dense"
    RULE = "rule"


class Vote(BaseModel):
    source: VoteSource
    label: Intent
    score: float = Field(ge=0.0, le=1.0)
    weight: float = 1.0


class IntentResult(BaseModel):
    label: Intent
    urgency: Urgency
    confidence: float
    votes: list[Vote] = Field(default_factory=list)
    margin: float = 0.0
    need_clarify: bool = False
    cached: bool = False
    latency_ms: int = 0


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(default_factory=lambda: new_id("call"))


class ToolResult(BaseModel):
    call_id: str
    name: str
    ok: bool
    data: Any = None
    error: str | None = None
    degraded: bool = False
    latency_ms: int = 0
    from_cache: bool = False


class RetrievedChunk(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    domain: str
    text: str
    score: float = 0.0
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    valid_until: str | None = None


class TraceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("ev"))
    trace_id: str
    type: str
    node: str | None = None
    ts: int = Field(default_factory=now_ms)
    payload: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_id: str = "u_1001"


class ChatResponse(BaseModel):
    trace_id: str
    session_id: str
    reply: str
    intent: IntentResult | None = None
    tools_used: list[str] = Field(default_factory=list)
    degraded: bool = False
    handoff: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class HandoffTicket(BaseModel):
    ticket_id: str
    reason: str
    urgency: Urgency
    summary: str
    intent: Intent
    session_id: str
    user_id: str
    created_at: int = Field(default_factory=now_ms)


Level = Literal["info", "warn", "error"]
