from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # 身份
    trace_id: str
    session_id: str
    user_id: str
    message: str

    # 记忆快照
    profile: dict[str, Any]
    history: list[str]
    summary: str
    slots: dict[str, str]
    episodes: list[dict[str, Any]]
    memory_report: dict[str, Any]

    # 意图与受控参数
    intent: Any
    task_type: str
    allowed_tools: list[str]
    remaining_tools: list[str]

    # 执行
    plan: dict[str, Any]
    tool_results: list[dict[str, Any]]
    iterations: int
    degraded: list[str]

    # 输出
    reply: str
    citations: list[dict[str, Any]]
    handoff: dict[str, Any] | None
    clarify: bool
    assembly: dict[str, Any]
    error: str | None
