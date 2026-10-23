"""FastAPI 接入层。

三个端点对应三种消费方式：
    POST /api/chat          同步拿最终答案（业务系统对接）
    POST /api/chat/stream   SSE 推 trace 事件（前端轨迹面板）
    GET  /api/trace/{id}    回看完整轨迹（客诉复盘 / 评测取样）

trace 是先产生的、答案后产生的 —— 所以流式端点先订阅再触发。
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from mooki.core.observability import trace_store
from mooki.core.schema import ChatRequest, ChatResponse, new_id
from mooki.graph.runtime import GRAPH_ENGINE, get_runtime

ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = ROOT / "web"

app = FastAPI(title="Mooki 售后 Agent", version="0.1.0")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def event_envelope(event) -> dict:
    """把 TraceEvent 包成 SSE 消息。

    必须嵌套在 `event` 字段下。早期版本用 `{"type": "trace_event", **event.model_dump()}`
    直接展开，TraceEvent 自带的 type（run_end / tool_call…）会把外层的 "trace_event" 覆盖掉，
    前端按 type 分流就永远匹配不上 —— 表现为面板全空、回复也拿不到。
    """
    return {"type": "trace_event", "event": event.model_dump()}


@app.get("/api/health")
def health() -> dict:
    runtime = get_runtime()
    return {
        "ok": True,
        "engine": GRAPH_ENGINE,
        "chunks": runtime.index.size,
        "domains": runtime.index.domains(),
        "llm_offline": runtime.settings.llm_offline,
    }


@app.get("/api/stats")
def stats() -> dict:
    return get_runtime().stats()


@app.get("/api/tools")
def tools() -> dict:
    """MCP 工具清单（就是 registry.manifest，可直接喂给任意 MCP client）。"""
    runtime = get_runtime()
    return {"count": len(runtime.registry.names()), "tools": runtime.registry.manifest()}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return get_runtime().chat(request)


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    runtime = get_runtime()
    trace_id = new_id("trace")
    channel = trace_store.subscribe(trace_id)

    def worker() -> None:
        try:
            runtime.chat(request, trace_id=trace_id)
        finally:
            trace_store.close(trace_id)

    threading.Thread(target=worker, daemon=True).start()

    def generator():
        yield _sse({"type": "accepted", "trace_id": trace_id, "engine": GRAPH_ENGINE})
        while True:
            try:
                item = channel.get(timeout=30)
            except queue.Empty:
                yield _sse({"type": "ping"})
                continue
            if item is None:
                break
            # 必须把事件嵌套进 event 字段，不能直接展开：
            # TraceEvent 自己也有 type（run_end / tool_call…），展开会把它覆盖掉，
            # 前端按 trace_event 分流就永远匹配不上（踩过一次）。
            yield _sse(event_envelope(item))
        yield _sse({"type": "closed", "trace_id": trace_id})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/trace/{trace_id}")
def trace_detail(trace_id: str) -> dict:
    events = trace_store.get(trace_id)
    if not events:
        raise HTTPException(status_code=404, detail="trace 不存在")
    return {"trace_id": trace_id, "events": [e.model_dump() for e in events]}


@app.get("/api/traces")
def trace_list(limit: int = 20) -> dict:
    return {"items": trace_store.recent(limit)}


@app.post("/api/eval/retrieval")
def eval_retrieval() -> dict:
    from mooki.eval.retrieval_eval import run_ablation

    runtime = get_runtime()
    return run_ablation(runtime.retriever, runtime.settings)


@app.post("/api/eval/intent")
def eval_intent() -> dict:
    from mooki.eval.intent_eval import run_intent_eval

    return run_intent_eval(get_runtime().classifier)


@app.get("/")
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="前端页面缺失")
    return FileResponse(page)
