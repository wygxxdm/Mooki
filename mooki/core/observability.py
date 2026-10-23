"""可观测性：Agent trace 事件模型 + 指标聚合。

设计要点：
1. 事件是唯一事实来源 —— 前端 trace 面板、离线分析、指标看板都读同一份事件流，
   避免"日志一套、监控一套、前端又一套"。
2. 订阅机制用 queue 实现，SSE 端点只是把 queue 里的东西吐出去，
   不侵入业务代码（这是 trace 面板能低成本做出来的原因）。
"""

from __future__ import annotations

import threading
import time
import queue
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Iterator

from mooki.core.schema import TraceEvent, new_id, now_ms

_MAX_EVENTS_PER_TRACE = 500


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = defaultdict(int)
        self.latencies: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=200))

    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[name] += value

    def observe(self, name: str, ms: int) -> None:
        with self._lock:
            self.latencies[name].append(ms)

    def percentile(self, name: str, p: float) -> float:
        with self._lock:
            data = sorted(self.latencies.get(name, []))
        if not data:
            return 0.0
        idx = min(len(data) - 1, int(round((p / 100) * (len(data) - 1))))
        return float(data[idx])

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            keys = set(self.latencies) | {
                k[: -len("_ms")] for k in self.counters if k.endswith("_ms")
            }
        out = {"counters": dict(self.counters), "latency": {}}
        for k in sorted(keys):
            out["latency"][k] = {
                "p50": round(self.percentile(k, 50), 1),
                "p95": round(self.percentile(k, 95), 1),
                "n": len(self.latencies.get(k, [])),
            }
        return out


class TraceStore:
    def __init__(self, max_traces: int = 500):
        self._events: dict[str, list[TraceEvent]] = {}
        self._order: deque[str] = deque(maxlen=max_traces)
        self._subs: dict[str, list[queue.Queue]] = defaultdict(list)
        self._lock = threading.Lock()
        self.metrics = Metrics()

    # ---------- 生命周期 ----------

    def start(self, trace_id: str | None = None, meta: dict[str, Any] | None = None) -> str:
        trace_id = trace_id or new_id("trace")
        with self._lock:
            if trace_id not in self._events:
                self._order.append(trace_id)
            self._events[trace_id] = []
        self.emit(trace_id, "run_start", payload=meta or {})
        return trace_id

    def finish(self, trace_id: str, payload: dict[str, Any] | None = None) -> None:
        self.emit(trace_id, "run_end", payload=payload or {})
        self.close(trace_id)

    def close(self, trace_id: str) -> None:
        with self._lock:
            subs = list(self._subs.pop(trace_id, []))
        for q in subs:
            q.put(None)

    # ---------- 写入 ----------

    def emit(
        self,
        trace_id: str,
        type_: str,
        node: str | None = None,
        payload: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            trace_id=trace_id, type=type_, node=node, payload=payload or {}, latency_ms=latency_ms
        )
        with self._lock:
            bucket = self._events.setdefault(trace_id, [])
            if len(bucket) < _MAX_EVENTS_PER_TRACE:
                bucket.append(event)
            subs = list(self._subs.get(trace_id, []))
        for q in subs:
            q.put(event)
        self.metrics.incr(f"event.{type_}")
        return event

    @contextmanager
    def span(self, trace_id: str, node: str, payload: dict[str, Any] | None = None) -> Iterator[TraceEvent]:
        started = time.time()
        self.emit(trace_id, "node_enter", node, payload)
        holder = TraceEvent(trace_id=trace_id, type="node_exit", node=node)
        try:
            yield holder
        finally:
            latency = int((time.time() - started) * 1000)
            self.emit(trace_id, "node_exit", node, holder.payload, latency_ms=latency)
            self.metrics.observe(node, latency)

    # ---------- 读取 ----------

    def get(self, trace_id: str) -> list[TraceEvent]:
        with self._lock:
            return list(self._events.get(trace_id, []))

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(self._order)[-n:]
        out = []
        for tid in reversed(ids):
            events = self.get(tid)
            if not events:
                continue
            start = events[0]
            end = next((e for e in events if e.type == "run_end"), None)
            out.append(
                {
                    "trace_id": tid,
                    "started_at": start.ts,
                    "user_input": start.payload.get("message", ""),
                    "duration_ms": end.latency_ms if end and end.latency_ms else None,
                    "events": len(events),
                    "degraded": any(e.payload.get("degraded") for e in events),
                }
            )
        return out

    def subscribe(self, trace_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subs[trace_id].append(q)
            backlog = list(self._events.get(trace_id, []))
        for e in backlog:
            q.put(e)
        return q


trace_store = TraceStore()


def span_now() -> int:
    return now_ms()
