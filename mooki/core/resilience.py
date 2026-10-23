"""熔断 / 超时 / 降级。

三个能力收敛在 call_guarded 一个入口，工具层和检索层共用同一套语义：
    熔断打开 -> 直接走 fallback（degraded=True），不打后端
    超时     -> 计为失败 + 走 fallback
    抛异常   -> 计为失败 + 走 fallback
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

_EXECUTOR = ThreadPoolExecutor(max_workers=32, thread_name_prefix="mooki-guard")


@dataclass
class GuardedResult:
    ok: bool
    value: Any = None
    error: str | None = None
    degraded: bool = False
    latency_ms: int = 0


class CircuitBreaker:
    def __init__(self, name: str, fail_threshold: int = 5, reset_s: float = 30.0):
        self.name = name
        self.fail_threshold = fail_threshold
        self.reset_s = reset_s
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False
        self._lock = threading.Lock()
        self.transitions: list[tuple[float, str]] = []

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if time.time() - self._opened_at >= self.reset_s:
                return "half_open"
            return "open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half_open":
            with self._lock:
                if self._probe_in_flight:
                    return False
                self._probe_in_flight = True
                return True
        return False

    def record_success(self) -> None:
        with self._lock:
            if self._opened_at is not None:
                self.transitions.append((time.time(), "closed"))
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._probe_in_flight = False
            if self._failures >= self.fail_threshold and self._opened_at is None:
                self._opened_at = time.time()
                self.transitions.append((time.time(), "open"))

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "failures": self._failures,
            "fail_threshold": self.fail_threshold,
        }


def call_guarded(
    fn: Callable[[], Any],
    *,
    timeout_s: float = 8.0,
    breaker: CircuitBreaker | None = None,
    fallback: Callable[[str], Any] | None = None,
    fallback_label: str = "fallback",
) -> GuardedResult:
    started = time.time()

    def _elapsed() -> int:
        return int((time.time() - started) * 1000)

    if breaker is not None and not breaker.allow():
        value = fallback(fallback_label + ":circuit_open") if fallback else None
        return GuardedResult(
            ok=fallback is not None,
            value=value,
            error="circuit_open",
            degraded=True,
            latency_ms=_elapsed(),
        )

    future = _EXECUTOR.submit(fn)
    try:
        value = future.result(timeout=timeout_s)
    except FutureTimeout:
        if breaker is not None:
            breaker.record_failure()
        value = fallback(fallback_label + ":timeout") if fallback else None
        return GuardedResult(
            ok=fallback is not None,
            value=value,
            error="timeout",
            degraded=True,
            latency_ms=_elapsed(),
        )
    except Exception as exc:  # noqa: BLE001 - 这里就是要兜住所有工具异常
        if breaker is not None:
            breaker.record_failure()
        value = fallback(f"{fallback_label}:{type(exc).__name__}") if fallback else None
        return GuardedResult(
            ok=fallback is not None,
            value=value,
            error=f"{type(exc).__name__}: {exc}",
            degraded=True,
            latency_ms=_elapsed(),
        )

    if breaker is not None:
        breaker.record_success()
    return GuardedResult(ok=True, value=value, latency_ms=_elapsed())
