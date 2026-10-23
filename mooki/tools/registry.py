"""MCP 风格工具层。

每个工具声明为一份 MCP tool descriptor（name / description / inputSchema），
registry.manifest() 直接就能喂给任何 MCP client。

统一的缓存 / 熔断 / 超时 / 降级在这里一次性实现，业务 handler 里不写这些横切逻辑 ——
这是"工具层"存在的意义，否则它就只是函数集合。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from mooki.core.cache import TTLCache
from mooki.core.config import Settings
from mooki.core.resilience import CircuitBreaker, call_guarded
from mooki.core.schema import Intent, ToolResult


@dataclass
class ToolContext:
    user_id: str
    session_id: str
    trace_id: str
    intent: Intent | None = None
    services: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], Any]
    timeout_s: float = 8.0
    cache_ttl: int | None = 60
    risk: str = "read"  # read | write

    def descriptor(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": self.input_schema}


class ToolPermissionError(PermissionError):
    pass


class ToolRegistry:
    def __init__(self, settings: Settings, cache: TTLCache):
        self.settings = settings
        self.cache = cache
        self._tools: dict[str, ToolSpec] = {}
        self._breakers: dict[str, CircuitBreaker] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        self._breakers[spec.name] = CircuitBreaker(
            f"tool.{spec.name}", self.settings.breaker_fail_threshold, self.settings.breaker_reset_s
        )

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def manifest(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        selected = names if names is not None else self.names()
        return [self._tools[n].descriptor() for n in selected if n in self._tools]

    def call(self, name: str, args: dict[str, Any], ctx: ToolContext, whitelist: list[str]) -> ToolResult:
        """唯一的调用入口：先做白名单裁决，再做降级包装。"""
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(call_id="-", name=name, ok=False, error="tool_not_found")
        if name not in whitelist:
            # 越权拦截属于预期行为，不计入熔断失败
            return ToolResult(call_id="-", name=name, ok=False, error="tool_not_in_whitelist")

        missing = [k for k in spec.input_schema.get("required", []) if not args.get(k)]
        if missing:
            return ToolResult(call_id="-", name=name, ok=False, error=f"missing_args:{missing}")

        cache_key = ""
        if spec.cache_ttl and spec.risk == "read":
            cache_key = f"tool|{name}|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return ToolResult(
                    call_id="-", name=name, ok=True, data=cached, from_cache=True, latency_ms=0
                )

        result = call_guarded(
            lambda: spec.handler(args, ctx),
            timeout_s=spec.timeout_s or self.settings.tool_timeout_s,
            breaker=self._breakers[name],
            fallback=lambda reason: {"degraded_reason": reason, "available": False},
        )

        if result.ok and not result.degraded and cache_key:
            self.cache.set(cache_key, result.value)

        return ToolResult(
            call_id="-",
            name=name,
            ok=result.ok,
            data=result.value,
            error=result.error,
            degraded=result.degraded,
            latency_ms=result.latency_ms,
        )
