"""LangGraph StateGraph 的最小等价实现（仅在 langgraph 未安装时启用）。

只实现本项目用到的子集：add_node / set_entry_point / add_edge /
add_conditional_edges / compile().invoke()。语义与 LangGraph 一致 ——
节点返回"状态增量"，由引擎合并；条件边由路由函数决定下一跳；
循环靠 recursion_limit 兜底，而不是靠模型自觉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

END = "__end__"


class StateGraph:
    def __init__(self, state_schema: Any = None):
        self.state_schema = state_schema
        self.nodes: dict[str, Callable[[dict], dict]] = {}
        self.edges: dict[str, str] = {}
        self.conditional: dict[str, tuple[Callable[[dict], str], dict[str, str] | None]] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: Callable[[dict], dict]) -> "StateGraph":
        self.nodes[name] = fn
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    def add_edge(self, start: str, end: str) -> "StateGraph":
        self.edges[start] = end
        return self

    def add_conditional_edges(
        self, start: str, router: Callable[[dict], str], mapping: dict[str, str] | None = None
    ) -> "StateGraph":
        self.conditional[start] = (router, mapping)
        return self

    def compile(self) -> "CompiledGraph":
        if self.entry is None:
            raise ValueError("entry point 未设置")
        return CompiledGraph(self)


@dataclass
class CompiledGraph:
    graph: StateGraph
    recursion_limit: int = 25
    trace: list[str] = field(default_factory=list)

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        limit = (config or {}).get("recursion_limit", self.recursion_limit)
        current = self.graph.entry
        data = dict(state)
        self.trace = []
        steps = 0
        while current and current != END:
            steps += 1
            if steps > limit:
                raise RuntimeError(f"图执行超过 recursion_limit={limit}，疑似死循环")
            fn = self.graph.nodes.get(current)
            if fn is None:
                raise KeyError(f"节点不存在：{current}")
            self.trace.append(current)
            update = fn(data) or {}
            data.update(update)

            if current in self.graph.conditional:
                router, mapping = self.graph.conditional[current]
                nxt = router(data)
                current = mapping.get(nxt, nxt) if mapping else nxt
            else:
                current = self.graph.edges.get(current)
        return data
