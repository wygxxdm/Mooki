"""受控单 Agent 运行时。

它不是 ReAct：拓扑是固定的，模型只在"白名单内的工具选择 + 参数生成"这一小段有自由度。
换来的是三件 ReAct 给不了的东西：
    1. 动作空间可枚举 —— 每个意图对应固定工具集，越权调用在 registry 层被拦；
    2. 循环可终止 —— 迭代次数硬上限，不靠模型自己说"我完成了"；
    3. 失败有保底 —— 校验不过就降级/转人工，不会卡死在半路。

节点职责划分（每个节点只做一件事，避免状态漂移）：
    load_context  只读记忆，不推理
    intent        只分类，不决策
    plan/act      只做工具选择与执行，不生成话术
    observe       只做结果校验与路由，不生成长文本
    respond       只生成话术，不再调工具
职责单一才能让每一步的 trace 有意义，也才能单独评测。
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from mooki.core.cache import TTLCache
from mooki.core.config import Settings, settings as default_settings
from mooki.core.embedder import build_embedder
from mooki.core.llm import build_llm
from mooki.core.observability import trace_store
from mooki.core.schema import (
    ChatRequest,
    ChatResponse,
    Intent,
    TaskType,
    Urgency,
    now_ms,
)
from mooki.core.storage import build_kv
from mooki.core.text import truncate_tokens
from mooki.graph.state import AgentState
from mooki.intent.classifier import IntentClassifier
from mooki.intent.labels import INTENT_SPECS, task_for, tools_for
from mooki.memory.assembler import ContextAssembler
from mooki.memory.episodic import Episode, EpisodicMemory
from mooki.memory.profile import ProfileStore
from mooki.memory.short_term import ShortTermMemory
from mooki.rag.ingest import load_index
from mooki.rag.retriever import Retriever
from mooki.rag.rewrite import QueryRewriter
from mooki.tools.builtin.service_tools import ServiceTools, build_registry
from mooki.tools.registry import ToolContext

try:  # 优先用真正的 LangGraph
    from langgraph.graph import END, StateGraph

    GRAPH_ENGINE = "langgraph"
except Exception:  # noqa: BLE001 - 未安装时退回等价最小实现，保证可离线运行
    from mooki.graph.mini_graph import END, StateGraph

    GRAPH_ENGINE = "mini"


# 会改变外部状态的写操作。规划层对它们做幂等拦截，避免多轮追问重放。
SIDE_EFFECT_TOOLS = frozenset(
    {"refund.apply", "refund.cancel", "return.create", "invoice.issue", "ticket.create"}
)
_APPLIED_TTL_S = 3600


class AgentRuntime:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        s = self.settings

        self.kv = build_kv(s.redis_url)
        self.embedder = build_embedder(s)
        self.llm = build_llm(s)
        self.index = load_index(s, self.embedder)

        self.intent_cache = TTLCache(s.intent_cache_size, s.intent_cache_ttl)
        self.retrieval_cache = TTLCache(2048, s.retrieval_cache_ttl)
        self.rewrite_cache = TTLCache(2048, s.rewrite_cache_ttl)
        self.tool_cache = TTLCache(4096, 120)

        self.rewriter = QueryRewriter(self.llm, self.rewrite_cache)
        self.retriever = Retriever(s, self.embedder, self.llm, self.index, self.retrieval_cache, self.rewriter)
        self.classifier = IntentClassifier(self.llm, self.embedder, self.intent_cache)

        self.short_term = ShortTermMemory(self.kv, window=s.short_term_window)
        self.episodic = EpisodicMemory(self.embedder, path=s.index_dir / "episodic", top_k=s.episodic_top_k)
        self.profiles = ProfileStore(self.kv)
        self.assembler = ContextAssembler(s.context_budget_tokens)

        self.services = ServiceTools(s.data_dir, ledger_path=s.ledger_file)
        self.registry = build_registry(s, self.tool_cache, self.services)
        self.trace = trace_store

        self.graph = self._build_graph()

    # ---------- 图构建 ----------

    def _wrap(self, name: str, fn: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
        def node(state: AgentState) -> dict:
            with self.trace.span(state["trace_id"], name):
                return fn(state) or {}

        return node

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("load_context", self._wrap("load_context", self._n_load_context))
        graph.add_node("intent", self._wrap("intent", self._n_intent))
        graph.add_node("plan", self._wrap("plan", self._n_plan))
        graph.add_node("act", self._wrap("act", self._n_act))
        graph.add_node("observe", self._wrap("observe", self._n_observe))
        graph.add_node("respond", self._wrap("respond", self._n_respond))
        graph.add_node("clarify", self._wrap("clarify", self._n_clarify))
        graph.add_node("handoff", self._wrap("handoff", self._n_handoff))
        graph.add_node("persist", self._wrap("persist", self._n_persist))

        graph.set_entry_point("load_context")
        graph.add_edge("load_context", "intent")
        graph.add_conditional_edges(
            "intent",
            self._route_after_intent,
            {"clarify": "clarify", "handoff": "handoff", "plan": "plan"},
        )
        graph.add_edge("plan", "act")
        graph.add_edge("act", "observe")
        graph.add_conditional_edges(
            "observe",
            self._route_after_observe,
            {"plan": "plan", "respond": "respond", "handoff": "handoff", "clarify": "clarify"},
        )
        graph.add_edge("respond", "persist")
        graph.add_edge("clarify", "persist")
        graph.add_edge("handoff", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

    # ---------- 路由 ----------

    def _route_after_intent(self, state: AgentState) -> str:
        intent = state["intent"]
        message = state["message"]
        # 用户显式要求人工，必须无条件满足，不能再去猜意图
        if any(k in message for k in ("转人工", "人工客服", "找真人", "真人客服", "要人工")):
            return "handoff"
        if intent.urgency is Urgency.P0:
            return "handoff"  # 资损/强投诉不进自动流程，这是安全阀
        if intent.label is Intent.OTHER:
            return "clarify"  # 落到 other 说明没听懂，追问比瞎猜便宜
        if intent.need_clarify and intent.urgency.rank >= 2:
            return "clarify"
        return "plan"

    def _route_after_observe(self, state: AgentState) -> str:
        results = state.get("tool_results") or []
        if not results:
            return "respond"
        last = results[-1]
        data = last.get("data") if isinstance(last.get("data"), dict) else {}
        if data.get("found") is False:
            return "respond"  # 查无此单，继续调工具没有意义
        iterations = state.get("iterations", 0)
        if iterations >= self.settings.max_tool_iterations:
            return "respond"
        used = {r["name"] for r in results}
        if not last["ok"]:
            return "plan" if "kb.search" not in used else "respond"
        if state.get("task_type") == TaskType.TRANSACTION.value and "kb.search" not in used:
            return "plan"  # 交易类必须补一次政策检索，否则容易答错规则
        return "respond"

    # ---------- 节点 ----------

    def _n_load_context(self, state: AgentState) -> dict:
        sid, uid = state["session_id"], state["user_id"]
        self.short_term.maybe_compress(sid, self.llm)
        memory = self.short_term.load(sid)
        profile = self.profiles.get(uid)
        episodes = self.episodic.recall(state["message"], uid)
        report = {
            "turns": len(memory.turns),
            "memory_tokens": memory.tokens,
            "summary_tokens": len(memory.summary),
            "slots": memory.slots,
            "episodes": len(episodes),
        }
        self.trace.emit(state["trace_id"], "memory_read", "load_context", report)
        return {
            "profile": profile.as_prompt(),
            "history": memory.as_history(),
            "summary": memory.summary,
            "slots": memory.slots,
            "episodes": [{"summary": e.summary, "intent": e.intent} for e in episodes],
            "memory_report": report,
        }

    def _n_intent(self, state: AgentState) -> dict:
        result = self.classifier.classify(state["message"])
        self.trace.emit(
            state["trace_id"],
            "intent",
            "intent",
            {
                "label": result.label.value,
                "urgency": result.urgency.value,
                "confidence": result.confidence,
                "margin": result.margin,
                "need_clarify": result.need_clarify,
                "cached": result.cached,
                "votes": [
                    {
                        "source": v.source.value,
                        "label": v.label.value,
                        "score": v.score,
                        "weight": v.weight,
                    }
                    for v in result.votes
                ],
            },
            latency_ms=result.latency_ms,
        )
        return {
            "intent": result,
            "task_type": task_for(result.label).value,
            "allowed_tools": tools_for(result.label),
        }

    def _n_plan(self, state: AgentState) -> dict:
        used = {r["name"] for r in state.get("tool_results", [])}
        remaining = [t for t in state["allowed_tools"] if t not in used]
        if not remaining:
            return {"plan": {"tool": "", "args": {}}, "remaining_tools": []}

        slots = state.get("slots") or {}
        probe = state["message"]
        if slots.get("order_id") and slots["order_id"] not in probe:
            probe = f"{probe}（订单号 {slots['order_id']}）"

        raw = self.llm.plan_tool(probe, remaining, intent_hint=state["intent"].label.value) or {}
        tool = str(raw.get("tool") or "")
        args = dict(raw.get("args") or {})
        if tool not in remaining:  # 防幻觉：模型可能编造工具名
            tool = ""
        if not tool:
            tool = "kb.search" if "kb.search" in remaining else remaining[0]
            if tool == "kb.search":
                args.setdefault("query", state["message"][:120])

        # 订单号兜底：模型漏填就从原话里抽，再退到槽位记忆
        if tool != "kb.search" and not args.get("order_id"):
            found = re.search(r"SO\d{9,}", state["message"].upper())
            args["order_id"] = (found.group(0) if found else "") or slots.get("order_id", "")

        # 参数仍不全就退回检索，避免把"缺参数"变成一次失败的工具调用
        spec = self.registry.get(tool)
        if spec:
            missing = [k for k in spec.input_schema.get("required", []) if not args.get(k)]
            if missing and tool != "kb.search" and "kb.search" in remaining:
                tool, args = "kb.search", {"query": state["message"][:120]}

        # 副作用幂等：账本里已经落过账的动作不在规划层重放。
        # 多轮里用户追问"钱多久到账"，意图仍是 refund，若不拦会再规划一次 refund.apply；
        # 账本幂等键能兜住不重复扣钱，但白跑一次写操作、答复也会退化成重复播报上次结论。
        skip_reason = self._side_effect_done(tool, args, state["message"])
        if skip_reason and tool != "kb.search" and "kb.search" in remaining:
            self.trace.emit(
                state["trace_id"], "plan_skip", "plan", {"tool": tool, "reason": skip_reason}
            )
            tool, args = "kb.search", {"query": state["message"][:120]}

        self.trace.emit(
            state["trace_id"],
            "plan",
            "plan",
            {"tool": tool, "args": args, "remaining": remaining, "iteration": state.get("iterations", 0)},
        )
        return {"plan": {"tool": tool, "args": args}, "remaining_tools": remaining}

    def _side_effect_done(self, tool: str, args: dict, message: str) -> str:
        """写操作是否已经生效。返回非空字符串表示应跳过，值即跳过原因。

        两级判断：
        1. 会话级登记表 —— 本次会话里已经成功执行过的写操作，覆盖 return.create /
           invoice.issue 这类没有订单状态机的工具；
        2. 账本级状态 —— refund.apply / refund.cancel 以订单状态机为准，
           它跨会话也成立（用户明天再来问，仍不该重复退）。
        """
        order_id = str(args.get("order_id") or "")
        if not order_id:
            return ""
        # 用户明确要求"再退一次/重新申请"，交给账本幂等键处理，不做规划层拦截
        if any(k in message for k in ("再退", "重新退", "再申请", "重新申请")):
            return ""
        if tool in SIDE_EFFECT_TOOLS:
            if self.kv.get(self._applied_key(args, tool)):
                return "session_applied"
        state = self.services.ledger.status(order_id)
        if tool == "refund.apply" and state in ("refund_pending", "refunded"):
            return f"ledger_state={state}"
        if tool == "refund.cancel" and state != "refund_pending":
            return f"ledger_state={state or 'none'}"
        return ""

    def _applied_key(self, args: dict, tool: str) -> str:
        order_id = str(args.get("order_id") or "")
        return f"applied:{tool}:{order_id}"

    def _n_act(self, state: AgentState) -> dict:
        plan = state.get("plan") or {}
        tool, args = plan.get("tool") or "", plan.get("args") or {}
        ctx = ToolContext(
            user_id=state["user_id"],
            session_id=state["session_id"],
            trace_id=state["trace_id"],
            intent=state.get("intent"),
            services={"retriever": self.retriever, "history": state.get("history") or []},
        )
        self.trace.emit(state["trace_id"], "tool_call", "act", {"tool": tool, "args": args})
        result = self.registry.call(tool, args, ctx, whitelist=state["allowed_tools"])

        # 写操作成功后登记，供下一轮规划层判断"这件事已经做过了"
        if tool in SIDE_EFFECT_TOOLS and result.ok:
            data = result.data if isinstance(result.data, dict) else {}
            if data.get("created") or data.get("found") or data.get("refund_id") or data.get("ticket_id"):
                self.kv.set(self._applied_key(args, tool), "1", ttl=_APPLIED_TTL_S)

        preview = str(result.data)[:400] if result.data is not None else ""
        self.trace.emit(
            state["trace_id"],
            "tool_result",
            "act",
            {
                "tool": tool,
                "ok": result.ok,
                "error": result.error,
                "degraded": result.degraded,
                "from_cache": result.from_cache,
                "preview": preview,
            },
            latency_ms=result.latency_ms,
        )
        records = list(state.get("tool_results") or [])
        records.append(
            {
                "name": result.name,
                "ok": result.ok,
                "data": result.data,
                "error": result.error,
                "degraded": result.degraded,
                "latency_ms": result.latency_ms,
            }
        )
        degraded = list(state.get("degraded") or [])
        if result.degraded:
            degraded.append(f"tool:{tool}:{result.error}")
        return {
            "tool_results": records,
            "iterations": state.get("iterations", 0) + 1,
            "degraded": degraded,
        }

    def _n_observe(self, state: AgentState) -> dict:
        records = state.get("tool_results") or []
        last = records[-1] if records else {}
        data = last.get("data") if isinstance(last.get("data"), dict) else {}
        verdict = "ok"
        if not last.get("ok"):
            verdict = "tool_failed"
        elif data.get("found") is False:
            verdict = "not_found"
        elif last.get("name") == "kb.search" and not data.get("results"):
            verdict = "empty_evidence"
        self.trace.emit(
            state["trace_id"],
            "observe",
            "observe",
            {"verdict": verdict, "iterations": state.get("iterations", 0)},
        )
        return {}

    def _collect_evidence(self, state: AgentState) -> tuple[list[str], list[dict], list[str]]:
        passages: list[str] = []
        citations: list[dict] = []
        decisions: list[str] = []  # 结论类（能不能退、能不能开票），优先给模型看
        details: list[str] = []    # 事实类（订单状态、物流轨迹）
        for record in state.get("tool_results") or []:
            data = record.get("data") if isinstance(record.get("data"), dict) else {}
            if record["name"] == "kb.search":
                for item in (data.get("results") or [])[:5]:
                    passages.append(f"{item['title']}：{item['text']}")
                    citations.append(
                        {"chunk_id": item["chunk_id"], "title": item["title"], "domain": item["domain"]}
                    )
            elif record["name"] == "order.query" and data.get("found"):
                details.append(
                    f"订单 {data['order_id']}：状态 {data['status_zh']}，商品 {data['sku']}，"
                    f"实付 {data['paid_amount']} 元，下单 {data['created_at']}，签收 {data.get('delivered_at') or '未签收'}"
                )
            elif record["name"] == "logistics.track" and data.get("found"):
                if data.get("has_logistics"):
                    details.append(
                        f"物流：{data.get('carrier')} {data.get('tracking_no')}，最新 {data.get('latest')}，"
                        f"更新于 {data.get('updated_at')}"
                    )
                else:
                    details.append("物流：商家尚未发货，暂无轨迹")
            elif record["name"] == "refund.apply" and data.get("found"):
                state_zh = "可退" if data.get("eligible") else "当前不可退"
                line = (
                    f"退款判断：{state_zh}，可退金额 {data.get('refund_amount')} 元，"
                    f"阻断项 {data.get('blockers') or '无'}，签收至今 {data.get('days_since_delivery')} 天"
                )
                if data.get("refund_id"):
                    line += f"，退款单号 {data['refund_id']}，当前状态 {data.get('state')}"
                if data.get("duplicate"):
                    line += "（幂等命中，本次未重复创建）"
                if data.get("compensated"):
                    line += "（下游失败，本地状态已回滚）"
                decisions.append(line)
            elif record["name"] == "return.create" and data.get("found"):
                if data.get("created"):
                    decisions.append(
                        f"退货单已创建 {data.get('return_id')}，签收至今 {data.get('days_since_delivery')} 天"
                    )
                else:
                    decisions.append(
                        f"退货单未创建，原因 {data.get('reason')}，签收至今 {data.get('days_since_delivery')} 天；"
                        f"{data.get('fallback', '')}"
                    )
            elif record["name"] == "promo.check" and data.get("found"):
                decisions.append(
                    f"价保判断：{'符合' if data.get('price_protection_eligible') else '不符合'}，"
                    f"窗口 {data.get('window_days')} 天，已过 {data.get('days_since_delivery')} 天"
                )
            elif record["name"] == "invoice.issue" and data.get("found"):
                decisions.append(
                    f"开票判断：{'可开' if data.get('issuable') else '不可开（' + str(data.get('reason')) + '）'}"
                )
        return passages, citations, decisions + details

    def _n_respond(self, state: AgentState) -> dict:
        passages, citations, facts = self._collect_evidence(state)

        if not passages:  # 没有政策证据就补一次检索，保证有据可依
            out = self.retriever.search(
                state["message"], top_k=self.settings.rerank_top_k, history=state.get("history")
            )
            passages = out.as_context()
            citations = [{"chunk_id": c.chunk_id, "title": c.title, "domain": c.domain} for c in out.chunks]
            self.trace.emit(
                state["trace_id"],
                "retrieval",
                "respond",
                {
                    "config": out.config,
                    "effective_query": out.effective_query,
                    "chunks": len(out.chunks),
                    "degraded": out.degraded,
                    "stages": out.stages,
                    "candidates": out.candidates,
                    "cache_hit": out.cache_hit,
                    "dropped_invalid": out.dropped_invalid,
                },
            )

        blocks_in = list(passages)
        if facts:
            blocks_in = ["单据数据：" + "；".join(facts)] + blocks_in

        _, report, blocks = self.assembler.assemble(
            system="售后客服回答生成",
            profile=state.get("profile") or {},
            summary=state.get("summary") or "",
            history=state.get("history") or [],
            retrieval=blocks_in,
            query=state["message"],
        )

        parts: list[str] = []
        if blocks.get("retrieval"):
            parts.append("参考资料：\n" + blocks["retrieval"])
        if blocks.get("summary"):
            parts.append(blocks["summary"])
        if blocks.get("history"):
            parts.append("最近对话：\n" + blocks["history"])
        parts.append("用户问题：" + state["message"])

        try:
            reply = self.llm.answer_conversation("\n\n".join(parts))
        except Exception as exc:  # noqa: BLE001 - 生成失败必须给兜底话术
            reply = "抱歉，我这边暂时没能组织出可靠的答复，已为你转接人工客服。"
            self.trace.emit(state["trace_id"], "error", "respond", {"error": str(exc)})
            state.setdefault("degraded", []).append("answer:llm_failed")

        self.trace.emit(state["trace_id"], "assembly", "respond", report.as_dict())
        return {"reply": reply, "citations": citations, "assembly": report.as_dict()}

    def _n_clarify(self, state: AgentState) -> dict:
        intent = state["intent"]
        spec = INTENT_SPECS[intent.label]
        hint = ""
        if intent.label in (Intent.ORDER_STATUS, Intent.REFUND, Intent.LOGISTICS, Intent.RETURN_EXCHANGE):
            hint = "（直接发订单号给我会更快，形如 SO20260512001）"
        reply = f"{spec.clarify}{hint}"
        self.trace.emit(
            state["trace_id"],
            "clarify",
            "clarify",
            {"reason": "low_margin", "confidence": intent.confidence, "margin": intent.margin},
        )
        return {"reply": reply, "clarify": True}

    def _n_handoff(self, state: AgentState) -> dict:
        intent = state["intent"]
        _, _, facts = self._collect_evidence(state)
        summary = f"用户诉求：{truncate_tokens(state['message'], 60)}"
        if facts:
            summary += "｜单据：" + "；".join(facts)[:200]
        ctx = ToolContext(
            user_id=state["user_id"],
            session_id=state["session_id"],
            trace_id=state["trace_id"],
            intent=intent,
        )
        result = self.registry.call(
            "ticket.create",
            {"reason": "high_urgency" if intent.urgency is Urgency.P0 else "auto_escalate",
             "summary": summary, "urgency": intent.urgency.value},
            ctx,
            whitelist=["ticket.create"],
        )
        ticket = result.data if isinstance(result.data, dict) else {}
        eta = "2 小时" if intent.urgency is Urgency.P0 else "24 小时"
        reply = (
            f"已为你转接人工客服并创建工单 {ticket.get('ticket_id', '（生成中）')}，"
            f"预计 {eta}内与你联系。你的问题我已完整记录，无需重复描述。"
        )
        self.trace.emit(
            state["trace_id"],
            "handoff",
            "handoff",
            {"ticket_id": ticket.get("ticket_id"), "urgency": intent.urgency.value, "ok": result.ok},
        )
        return {"reply": reply, "handoff": ticket}

    def _n_persist(self, state: AgentState) -> dict:
        sid, uid = state["session_id"], state["user_id"]
        intent = state.get("intent")
        self.short_term.append(sid, "user", state["message"], intent.label.value if intent else None)
        self.short_term.append(sid, "assistant", state.get("reply", ""), None)
        if intent is not None:
            self.profiles.bump(uid, intent.label.value)
        resolved = not state.get("handoff") and not state.get("clarify")
        if resolved and intent is not None:
            self.episodic.add(
                Episode(
                    episode_id=f"ep_{state['trace_id']}",
                    user_id=uid,
                    session_id=sid,
                    summary=f"{state['message'][:60]} → {state.get('reply', '')[:100]}",
                    intent=intent.label.value,
                )
            )
        self.trace.emit(
            state["trace_id"],
            "memory_write",
            "persist",
            {"resolved": resolved, "intent": intent.label.value if intent else None},
        )
        return {}

    # ---------- 对外入口 ----------

    def chat(self, request: ChatRequest, trace_id: str | None = None) -> ChatResponse:
        started = time.time()
        llm_calls_before = self.llm.calls
        trace_id = self.trace.start(
            trace_id,
            meta={
                "message": request.message,
                "session_id": request.session_id,
                "user_id": request.user_id,
                "engine": GRAPH_ENGINE,
            },
        )
        state: dict[str, Any] = {
            "trace_id": trace_id,
            "session_id": request.session_id,
            "user_id": request.user_id,
            "message": request.message,
            "tool_results": [],
            "iterations": 0,
            "degraded": [],
        }
        try:
            final = self.graph.invoke(state, config={"recursion_limit": 25})
        except Exception as exc:  # noqa: BLE001
            self.trace.emit(trace_id, "error", payload={"error": f"{type(exc).__name__}: {exc}"})
            final = {
                "reply": "系统开小差了，已记录本次会话，请稍后重试或直接转人工。",
                "degraded": (state.get("degraded") or []) + ["graph:crash"],
            }

        duration = int((time.time() - started) * 1000)
        intent = final.get("intent")
        response = ChatResponse(
            trace_id=trace_id,
            session_id=request.session_id,
            reply=final.get("reply", ""),
            intent=intent if intent else None,
            tools_used=[r["name"] for r in final.get("tool_results") or []],
            degraded=bool(final.get("degraded")),
            handoff=final.get("handoff"),
            citations=final.get("citations") or [],
            usage={
                "latency_ms": duration,
                "llm_calls": self.llm.calls - llm_calls_before,
                "assembly": final.get("assembly") or {},
                "degraded": final.get("degraded") or [],
                "memory": final.get("memory_report") or {},
                "engine": GRAPH_ENGINE,
            },
        )
        self.trace.finish(
            trace_id,
            {
                "duration_ms": duration,
                "reply": response.reply,
                "reply_preview": response.reply[:120],
                "degraded": response.degraded,
                "intent": response.intent.label.value if response.intent else None,
                "tools": response.tools_used,
            },
        )
        self.trace.emit(trace_id, "usage", payload=response.usage, latency_ms=duration)
        return response

    def stats(self) -> dict[str, Any]:
        return {
            "engine": GRAPH_ENGINE,
            "llm": {"calls": self.llm.calls, "prompt_tokens": self.llm.prompt_tokens,
                    "completion_tokens": self.llm.completion_tokens},
            "intent_cache": self.intent_cache.stats.as_dict(),
            "retrieval_cache": self.retrieval_cache.stats.as_dict(),
            "tool_cache": self.tool_cache.stats.as_dict(),
            "breaker": {
                "bm25": self.retriever.bm25_breaker.snapshot(),
                "dense": self.retriever.dense_breaker.snapshot(),
                "rerank": self.retriever.rerank_breaker.snapshot(),
            },
            "index": {"chunks": self.index.size, "domains": self.index.domains()},
            "memory": {"compressions": self.short_term.compressions},
            "metrics": self.trace.metrics.snapshot(),
            "ts": now_ms(),
        }


_runtime: AgentRuntime | None = None


def get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = AgentRuntime()
    return _runtime
