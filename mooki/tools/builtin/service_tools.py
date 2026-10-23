"""售后业务工具实现。

约定：
- 所有 handler 只关心业务，不写缓存/熔断/超时（那是 registry 的事）；
- 查询类工具对"无权访问"统一返回 found=False，不区分"不存在"和"不属于你"，
  避免通过订单号探测他人数据；
- 交易类工具不落库改状态（本项目无真实交易系统），而是返回可执行性与阻断原因，
  由 Agent 组织话术，真实环境在这里接工单/支付网关。
"""

from __future__ import annotations

import datetime as dt
import json
import random
import uuid
from pathlib import Path
from typing import Any

from mooki.tools.ledger import OrderLedger, derive_idem
from mooki.tools.registry import ToolContext, ToolRegistry, ToolSpec

NO_REASON_DAYS = 7

# 账本里出现的新状态 -> 中文
EXTRA_STATUS_ZH = {
    "refund_pending": "退款处理中",
    "refunded": "已退款",
}


class ServiceTools:
    def __init__(
        self,
        data_dir: Path,
        ticket_path: Path | None = None,
        ledger_path: Path | None = None,
        refund_fault_rate: float = 0.0,
    ):
        order_file = data_dir / "orders.json"
        self.orders: dict[str, dict] = json.loads(order_file.read_text(encoding="utf-8"))
        self.ticket_path = ticket_path or (data_dir / "tickets.jsonl")
        self.ledger = OrderLedger(ledger_path or (data_dir / "ledger.jsonl"))
        self.refund_fault_rate = refund_fault_rate
        self._gateway_fail_next = False  # 供测试注入确定性故障

    # ---------- helpers ----------

    def _order(self, order_id: str | None) -> dict | None:
        if not order_id:
            return None
        return self.orders.get(order_id)

    @staticmethod
    def _days_since(date_str: str | None) -> int | None:
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                then = dt.datetime.strptime(date_str.strip(), fmt)
                return (dt.datetime.now() - then).days
            except ValueError:
                continue
        return None

    # ---------- 查询类 ----------

    def order_query(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        order = self._order(args.get("order_id"))
        if order is None:
            return {"found": False}
        if order["user_id"] != ctx.user_id:
            return {"found": False}
        # 账本状态覆盖单据状态：退款中/已退款 只存在于账本，orders.json 是快照
        status = self.ledger.status(order["order_id"], order["status"]) or order["status"]
        snap = self.ledger.snapshot(order["order_id"])
        return {
            "found": True,
            "order_id": order["order_id"],
            "status": status,
            "status_zh": EXTRA_STATUS_ZH.get(status) or order["status_zh"],
            "sku": order["sku"],
            "amount": order["amount"],
            "paid_amount": order["paid_amount"],
            "created_at": order["created_at"],
            "delivered_at": order["delivered_at"],
            "invoice_status": order["invoice"].get("status"),
            "logistics_no": (order.get("logistics") or {}).get("tracking_no"),
            "refund_id": snap.get("refund_id"),
            "refunded": bool(snap.get("refunded")),
        }

    def logistics_track(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        order = self._order(args.get("order_id"))
        if order is None or order["user_id"] != ctx.user_id:
            return {"found": False}
        logistics = order.get("logistics")
        if not logistics:
            return {"found": True, "has_logistics": False, "hint": "商家尚未发货，暂无物流信息"}
        return {"found": True, "has_logistics": True, **logistics}

    def _gateway_settle(self, order_id: str, amount: float) -> tuple[bool, str]:
        """模拟支付网关预授权。故障可注入，用于验证补偿链路。"""
        if self._gateway_fail_next:
            self._gateway_fail_next = False
            return False, "gateway_timeout"
        if self.refund_fault_rate and random.random() < self.refund_fault_rate:
            return False, "gateway_rejected"
        return True, ""

    def refund_apply(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        order = self._order(args.get("order_id"))
        if order is None or order["user_id"] != ctx.user_id:
            return {"found": False}
        oid = order["order_id"]
        blockers: list[str] = []
        if order["status"] == "unpaid":
            blockers.append("order_unpaid")
        if not order["refundable"]:
            blockers.append("not_refundable")
        if order["invoice"].get("status") == "issued":
            blockers.append("invoice_issued_need_reversal")

        days = self._days_since(order.get("delivered_at"))
        reason = args.get("reason") or "no_reason"
        if order["status"] == "delivered" and reason == "no_reason":
            if days is not None and days > NO_REASON_DAYS:
                blockers.append(f"exceed_no_reason_window:{days}d")

        base = {
            "found": True,
            "order_id": oid,
            "refund_amount": order["paid_amount"],
            "days_since_delivery": days,
        }
        if blockers:
            return {**base, "eligible": False, "blockers": blockers, "next_steps": []}

        # 走到这里才真正落账：先写 refund.request，再打网关，失败则补偿回滚
        idem = args.get("idempotency_key") or derive_idem(ctx.user_id, oid, "refund.request")
        record, error, duplicate = self.ledger.apply(
            "refund.request",
            oid,
            idem,
            {"refund_id": "RF" + uuid.uuid4().hex[:10].upper(), "amount": order["paid_amount"], "reason": reason},
            base_status=order["status"],
        )
        if error:
            return {**base, "eligible": False, "blockers": [error], "next_steps": []}

        refund_id = record.payload.get("refund_id") if record else None
        settled, gw_error = self._gateway_settle(oid, order["paid_amount"])
        if not settled:
            # 补偿：本地状态回滚，避免"平台退款中、网关没收到"的脏数据
            comp_key = derive_idem(ctx.user_id, oid, "refund.cancel", gw_error)
            self.ledger.apply(
                "refund.cancel", oid, comp_key, {"reason": f"gateway_failed:{gw_error}"},
                base_status="refund_pending",
            )
            return {
                **base,
                "eligible": False,
                "blockers": [f"gateway:{gw_error}"],
                "refund_id": refund_id,
                "compensated": True,
                "state": "rolled_back",
                "next_steps": ["稍后重试", "或转人工由专员处理"],
            }

        return {
            **base,
            "eligible": True,
            "blockers": [],
            "refund_id": refund_id,
            "state": self.ledger.status(oid) or "refund_pending",
            "duplicate": duplicate,
            "next_steps": ["等待商家审核（48 小时内）", "审核通过后原路退回"],
        }

    def refund_cancel(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        order = self._order(args.get("order_id"))
        if order is None or order["user_id"] != ctx.user_id:
            return {"found": False}
        oid = order["order_id"]
        idem = args.get("idempotency_key") or derive_idem(ctx.user_id, oid, "refund.cancel")
        record, error, duplicate = self.ledger.apply(
            "refund.cancel", oid, idem, {"reason": args.get("reason") or "user_cancel"},
            base_status=order["status"],
        )
        if error:
            return {"found": True, "cancelled": False, "reason": error}
        return {
            "found": True,
            "cancelled": True,
            "duplicate": duplicate,
            "state": self.ledger.status(oid),
            "refund_id": record.payload.get("refund_id") if record else None,
        }

    def return_create(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        order = self._order(args.get("order_id"))
        if order is None or order["user_id"] != ctx.user_id:
            return {"found": False}
        if order["status"] != "delivered":
            return {"found": True, "created": False, "reason": "not_delivered"}
        days = self._days_since(order.get("delivered_at"))
        reason = args.get("reason") or "no_reason"
        if reason == "no_reason" and days is not None and days > NO_REASON_DAYS:
            return {
                "found": True,
                "created": False,
                "reason": f"exceed_no_reason_window:{days}d",
                "days_since_delivery": days,
                "window_days": NO_REASON_DAYS,
                "fallback": "如存在质量问题，可按质量问题申请退换，不受 7 天限制",
            }
        return {
            "found": True,
            "created": True,
            "return_id": "RT" + uuid.uuid4().hex[:10].upper(),
            "reason": reason,
            "days_since_delivery": days,
            "need_ship_back": True,
            "address_hint": "退回地址将在审核通过后展示",
        }

    def invoice_issue(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        order = self._order(args.get("order_id"))
        if order is None or order["user_id"] != ctx.user_id:
            return {"found": False}
        status = order["invoice"].get("status")
        if status == "issued":
            return {"found": True, "issuable": False, "reason": "already_issued"}
        return {
            "found": True,
            "issuable": True,
            "supported_types": ["electronic"],
            "hint": "申请后 24 小时内开具，可在发票管理下载",
        }

    def promo_check(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        order = self._order(args.get("order_id"))
        if order is None or order["user_id"] != ctx.user_id:
            return {"found": False}
        days = self._days_since(order.get("delivered_at"))
        within = days is not None and days <= 15
        return {
            "found": True,
            "price_protection_eligible": within and order["status"] == "delivered",
            "days_since_delivery": days,
            "window_days": 15,
        }

    # ---------- 知识检索 ----------

    def kb_search(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        from mooki.intent.labels import domain_for

        retriever = ctx.services.get("retriever")
        if retriever is None:
            return {"available": False, "results": []}
        query = str(args.get("query") or "").strip()
        if not query:
            return {"available": True, "results": [], "hint": "empty_query"}
        filters = args.get("filters") or None
        intent = getattr(ctx.intent, "label", None)
        out = retriever.search(
            query,
            top_k=int(args.get("top_k") or 5),
            # 默认 full 而非 full_prior：域先验在 R@1 上只有 +0.025（40 条里多命中 1 条），
            # 属于噪声量级，不够格换默认值。等语料规模上去再重新评估。
            config_name=args.get("config") or "full",
            filters=filters,
            domain_prior=domain_for(intent),
            # 多轮里 plan 阶段的检索必须带上历史：追问句（"三天没更新了怎么办"）
            # 本身没有领域信息，不带历史会一路漂到退款/改地址那些通用条款上。
            history=ctx.services.get("history") or [],
        )
        return {
            "available": True,
            "effective_query": out.effective_query,
            "config": out.config,
            "domain_prior": domain_for(intent),
            "degraded": out.degraded,
            "dropped_invalid": out.dropped_invalid,
            "results": [
                {
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "domain": c.domain,
                    "text": c.text,
                    "score": round(c.rrf_score, 6),
                }
                for c in out.chunks
            ],
        }

    # ---------- 转人工 ----------

    def ticket_create(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        ticket = {
            "ticket_id": "TK" + uuid.uuid4().hex[:10].upper(),
            "user_id": ctx.user_id,
            "session_id": ctx.session_id,
            "trace_id": ctx.trace_id,
            "reason": args.get("reason") or "customer_request",
            "summary": args.get("summary") or "",
            "urgency": args.get("urgency") or "P2",
            "intent": getattr(getattr(ctx.intent, "label", None), "value", "other"),
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        self.ticket_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ticket_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(ticket, ensure_ascii=False) + "\n")
        return ticket


# ---------- 注册 ----------

def build_registry(settings, cache, service: ServiceTools) -> ToolRegistry:
    registry = ToolRegistry(settings, cache)
    read_ttl = 60
    registry.register(
        ToolSpec(
            name="order.query",
            description="按订单号查询订单状态、金额、发货与签收时间。当用户未提供订单号时返回 need_order_id。",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单号，形如 SO 开头加数字"}},
                "required": [],
            },
            handler=service.order_query,
            cache_ttl=read_ttl,
        )
    )
    registry.register(
        ToolSpec(
            name="logistics.track",
            description="查询订单的物流轨迹与最新状态。",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            handler=service.logistics_track,
            cache_ttl=30,
        )
    )
    registry.register(
        ToolSpec(
            name="refund.apply",
            description="判断订单是否可退款；可退则写入账本创建退款单，并按幂等键防重复提交。",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string", "enum": ["no_reason", "quality", "not_received", "wrong_item"]},
                    "idempotency_key": {"type": "string", "description": "客户端幂等键，缺省按用户+单据+动作+5 分钟桶推导"},
                },
                "required": ["order_id"],
            },
            handler=service.refund_apply,
            risk="write",
            cache_ttl=None,
        )
    )
    registry.register(
        ToolSpec(
            name="refund.cancel",
            description="撤销退款申请（补偿动作），把订单从退款处理中回滚到原状态。",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
            handler=service.refund_cancel,
            risk="write",
            cache_ttl=None,
        )
    )
    registry.register(
        ToolSpec(
            name="return.create",
            description="创建退货单，仅已签收订单可用。",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
            handler=service.return_create,
            risk="write",
            cache_ttl=None,
        )
    )
    registry.register(
        ToolSpec(
            name="invoice.issue",
            description="判断订单能否开票并返回支持的发票类型。",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            handler=service.invoice_issue,
            risk="write",
            cache_ttl=None,
        )
    )
    registry.register(
        ToolSpec(
            name="promo.check",
            description="查询订单的价保资格与剩余窗口天数。",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            handler=service.promo_check,
            cache_ttl=read_ttl,
        )
    )
    registry.register(
        ToolSpec(
            name="kb.search",
            description="检索平台售后知识库（退款、退换货、物流、发票、优惠、保修等政策）。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询，建议使用改写后的短查询"},
                    "top_k": {"type": "integer", "default": 5},
                    "filters": {"type": "object", "description": "metadata 过滤，如 {\"domain\":\"refund\"}"},
                },
                "required": ["query"],
            },
            handler=service.kb_search,
            cache_ttl=120,
        )
    )
    registry.register(
        ToolSpec(
            name="ticket.create",
            description="创建转人工工单，写入工单库并返回工单号。",
            input_schema={
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "summary": {"type": "string"},
                    "urgency": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
                },
                "required": ["reason"],
            },
            handler=service.ticket_create,
            risk="write",
            cache_ttl=None,
        )
    )
    return registry
