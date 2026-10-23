"""多轮会话评测：验的是"跨轮的东西没丢、也没重复做"。

单轮评测看不出下面这两类错，而它们恰好是售后多轮里最容易出事的地方：

1. 槽位丢失 —— 第二句"那钱多久到账"不带单号，系统得自己记得 SO 号。
   用滚动摘要压缩历史时，订单号是第一个被摘要抹掉的东西。
2. 副作用重放 —— 同一诉求追问一次，规划层又照着链路重跑一遍 refund.apply。
   账本幂等键能挡住重复扣钱，但工具白跑 + 答复退化成复读，用户是能感知的。

所以这里断言三件事：每轮都要有答复、关键槽位在后续轮次仍可解析、
写操作整段会话只出现一次。
"""

from __future__ import annotations

import json
from pathlib import Path

from mooki.core.schema import ChatRequest

GOLDEN = Path(__file__).resolve().parent / "golden" / "multiturn.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "eval_out"

SIDE_EFFECT_TOOLS = {"refund.apply", "refund.cancel", "return.create", "invoice.issue", "ticket.create"}
ORDER_SCOPED_TOOLS = {"order.query", "logistics.track", "refund.apply", "refund.cancel", "return.create", "invoice.issue", "promo.check"}


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_multiturn_eval(runtime, session_prefix: str = "mt", save: bool = True) -> dict:
    cases = []
    for case in load_golden():
        session_id = f"{session_prefix}-{case['id']}"
        turns = []
        for idx, text in enumerate(case["turns"]):
            resp = runtime.chat(
                ChatRequest(message=text, session_id=session_id, user_id=case.get("user_id", "u_1001"))
            )
            turns.append(
                {
                    "turn": idx + 1,
                    "message": text,
                    "reply": resp.reply,
                    "intent": resp.intent.label.value if resp.intent else None,
                    "tools": resp.tools_used,
                    "handoff": bool(resp.handoff),
                }
            )

        # 断言 1：每轮都得有非空答复，不允许静默失败
        empty_turns = [t["turn"] for t in turns if not (t["reply"] or "").strip()]

        # 断言 2：槽位在后几轮仍可用 —— 看该轮是否还能规划出「需要订单号」的工具。
        # 不比对具体单号字符串，是因为答复模板不一定复述单号；工具选择才是槽位真正生效的证据。
        slot_failures = []
        for check in case.get("slot_checks", []):
            probe_turn = turns[check["turn"] - 1]
            if check.get("order_scoped") and not (set(probe_turn["tools"]) & ORDER_SCOPED_TOOLS):
                slot_failures.append({"turn": check["turn"], "missing": "order_scoped_tool"})

        # 断言 3：写操作整段会话只允许出现一次
        side_effect_counts: dict[str, int] = {}
        for t in turns:
            for name in t["tools"]:
                if name in SIDE_EFFECT_TOOLS:
                    side_effect_counts[name] = side_effect_counts.get(name, 0) + 1
        replayed = {k: v for k, v in side_effect_counts.items() if v > 1}

        # 断言 4：指定轮次必须命中关键词
        keyword_failures = []
        for check in case.get("must_include", []):
            probe_turn = turns[check["turn"] - 1]
            missing = [kw for kw in check["keywords"] if kw not in probe_turn["reply"]]
            if missing:
                keyword_failures.append({"turn": check["turn"], "missing": missing})

        cases.append(
            {
                "id": case["id"],
                "desc": case.get("desc", ""),
                "turns": turns,
                "slot_checks": len(case.get("slot_checks", [])),
                "empty_turns": empty_turns,
                "slot_failures": slot_failures,
                "replayed_side_effects": replayed,
                "keyword_failures": keyword_failures,
                "passed": not empty_turns and not slot_failures and not replayed and not keyword_failures,
            }
        )

    slot_total = sum(c["slot_checks"] for c in cases)
    slot_miss = sum(len(c["slot_failures"]) for c in cases)
    report = {
        "n_cases": len(cases),
        "n_turns": sum(len(c["turns"]) for c in cases),
        "pass_rate": round(sum(1 for c in cases if c["passed"]) / len(cases), 4) if cases else 0.0,
        "slot_retention_rate": round(1 - slot_miss / slot_total, 4) if slot_total else 1.0,
        "slot_checks": slot_total,
        "side_effect_replay_cases": [c["id"] for c in cases if c["replayed_side_effects"]],
        "failed_cases": [c["id"] for c in cases if not c["passed"]],
        "details": cases,
    }

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "multiturn_eval.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT_DIR / "multiturn_eval.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    lines = [
        f"## 多轮会话评测（{report['n_cases']} 组会话 / {report['n_turns']} 轮）",
        "",
        f"- 会话通过率: **{report['pass_rate']:.2%}**",
        f"- 槽位保留率: **{report['slot_retention_rate']:.2%}**（{report['slot_checks']} 项断言）",
        f"- 写操作重放会话: {report['side_effect_replay_cases'] or '无'}",
        f"- 未通过会话: {report['failed_cases'] or '无'}",
        "",
        "| 会话 | 轮数 | 槽位 | 写操作次数 | 结论 |",
        "|---|---|---|---|---|",
    ]
    for c in report["details"]:
        slots = "✓" if not c["slot_failures"] else f"缺 {[f['missing'] for f in c['slot_failures']]}"
        effects = ", ".join(f"{k}×{v}" for k, v in c["replayed_side_effects"].items()) or "-"
        lines.append(
            f"| {c['id']} | {len(c['turns'])} | {slots} | {effects} | {'通过' if c['passed'] else '未通过'} |"
        )
    lines.append("")
    lines.append("> 写操作次数为整段会话内的调用次数，>1 即视为重放。幂等键保证不重复扣钱，")
    lines.append("> 但重放本身说明规划层没有读账本状态。")
    return "\n".join(lines) + "\n"
