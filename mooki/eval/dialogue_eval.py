"""端到端对话评测：跑完整 Agent 链路，用 LLM-as-Judge 打分。

为什么必须同时报"关键词命中率"：
LLM-as-Judge 与人工的一致性本身要验证，否则就是一个不可信的打分器。
这里用 must_have 关键词做一个廉价的、完全确定性的下界检验 ——
如果 judge 打了 5 分但关键词一个没命中，说明 judge 不可信，这条要标出来。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from mooki.core.schema import ChatRequest

GOLDEN = Path(__file__).resolve().parent / "golden" / "dialogue.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "eval_out"
PASS_SCORE = 4


def load_golden() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_dialogue_eval(runtime, session_prefix: str = "eval", save: bool = True) -> dict:
    rows = load_golden()
    details = []
    for row in rows:
        response = runtime.chat(
            ChatRequest(message=row["message"], session_id=f"{session_prefix}-{row['id']}", user_id="u_1001")
        )
        judge = runtime.llm.judge(row["message"], response.reply, row["reference"]) or {}
        score = int(judge.get("score", 0) or 0)
        hits = [kw for kw in row.get("must_have", []) if kw in response.reply]
        details.append(
            {
                "id": row["id"],
                "message": row["message"],
                "reply": response.reply,
                "score": score,
                "judge_reason": judge.get("reason", ""),
                "must_have": row.get("must_have", []),
                "must_have_hit": hits,
                "keyword_pass": len(hits) == len(row.get("must_have", [])),
                "intent": response.intent.label.value if response.intent else None,
                "tools": response.tools_used,
                "degraded": response.degraded,
                "latency_ms": response.usage.get("latency_ms"),
            }
        )

    scores = [d["score"] for d in details]
    report = {
        "n": len(details),
        "mean_score": round(statistics.mean(scores), 3) if scores else 0.0,
        "median_score": statistics.median(scores) if scores else 0,
        "pass_rate": round(sum(1 for s in scores if s >= PASS_SCORE) / len(scores), 4) if scores else 0.0,
        "keyword_pass_rate": round(
            sum(1 for d in details if d["keyword_pass"]) / len(details), 4
        ) if details else 0.0,
        "judge_keyword_conflict": [
            d["id"] for d in details if d["score"] >= PASS_SCORE and not d["keyword_pass"]
        ],
        "degraded_rate": round(sum(1 for d in details if d["degraded"]) / len(details), 4) if details else 0.0,
        "p95_latency_ms": sorted(d["latency_ms"] or 0 for d in details)[
            min(len(details) - 1, int(0.95 * len(details)))
        ] if details else 0,
        "details": details,
    }

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "dialogue_eval.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT_DIR / "dialogue_eval.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    lines = [
        f"## 端到端对话评测（n={report['n']}，LLM-as-Judge 1-5 分）",
        "",
        f"- 平均分: **{report['mean_score']}**，中位数 {report['median_score']}",
        f"- 通过率（≥{PASS_SCORE} 分）: **{report['pass_rate']:.2%}**",
        f"- 关键词命中率（确定性下界）: {report['keyword_pass_rate']:.2%}",
        f"- Judge 与关键词冲突条目: {report['judge_keyword_conflict'] or '无'}",
        f"- 降级率: {report['degraded_rate']:.2%}，P95 延迟 {report['p95_latency_ms']} ms",
        "",
        "| # | 问题 | 分 | 意图 | 工具 | 关键词 |",
        "|---|---|---|---|---|---|",
    ]
    for d in report["details"]:
        tools = ",".join(d["tools"]) or "-"
        lines.append(
            f"| {d['id']} | {d['message'][:24]} | {d['score']} | {d['intent']} | {tools} | "
            f"{'✓' if d['keyword_pass'] else '✗'} |"
        )
    return "\n".join(lines) + "\n"


render_dialogue_markdown = render_markdown
