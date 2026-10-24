"""在线模式评测：换真实 LLM 与 embedding 跑同一套题，看延迟、成本与观感。

与离线评测的分工要说清楚：
- 离线评测（scripts/run_eval.py）用来比较配置 —— 确定性替身，同一份输入永远
  同一份输出，消融才成立；
- 在线评测用来回答"真模型接上去到底行不行"，报的是绝对观感和成本，
  **不能**拿来和离线数字比大小，也不该拿来给多个配置排名 —— 温度不为 0，
  跑两次结果就不一样。

用法：
    cp .env.example .env && 填 MOOKI_LLM_BASE_URL / MOOKI_LLM_API_KEY
    MOOKI_LLM_OFFLINE=false python -m scripts.run_online_eval
    python -m scripts.run_online_eval --limit 8   # 先小样本试跑，省 token

没有 key 时脚本直接退出，不会静默降级成离线 —— 否则报告会看起来"跑过了"。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "eval_out"
DIALOGUE = ROOT / "mooki" / "eval" / "golden" / "dialogue.jsonl"
RETRIEVAL = ROOT / "mooki" / "eval" / "golden" / "retrieval.jsonl"

# 单价从配置读（MOOKI_PRICE_IN_PER_1K / MOOKI_PRICE_OUT_PER_1K），
# 没配就按 0 报，避免拿拍脑袋的数字当账单。


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return round(ordered[idx], 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条端到端样本")
    args = parser.parse_args()

    from mooki.core.config import Settings
    from mooki.core.schema import ChatRequest
    from mooki.graph.runtime import AgentRuntime

    settings = Settings()
    if settings.llm_offline or not settings.llm_api_key:
        print(
            "未配置在线模型。请在 .env 里填 MOOKI_LLM_OFFLINE=false、MOOKI_LLM_BASE_URL、"
            "MOOKI_LLM_API_KEY 后重跑。",
            file=sys.stderr,
        )
        return 2

    ledger = Path(tempfile.mkdtemp(prefix="mooki-online-")) / "ledger.db"
    runtime = AgentRuntime(settings=Settings(ledger_file=ledger))
    print(f"在线模式：model={settings.llm_model} embed={settings.embed_backend}\n")

    rows = _load_jsonl(DIALOGUE)
    if args.limit:
        rows = rows[: args.limit]

    details, latencies = [], []
    for row in rows:
        t0 = time.perf_counter()
        resp = runtime.chat(
            ChatRequest(message=row["message"], session_id=f"online-{row['id']}", user_id="u_1001")
        )
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        judge = runtime.llm.judge(row["message"], resp.reply, row["reference"]) or {}
        hits = [kw for kw in row.get("must_have", []) if kw in resp.reply]
        details.append(
            {
                "id": row["id"],
                "message": row["message"],
                "reply": resp.reply,
                "score": int(judge.get("score", 0) or 0),
                "judge_reason": judge.get("reason", ""),
                "keyword_pass": len(hits) == len(row.get("must_have", [])),
                "intent": resp.intent.label.value if resp.intent else None,
                "tools": resp.tools_used,
                "latency_ms": round(ms, 1),
                "llm_calls": resp.usage.get("llm_calls"),
            }
        )

    usage = runtime.llm.usage()
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cost = (
        prompt_tokens / 1000 * settings.price_in_per_1k
        + completion_tokens / 1000 * settings.price_out_per_1k
    )
    scores = [d["score"] for d in details]

    report = {
        "mode": "online",
        "model": settings.llm_model,
        "embed_backend": settings.embed_backend,
        "n": len(details),
        "mean_score": round(statistics.mean(scores), 3) if scores else 0.0,
        "pass_rate": round(sum(1 for s in scores if s >= 4) / len(scores), 4) if scores else 0.0,
        "keyword_pass_rate": round(
            sum(1 for d in details if d["keyword_pass"]) / len(details), 4
        ) if details else 0.0,
        "latency_p50_ms": _pct(latencies, 0.5),
        "latency_p95_ms": _pct(latencies, 0.95),
        "llm_calls": usage.get("calls"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "est_cost_cny": round(cost, 4),
        "est_cost_per_turn_cny": round(cost / len(details), 4) if details else 0.0,
        "details": details,
    }
    _write(report)
    print(f"端到端 {report['n']} 条：均分 {report['mean_score']}，"
          f"通过率 {report['pass_rate']:.1%}，P95 {report['latency_p95_ms']} ms，"
          f"单轮约 {report['est_cost_per_turn_cny']} 元")
    return 0


def _write(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "online_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"## 在线模式评测（{report['model']}，n={report['n']}）",
        "",
        "> 与离线报告不可比：温度不为 0，同一句问两次答复不同；这里看的是绝对观感与成本。",
        "",
        f"- 平均分 **{report['mean_score']}**，通过率 {report['pass_rate']:.1%}",
        f"- 关键词命中率 {report['keyword_pass_rate']:.1%}",
        f"- 延迟 P50 **{report['latency_p50_ms']} ms** / P95 **{report['latency_p95_ms']} ms**",
        f"- LLM 调用 {report['llm_calls']} 次，prompt {report['prompt_tokens']} / completion {report['completion_tokens']} tokens",
        f"- 本轮总成本约 {report['est_cost_cny']} 元，单轮约 {report['est_cost_per_turn_cny']} 元",
        "",
        "| # | 问题 | 分 | 意图 | 工具 | 延迟 | 关键词 |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in report["details"]:
        lines.append(
            f"| {d['id']} | {d['message'][:22]} | {d['score']} | {d['intent']} | "
            f"{','.join(d['tools']) or '-'} | {d['latency_ms']:.0f} ms | "
            f"{'✓' if d['keyword_pass'] else '✗'} |"
        )
    (OUT_DIR / "online_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
