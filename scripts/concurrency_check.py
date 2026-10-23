"""退款幂等并发压测。

要回答的问题只有一个：同一笔退款请求被并发重试时，会不会退两次。

实测结论比预期更糟，也更值得记下来：

- jsonl 后端连**线程**都挡不住。`threading.Lock` 是 OrderLedger 实例上的锁，
  而每个请求各建一个 ServiceTools（多 worker、或每请求新建）就各持一把锁，
  8 并发直接落 8 条退款事实、8 个退款单号。
- SQLite 后端线程与进程下都只有 1 条成功，其余命中 `idem_key UNIQUE` 返回首次结果。

所以问题不在于"线程安全没做好"，而在于**互斥放错了层**：幂等键的唯一性
本来就该由存储层保证，进程内锁只能算优化。

用法：python -m scripts.concurrency_check [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path

from mooki.core.schema import IntentResult, Intent, Urgency
from mooki.tools.builtin.service_tools import ServiceTools
from mooki.tools.registry import ToolContext

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = ROOT / "eval_out"
ORDER_ID = "SO20261018006"
USER_ID = "u_1001"


def _ctx() -> ToolContext:
    return ToolContext(
        user_id=USER_ID,
        session_id="concurrency",
        trace_id="concurrency",
        intent=IntentResult(label=Intent.REFUND, urgency=Urgency.P2, confidence=1.0, margin=1.0),
    )


def _hit(ledger_path: str) -> dict:
    tools = ServiceTools(DATA, ledger_path=Path(ledger_path))
    out = tools.refund_apply({"order_id": ORDER_ID, "reason": "no_reason"}, _ctx())
    return {
        "pid": os.getpid(),
        "ok": bool(out.get("eligible")),
        "duplicate": bool(out.get("duplicate")),
        "refund_id": out.get("refund_id"),
        "blockers": out.get("blockers") or [],
    }


def _thread_round(ledger_path: Path, workers: int) -> dict:
    results: list[dict] = []
    lock = threading.Lock()

    def run() -> None:
        r = _hit(str(ledger_path))
        with lock:
            results.append(r)

    threads = [threading.Thread(target=run) for _ in range(workers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = (time.perf_counter() - t0) * 1000
    return _summarize(results, elapsed, workers, "threads")


def _process_round(ledger_path: Path, workers: int) -> dict:
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(_hit, [str(ledger_path)] * workers)
    elapsed = (time.perf_counter() - t0) * 1000
    return _summarize(results, elapsed, workers, "processes")


def _summarize(results: list[dict], elapsed_ms: float, workers: int, mode: str) -> dict:
    created = [r for r in results if r["ok"] and not r["duplicate"]]
    duplicated = [r for r in results if r["duplicate"]]
    refund_ids = {r["refund_id"] for r in created if r["refund_id"]}
    return {
        "mode": mode,
        "workers": workers,
        "elapsed_ms": round(elapsed_ms, 1),
        "created": len(created),
        "duplicate": len(duplicated),
        "distinct_refund_ids": len(refund_ids),
        "blocked": len([r for r in results if not r["ok"] and not r["duplicate"]]),
        "double_refund": len(refund_ids) > 1,
    }


def run(workers: int = 16) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="mooki-concurrency-"))
    rows: list[dict] = []

    for backend, suffix in (("sqlite", ".db"), ("jsonl", ".jsonl")):
        # 每轮都用一个全新账本，避免上一轮的状态影响下一轮
        for mode, fn in (("threads", _thread_round), ("processes", _process_round)):
            path = tmp / f"ledger-{backend}-{mode}{suffix}"
            rows.append({"backend": backend, **fn(path, workers)})

    report = {
        "workers": workers,
        "order_id": ORDER_ID,
        "rounds": rows,
        "conclusion": {
            "sqlite_process_safe": all(
                not r["double_refund"] for r in rows if r["backend"] == "sqlite"
            ),
            "jsonl_cross_process_unsafe": any(
                r["double_refund"] for r in rows
                if r["backend"] == "jsonl" and r["mode"] == "processes"
            ),
        },
    }
    _write(report)
    return report


def _write(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "concurrency.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"## 退款幂等并发压测（{report['workers']} 并发，同一订单 {report['order_id']}）",
        "",
        "| 账本后端 | 并发方式 | 成功创建 | 命中幂等 | 退款单号数 | 是否重复退款 | 耗时 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["rounds"]:
        lines.append(
            f"| {r['backend']} | {r['mode']} | {r['created']} | {r['duplicate']} | "
            f"{r['distinct_refund_ids']} | {'**是**' if r['double_refund'] else '否'} | "
            f"{r['elapsed_ms']:.0f} ms |"
        )
    lines += [
        "",
        f"- SQLite 后端（线程 + 进程）：{'全部通过' if report['conclusion']['sqlite_process_safe'] else '存在重复退款'}",
        f"- jsonl 后端跨进程："
        f"{'复现出重复退款（预期）' if report['conclusion']['jsonl_cross_process_unsafe'] else '未复现（机器太快，建议加大并发）'}",
        "",
        "> jsonl 版的 `threading.Lock` 是账本实例上的锁，而每个请求各建一个 ServiceTools",
        "> 就各持一把锁 —— 8 并发直接落 8 条退款事实。问题不是并发压力不够，",
        "> 是互斥放错了层：幂等键的唯一性应该由存储层保证。",
    ]
    (OUT_DIR / "concurrency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    report = run(args.workers)
    for r in report["rounds"]:
        print(
            f"{r['backend']:>6} {r['mode']:>9}  created={r['created']} dup={r['duplicate']} "
            f"ids={r['distinct_refund_ids']} double_refund={r['double_refund']} {r['elapsed_ms']}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
