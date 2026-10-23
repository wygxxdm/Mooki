"""在线检索对照：把检索链路里的 LLM 侧（改写 + 重排）换成真实模型。

离线消融只能比较配置之间的相对差异，回答不了最要紧的那个问题：
**"短口语问句检索不准，到底是架构问题还是离线替身的问题？"**

这里拿同一批 query、同一个索引（LSA 向量不动），只把改写与重排换成
DeepSeek，看 Recall 有没有变化。样本默认取全集的 1/3，控制调用次数 ——
每条 query 要 2 次 LLM 调用（改写 + 重排），全量 137 条会跑二十分钟。

用法：
    python -m scripts.online_retrieval_ablation            # 抽样 1/3
    python -m scripts.online_retrieval_ablation --all      # 全量
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

from mooki.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from mooki.eval.retrieval_eval import load_golden, resolve_relevant

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "eval_out"
TOP_K = 5


def _eval(rows, retriever, index) -> list[dict]:
    out = []
    for row in rows:
        relevant = resolve_relevant(index, row["doc"], row["section"])
        if not relevant:
            continue
        result = retriever.search(row["query"], top_k=TOP_K)
        ids = [c.chunk_id for c in result.chunks]
        out.append(
            {
                "query": row["query"],
                "effective_query": result.effective_query,
                "recall@1": recall_at_k(ids, relevant, 1),
                "recall@5": recall_at_k(ids, relevant, TOP_K),
                "mrr": reciprocal_rank(ids, relevant),
                "ndcg@5": ndcg_at_k(ids, relevant, TOP_K),
            }
        )
    return out


def _summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "recall@1": round(statistics.mean(r["recall@1"] for r in rows), 4),
        "recall@5": round(statistics.mean(r["recall@5"] for r in rows), 4),
        "mrr": round(statistics.mean(r["mrr"] for r in rows), 4),
        "ndcg@5": round(statistics.mean(r["ndcg@5"] for r in rows), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="跑全量 137 条（约 20 分钟）")
    parser.add_argument("--stride", type=int, default=3, help="抽样步长，1 等于全量")
    args = parser.parse_args()

    from mooki.core.config import Settings
    from mooki.graph.runtime import AgentRuntime

    settings = Settings()
    if settings.llm_offline or not settings.llm_api_key:
        print("未配置在线模型，先填 .env 再来。", file=sys.stderr)
        return 2

    rows = load_golden()
    if not args.all and args.stride > 1:
        rows = rows[:: args.stride]

    tmp = Path(tempfile.mkdtemp(prefix="mooki-online-retrieval-"))
    offline = AgentRuntime(settings=Settings(llm_offline=True, ledger_file=tmp / "a.db"))
    online = AgentRuntime(settings=Settings(ledger_file=tmp / "b.db"))

    print(f"样本 {len(rows)} 条；先跑离线，再跑在线\n")
    t0 = time.perf_counter()
    base = _eval(rows, offline.retriever, offline.index)
    t1 = time.perf_counter()
    llm = _eval(rows, online.retriever, online.index)
    t2 = time.perf_counter()

    base_s, llm_s = _summary(base), _summary(llm)
    # 逐条对比：哪些样本因为换了真实 LLM 而改善/退化
    deltas = []
    for b, l in zip(base, llm):
        deltas.append(
            {
                "query": b["query"],
                "delta@5": round(l["recall@5"] - b["recall@5"], 4),
                "effective_query": l["effective_query"],
            }
        )
    improved = [d for d in deltas if d["delta@5"] > 0]
    regressed = [d for d in deltas if d["delta@5"] < 0]

    report = {
        "n": len(base),
        "offline": base_s,
        "online": llm_s,
        "offline_seconds": round(t1 - t0, 1),
        "online_seconds": round(t2 - t1, 1),
        "improved": len(improved),
        "regressed": len(regressed),
        "unchanged": len(deltas) - len(improved) - len(regressed),
        "samples": deltas,
        "llm_usage": online.llm.usage(),
    }
    _write(report)
    print(json.dumps({"offline": base_s, "online": llm_s}, ensure_ascii=False, indent=2))
    print(f"改善 {len(improved)} / 退化 {len(regressed)} / 持平 {report['unchanged']}")
    return 0


def _write(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "online_retrieval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    o, l = report["offline"], report["online"]
    lines = [
        f"## 在线检索对照（n={report['n']}，仅替换 LLM 侧，索引与向量不变）",
        "",
        "| 配置 | R@1 | R@5 | MRR | NDCG@5 | 耗时 |",
        "|---|---|---|---|---|---|",
        f"| 离线替身（同义词改写 + bigram 重排） | {o['recall@1']} | {o['recall@5']} | {o['mrr']} | {o['ndcg@5']} | {report['offline_seconds']}s |",
        f"| DeepSeek 改写 + 重排 | {l['recall@1']} | {l['recall@5']} | {l['mrr']} | {l['ndcg@5']} | {report['online_seconds']}s |",
        "",
        f"- 逐条：改善 {report['improved']} 条，退化 {report['regressed']} 条，持平 {report['unchanged']} 条",
        f"- LLM 调用 {report['llm_usage']['calls']} 次，prompt {report['llm_usage']['prompt_tokens']} tokens",
        "",
        "> 结论口径：向量通道没动（仍是 LSA），所以这里测的只是「改写 + 重排交给真模型」值多少。",
        "> 真实 embedding 接进来之后要重跑 `make embedder`，向量通道的收益另算。",
    ]
    if report["regressed"]:
        lines += ["", "### 退化的样本（换真模型反而更差）", ""]
        for d in report["samples"]:
            if d["delta@5"] < 0:
                lines.append(f"- `{d['query']}` → 改写为 `{d['effective_query'][:50]}`")
    (OUT_DIR / "online_retrieval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
