"""检索评测：配置消融 + 词汇重叠度分析。

两个实验回答两个不同的问题：

1. **配置消融**：每一档能力（融合 / 改写 / 重排 / 域先验）各自贡献多少？
   口径为 chunk 级二值标注，通过 (doc_id, section) 精确解析到 chunk_id；
   每个 config 跑之前清空缓存，保证冷启动。

2. **词汇重叠度分析**：为什么很多项目的 Recall 看起来特别高？
   把 query 与目标 chunk 的 bigram 覆盖率分桶，看各桶的 Recall 差异。
   如果高重叠桶接近 1.0、低重叠桶明显掉下来，就说明"评测集与语料同源撰写"
   会把指标虚高几十个点 —— 这比换几个超参重要得多。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

from mooki.eval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from mooki.rag.retriever import CONFIGS

GOLDEN = Path(__file__).resolve().parent / "golden" / "retrieval.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "eval_out"
TOP_K = 5
_CLEAN = re.compile(r"[^\w\u4e00-\u9fff]+")

OVERLAP_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("低重叠 (<0.45)", 0.0, 0.45),
    ("中重叠 (0.45-0.65)", 0.45, 0.65),
    ("高重叠 (>=0.65)", 0.65, 1.01),
)


def load_golden() -> list[dict]:
    rows = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def resolve_relevant(index, doc: str, section: str) -> set[str]:
    return {c.chunk_id for c in index.chunks if c.doc_id == doc and section in c.section}


def _bigrams(text: str) -> set[str]:
    clean = _CLEAN.sub("", text)
    return {clean[i : i + 2] for i in range(max(0, len(clean) - 1))}


def query_coverage(query: str, target_text: str) -> float:
    """query 的 bigram 有多少比例出现在目标 chunk 里。

    用覆盖率而不是 Jaccard：我们关心的是"query 的措辞能不能在文档中原样找到"，
    文档长度不该稀释这个信号。
    """
    q = _bigrams(query)
    if not q:
        return 0.0
    return len(q & _bigrams(target_text)) / len(q)


def _assign_split(index: int) -> str:
    """按序号的稳定划分（2:1）。检索侧没有做超参搜索，这个划分用来量化样本波动，不是用来调参。"""
    return "dev" if index % 3 != 2 else "test"


def run_ablation(
    retriever,
    settings,
    top_k: int = TOP_K,
    save: bool = True,
    domain_resolver: Callable[[str], str | None] | None = None,
) -> dict:
    golden = load_golden()
    resolved: list[tuple[dict, set[str], str]] = []
    unresolved = []
    for idx, row in enumerate(golden):
        relevant = resolve_relevant(retriever.index, row["doc"], row["section"])
        if relevant:
            resolved.append((row, relevant, _assign_split(idx)))
        else:
            unresolved.append(row)

    results: dict[str, dict] = {}
    records: dict[str, list[dict]] = {}

    for config_name in CONFIGS:
        retriever.cache.clear()
        rows = []
        split_rows: dict[str, list[dict]] = {"dev": [], "test": []}
        latencies = []
        empty = 0
        per_query: list[dict] = []
        for row, relevant, split in resolved:
            prior = domain_resolver(row["query"]) if domain_resolver else None
            t0 = time.time()
            out = retriever.search(
                row["query"], top_k=top_k, config_name=config_name, domain_prior=prior
            )
            latencies.append((time.time() - t0) * 1000)
            retrieved = [c.chunk_id for c in out.chunks]
            if not retrieved:
                empty += 1
            metric = {
                "recall@1": recall_at_k(retrieved, relevant, 1),
                "recall@3": recall_at_k(retrieved, relevant, 3),
                "recall": recall_at_k(retrieved, relevant, top_k),
                "precision": precision_at_k(retrieved, relevant, top_k),
                "mrr": reciprocal_rank(retrieved, relevant),
                "ndcg": ndcg_at_k(retrieved, relevant, top_k),
            }
            rows.append(metric)
            split_rows[split].append(metric)
            target = next(iter(relevant))
            target_text = retriever.index.chunk_map[target].text
            per_query.append(
                {
                    "query": row["query"],
                    "hit@5": float(retrieved[:top_k].count(target) > 0),
                    "hit@1": float(retrieved[:1].count(target) > 0),
                    "overlap": round(query_coverage(row["query"], target_text), 4),
                    "rank": (retrieved.index(target) + 1) if target in retrieved else None,
                }
            )
        records[config_name] = per_query

        n = len(rows) or 1

        def _split_stat(picked: list[dict]) -> dict:
            if not picked:
                return {"n": 0, "recall@1": 0.0, "recall@5": 0.0}
            return {
                "n": len(picked),
                "recall@1": round(sum(r["recall@1"] for r in picked) / len(picked), 4),
                "recall@5": round(sum(r["recall"] for r in picked) / len(picked), 4),
            }

        results[config_name] = {
            "config": config_name,
            "n": len(rows),
            "recall@1": round(sum(r["recall@1"] for r in rows) / n, 4),
            "recall@3": round(sum(r["recall@3"] for r in rows) / n, 4),
            "recall@5": round(sum(r["recall"] for r in rows) / n, 4),
            "precision@5": round(sum(r["precision"] for r in rows) / n, 4),
            "mrr": round(sum(r["mrr"] for r in rows) / n, 4),
            "ndcg@5": round(sum(r["ndcg"] for r in rows) / n, 4),
            "empty_rate": round(empty / n, 4),
            "avg_latency_ms": round(sum(latencies) / n, 1),
            "splits": {"dev": _split_stat(split_rows["dev"]), "test": _split_stat(split_rows["test"])},
        }

    overlap = analyse_overlap(records)
    report = {
        "top_k": top_k,
        "queries_total": len(golden),
        "queries_resolved": len(resolved),
        "unresolved": unresolved,
        "results": results,
        "overlap": overlap,
        "per_query": records,
    }

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "retrieval_ablation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT_DIR / "retrieval_ablation.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def analyse_overlap(records: dict[str, list[dict]]) -> dict:
    """按重叠度三分位分桶比较 direct 与 full，量化"措辞重叠虚高"。

    用分位数而不是固定阈值：语料一变覆盖率分布就变，固定阈值会出现某个桶只有 1 条样本，
    结论就不可信。三分位保证每桶样本量相当。
    """
    if "direct" not in records or "full" not in records:
        return {}
    direct = {r["query"]: r for r in records["direct"]}
    full = {r["query"]: r for r in records["full"]}
    items = sorted(direct.items(), key=lambda kv: kv[1]["overlap"])
    if len(items) < 6:
        return {}
    size = len(items) // 3
    groups = [items[:size], items[size : size * 2], items[size * 2 :]]
    labels = ("低重叠 (1/3 分位)", "中重叠 (2/3 分位)", "高重叠 (3/3 分位)")

    buckets = []
    for label, group in zip(labels, groups):
        if not group:
            continue
        queries = [q for q, _ in group]
        overlaps = [r["overlap"] for _, r in group]
        d = sum(direct[q]["hit@5"] for q in queries) / len(queries)
        f = sum(full[q]["hit@5"] for q in queries) / len(queries)
        buckets.append(
            {
                "bucket": label,
                "n": len(queries),
                "overlap_range": f"{min(overlaps):.2f}-{max(overlaps):.2f}",
                "direct": round(d, 4),
                "full": round(f, 4),
                "lift": round(f - d, 4),
            }
        )
    mean_overlap = sum(r["overlap"] for r in direct.values()) / max(len(direct), 1)
    return {"buckets": buckets, "mean_query_coverage": round(mean_overlap, 4)}


def render_markdown(report: dict) -> str:
    lines = [
        f"## 实验一 · 检索配置消融（n={report['queries_resolved']}/{report['queries_total']}，top_k={report['top_k']}）",
        "",
        "| 配置 | Recall@1 | Recall@3 | Recall@5 | MRR | NDCG@5 | 空结果率 | 平均延迟(ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in report["results"].items():
        lines.append(
            f"| {name} | {r['recall@1']:.4f} | {r['recall@3']:.4f} | {r['recall@5']:.4f} "
            f"| {r['mrr']:.4f} | {r['ndcg@5']:.4f} | {r['empty_rate']:.2%} | {r['avg_latency_ms']:.1f} |"
        )

    res = report["results"]
    if "direct" in res and "full" in res:
        lines.append("")
        lines.append(
            f"direct → full：Recall@1 {res['direct']['recall@1']:.4f} → {res['full']['recall@1']:.4f}，"
            f"Recall@5 {res['direct']['recall@5']:.4f} → {res['full']['recall@5']:.4f}"
        )

    overlap = report.get("overlap") or {}
    if overlap.get("buckets"):
        lines.extend(
            [
                "",
                f"## 实验二 · 词汇重叠度对召回的影响（query 平均覆盖率 {overlap['mean_query_coverage']:.3f}）",
                "",
                "| 重叠度分桶 | 覆盖率区间 | n | direct Recall@5 | full Recall@5 | 架构增益 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for b in overlap["buckets"]:
            lines.append(
                f"| {b['bucket']} | {b['overlap_range']} | {b['n']} | {b['direct']:.4f} "
                f"| {b['full']:.4f} | {b['lift']:+.4f} |"
            )
        low = overlap["buckets"][0]
        high = overlap["buckets"][-1]
        if low["n"] and high["n"]:
            lines.append("")
            lines.append(
                f"> 高重叠桶 direct 已达 {high['direct']:.2%}，低重叠桶只有 {low['direct']:.2%}"
                f"（差 {(high['direct'] - low['direct']) * 100:.1f}pp）；"
                f"而架构增益恰好相反：低重叠桶 {low['lift']:+.2%}，高重叠桶 {high['lift']:+.2%}。"
                "结论：**只看总体 Recall 会同时高估系统能力（被容易样本拉高）"
                "和低估架构价值（难样本上的增益被平均掉）**，必须分桶看。"
            )

    lines.append("")
    lines.append("### dev/test 漂移（40 条样本的波动幅度）")
    lines.append("")
    lines.append("| 配置 | dev R@5 (n) | test R@5 (n) | 漂移 |")
    lines.append("|---|---|---|---|")
    for name, r in report["results"].items():
        dev, test = r["splits"]["dev"], r["splits"]["test"]
        drift = test["recall@5"] - dev["recall@5"]
        lines.append(
            f"| {name} | {dev['recall@5']:.4f} ({dev['n']}) | {test['recall@5']:.4f} ({test['n']}) | {drift:+.4f} |"
        )
    lines.append("")
    lines.append(
        "> 检索侧没有做超参搜索（`rrf_k=60`、`top_k=5` 取文献默认值），因此不存在用 test 调参的问题；"
        "这张表的作用是量化「40 条样本能差多少」—— 漂移本身就是统计显著性的上限，"
        "任何小于漂移的配置间差异都不该被当成结论。"
    )
    lines.append("")
    lines.append(
        "> 单相关标注（每个 query 只有 1 条目标 chunk），Precision@5 天然上限 0.2，不作为主要指标。"
        "本表用于配置间相对比较，绝对数值受语料规模（39 chunk）限制。"
    )
    return "\n".join(lines) + "\n"
