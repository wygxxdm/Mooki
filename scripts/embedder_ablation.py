"""向量后端对比：hashing / lsa / api 在检索任务上的差异。

存在的理由：README 里写着"LSA 不是真语义模型，接真实 embedding 后要重测"。
这句话如果是纯文字，就永远没人重测。做成脚本后，换 embedding 服务只需一条命令。

用法：
    python -m scripts.embedder_ablation
    python -m scripts.embedder_ablation --api       # 需要 .env 里配好 MOOKI_EMBED_BASE_URL
    python -m scripts.embedder_ablation --no-local  # 不跑本地 bge（没装 fastembed 时）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from mooki.core.cache import TTLCache
from mooki.core.config import Settings
from mooki.core.embedder import APIEmbedder, FastEmbedEmbedder, HashingEmbedder, LSAEmbedder
from mooki.eval.retrieval_eval import run_ablation
from mooki.graph.runtime import AgentRuntime
from mooki.rag.bm25 import BM25Index
from mooki.rag.ingest import IndexBundle
from mooki.rag.retriever import Retriever
from mooki.rag.rewrite import QueryRewriter
from mooki.rag.vector_store import LocalVectorStore

OUT_DIR = Path(__file__).resolve().parent.parent / "eval_out"
CONFIGS = ("direct", "bm25", "full")


def build_retriever(runtime: AgentRuntime, embedder, settings: Settings) -> Retriever:
    chunks = list(runtime.index.chunks)
    texts = [f"{c.title}\n{c.text}" for c in chunks]
    if hasattr(embedder, "fit"):
        embedder.fit(texts)
    vectors = embedder.embed(texts)
    bm25 = BM25Index()
    bm25.fit([(c.chunk_id, f"{c.title}\n{c.text}") for c in chunks])
    store = LocalVectorStore(None)
    store.add(
        [c.chunk_id for c in chunks],
        np.asarray(vectors, dtype=np.float32),
        [{"domain": c.domain, "chunk_id": c.chunk_id} for c in chunks],
    )
    index = IndexBundle(chunks=chunks, chunk_map={c.chunk_id: c for c in chunks}, bm25=bm25, store=store)
    return Retriever(
        settings,
        embedder,
        runtime.llm,
        index,
        TTLCache(2048, 60),
        QueryRewriter(runtime.llm, TTLCache(512, 60)),
    )


def main(argv: list[str]) -> int:
    settings = Settings(llm_offline=True)  # 评测必须离线可复现
    runtime = AgentRuntime(settings=settings)

    backends: list[tuple[str, object]] = [
        ("hashing-512", HashingEmbedder(512)),
        # 必须与生产配置同参（含 min_df 默认值），否则比的是两套词表而不是两套算法
        ("lsa-" + str(settings.embed_dim), LSAEmbedder(dim=settings.embed_dim)),
    ]
    if "--api" in argv and settings.embed_base_url:
        backends.append(("api", APIEmbedder(settings)))
    if "--no-local" not in argv:
        try:
            backends.append(("bge-small-zh", FastEmbedEmbedder(settings.embed_local_model)))
        except Exception as exc:  # noqa: BLE001 - 没装 fastembed 就跳过，不影响其它后端
            print(f"跳过本地语义模型：{exc}", file=sys.stderr)

    rows: dict[str, dict] = {}
    for name, embedder in backends:
        retriever = build_retriever(runtime, embedder, settings)
        report = run_ablation(retriever, settings, save=False)
        rows[name] = {
            config: {
                "recall@1": report["results"][config]["recall@1"],
                "recall@5": report["results"][config]["recall@5"],
                "ndcg@5": report["results"][config]["ndcg@5"],
            }
            for config in CONFIGS
        }

    lines = [
        "## 实验七 · 向量后端对比（同一语料、同一流水线，只换 embedding）",
        "",
        "| 后端 | direct R@1 | direct R@5 | direct NDCG@5 | full R@5 | full-direct | full NDCG@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, data in rows.items():
        gap = data["full"]["recall@5"] - data["direct"]["recall@5"]
        lines.append(
            f"| {name} | {data['direct']['recall@1']:.4f} | {data['direct']['recall@5']:.4f} "
            f"| {data['direct']['ndcg@5']:.4f} | {data['full']['recall@5']:.4f} "
            f"| {gap:+.4f} | {data['full']['ndcg@5']:.4f} |"
        )
    lines.append("")
    lines.append(
        "> direct 三列直接反映 embedding 质量：它只走向量、不改写不重排。"
        "full-direct 是整条链路（改写 + BM25 + RRF + 重排）在这套向量上的增量。"
    )
    lines.append("")
    lines.append(
        "> 读法：向量够强时 direct 已经接近上限，full 加不动（增量趋近 0）——"
        "说明这条链路原本是在补偿弱向量，不是永远必要。"
        "向量弱时（hashing / lsa）增量才有十几个点，那才是「架构价值」的真实来源。"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "embedder_ablation.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "embedder_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
