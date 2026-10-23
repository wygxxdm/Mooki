"""向量后端对比：hashing / lsa / api 在检索任务上的差异。

存在的理由：README 里写着"LSA 不是真语义模型，接真实 embedding 后要重测"。
这句话如果是纯文字，就永远没人重测。做成脚本后，换 embedding 服务只需一条命令。

用法：
    python -m scripts.embedder_ablation
    python -m scripts.embedder_ablation --api   # 需要 .env 里配好 MOOKI_EMBED_BASE_URL
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from mooki.core.cache import TTLCache
from mooki.core.config import Settings
from mooki.core.embedder import APIEmbedder, LSAEmbedder, HashingEmbedder
from mooki.eval.retrieval_eval import run_ablation
from mooki.graph.runtime import AgentRuntime
from mooki.rag.bm25 import BM25Index
from mooki.rag.ingest import IndexBundle
from mooki.rag.retriever import Retriever
from mooki.rag.rewrite import QueryRewriter
from mooki.rag.vector_store import LocalVectorStore

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "eval_out"
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
        "| 后端 | direct R@5 | bm25 R@5 | full R@5 | full-direct | full NDCG@5 |",
        "|---|---|---|---|---|---|",
    ]
    for name, data in rows.items():
        gap = data["full"]["recall@5"] - data["direct"]["recall@5"]
        lines.append(
            f"| {name} | {data['direct']['recall@5']:.4f} | {data['bm25']['recall@5']:.4f} "
            f"| {data['full']['recall@5']:.4f} | {gap:+.4f} | {data['full']['ndcg@5']:.4f} |"
        )
    lines.append("")
    lines.append(
        "> direct 一列直接反映 embedding 质量：它只走向量、不改写不重排。"
        "换真实 embedding 服务后重跑本脚本即可验证——**这也是把「已知局限」变成「可复测项」的做法**。"
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
