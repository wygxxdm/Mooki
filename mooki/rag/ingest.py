from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mooki.core.config import Settings
from mooki.core.embedder import Embedder
from mooki.rag.bm25 import BM25Index
from mooki.rag.chunker import Chunk, chunk_document
from mooki.rag.vector_store import ChromaVectorStore, LocalVectorStore, VectorStore


@dataclass
class IndexBundle:
    chunks: list[Chunk]
    chunk_map: dict[str, Chunk]
    bm25: BM25Index
    store: VectorStore

    @property
    def size(self) -> int:
        return len(self.chunks)

    def domains(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.domain] = out.get(c.domain, 0) + 1
        return dict(sorted(out.items()))


def _read_docs(kb_dir: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for path in sorted(kb_dir.glob("*.md")):
        docs.append((path.stem, path.read_text(encoding="utf-8")))
    return docs


def build_index(settings: Settings, embedder: Embedder) -> IndexBundle:
    docs = _read_docs(settings.kb_dir)
    if not docs:
        raise RuntimeError(f"知识库为空：{settings.kb_dir}")

    chunks: list[Chunk] = []
    for doc_id, raw in docs:
        chunks.extend(
            chunk_document(doc_id, raw, max_chars=settings.chunk_size, overlap=settings.chunk_overlap)
        )
    chunks.sort(key=lambda c: (c.doc_id, c.order))

    bm25 = BM25Index()
    bm25.fit([(c.chunk_id, f"{c.title}\n{c.text}") for c in chunks])

    texts = [f"{c.title}\n{c.text}" for c in chunks]
    if hasattr(embedder, "fit"):
        embedder.fit(texts)  # LSA 必须在语料内训练，否则 dense 路只是随机投影
    vectors = embedder.embed(texts)
    metadatas = [
        {
            "doc_id": c.doc_id,
            "chunk_id": c.chunk_id,
            "title": c.title,
            "domain": c.domain,
            "section": c.section,
        }
        for c in chunks
    ]
    if settings.vector_backend == "chroma":
        store: VectorStore = ChromaVectorStore(settings.chroma_dir)
    else:
        store = LocalVectorStore(settings.index_dir / "vectors")
    store.add([c.chunk_id for c in chunks], vectors, metadatas)

    index_dir = settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(index_dir / "bm25.json")
    (index_dir / "chunks.json").write_text(
        json.dumps([c.__dict__ for c in chunks], ensure_ascii=False), encoding="utf-8"
    )
    store.persist()
    if hasattr(embedder, "save"):
        embedder.save(index_dir)
    return IndexBundle(chunks=chunks, chunk_map={c.chunk_id: c for c in chunks}, bm25=bm25, store=store)


def load_index(settings: Settings, embedder: Embedder) -> IndexBundle:
    index_dir = settings.index_dir
    bm25_path = index_dir / "bm25.json"
    chunks_path = index_dir / "chunks.json"
    if not bm25_path.exists() or not chunks_path.exists():
        return build_index(settings, embedder)

    rows = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [Chunk(**row) for row in rows]
    bm25 = BM25Index.load(bm25_path)
    if settings.vector_backend == "chroma":
        store: VectorStore = ChromaVectorStore(settings.chroma_dir)
    else:
        store = LocalVectorStore.load(index_dir / "vectors")
        if store.count() == 0:
            return build_index(settings, embedder)

    # 向量化器必须与建索引时同一份，否则查询向量与库里向量不在一个空间
    if hasattr(embedder, "restore") and not embedder.restore(index_dir):  # type: ignore[attr-defined]
        return build_index(settings, embedder)
    return IndexBundle(chunks=chunks, chunk_map={c.chunk_id: c for c in chunks}, bm25=bm25, store=store)
