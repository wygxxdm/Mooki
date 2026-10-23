"""向量存储抽象。

默认 LocalVectorStore：numpy 矩阵 + jsonl 落盘，零依赖、可测。
配置 vector_backend=chroma 时切 ChromaDB —— 上层检索代码不改一行，
这是把"用什么向量库"降级成实现细节的常规做法。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None: ...
    def search(self, vector: np.ndarray, top_k: int, where: dict | None = None) -> list[Hit]: ...
    def count(self) -> int: ...
    def persist(self) -> None: ...


class LocalVectorStore:
    name = "local"

    def __init__(self, path: Path | None = None):
        self.path = path
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._meta: list[dict] = []

    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._vectors is None or self._vectors.size == 0:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)
        self._meta.extend(metadatas)

    def search(self, vector: np.ndarray, top_k: int = 20, where: dict | None = None) -> list[Hit]:
        if self._vectors is None or not self._ids:
            return []
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        sims = self._vectors @ vec
        order = np.argsort(-sims)
        hits: list[Hit] = []
        for i in order:
            meta = self._meta[int(i)]
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            hits.append(Hit(id=self._ids[int(i)], score=float(sims[int(i)]), metadata=meta))
            if len(hits) >= top_k:
                break
        return hits

    def count(self) -> int:
        return len(self._ids)

    def persist(self) -> None:
        if self.path is None or self._vectors is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.path.with_suffix(".npy"), self._vectors)
        payload = [{"id": i, "meta": m} for i, m in zip(self._ids, self._meta)]
        self.path.with_suffix(".jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in payload), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "LocalVectorStore":
        store = cls(path)
        npy = path.with_suffix(".npy")
        if npy.exists():
            store._vectors = np.load(npy)
            rows = path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
            store._ids = [json.loads(r)["id"] for r in rows if r]
            store._meta = [json.loads(r)["meta"] for r in rows if r]
        return store


class ChromaVectorStore:
    """ChromaDB 适配器。仅在安装了 chromadb 时可用。"""

    name = "chroma"

    def __init__(self, path: Path, collection: str = "mooki_kb"):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None:
        self._col.add(ids=ids, embeddings=np.asarray(vectors).tolist(), metadatas=metadatas)

    def search(self, vector: np.ndarray, top_k: int = 20, where: dict | None = None) -> list[Hit]:
        res = self._col.query(
            query_embeddings=[np.asarray(vector).tolist()],
            n_results=top_k,
            where=where or None,
        )
        hits: list[Hit] = []
        for i, cid in enumerate(res["ids"][0]):
            distance = res["distances"][0][i] if res.get("distances") else 0.0
            hits.append(
                Hit(id=cid, score=1.0 - float(distance), metadata=res["metadatas"][0][i])
            )
        return hits

    def count(self) -> int:
        return self._col.count()

    def persist(self) -> None:
        return None
