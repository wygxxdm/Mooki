"""长期情景记忆：把每轮会话的摘要向量化存起来，下次同类问题先捞历史。

用途很具体：老客户重复咨询同一笔订单时，不必从头问一遍订单号；
换个渠道进来，也能接上上次的结论。这与"知识库检索"是两件事 ——
知识库存政策，情景库存"这个人上次是怎么解决的"。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from mooki.core.async_writer import BackgroundWriter
from mooki.core.embedder import Embedder
from mooki.rag.vector_store import LocalVectorStore


@dataclass
class Episode:
    episode_id: str
    user_id: str
    session_id: str
    summary: str
    intent: str
    resolved: bool = True
    ts: int = field(default_factory=lambda: int(time.time()))
    meta: dict = field(default_factory=dict)


class EpisodicMemory:
    def __init__(self, embedder: Embedder, path: Path | None = None, top_k: int = 3, flush_every: int = 32):
        self.embedder = embedder
        self.top_k = top_k
        self.path = path
        self.flush_every = flush_every
        self._pending = 0
        self._store = LocalVectorStore(path)
        self._episodes: dict[str, Episode] = {}
        # 向量索引是派生态（可从 jsonl 重建），走后台线程；jsonl 本身仍然同步写
        self.writer = BackgroundWriter("episodic")
        if path is not None:
            self._load_meta()

    def _load_meta(self) -> None:
        meta_path = self.path.with_suffix(".episodes.jsonl") if self.path else None
        if meta_path and meta_path.exists():
            for line in meta_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                ep = Episode(**row)
                self._episodes[ep.episode_id] = ep

    def add(self, episode: Episode) -> None:
        if episode.episode_id in self._episodes:
            return
        vec = self.embedder.embed([episode.summary])[0]
        self._store.add(
            [episode.episode_id],
            np.asarray([vec], dtype=np.float32),
            [{"user_id": episode.user_id, "intent": episode.intent, "ts": episode.ts}],
        )
        self._episodes[episode.episode_id] = episode
        if self.path is not None:
            meta_path = self.path.with_suffix(".episodes.jsonl")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            # 事实源：同步追加。这条不能进队列 —— 丢了无法从别处重建。
            with meta_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")
            # 派生索引：攒批后交给后台线程重写，避免把整库 np.save 挂在响应路径上
            self._pending += 1
            if self._pending >= self.flush_every:
                self._pending = 0
                self.writer.submit(self._persist_store)

    def _persist_store(self) -> None:
        self._store.persist()

    def flush(self) -> None:
        if self.path is not None:
            self.writer.submit(self._persist_store)
            self.writer.flush()
            self._pending = 0

    def close(self) -> None:
        self.flush()
        self.writer.close()

    def recall(self, query: str, user_id: str, top_k: int | None = None) -> list[Episode]:
        top_k = top_k or self.top_k
        if not self._episodes or not query.strip():
            return []
        vec = self.embedder.embed([query])[0]
        hits = self._store.search(vec, top_k=max(top_k * 4, 8), where={"user_id": user_id})
        out: list[Episode] = []
        for hit in hits:
            ep = self._episodes.get(hit.id)
            if ep is not None:
                out.append(ep)
            if len(out) >= top_k:
                break
        return out
