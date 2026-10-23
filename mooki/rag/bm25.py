"""BM25 索引（自实现，不依赖 rank_bm25）。

中文没有空格，这里用「字符 bigram + 英文/数字整词」切分：
- 不做分词器依赖，离线可跑；
- bigram 天然带一点词序信息，对"退款 / 退钱"这类同义不同词仍会漏，
  这正是需要向量路兜的地方 —— 混合检索的收益来源在这里是可解释的。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from mooki.core.text import tokenize

__all__ = ["BM25Index", "tokenize"]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    ids: list[str] = field(default_factory=list)
    _tfs: list[dict[str, int]] = field(default_factory=list)
    _lens: list[int] = field(default_factory=list)
    _df: dict[str, int] = field(default_factory=dict)
    _avgdl: float = 0.0
    _idf: dict[str, float] = field(default_factory=dict)

    def fit(self, docs: list[tuple[str, str]]) -> None:
        """docs: [(id, text)]"""
        self.ids = [d[0] for d in docs]
        self._tfs, self._lens, self._df = [], [], {}
        for _, text in docs:
            toks = tokenize(text)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._tfs.append(tf)
            self._lens.append(len(toks))
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
        n = len(docs)
        self._avgdl = (sum(self._lens) / n) if n else 0.0
        self._idf = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in self._df.items()
        }

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.ids:
            return []
        scores: list[float] = []
        avgdl = self._avgdl or 1.0
        for i, tf in enumerate(self._tfs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._lens[i] / avgdl)
            for t in q_tokens:
                f = tf.get(t)
                if not f:
                    continue
                idf = self._idf.get(t, 0.0)
                score += idf * (f * (self.k1 + 1)) / (f + norm)
            scores.append(score)
        ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.ids[i], float(s)) for i, s in ranked if s > 0]

    # ---------- 持久化 ----------

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "ids": self.ids,
                    "tfs": self._tfs,
                    "lens": self._lens,
                    "df": self._df,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = cls(k1=data["k1"], b=data["b"])
        idx.ids = data["ids"]
        idx._tfs = data["tfs"]
        idx._lens = data["lens"]
        idx._df = data["df"]
        n = len(idx.ids)
        idx._avgdl = (sum(idx._lens) / n) if n else 0.0
        idx._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in idx._df.items()}
        return idx
