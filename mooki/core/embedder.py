"""Embedding 抽象。

三档实现，按"是否需要外部模型"排序：

1. `LSAEmbedder`（默认）—— 在语料内 fit 的 TF-IDF + 截断 SVD（Latent Semantic Analysis）。
   它把「词共现」压进低维空间，因此同义、近义、上下位词在向量空间里会靠近，
   这才是 dense 路该有的语义性。纯 numpy，无需下载模型，且**可持久化、可复现**。

   为什么不用哈希向量：hashing 只是把词袋随机投影到固定维度，本质仍是词法匹配，
   和 BM25 同源 —— 两路同源的 RRF 融合没有信息增益，hybrid 实验就是自欺。

2. `APIEmbedder` —— 接真实 embedding 服务（OpenAI 兼容协议）。
3. `HashingEmbedder` —— 兜底，供单测与无 fit 场景使用。

三者对上层完全等价：只要满足 `fit(可选) + embed`，检索代码不改一行。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from mooki.core.config import Settings
from mooki.core.text import tokenize

_LSA_VOCAB = "lsa_vocab.json"
_LSA_IDF = "lsa_idf.npy"
_LSA_PROJ = "lsa_proj.npy"


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """词法向量的确定性兜底实现，不依赖语料，也不需要 fit。"""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                # 不能用内置 hash()：它按进程随机加盐，跨进程不可复现
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                mat[i, idx] += sign
            row = mat[i]
            mat[i] = np.sign(row) * np.log1p(np.abs(row))
            norm = float(np.linalg.norm(mat[i]))
            if norm > 0:
                mat[i] /= norm
        return mat

    def fit(self, texts: list[str]) -> None:  # 兼容接口
        return None


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dim: int = 128, min_df: int = 2):
        self.dim = dim
        self.target_dim = dim  # 语料太小时 SVD 会把 dim 压小，克隆时要用原始目标维度
        self.min_df = min_df
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._proj: np.ndarray | None = None
        self.fitted = False
        self.explained_variance = 0.0

    # ---------- 训练 ----------

    def fit(self, texts: list[str]) -> None:
        tokenized = [tokenize(t) for t in texts]
        df: Counter[str] = Counter()
        for toks in tokenized:
            df.update(set(toks))
        keep = [t for t, c in df.items() if c >= self.min_df] or list(df)
        keep.sort()
        self._vocab = {t: i for i, t in enumerate(keep)}
        n_docs, n_vocab = len(tokenized), len(keep)
        if n_docs == 0 or n_vocab == 0:
            raise ValueError("LSA 训练语料为空")

        self._idf = np.array(
            [math.log((n_docs + 1) / (df[t] + 1)) + 1.0 for t in keep], dtype=np.float32
        )

        matrix = np.zeros((n_docs, n_vocab), dtype=np.float32)
        for row, toks in enumerate(tokenized):
            for token, count in Counter(toks).items():
                col = self._vocab.get(token)
                if col is not None:
                    matrix[row, col] = 1.0 + math.log(count)
        matrix *= self._idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms

        k = max(2, min(self.dim, min(matrix.shape) - 1))
        _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        proj = vt[:k].T
        # SVD 的符号不唯一，固定成"每个主成分绝对值最大的分量为正"，保证跨进程可复现
        pivot = np.argmax(np.abs(proj), axis=0)
        signs = np.sign(proj[pivot, np.arange(proj.shape[1])])
        signs[signs == 0] = 1.0
        self._proj = (proj * signs).astype(np.float32)
        self.dim = k

        energy = float((singular**2).sum())
        self.explained_variance = float((singular[:k] ** 2).sum() / energy) if energy else 0.0
        self.fitted = True
        self.name = f"lsa-{k}"

    # ---------- 推理 ----------

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for token, count in Counter(tokenize(text)).items():
            col = self._vocab.get(token)
            if col is not None:
                vec[col] = 1.0 + math.log(count)
        vec *= self._idf
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self.fitted or self._proj is None:
            raise RuntimeError("LSAEmbedder 未训练，请先调用 fit(corpus)")
        if not texts:
            return np.zeros((0, self._proj.shape[1]), dtype=np.float32)
        mat = np.vstack([self._vectorize(t) for t in texts]) @ self._proj
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)

    # ---------- 持久化 ----------

    def save(self, directory: Path) -> None:
        if not self.fitted:
            return
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _LSA_VOCAB).write_text(
            json.dumps({"dim": self.dim, "vocab": self._vocab}, ensure_ascii=False), encoding="utf-8"
        )
        np.save(directory / _LSA_IDF, self._idf)
        np.save(directory / _LSA_PROJ, self._proj)

    @classmethod
    def load(cls, directory: Path) -> "LSAEmbedder | None":
        meta_path = directory / _LSA_VOCAB
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        emb = cls(dim=int(meta["dim"]))
        emb._vocab = {t: int(i) for t, i in meta["vocab"].items()}
        emb._idf = np.load(directory / _LSA_IDF)
        emb._proj = np.load(directory / _LSA_PROJ)
        emb.dim = int(meta["dim"])
        emb.fitted = True
        emb.name = f"lsa-{emb.dim}"
        return emb

    def restore(self, directory: Path) -> bool:
        """原地恢复训练态。

        必须原地改而不是返回新对象 —— 检索器、分类器都持有同一个 embedder 引用，
        换对象会让它们继续用未训练的那一份。
        """
        loaded = type(self).load(directory)
        if loaded is None:
            return False
        self._vocab = loaded._vocab
        self._idf = loaded._idf
        self._proj = loaded._proj
        self.dim = loaded.dim
        self.fitted = True
        self.name = loaded.name
        return True


class APIEmbedder:
    def __init__(self, settings: Settings):
        import httpx

        self.settings = settings
        self.dim = settings.embed_dim
        self.name = settings.embed_model
        base = settings.embed_base_url.rstrip("/")
        self._url = base + ("/embeddings" if base.endswith("/v1") else "/v1/embeddings")
        self._client = httpx.Client(timeout=settings.llm_timeout_s)

    def fit(self, texts: list[str]) -> None:  # 远端模型无需本地 fit
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.post(
            self._url,
            json={"model": self.settings.embed_model, "input": texts},
            headers={"Authorization": f"Bearer {self.settings.embed_api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        mat = np.array([d["embedding"] for d in data], dtype=np.float32)
        self.dim = mat.shape[1]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


def build_embedder(settings: Settings) -> Embedder:
    if settings.embed_base_url:
        return APIEmbedder(settings)
    if settings.embed_backend == "hashing":
        return HashingEmbedder(settings.embed_dim)
    return LSAEmbedder(dim=settings.embed_dim)


def clone_unfitted(embedder: Embedder) -> Embedder:
    """克隆一个同类型、未训练的实例。

    为什么需要：向量空间是"训练语料决定"的。知识库的 LSA 只在 KB 语料上训练过，
    拿它去编码意图样本，样本里大量的词是 OOV，编码出来接近零向量 ——
    dense 路会静默退化成"全投同一票"。所以意图分类必须用自己的特征空间。
    """
    if isinstance(embedder, LSAEmbedder):
        return LSAEmbedder(dim=embedder.target_dim, min_df=1)
    if isinstance(embedder, HashingEmbedder):
        return HashingEmbedder(embedder.dim)
    return embedder  # 远端 embedding 服务不需要本地 fit，直接复用
