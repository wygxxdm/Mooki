"""向量后端的选择逻辑。

重点不是"能不能用 bge"，而是**换后端时上层一行不改**：
Retriever 只依赖 `fit(可选) + embed`，LSA / 本地 ONNX / 远端 API 三者等价。
这条如果破了，README 里"换 embedding 只要一条命令"就是空话。
"""

from __future__ import annotations

import importlib.util
import unittest

from mooki.core.config import Settings
from mooki.core.embedder import (
    FastEmbedEmbedder,
    HashingEmbedder,
    LSAEmbedder,
    build_embedder,
    clone_unfitted,
)

HAS_FASTEMBED = importlib.util.find_spec("fastembed") is not None


class TestBackendSelection(unittest.TestCase):
    def test_default_is_lsa(self):
        self.assertIsInstance(build_embedder(Settings(embed_backend="lsa")), LSAEmbedder)

    def test_hashing(self):
        self.assertIsInstance(build_embedder(Settings(embed_backend="hashing")), HashingEmbedder)

    def test_base_url_wins_over_backend(self):
        """配了远端地址就以远端为准，不再看 backend 字段。"""
        settings = Settings(embed_backend="lsa", embed_base_url="https://example.com/v1")
        self.assertEqual(build_embedder(settings).__class__.__name__, "APIEmbedder")

    @unittest.skipUnless(HAS_FASTEMBED, "未安装 fastembed")
    def test_bge_backend_requires_local_model(self):
        """只验证分派，不触发下载 —— 真跑模型由 make embedder 负责。"""
        settings = Settings(embed_backend="bge", embed_local_model="BAAI/bge-small-zh-v1.5")
        picked = build_embedder(settings)
        self.assertIsInstance(picked, FastEmbedEmbedder)
        self.assertIn("bge-small-zh", picked.name)


class TestCloneUnfitted(unittest.TestCase):
    def test_lsa_clone_is_fresh(self):
        original = LSAEmbedder(dim=64)
        clone = clone_unfitted(original)
        self.assertIsInstance(clone, LSAEmbedder)
        self.assertIsNot(clone, original)

    def test_hashing_clone_keeps_dim(self):
        clone = clone_unfitted(HashingEmbedder(128))
        self.assertEqual(clone.dim, 128)

    @unittest.skipUnless(HAS_FASTEMBED, "未安装 fastembed")
    def test_pretrained_clone_is_reused(self):
        """预训练模型没有 fit 过程，克隆时直接复用，避免重复加载模型。"""
        class Dummy(FastEmbedEmbedder):
            def __init__(self):  # 绕过真实模型加载
                self.name = "dummy"
                self.dim = 8

        dummy = Dummy()
        self.assertIs(clone_unfitted(dummy), dummy)


if __name__ == "__main__":
    unittest.main()
