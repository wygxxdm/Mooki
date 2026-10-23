"""三路权重随「LLM 是不是替身」切换。

这组权重是超参，而且绑在通道上而非业务上：
- 离线替身（词袋假扮 LLM）下，dev 标定出 rule 0.65 / llm 0.25；
- 真模型下同一份数据重新标定，最优回到 llm 0.55 / rule 0.40。

如果默认值不跟着模型切换，就会出现"换了模型但权重还是替身时期那一套"，
把模型本可以判对的意图压掉（在线评测 d22：发票问题被判成物流）。
"""

from __future__ import annotations

import unittest

from mooki.core.config import Settings
from mooki.core.schema import VoteSource
from mooki.intent.classifier import base_weights


class TestBaseWeights(unittest.TestCase):
    def test_offline_uses_rule_heavy(self):
        w = base_weights(Settings(llm_offline=True))
        self.assertGreater(w[VoteSource.RULE], w[VoteSource.LLM])
        self.assertAlmostEqual(w[VoteSource.RULE], 0.65)

    def test_online_uses_llm_heavy(self):
        w = base_weights(Settings(llm_offline=False))
        self.assertGreater(w[VoteSource.LLM], w[VoteSource.RULE])
        self.assertAlmostEqual(w[VoteSource.LLM], 0.55)

    def test_explicit_config_wins(self):
        s = Settings(llm_offline=False, intent_weight_llm=0.8, intent_weight_dense=0.1, intent_weight_rule=0.1)
        w = base_weights(s)
        self.assertAlmostEqual(w[VoteSource.LLM], 0.8)
        self.assertAlmostEqual(w[VoteSource.RULE], 0.1)

    def test_weights_sum_to_one(self):
        for offline in (True, False):
            w = base_weights(Settings(llm_offline=offline))
            self.assertAlmostEqual(sum(w.values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
