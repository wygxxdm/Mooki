"""离线意图替身：多项式朴素贝叶斯。

为什么不用关键词：原来的离线替身和规则路都是"手写关键词命中"，两条路同源，
投票就成了重复计票（互补性矩阵会显示 rule/llm 互补率只有 6%，且融合反而不如单路）。

换成 NB 之后，三路的假设空间真正分开了：
    规则路    手写关键词 + 优先级       （人的先验）
    NB 路     词袋上的生成式概率模型     （从样本学到的统计规律）
    向量路    LSA 低维语义空间上的 kNN   （共现结构）
这样才能验证"投票"本身的价值，而不是验证"三份关键词表的一致性"。

注意：NB 在 exemplar 上训练，而 golden query 与 exemplar 同源，指标仍然偏乐观；
这一点在 README 的已知局限里单列，不能靠换个模型就宣称解决。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from mooki.core.schema import Urgency
from mooki.core.text import tokenize
from mooki.intent.labels import INTENT_SPECS, rule_urgency


class NaiveBayesIntentModel:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.class_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.class_totals: dict[str, int] = defaultdict(int)
        self.vocab: set[str] = set()
        self.n_docs = 0

    def fit(self, texts: list[str], labels: list[str]) -> "NaiveBayesIntentModel":
        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            self.class_counts[label] += 1
            self.n_docs += 1
            for tok in tokens:
                self.token_counts[label][tok] += 1
                self.class_totals[label] += 1
                self.vocab.add(tok)
        return self

    def _log_prob(self, text: str) -> dict[str, float]:
        tokens = [t for t in tokenize(text) if t in self.vocab]
        v = max(len(self.vocab), 1)
        scores: dict[str, float] = {}
        for label, count in self.class_counts.items():
            score = math.log(count / self.n_docs)
            denom = self.class_totals[label] + self.alpha * v
            for tok in tokens:
                score += math.log((self.token_counts[label][tok] + self.alpha) / denom)
            scores[label] = score
        return scores

    def predict(self, text: str) -> tuple[str, float]:
        scores = self._log_prob(text)
        if not scores:
            return "other", 0.35
        best = max(scores.values())
        # softmax 归一化，温度压低避免过度自信
        exps = {k: math.exp((v - best) / 1.5) for k, v in scores.items()}
        total = sum(exps.values()) or 1.0
        label = max(exps.items(), key=lambda kv: kv[1])[0]
        prob = exps[label] / total
        # 压到 [0.3, 0.9]：短文本上 NB 的概率不可当真实置信度用
        return label, round(min(0.9, max(0.3, prob)), 4)


_MODEL: NaiveBayesIntentModel | None = None


def build_model(exclude: set[str] | None = None) -> NaiveBayesIntentModel:
    """从 exemplar 训练。exclude 用于留出实验：被排除的样本当查询集，不能参与训练。"""
    exclude = exclude or set()
    texts: list[str] = []
    labels: list[str] = []
    for intent, spec in INTENT_SPECS.items():
        for ex in spec.exemplars:
            if ex in exclude:
                continue
            texts.append(ex)
            labels.append(intent.value)
    return NaiveBayesIntentModel().fit(texts, labels)


def install_model(model: NaiveBayesIntentModel | None) -> None:
    global _MODEL
    _MODEL = model


def predict_intent(text: str) -> dict:
    """离线意图预测：返回与在线 LLM 同构的 JSON 结构。"""
    global _MODEL
    if _MODEL is None:
        _MODEL = build_model()
    label, confidence = _MODEL.predict(text)

    urgency = Urgency.P2
    rule = rule_urgency(text)
    if rule is not None:
        urgency = rule
    return {"label": label, "confidence": confidence, "urgency": urgency.value}
