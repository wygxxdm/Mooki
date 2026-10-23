"""多路融合意图识别。

核心设计：
1. 三路各自"投一票"，而不是三路各出一套概率再平均 —— 因为三路输出的东西量纲不同：
   LLM 是类别 + 置信度，向量是余弦相似度，规则是关键词命中强度。投票是唯一能统一语义的形式。
2. 权重随证据质量自适应：LLM 低置信时把权重让给向量和规则；某路缺失时权重重新归一化。
3. 冲突不强行裁决：top1/top2 间隔小于阈值时输出 need_clarify，交给上层追问，
   而不是赌一个。客服场景错分一次的成本远高于多问一句。
4. 紧急度是完全独立的任务，规则命中 P0 信号时硬覆盖 LLM 判断（资损/投诉不能赌）。
"""

from __future__ import annotations

import time

import numpy as np

from mooki.core.cache import TTLCache
from mooki.core.config import settings as default_settings
from mooki.core.embedder import clone_unfitted
from mooki.core.schema import Intent, IntentResult, Urgency, Vote, VoteSource
from mooki.core.text import normalize_text
from mooki.intent.labels import INTENT_SPECS, rule_urgency
from mooki.intent.rules import rule_scores

# 离线替身下的标定结果：LLM 是词袋假扮的，规则通道最稳
OFFLINE_WEIGHTS = {"llm": 0.25, "dense": 0.10, "rule": 0.65}
# 真实模型下的标定结果（105 条 query 三路投票重新取一次，dev 网格搜索，test 只报一次）
ONLINE_WEIGHTS = {"llm": 0.55, "dense": 0.05, "rule": 0.40}


def base_weights(settings=None) -> dict[VoteSource, float]:
    """三路权重。

    这是**超参，而且绑在"哪套通道"上**，不是绑在业务上：
    - 离线替身表达能力弱，dev 标定出 rule 0.65 / llm 0.25，融合 0.85；
    - 换成真模型后同一份数据重新标定，最优回到 llm 0.55 / rule 0.40，test 0.969；
      而沿用离线那组权重只有 0.875。

    所以没有显式配置时，按当前 LLM 是不是替身给一组默认值；显式设了
    MOOKI_INTENT_WEIGHT_* 就以配置为准。
    """
    st = settings or default_settings
    explicit = st.model_fields_set
    if "intent_weight_llm" not in explicit:
        chosen = OFFLINE_WEIGHTS if st.llm_offline else ONLINE_WEIGHTS
        return {
            VoteSource.LLM: chosen["llm"],
            VoteSource.DENSE: chosen["dense"],
            VoteSource.RULE: chosen["rule"],
        }
    return {
        VoteSource.LLM: st.intent_weight_llm,
        VoteSource.DENSE: st.intent_weight_dense,
        VoteSource.RULE: st.intent_weight_rule,
    }


class IntentClassifier:
    def __init__(self, llm, embedder, cache: TTLCache, exemplar_exclude: set[str] | None = None,
                 settings=None):
        self.llm = llm
        self.settings = settings or default_settings
        self.cache = cache
        self.exemplar_exclude = exemplar_exclude or set()
        # 意图样本走独立的特征空间，避免复用知识库的 LSA 导致大量 OOV
        self.embedder = clone_unfitted(embedder)
        self._build_exemplar_bank()

    def _build_exemplar_bank(self) -> None:
        texts: list[str] = []
        labels: list[Intent] = []
        for intent, spec in INTENT_SPECS.items():
            for ex in spec.exemplars:
                if ex in self.exemplar_exclude:
                    continue  # 留出实验：被排除的样本不能进底库，否则等于考训练集
                texts.append(ex)
                labels.append(intent)
        if hasattr(self.embedder, "fit"):
            self.embedder.fit(texts)
        self._ex_matrix = self.embedder.embed(texts)
        self._ex_labels = labels

    # ---------- 三路 ----------

    def llm_route(self, text: str) -> tuple[Vote | None, Urgency | None]:
        """一次 LLM 调用同时拿到意图票与紧急度，避免重复推理。"""
        out = self.llm.classify_intent(text)
        if not out:
            return None, None
        try:
            label = Intent(out["label"])
        except (KeyError, ValueError):
            return None, None
        conf = min(1.0, max(0.0, float(out.get("confidence", 0.5))))
        urgency = None
        raw_urgency = out.get("urgency")
        if raw_urgency in {u.value for u in Urgency}:
            urgency = Urgency(raw_urgency)
        return Vote(source=VoteSource.LLM, label=label, score=conf), urgency

    def dense_vote(self, text: str) -> Vote | None:
        if not text.strip():
            return None
        vec = self.embedder.embed([text])[0]
        if float(np.linalg.norm(vec)) == 0.0:
            return None  # 全是 OOV，没有可用特征就不投票，而不是投一个 0 分的空票
        sims = self._ex_matrix @ vec
        best_per_intent: dict[Intent, float] = {}
        for sim, label in zip(sims.tolist(), self._ex_labels):
            if sim > best_per_intent.get(label, -1.0):
                best_per_intent[label] = sim
        label, sim = max(best_per_intent.items(), key=lambda kv: kv[1])
        # 余弦下界裁到 0：负相关不该贡献正向票
        return Vote(source=VoteSource.DENSE, label=label, score=round(max(0.0, sim), 4))

    def rule_vote(self, text: str) -> Vote | None:
        scores = rule_scores(text)
        if not scores:
            return None
        label, score = max(scores.items(), key=lambda kv: kv[1])
        return Vote(source=VoteSource.RULE, label=label, score=score)

    # ---------- 融合 ----------

    def _weights(self, votes: list[Vote]) -> dict[VoteSource, float]:
        present = {v.source for v in votes}
        base = base_weights(self.settings)
        weights = {s: base[s] for s in present}
        llm_vote = next((v for v in votes if v.source is VoteSource.LLM), None)
        if llm_vote is not None and llm_vote.score < 0.5 and len(weights) > 1:
            # LLM 不确定，把它的一半权重让给其它两路
            released = weights[VoteSource.LLM] * 0.5
            weights[VoteSource.LLM] -= released
            others = [s for s in weights if s is not VoteSource.LLM]
            for s in others:
                weights[s] += released / len(others)
        total = sum(weights.values()) or 1.0
        return {s: w / total for s, w in weights.items()}

    def fuse(self, votes: list[Vote]) -> tuple[Intent, float, float]:
        if not votes:
            return Intent.OTHER, 0.0, 0.0
        weights = self._weights(votes)
        fused: dict[Intent, float] = {}
        for v in votes:
            fused[v.label] = fused.get(v.label, 0.0) + weights[v.source] * v.score
            v.weight = round(weights[v.source], 4)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return top_label, round(top_score, 4), round(top_score - second, 4)

    # ---------- 对外 ----------

    def classify(self, text: str, routes: set[VoteSource] | None = None) -> IntentResult:
        """routes=None 走完整三路融合；传入子集时只跑指定通道（用于消融，且绕过缓存）。"""
        started = time.time()
        ablation = routes is not None
        key = normalize_text(text)
        if not ablation:
            cached = self.cache.get(key)
            if cached is not None:
                return cached.model_copy(
                    update={"cached": True, "latency_ms": int((time.time() - started) * 1000)}
                )

        use = routes if routes is not None else {VoteSource.LLM, VoteSource.DENSE, VoteSource.RULE}
        votes: list[Vote] = []
        llm_urgency: Urgency | None = None
        if VoteSource.LLM in use:
            llm_vote, llm_urgency = self.llm_route(text)
            if llm_vote is not None:
                votes.append(llm_vote)
        if VoteSource.DENSE in use:
            dense = self.dense_vote(text)
            if dense is not None:
                votes.append(dense)
        if VoteSource.RULE in use:
            rule = self.rule_vote(text)
            if rule is not None:
                votes.append(rule)

        label, confidence, margin = self.fuse(votes)

        # 紧急度独立判定：规则与 LLM 取更紧急的一方（宁可高估），都没有则 P2
        candidates = [u for u in (rule_urgency(text), llm_urgency) if u is not None]
        urgency = min(candidates, key=lambda u: u.rank) if candidates else Urgency.P2

        result = IntentResult(
            label=label,
            urgency=urgency,
            confidence=confidence,
            votes=votes,
            margin=margin,
            need_clarify=(margin < 0.12) or (confidence < 0.45),
            latency_ms=int((time.time() - started) * 1000),
        )
        if not ablation:
            self.cache.set(key, result)
        return result
