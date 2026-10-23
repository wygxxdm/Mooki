"""规则路投票：关键词命中。

规则路的价值不在于覆盖率高，而在于高精度 + 可解释 + 零成本。
所以它只输出"最高命中"的那一个意图，不铺开打分，避免污染投票。
"""

from __future__ import annotations

from mooki.core.schema import Intent
from mooki.core.text import normalize_text
from mooki.intent.labels import INTENT_SPECS


def rule_scores(text: str) -> dict[Intent, float]:
    """返回意图 -> 置信度（只保留命中意图）。"""
    norm = normalize_text(text)
    hits: dict[Intent, float] = {}
    for intent, spec in INTENT_SPECS.items():
        matched = [kw for kw in spec.keywords if kw in norm]
        if not matched:
            continue
        # 长关键词更具体，权重更高；命中越多越可信，上限 0.92
        longest = max(len(kw) for kw in matched)
        score = min(0.92, 0.5 + 0.1 * len(matched) + 0.04 * longest)
        hits[intent] = round(score, 4)
    return hits


def rule_vote(text: str) -> tuple[Intent, float] | None:
    scores = rule_scores(text)
    if not scores:
        return None
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0], best[1]
