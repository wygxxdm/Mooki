"""用户画像：结构化 KV，不走向量。

画像只放"稳定属性"（会员等级、常用收货城市、历史售后次数、偏好渠道），
不放对话内容 —— 那是情景记忆的职责。混在一起会导致每次都要做相似度检索，
反而更慢更贵。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from mooki.core.storage import KVStore

_TTL = 60 * 60 * 24 * 30


@dataclass
class Profile:
    user_id: str
    vip_level: str = "normal"
    city: str = ""
    total_orders: int = 0
    refund_count: int = 0
    complaint_count: int = 0
    last_intent: str = ""
    tags: list[str] = field(default_factory=list)
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def as_prompt(self) -> dict:
        """只把对回答有用的字段喂给模型，避免把整份画像塞进 prompt。"""
        return {
            "vip_level": self.vip_level,
            "history_refunds": self.refund_count,
            "history_complaints": self.complaint_count,
            "last_intent": self.last_intent,
        }


class ProfileStore:
    def __init__(self, kv: KVStore):
        self.kv = kv

    @staticmethod
    def _key(user_id: str) -> str:
        return f"profile:{user_id}"

    def get(self, user_id: str) -> Profile:
        raw = self.kv.get(self._key(user_id))
        if not raw:
            return Profile(user_id=user_id)
        try:
            return Profile(**json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return Profile(user_id=user_id)

    def save(self, profile: Profile) -> None:
        profile.updated_at = int(time.time())
        self.kv.set(self._key(profile.user_id), json.dumps(asdict(profile), ensure_ascii=False), ttl=_TTL)

    def bump(self, user_id: str, intent: str) -> Profile:
        profile = self.get(user_id)
        profile.last_intent = intent
        if intent == "refund":
            profile.refund_count += 1
        if intent == "complaint":
            profile.complaint_count += 1
        if profile.complaint_count >= 2 and "high_risk" not in profile.tags:
            profile.tags.append("high_risk")
        self.save(profile)
        return profile
