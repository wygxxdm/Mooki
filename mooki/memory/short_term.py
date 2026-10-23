"""短期工作记忆：滑动窗口 + 结构化槽位 + 摘要压缩。

关键点（也是客服场景真正的坑）：
纯 rolling summary 会把订单号、金额压没，一旦丢了订单号，后面所有工具都调不动。
所以这里把记忆拆成两部分：
    槽位（slots）  结构化 KV，存订单号 / 金额 / 商品 / 诉求，永不参与压缩，永不丢
    摘要（summary）只压叙述性内容
窗口只保留最近 N 轮原文，超出后最老的一半折叠进摘要。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from mooki.core.storage import KVStore
from mooki.core.text import estimate_tokens

_ORDER_ID = re.compile(r"SO\d{9,}")
_AMOUNT = re.compile(r"(\d+(?:\.\d{1,2})?)\s*(?:元|块)")
_SLOT_TTL = 60 * 60 * 24


@dataclass
class Turn:
    role: str
    content: str
    ts: int = field(default_factory=lambda: int(time.time()))
    intent: str | None = None


@dataclass
class WorkingMemory:
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    slots: dict[str, str] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return sum(estimate_tokens(t.content) for t in self.turns) + estimate_tokens(self.summary)

    def as_history(self) -> list[str]:
        return [f"{t.role}: {t.content}" for t in self.turns]


class ShortTermMemory:
    def __init__(self, kv: KVStore, window: int = 8, compress_at: int = 900):
        self.kv = kv
        self.window = window
        self.compress_at = compress_at  # 窗口 token 超过该值时触发压缩
        self.compressions = 0

    @staticmethod
    def _turns_key(session_id: str) -> str:
        return f"st:{session_id}:turns"

    @staticmethod
    def _slots_key(session_id: str) -> str:
        return f"st:{session_id}:slots"

    @staticmethod
    def _summary_key(session_id: str) -> str:
        return f"st:{session_id}:summary"

    # ---------- 读写 ----------

    def load(self, session_id: str) -> WorkingMemory:
        raw_turns = self.kv.lrange(self._turns_key(session_id), 0, -1)
        turns = []
        for item in raw_turns:
            try:
                turns.append(Turn(**json.loads(item)))
            except (json.JSONDecodeError, TypeError):
                continue
        slots = {}
        raw_slots = self.kv.get(self._slots_key(session_id))
        if raw_slots:
            try:
                slots = json.loads(raw_slots)
            except json.JSONDecodeError:
                slots = {}
        return WorkingMemory(
            turns=turns, summary=self.kv.get(self._summary_key(session_id)) or "", slots=slots
        )

    def append(self, session_id: str, role: str, content: str, intent: str | None = None) -> WorkingMemory:
        turn = Turn(role=role, content=content, intent=intent)
        self.kv.rpush(self._turns_key(session_id), json.dumps(turn.__dict__, ensure_ascii=False), ttl=_SLOT_TTL)
        # 注意：这里不裁剪，也不压摘要。
        # 早期版本在 append 里直接 ltrim，会把超窗的历史"静默丢弃"——
        # 既不进摘要也不留槽位，等于信息凭空消失。折叠必须走 maybe_compress，
        # 因为只有那里拿得到 llm，能把老轮次先总结再丢。
        self._update_slots(session_id, content)
        return self.load(session_id)

    def _update_slots(self, session_id: str, content: str) -> None:
        """从用户话术里抽实体。规则优先，避免为了两个字段多调一次 LLM。"""
        slots = self.load(session_id).slots
        changed = False
        if m := _ORDER_ID.search(content.upper()):
            slots["order_id"] = m.group(0)
            changed = True
        if m := _AMOUNT.search(content):
            slots["amount"] = m.group(1)
            changed = True
        for keyword, key in (("退款", "demand"), ("退货", "demand"), ("换货", "demand"), ("发票", "demand")):
            if keyword in content:
                slots[key] = keyword
                changed = True
        if changed:
            self.kv.set(self._slots_key(session_id), json.dumps(slots, ensure_ascii=False), ttl=_SLOT_TTL)

    # ---------- 压缩 ----------

    def maybe_compress(self, session_id: str, llm) -> bool:
        """窗口超限或 token 超阈值时，把最老的一段折叠进摘要，原文丢弃。

        触发条件有两条，缺一不可：
          轮数超过 window*2（结构化超限）       -> 按条数折叠
          token 超过 compress_at（非结构化超限） -> 折叠一半
        只按 token 一条会出问题：短句多时轮数早就爆了但 token 还没到阈值，
        窗口就会无限长。

        返回是否发生了压缩，便于在 trace 里标注 token 变化。
        """
        memory = self.load(session_id)
        limit = self.window * 2
        overflow = len(memory.turns) - limit
        over_token = memory.tokens > self.compress_at
        if overflow <= 0 and not over_token:
            return False
        fold = overflow if overflow > 0 else len(memory.turns) // 2
        if fold <= 0:
            return False

        old, keep = memory.turns[:fold], memory.turns[fold:]
        merged = (memory.summary + "\n" + "\n".join(f"{t.role}: {t.content}" for t in old)).strip()
        try:
            new_summary = llm.summarize([merged])[:400]
        except Exception:  # noqa: BLE001 - 摘要失败就保留原文，宁可多花 token 也不能丢信息
            return False

        key = self._turns_key(session_id)
        self.kv.delete(key)
        for turn in keep:
            self.kv.rpush(key, json.dumps(turn.__dict__, ensure_ascii=False), ttl=_SLOT_TTL)
        self.kv.set(self._summary_key(session_id), new_summary, ttl=_SLOT_TTL)
        self.compressions += 1
        return True
