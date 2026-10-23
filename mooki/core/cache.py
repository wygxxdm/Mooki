"""进程内 TTL + LRU 缓存，带命中率统计。

为什么不用 functools.lru_cache：它没有 TTL，而意图结果和检索结果都必须随知识库更新失效。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": round(self.hit_rate, 4),
        }


class TTLCache:
    def __init__(self, maxsize: int = 1024, ttl: float = 600.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.stats.misses += 1
                return None
            expire_at, value = item
            if expire_at < time.time():
                self._data.pop(key, None)
                self.stats.expirations += 1
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            self.stats.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            elif len(self._data) >= self.maxsize:
                self._data.popitem(last=False)
                self.stats.evictions += 1
            self._data[key] = (time.time() + self.ttl, value)

    def get_or_set(self, key: str, factory) -> tuple[Any, bool]:
        """返回 (value, from_cache)。"""
        hit = self.get(key)
        if hit is not None:
            return hit, True
        value = factory()
        self.set(key, value)
        return value, False

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
