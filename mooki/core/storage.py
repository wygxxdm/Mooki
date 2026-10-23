"""KV / List 存储抽象。

默认进程内实现，配置 redis_url 后自动切 Redis —— 记忆模块不感知后端差异。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol


class KVStore(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def rpush(self, key: str, value: str, ttl: int | None = None) -> None: ...
    def lrange(self, key: str, start: int, stop: int) -> list[str]: ...
    def ltrim(self, key: str, start: int, stop: int) -> None: ...
    def llen(self, key: str) -> int: ...


class MemoryKV:
    """进程内实现。仅用于本地开发与单测；多实例部署必须用 Redis。"""

    name = "memory"

    def __init__(self) -> None:
        self._kv: dict[str, tuple[str, float | None]] = {}
        self._lists: dict[str, tuple[list[str], float | None]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _expired(expire_at: float | None) -> bool:
        return expire_at is not None and expire_at < time.time()

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._kv.get(key)
            if item is None:
                return None
            value, expire_at = item
            if self._expired(expire_at):
                self._kv.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        with self._lock:
            self._kv[key] = (value, time.time() + ttl if ttl else None)

    def delete(self, key: str) -> None:
        with self._lock:
            self._kv.pop(key, None)
            self._lists.pop(key, None)

    def rpush(self, key: str, value: str, ttl: int | None = None) -> None:
        with self._lock:
            items, _ = self._lists.get(key, ([], None))
            items.append(value)
            self._lists[key] = (items, time.time() + ttl if ttl else None)

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        with self._lock:
            items, expire_at = self._lists.get(key, ([], None))
            if self._expired(expire_at):
                self._lists.pop(key, None)
                return []
            if stop == -1:
                return list(items[start:])
            return list(items[start : stop + 1])

    def ltrim(self, key: str, start: int, stop: int) -> None:
        with self._lock:
            items, expire_at = self._lists.get(key, ([], None))
            kept = items[start:] if stop == -1 else items[start : stop + 1]
            self._lists[key] = (list(kept), expire_at)

    def llen(self, key: str) -> int:
        return len(self.lrange(key, 0, -1))


class RedisKV:
    name = "redis"

    def __init__(self, url: str):
        import redis  # 延迟导入，未安装也不影响进程内模式

        self._r = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> str | None:
        return self._r.get(key)

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        self._r.set(key, value, ex=ttl)

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def rpush(self, key: str, value: str, ttl: int | None = None) -> None:
        self._r.rpush(key, value)
        if ttl:
            self._r.expire(key, ttl)

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        return self._r.lrange(key, start, stop)

    def ltrim(self, key: str, start: int, stop: int) -> None:
        self._r.ltrim(key, start, stop)

    def llen(self, key: str) -> int:
        return int(self._r.llen(key))


def build_kv(redis_url: str = "") -> KVStore:
    if redis_url:
        return RedisKV(redis_url)
    return MemoryKV()
