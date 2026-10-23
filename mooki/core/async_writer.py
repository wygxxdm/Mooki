"""后台批量写盘。

解决的问题：`persist` 节点原本同步写 jsonl + 重写向量库，P50 4ms，是离线链路里最大的一项；
线上的真实 IO 更慢，还会把尾延迟拉长。日志类写入没必要阻塞响应。

语义取舍（必须说清楚，否则就是掩盖问题）：
- 本模块只用于**可重建的派生态**：情景记忆的向量库、trace 快照。
- 事实源（episodes.jsonl / ledger.jsonl）仍然是**同步追加写** —— 丢了就真的丢了，
  不能进队列。所以这里省的是"索引重建"，不是"事实落盘"。
- 进程退出时 flush，队列满时降级为同步写，避免静默丢任务。
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

logger = logging.getLogger("mooki.writer")


class BackgroundWriter:
    def __init__(self, name: str = "writer", max_queue: int = 1024):
        self.name = name
        self._queue: queue.Queue[tuple[Callable[..., Any], tuple, dict] | None] = queue.Queue(
            maxsize=max_queue
        )
        self._thread = threading.Thread(target=self._run, name=f"mooki-{name}", daemon=True)
        self._thread.start()
        self.submitted = 0
        self.executed = 0
        self.failed = 0
        self.sync_fallbacks = 0

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        self.submitted += 1
        try:
            self._queue.put_nowait((fn, args, kwargs))
            return True
        except queue.Full:
            # 队列满说明后端持续跟不上，宁可拖慢响应也不能丢写
            self.sync_fallbacks += 1
            self._execute(fn, args, kwargs)
            return False

    def _execute(self, fn: Callable[..., Any], args: tuple, kwargs: dict) -> None:
        try:
            fn(*args, **kwargs)
            self.executed += 1
        except Exception as exc:  # noqa: BLE001 - 后台线程不能因为单次写失败就退出
            self.failed += 1
            logger.warning("background write failed: %s", exc)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            fn, args, kwargs = item
            self._execute(fn, args, kwargs)
            self._queue.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        """等待队列清空。用于优雅关闭与测试。"""
        deadline = threading.Event()
        done = threading.Event()

        def _waiter() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=_waiter, daemon=True).start()
        done.wait(timeout)
        deadline.set()

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def stats(self) -> dict[str, int]:
        return {
            "submitted": self.submitted,
            "executed": self.executed,
            "failed": self.failed,
            "sync_fallbacks": self.sync_fallbacks,
            "pending": self._queue.qsize(),
        }
