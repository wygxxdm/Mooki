"""append-only 账本 + 状态机 + 幂等 + 补偿。

为什么不是"直接改订单 JSON"：
1. **改状态无法审计**。售后资损场景要能回答"谁在什么时候把单子改成退款了"，
   所以只追加事实（event），当前状态由重放推导（event sourcing 的最小形态）。
2. **重试必须幂等**。网络超时后客户端重发，不能退两次钱。
   幂等键落库，重复请求直接返回首次结果（带 duplicate=True），而不是再执行一次。
3. **失败必须补偿**。退款是跨系统操作（平台单 → 支付网关），本地成功、下游失败时
   必须把本地状态回滚，否则会出现"平台显示退款中、网关没收到"的脏数据。

状态机：
    paid / shipped / delivered --refund.request--> refund_pending
    refund_pending --refund.approve--> refunded
    refund_pending --refund.reject/cancel--> 回到 prev_status
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 动作 -> 允许的源状态
TRANSITIONS: dict[str, set[str]] = {
    "refund.request": {"paid", "shipped", "delivered"},
    "refund.approve": {"refund_pending"},
    "refund.reject": {"refund_pending"},
    "refund.cancel": {"refund_pending"},
}

# 动作 -> 目标状态；None 表示回滚到 payload.prev_status
TARGET: dict[str, str | None] = {
    "refund.request": "refund_pending",
    "refund.approve": "refunded",
    "refund.reject": None,
    "refund.cancel": None,
}


@dataclass
class LedgerRecord:
    seq: int
    ts: float
    action: str
    order_id: str
    idem_key: str
    payload: dict = field(default_factory=dict)


class OrderLedger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._seq = 0
        self._idem: dict[str, LedgerRecord] = {}
        self._state: dict[str, dict] = {}  # order_id -> {status, refund_id, refund_amount, refunded}
        self._load()

    # ---------- 加载与重放 ----------

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = LedgerRecord(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            self._seq = max(self._seq, record.seq)
            self._idem[record.idem_key] = record
            self._apply_to_state(record)

    def _apply_to_state(self, record: LedgerRecord) -> None:
        state = self._state.setdefault(record.order_id, {"status": None})
        target = TARGET.get(record.action)
        if target is None:
            # 回滚要回到"进入退款流程之前"的状态，而不是上一跳状态。
            # 若直接取 payload.prev_status，refund.cancel 的 prev 是 refund_pending，
            # 回滚后仍停在退款中 —— 补偿等于没做。
            state["status"] = state.get("origin_status") or record.payload.get("prev_status") or state.get("status")
        else:
            state["status"] = target
        if record.action == "refund.request":
            state["origin_status"] = record.payload.get("prev_status")
            state["refund_id"] = record.payload.get("refund_id")
            state["refund_amount"] = record.payload.get("amount")
        if record.action == "refund.approve":
            state["refunded"] = True
        if record.action in ("refund.cancel", "refund.reject"):
            state["refunded"] = False

    # ---------- 查询 ----------

    def status(self, order_id: str, base_status: str | None = None) -> str | None:
        with self._lock:
            state = self._state.get(order_id)
            return (state or {}).get("status") or base_status

    def snapshot(self, order_id: str) -> dict:
        with self._lock:
            return dict(self._state.get(order_id) or {})

    def find(self, idem_key: str) -> LedgerRecord | None:
        with self._lock:
            return self._idem.get(idem_key)

    def history(self, order_id: str) -> list[LedgerRecord]:
        with self._lock:
            return [r for r in self._idem.values() if r.order_id == order_id]

    # ---------- 写入 ----------

    def apply(
        self,
        action: str,
        order_id: str,
        idem_key: str,
        payload: dict | None = None,
        base_status: str | None = None,
    ) -> tuple[LedgerRecord | None, str | None, bool]:
        """返回 (record, error, duplicate)。

        duplicate=True 表示命中了幂等键，没有重复执行 —— 调用方应把它当成成功返回。
        """
        if action not in TRANSITIONS:
            return None, f"unknown_action:{action}", False

        with self._lock:
            existing = self._idem.get(idem_key)
            if existing is not None:
                return existing, None, True

            current = (self._state.get(order_id) or {}).get("status") or base_status
            if current is None:
                return None, "order_not_found", False
            if current not in TRANSITIONS[action]:
                return None, f"illegal_transition:{current}->{action}", False

            self._seq += 1
            record = LedgerRecord(
                seq=self._seq,
                ts=time.time(),
                action=action,
                order_id=order_id,
                idem_key=idem_key,
                payload={**(payload or {}), "prev_status": current},
            )
            # 事实源同步追加：这条不能异步，否则崩溃就查不到是谁改的
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

            self._idem[idem_key] = record
            self._apply_to_state(record)
            return record, None, False


def derive_idem(*parts: str, bucket_seconds: int = 300) -> str:
    """没有客户端幂等键时的兜底：同一主体 + 同一动作，在时间桶内视为同一次请求。

    真实系统应该由客户端生成 UUID 放在 header 里传进来 —— 服务端推导的键
    只能防"短时间重复提交"，防不了"隔很久的重复提交"。
    """
    bucket = int(time.time() // bucket_seconds)
    return "|".join([*parts, str(bucket)])


class SqliteLedger(OrderLedger):
    """SQLite 版账本：把幂等键的唯一性交给数据库约束，而不是进程内的锁。

    jsonl 版的 `self._lock` 只能挡住同一个进程内的并发。多 worker（uvicorn
    --workers、多容器）同时收到同一笔退款重试时，两边都读不到对方的幂等键，
    会各写一条事实 —— 钱就退两次。这里用 `idem_key UNIQUE` + `BEGIN IMMEDIATE`
    把互斥交给存储层，进程内锁只作为降低冲突的优化。

    其余语义（状态机、事件重放、补偿回滚）与 jsonl 版完全一致，两边接口相同。
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._seq = 0
        self._idem: dict[str, LedgerRecord] = {}
        self._state: dict[str, dict] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_events (
                seq       INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL    NOT NULL,
                action    TEXT    NOT NULL,
                order_id  TEXT    NOT NULL,
                idem_key  TEXT    NOT NULL UNIQUE,
                payload   TEXT    NOT NULL
            )
            """
        )
        self._conn.commit()
        self._load()

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT seq, ts, action, order_id, idem_key, payload FROM ledger_events ORDER BY seq"
        ).fetchall()
        for seq, ts, action, order_id, idem_key, payload in rows:
            record = LedgerRecord(
                seq=seq,
                ts=ts,
                action=action,
                order_id=order_id,
                idem_key=idem_key,
                payload=json.loads(payload),
            )
            self._seq = max(self._seq, record.seq)
            self._idem[record.idem_key] = record
            self._apply_to_state(record)

    def apply(
        self,
        action: str,
        order_id: str,
        idem_key: str,
        payload: dict | None = None,
        base_status: str | None = None,
    ) -> tuple[LedgerRecord | None, str | None, bool]:
        if action not in TRANSITIONS:
            return None, f"unknown_action:{action}", False

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT seq, ts, action, order_id, idem_key, payload "
                    "FROM ledger_events WHERE idem_key = ?",
                    (idem_key,),
                ).fetchone()
                if row is not None:
                    self._conn.execute("ROLLBACK")
                    existing = self._idem.get(idem_key) or LedgerRecord(
                        seq=row[0],
                        ts=row[1],
                        action=row[2],
                        order_id=row[3],
                        idem_key=row[4],
                        payload=json.loads(row[5]),
                    )
                    self._idem[idem_key] = existing
                    return existing, None, True

                current = (self._state.get(order_id) or {}).get("status") or base_status
                if current is None:
                    self._conn.execute("ROLLBACK")
                    return None, "order_not_found", False
                if current not in TRANSITIONS[action]:
                    self._conn.execute("ROLLBACK")
                    return None, f"illegal_transition:{current}->{action}", False

                self._seq += 1
                record = LedgerRecord(
                    seq=self._seq,
                    ts=time.time(),
                    action=action,
                    order_id=order_id,
                    idem_key=idem_key,
                    payload={**(payload or {}), "prev_status": current},
                )
                try:
                    self._conn.execute(
                        "INSERT INTO ledger_events (seq, ts, action, order_id, idem_key, payload) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            record.seq,
                            record.ts,
                            record.action,
                            record.order_id,
                            record.idem_key,
                            json.dumps(record.payload, ensure_ascii=False),
                        ),
                    )
                except sqlite3.IntegrityError:
                    # 另一个进程抢先写入了同一个幂等键：以对方的事实为准
                    self._conn.execute("ROLLBACK")
                    existing = self.find(idem_key)
                    return existing, None, True
                self._conn.execute("COMMIT")
                self._idem[idem_key] = record
                self._apply_to_state(record)
                return record, None, False
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        self._conn.close()


def build_ledger(path: Path) -> OrderLedger:
    """按文件后缀选后端：.db/.sqlite 走 SQLite，其余走 jsonl。"""
    return SqliteLedger(path) if path.suffix in (".db", ".sqlite") else OrderLedger(path)
