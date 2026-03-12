from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

from loguru import logger


STATUS_SIGNAL_GENERATED = "signal_generated"
STATUS_EXECUTED = "executed"
STATUS_MISSED = "missed"
STATUS_TARGET_HIT = "target_hit"
STATUS_STOP_LOSS_HIT = "stop_loss_hit"
STATUS_MANUAL_EXIT = "manual_exit"
STATUS_REVERSAL_EXIT = "reversal_exit"

OPEN_STATUSES = {STATUS_EXECUTED}


@dataclass
class TradeJournalEntry:
    id: int
    timestamp: str
    signal_type: str
    entry_price: float
    stop_loss: float
    target: float
    confidence: float
    trade_executed: bool
    execution_price: float | None
    exit_price: float | None
    pnl: float | None
    quantity: int | None
    status: str
    decision_score: int | None
    rationale: str | None
    metadata: dict[str, Any]
    max_profit: float | None
    max_drawdown: float | None
    target_hit: bool
    stop_loss_hit: bool


class TradeJournal:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    target REAL NOT NULL,
                    confidence REAL NOT NULL,
                    trade_executed INTEGER NOT NULL DEFAULT 0,
                    execution_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    quantity INTEGER,
                    status TEXT NOT NULL,
                    decision_score INTEGER,
                    rationale TEXT,
                    metadata TEXT,
                    max_profit REAL,
                    max_drawdown REAL,
                    target_hit INTEGER NOT NULL DEFAULT 0,
                    stop_loss_hit INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)"
            )
            self._conn.commit()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> TradeJournalEntry:
        return TradeJournalEntry(
            id=int(row["id"]),
            timestamp=str(row["timestamp"]),
            signal_type=str(row["signal_type"]),
            entry_price=float(row["entry_price"]),
            stop_loss=float(row["stop_loss"]),
            target=float(row["target"]),
            confidence=float(row["confidence"]),
            trade_executed=bool(row["trade_executed"]),
            execution_price=row["execution_price"],
            exit_price=row["exit_price"],
            pnl=row["pnl"],
            quantity=row["quantity"],
            status=str(row["status"]),
            decision_score=row["decision_score"],
            rationale=row["rationale"],
            metadata=json.loads(row["metadata"] or "{}"),
            max_profit=row["max_profit"],
            max_drawdown=row["max_drawdown"],
            target_hit=bool(row["target_hit"]),
            stop_loss_hit=bool(row["stop_loss_hit"]),
        )

    def record_signal(
        self,
        *,
        timestamp: datetime,
        signal_type: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        confidence: float,
        decision_score: int,
        rationale: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        payload = json.dumps(metadata or {}, default=str)
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO trades (
                    timestamp, signal_type, entry_price, stop_loss, target, confidence,
                    trade_executed, status, decision_score, rationale, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    timestamp.isoformat(),
                    signal_type,
                    entry_price,
                    stop_loss,
                    target,
                    confidence,
                    STATUS_SIGNAL_GENERATED,
                    decision_score,
                    rationale,
                    payload,
                ),
            )
            self._conn.commit()
            trade_id = int(cursor.lastrowid)
        logger.info(f"Trade journal recorded signal trade_id={trade_id} signal_type={signal_type}.")
        return trade_id

    def record_execution(self, signal_id: int, execution_price: float, quantity: int) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE trades
                SET trade_executed = 1, execution_price = ?, quantity = ?, status = ?
                WHERE id = ?
                """,
                (execution_price, quantity, STATUS_EXECUTED, signal_id),
            )
            self._conn.commit()
        logger.info(f"Trade journal marked trade_id={signal_id} as executed.")

    def mark_trade_missed(self, signal_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE trades SET status = ? WHERE id = ? AND trade_executed = 0",
                (STATUS_MISSED, signal_id),
            )
            self._conn.commit()
        logger.info(f"Trade journal marked trade_id={signal_id} as missed.")

    def close_trade(self, signal_id: int, *, status: str, exit_price: float, pnl: float | None) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE trades
                SET status = ?, exit_price = ?, pnl = ?
                WHERE id = ?
                """,
                (status, exit_price, pnl, signal_id),
            )
            self._conn.commit()
        logger.info(f"Trade journal closed trade_id={signal_id} with status={status}.")

    def update_simulation(
        self,
        signal_id: int,
        *,
        max_profit: float,
        max_drawdown: float,
        target_hit: bool,
        stop_loss_hit: bool,
        pnl: float | None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE trades
                SET max_profit = ?, max_drawdown = ?, target_hit = ?, stop_loss_hit = ?,
                    pnl = COALESCE(pnl, ?)
                WHERE id = ?
                """,
                (max_profit, max_drawdown, int(target_hit), int(stop_loss_hit), pnl, signal_id),
            )
            self._conn.commit()
        logger.info(f"Trade journal stored simulation metrics for trade_id={signal_id}.")

    def get_trade(self, signal_id: int) -> TradeJournalEntry | None:
        row = self._conn.execute("SELECT * FROM trades WHERE id = ?", (signal_id,)).fetchone()
        return self._row_to_entry(row) if row else None

    def get_open_trades(self) -> list[TradeJournalEntry]:
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE status = ? ORDER BY timestamp ASC",
            (STATUS_EXECUTED,),
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_recent_signals(self, limit: int = 20) -> list[TradeJournalEntry]:
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_all_trades(self) -> list[TradeJournalEntry]:
        rows = self._conn.execute("SELECT * FROM trades ORDER BY timestamp ASC").fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_trades_by_status(self, statuses: Iterable[str]) -> list[TradeJournalEntry]:
        statuses = list(statuses)
        placeholders = ",".join("?" for _ in statuses)
        rows = self._conn.execute(
            f"SELECT * FROM trades WHERE status IN ({placeholders}) ORDER BY timestamp DESC",
            tuple(statuses),
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def serialize_entries(entries: list[TradeJournalEntry]) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in entries]
