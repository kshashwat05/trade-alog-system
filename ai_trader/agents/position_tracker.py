from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from ai_trader.data.trade_journal import TradeJournal, TradeJournalEntry


@dataclass
class PositionSyncReport:
    synced_positions: int
    mismatches: int
    details: list[str]


class PositionTracker:
    """Checks journal consistency between intended and recorded execution symbols."""

    @staticmethod
    def _normalize_symbol(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = "".join(ch for ch in value.upper() if ch.isalnum())
        return normalized or None

    @staticmethod
    def _extract_signal_symbol(trade: TradeJournalEntry) -> str | None:
        symbol = trade.metadata.get("instrument_symbol")
        return PositionTracker._normalize_symbol(str(symbol)) if symbol else None

    @staticmethod
    def _extract_executed_symbol(trade: TradeJournalEntry) -> str | None:
        key = trade.metadata.get("instrument_key")
        if not key or ":" not in str(key):
            return None
        return PositionTracker._normalize_symbol(str(key).split(":", 1)[1])

    def sync_with_journal(self, journal: TradeJournal) -> PositionSyncReport:
        details: list[str] = []
        executed = [trade for trade in journal.get_all_trades() if trade.trade_executed]
        mismatch_count = 0
        for trade in executed:
            signal_symbol = self._extract_signal_symbol(trade)
            exec_symbol = self._extract_executed_symbol(trade)
            if signal_symbol and exec_symbol and signal_symbol != exec_symbol:
                mismatch_count += 1
                details.append(
                    f"trade_id={trade.id} signal_symbol={signal_symbol} executed_symbol={exec_symbol}"
                )

        report = PositionSyncReport(
            synced_positions=len(executed),
            mismatches=mismatch_count,
            details=details,
        )
        logger.info(
            f"PositionTracker sync complete: synced={report.synced_positions}, mismatches={report.mismatches}"
        )
        return report

    def record_alert_delivery(self, journal: TradeJournal, trade_id: int | None) -> None:
        if trade_id is None:
            return
        try:
            journal.merge_metadata(
                trade_id,
                {
                    "alert_sent_at": datetime.utcnow().isoformat(),
                    "tracked_by_position_tracker": True,
                },
            )
            logger.info(f"PositionTracker marked alert-delivered for trade_id={trade_id}.")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"PositionTracker failed to update trade metadata for trade_id={trade_id}: {exc}")
