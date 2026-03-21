from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from ai_trader.config.settings import settings
from ai_trader.data.trade_journal import TradeJournal, TradeJournalEntry


@dataclass
class KillSwitchDecision:
    blocked: bool
    reason: str
    consecutive_losses: int


class KillSwitchAgent:
    """Blocks new trades after consecutive realized losses."""

    @staticmethod
    def _realized_at(entry: TradeJournalEntry) -> datetime:
        return datetime.fromisoformat(entry.closed_at or entry.timestamp)

    @staticmethod
    def _closed_realized_trades(entries: list[TradeJournalEntry]) -> list[TradeJournalEntry]:
        closed = [entry for entry in entries if entry.trade_executed and entry.pnl is not None and entry.status != "executed"]
        if not closed:
            return []
        latest_day = KillSwitchAgent._realized_at(closed[-1]).date()
        return [entry for entry in closed if KillSwitchAgent._realized_at(entry).date() == latest_day]

    def evaluate(self, journal: TradeJournal) -> KillSwitchDecision:
        closed = self._closed_realized_trades(journal.get_all_trades())
        threshold = max(1, settings.kill_switch_consecutive_losses)
        consecutive = 0
        for trade in reversed(closed):
            if float(trade.pnl or 0.0) < 0:
                consecutive += 1
            else:
                break
        if consecutive >= threshold:
            decision = KillSwitchDecision(
                blocked=True,
                reason=f"Kill switch triggered after {consecutive} consecutive losses.",
                consecutive_losses=consecutive,
            )
            logger.warning(f"KillSwitch blocked: {decision.reason}")
            return decision
        decision = KillSwitchDecision(
            blocked=False,
            reason="Kill switch not triggered.",
            consecutive_losses=consecutive,
        )
        logger.info(f"KillSwitch passed: losses={consecutive}/{threshold}")
        return decision
