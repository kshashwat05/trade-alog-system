from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from loguru import logger

from ai_trader.agents.exit_intelligence_agent import ExitIntelligenceSuggestion
from ai_trader.data.trade_journal import (
    STATUS_REVERSAL_EXIT,
    STATUS_STOP_LOSS_HIT,
    STATUS_TARGET_HIT,
    TradeJournal,
    TradeJournalEntry,
)


@dataclass
class PositionMonitorResult:
    signal_id: int
    status: str
    current_price: float
    pnl: float
    message: str
    advisory_only: bool = False
    exit_suggestion: ExitIntelligenceSuggestion | None = None


class PositionMonitorAgent:
    def __init__(
        self,
        journal: TradeJournal,
        price_fetcher: Callable[[TradeJournalEntry], float | None],
        exit_intelligence_cb: Callable[[TradeJournalEntry, float], ExitIntelligenceSuggestion] | None = None,
        exit_cleanup_cb: Callable[[int], None] | None = None,
    ) -> None:
        self.journal = journal
        self.price_fetcher = price_fetcher
        self.exit_intelligence_cb = exit_intelligence_cb
        self.exit_cleanup_cb = exit_cleanup_cb

    @staticmethod
    def _determine_exit(entry: TradeJournalEntry, current_price: float) -> tuple[str | None, float]:
        execution_price = entry.execution_price or entry.entry_price
        quantity = entry.quantity or 1
        pnl = (current_price - execution_price) * quantity
        reversal_threshold = execution_price - ((execution_price - entry.stop_loss) * 0.5)
        if current_price >= entry.target:
            return STATUS_TARGET_HIT, pnl
        if current_price <= entry.stop_loss:
            return STATUS_STOP_LOSS_HIT, pnl
        if current_price <= reversal_threshold:
            return STATUS_REVERSAL_EXIT, pnl
        return None, pnl

    def monitor_once(self) -> list[PositionMonitorResult]:
        results: list[PositionMonitorResult] = []
        for trade in self.journal.get_open_trades():
            instrument_key = trade.metadata.get("instrument_key")
            if not instrument_key:
                logger.info(
                    f"Position monitor skipped trade_id={trade.id} because no instrument_key metadata is available."
                )
                continue
            current_price = self.price_fetcher(trade)
            if current_price is None:
                logger.warning(f"Position monitor could not fetch current price for trade_id={trade.id}.")
                continue

            suggestion: ExitIntelligenceSuggestion | None = None
            exit_status, pnl = self._determine_exit(trade, current_price)
            if exit_status is None:
                if self.exit_intelligence_cb is not None:
                    try:
                        suggestion = self.exit_intelligence_cb(trade, current_price)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(f"Exit intelligence callback failed for trade_id={trade.id}: {exc}")
                if suggestion is not None and suggestion.action != "NONE":
                    advisory_message = (
                        "NIFTY EXIT ADVISORY\n\n"
                        f"Trade ID: {trade.id}\n"
                        f"Signal: {trade.signal_type}\n"
                        f"Current Premium: {current_price:.2f}\n\n"
                        f"Suggestion: {suggestion.action}\n"
                        f"Confidence: {suggestion.confidence * 100:.0f}%\n"
                        f"Reason: {suggestion.reason}"
                    )
                    results.append(
                        PositionMonitorResult(
                            signal_id=trade.id,
                            status=suggestion.action,
                            current_price=current_price,
                            pnl=pnl,
                            message=advisory_message,
                            advisory_only=True,
                            exit_suggestion=suggestion,
                        )
                    )
                    logger.info(
                        f"Exit intelligence advisory for trade_id={trade.id}: "
                        f"{suggestion.action} ({suggestion.reason})"
                    )
                continue

            message = (
                "NIFTY TRADE EXIT\n\n"
                f"{trade.signal_type.replace('_', ' ')}\n\n"
                f"Entry: {(trade.execution_price or trade.entry_price):.2f}\n"
                f"Current: {current_price:.2f}\n\n"
                f"{exit_status.replace('_', ' ').upper()}\n\n"
                + ("Book profit now." if exit_status == STATUS_TARGET_HIT else "Exit now.")
            )
            self.journal.close_trade(
                trade.id,
                status=exit_status,
                exit_price=current_price,
                pnl=pnl,
            )
            if self.exit_cleanup_cb is not None:
                try:
                    self.exit_cleanup_cb(trade.id)
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Exit cleanup callback failed for trade_id={trade.id}: {exc}")
            result = PositionMonitorResult(
                signal_id=trade.id,
                status=exit_status,
                current_price=current_price,
                pnl=pnl,
                message=message,
                advisory_only=False,
                exit_suggestion=suggestion,
            )
            results.append(result)
            logger.info(f"Position monitor exit for trade_id={trade.id}: {result}")
        return results
