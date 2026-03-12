from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from ai_trader.data.kite_client import KiteClient
from ai_trader.data.trade_journal import TradeJournal, TradeJournalEntry


@dataclass
class MissedTradeAnalysis:
    signal_id: int
    max_profit: float
    max_drawdown: float
    target_hit: bool
    stop_loss_hit: bool
    final_pnl: float | None


class MissedTradeAnalyzer:
    def __init__(self, journal: TradeJournal, client: KiteClient | None = None) -> None:
        self.journal = journal
        self.client = client or KiteClient()

    @staticmethod
    def _analyze_series(trade: TradeJournalEntry, prices: list[float]) -> MissedTradeAnalysis:
        if not prices:
            return MissedTradeAnalysis(
                signal_id=trade.id,
                max_profit=0.0,
                max_drawdown=0.0,
                target_hit=False,
                stop_loss_hit=False,
                final_pnl=None,
            )

        entry = trade.entry_price
        pnl_values: list[float] = []
        target_hit = False
        stop_loss_hit = False
        for price in prices:
            pnl = price - entry if trade.signal_type == "BUY_CE" else entry - price
            pnl_values.append(pnl)
            if trade.signal_type == "BUY_CE":
                target_hit = target_hit or price >= trade.target
                stop_loss_hit = stop_loss_hit or price <= trade.stop_loss
            else:
                target_hit = target_hit or price <= trade.target
                stop_loss_hit = stop_loss_hit or price >= trade.stop_loss

        return MissedTradeAnalysis(
            signal_id=trade.id,
            max_profit=max(pnl_values),
            max_drawdown=min(pnl_values),
            target_hit=target_hit,
            stop_loss_hit=stop_loss_hit,
            final_pnl=pnl_values[-1],
        )

    def analyze_trade(self, signal_id: int) -> MissedTradeAnalysis:
        trade = self.journal.get_trade(signal_id)
        if trade is None:
            raise ValueError(f"Unknown trade id {signal_id}")

        candles = self.client.fetch_nifty_intraday(days=1).df
        if candles.empty:
            analysis = self._analyze_series(trade, [])
        else:
            signal_timestamp = datetime.fromisoformat(trade.timestamp)
            series = candles[candles["date"] >= signal_timestamp]["close"].tolist()
            analysis = self._analyze_series(trade, series)

        self.journal.update_simulation(
            signal_id,
            max_profit=analysis.max_profit,
            max_drawdown=analysis.max_drawdown,
            target_hit=analysis.target_hit,
            stop_loss_hit=analysis.stop_loss_hit,
            pnl=analysis.final_pnl,
        )
        logger.info(f"Missed trade analysis completed for trade_id={signal_id}: {analysis}")
        return analysis
