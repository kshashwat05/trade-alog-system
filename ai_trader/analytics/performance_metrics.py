from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from ai_trader.data.trade_journal import TradeJournalEntry, STATUS_EXECUTED, STATUS_MISSED


@dataclass
class PerformanceSummary:
    trade_count: int
    win_rate: float
    average_profit: float
    average_loss: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    total_pnl: float


def _build_equity_curve(pnls: list[float]) -> list[float]:
    equity = []
    running = 0.0
    for pnl in pnls:
        running += pnl
        equity.append(running)
    return equity


def calculate_performance_metrics(trades: Iterable[TradeJournalEntry]) -> PerformanceSummary:
    pnl_values = [float(trade.pnl) for trade in trades if trade.pnl is not None]
    if not pnl_values:
        return PerformanceSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    profits = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    trade_count = len(pnl_values)
    win_rate = len(profits) / trade_count if trade_count else 0.0
    average_profit = sum(profits) / len(profits) if profits else 0.0
    average_loss = sum(losses) / len(losses) if losses else 0.0
    gross_profit = sum(profits)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = sum(pnl_values) / trade_count if trade_count else 0.0

    equity_curve = _build_equity_curve(pnl_values)
    peak = 0.0
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value - peak)

    return PerformanceSummary(
        trade_count=trade_count,
        win_rate=win_rate,
        average_profit=average_profit,
        average_loss=average_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        max_drawdown=abs(max_drawdown),
        total_pnl=sum(pnl_values),
    )


def summarize_by_outcome(trades: Iterable[TradeJournalEntry]) -> dict[str, dict[str, float | int]]:
    trades = list(trades)
    executed = [trade for trade in trades if trade.trade_executed or trade.status == STATUS_EXECUTED]
    missed = [trade for trade in trades if not trade.trade_executed or trade.status == STATUS_MISSED]
    return {
        "executed": asdict(calculate_performance_metrics(executed)),
        "missed": asdict(calculate_performance_metrics(missed)),
        "overall": asdict(calculate_performance_metrics(trades)),
    }
