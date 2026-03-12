from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from threading import Lock

from loguru import logger

from ai_trader.config.settings import settings
from ai_trader.agents.trigger_agent import TradeSignal


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str | None = None


class RiskManagerAgent:
    """Applies hard risk limits to proposed trades."""

    def __init__(self) -> None:
        self._trade_count_by_day: dict[date, int] = {}
        self._pnl_by_day: dict[date, float] = {}
        self._lock = Lock()

    def record_trade_result(self, trade_date: date, pnl: float) -> None:
        with self._lock:
            self._pnl_by_day[trade_date] = self._pnl_by_day.get(trade_date, 0.0) + pnl

    def record_trade_open(self, trade_date: date) -> None:
        with self._lock:
            self._trade_count_by_day[trade_date] = self._trade_count_by_day.get(trade_date, 0) + 1

    def _get_today(self) -> date:
        return date.today()

    def record_trade_open_today(self) -> None:
        self.record_trade_open(self._get_today())

    def check(self, signal: TradeSignal, lots: int = 1) -> RiskCheckResult:
        today = self._get_today()
        with self._lock:
            day_trades = self._trade_count_by_day.get(today, 0)
            day_pnl = self._pnl_by_day.get(today, 0.0)

        if day_pnl <= -settings.max_daily_loss:
            reason = f"Daily loss limit exceeded: {day_pnl} <= -{settings.max_daily_loss}"
            logger.warning(reason)
            return RiskCheckResult(allowed=False, reason=reason)

        if day_trades >= settings.max_trades_per_day:
            reason = f"Max trades per day exceeded: {day_trades} >= {settings.max_trades_per_day}"
            logger.warning(reason)
            return RiskCheckResult(allowed=False, reason=reason)

        if lots > settings.max_position_lots:
            reason = f"Requested lots {lots} > max_position_lots {settings.max_position_lots}"
            logger.warning(reason)
            return RiskCheckResult(allowed=False, reason=reason)

        if signal.signal == "NONE":
            return RiskCheckResult(allowed=False, reason="No trade signal.")

        if signal.entry <= 0:
            return RiskCheckResult(allowed=False, reason="Signal entry price must be positive.")

        if signal.signal == "BUY_CE":
            risk_per_unit = signal.entry - signal.stop_loss
            reward_per_unit = signal.target - signal.entry
        else:
            risk_per_unit = signal.stop_loss - signal.entry
            reward_per_unit = signal.entry - signal.target

        if risk_per_unit <= 0 or reward_per_unit <= 0:
            reason = (
                f"Invalid trade geometry. risk_per_unit={risk_per_unit}, "
                f"reward_per_unit={reward_per_unit}"
            )
            logger.warning(reason)
            return RiskCheckResult(allowed=False, reason=reason)

        logger.info("RiskManagerAgent: trade allowed.")
        return RiskCheckResult(allowed=True, reason=None)
