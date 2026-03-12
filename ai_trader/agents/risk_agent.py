from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from threading import Lock
from typing import Callable

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
        self._now_provider: Callable[[], datetime] = datetime.now
        self._trade_count_by_day: dict[date, int] = {}
        self._pnl_by_day: dict[date, float] = {}
        self._lock = Lock()
        self._last_signal_at: datetime | None = None

    def record_trade_result(self, trade_date: date, pnl: float) -> None:
        with self._lock:
            self._pnl_by_day[trade_date] = self._pnl_by_day.get(trade_date, 0.0) + pnl

    def record_trade_open(self, trade_date: date) -> None:
        with self._lock:
            self._trade_count_by_day[trade_date] = self._trade_count_by_day.get(trade_date, 0) + 1

    def _get_today(self) -> date:
        return self._now_provider().date()

    def set_now_provider(self, provider: Callable[[], datetime]) -> None:
        with self._lock:
            self._now_provider = provider

    def record_trade_open_today(self) -> None:
        self.record_trade_open(self._get_today())

    def record_signal_emitted(self, emitted_at: datetime | None = None) -> None:
        with self._lock:
            self._last_signal_at = emitted_at or self._now_provider()

    def _evaluate_locked(
        self,
        signal: TradeSignal,
        lots: int,
        *,
        open_trades: int,
        liquidity: str,
        volatility: str,
        news_risk: str,
        reserve: bool,
    ) -> RiskCheckResult:
        today = self._get_today()
        day_trades = self._trade_count_by_day.get(today, 0)
        day_pnl = self._pnl_by_day.get(today, 0.0)
        last_signal_at = self._last_signal_at
        now = self._now_provider()

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

        if open_trades >= settings.max_open_trades:
            return RiskCheckResult(
                allowed=False,
                reason=f"Open trades limit reached: {open_trades} >= {settings.max_open_trades}",
            )

        if liquidity == "low":
            return RiskCheckResult(allowed=False, reason="Liquidity too low for execution.")

        if volatility == "low":
            return RiskCheckResult(allowed=False, reason="Volatility too low for institutional setup.")

        if news_risk == "high":
            return RiskCheckResult(allowed=False, reason="News risk too high.")

        if last_signal_at is not None:
            cooldown = timedelta(minutes=settings.signal_cooldown_minutes)
            if now - last_signal_at < cooldown:
                reason = (
                    f"Signal cooldown active until "
                    f"{(last_signal_at + cooldown).isoformat(timespec='seconds')}"
                )
                logger.warning(reason)
                return RiskCheckResult(allowed=False, reason=reason)

        if signal.entry <= 0:
            return RiskCheckResult(allowed=False, reason="Signal entry price must be positive.")

        risk_per_unit = signal.entry - signal.stop_loss
        reward_per_unit = signal.target - signal.entry

        if risk_per_unit <= 0 or reward_per_unit <= 0:
            reason = (
                f"Invalid trade geometry. risk_per_unit={risk_per_unit}, "
                f"reward_per_unit={reward_per_unit}"
            )
            logger.warning(reason)
            return RiskCheckResult(allowed=False, reason=reason)

        if reserve:
            self._trade_count_by_day[today] = day_trades + 1
            self._last_signal_at = now
            logger.info("RiskManagerAgent: trade allowed and signal slot reserved.")
        else:
            logger.info("RiskManagerAgent: trade allowed.")
        return RiskCheckResult(allowed=True, reason=None)

    def check(
        self,
        signal: TradeSignal,
        lots: int = 1,
        *,
        open_trades: int = 0,
        liquidity: str = "medium",
        volatility: str = "medium",
        news_risk: str = "medium",
    ) -> RiskCheckResult:
        with self._lock:
            return self._evaluate_locked(
                signal,
                lots,
                open_trades=open_trades,
                liquidity=liquidity,
                volatility=volatility,
                news_risk=news_risk,
                reserve=False,
            )

    def authorize_signal(
        self,
        signal: TradeSignal,
        lots: int = 1,
        *,
        open_trades: int = 0,
        liquidity: str = "medium",
        volatility: str = "medium",
        news_risk: str = "medium",
    ) -> RiskCheckResult:
        with self._lock:
            return self._evaluate_locked(
                signal,
                lots,
                open_trades=open_trades,
                liquidity=liquidity,
                volatility=volatility,
                news_risk=news_risk,
                reserve=True,
            )
