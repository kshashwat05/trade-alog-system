from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from ai_trader.agents.trigger_agent import TradeSignal
from ai_trader.config.settings import settings
from ai_trader.data.market_data_context import MarketDataContext


@dataclass
class ExecutionIntelligenceDecision:
    allowed: bool
    reason: str
    entry_style: str
    consolidation_detected: bool
    extended_move: bool


class ExecutionIntelligenceAgent:
    """Adds execution-aware filters after signal generation."""

    @staticmethod
    def _in_consolidation(df: pd.DataFrame) -> bool:
        lookback = max(5, settings.consolidation_lookback_candles)
        window = df.tail(lookback)
        if window.empty:
            return True
        high = float(window["high"].max())
        low = float(window["low"].min())
        close = float(window.iloc[-1]["close"])
        if close <= 0:
            return True
        range_pct = ((high - low) / close) * 100.0
        return range_pct <= max(0.05, settings.consolidation_range_threshold_pct)

    @staticmethod
    def _entry_style(df: pd.DataFrame, signal: TradeSignal) -> str:
        if len(df) < 6:
            return "unknown"
        recent = df.tail(6)
        last = recent.iloc[-1]
        prev = recent.iloc[:-1]
        prior_high = float(prev["high"].max())
        prior_low = float(prev["low"].min())
        close = float(last["close"])
        high = float(last["high"])
        low = float(last["low"])
        open_price = float(last["open"])
        body = abs(close - open_price)
        full = max(high - low, 1e-6)
        lower_wick = min(open_price, close) - low
        upper_wick = high - max(open_price, close)

        if signal.signal == "BUY_CE":
            if close > prior_high:
                return "breakout"
            if lower_wick / full > 0.4 and body / full < 0.6:
                return "rejection"
        if signal.signal == "BUY_PE":
            if close < prior_low:
                return "breakout"
            if upper_wick / full > 0.4 and body / full < 0.6:
                return "rejection"
        return "mid_range"

    @staticmethod
    def _is_extended(df: pd.DataFrame, signal: TradeSignal) -> bool:
        if len(df) < max(10, settings.extension_ema_period):
            return False
        ema = df["close"].ewm(span=max(3, settings.extension_ema_period), adjust=False).mean()
        last_close = float(df.iloc[-1]["close"])
        last_ema = float(ema.iloc[-1])
        if last_ema <= 0:
            return False
        extension_pct = abs((last_close - last_ema) / last_ema) * 100.0
        if extension_pct < max(0.1, settings.extension_threshold_pct):
            return False
        recent = df.tail(4)["close"].tolist()
        if signal.signal == "BUY_CE":
            return all(recent[i] > recent[i - 1] for i in range(1, len(recent)))
        if signal.signal == "BUY_PE":
            return all(recent[i] < recent[i - 1] for i in range(1, len(recent)))
        return False

    def evaluate(
        self,
        signal: TradeSignal,
        market_context: MarketDataContext | None,
    ) -> ExecutionIntelligenceDecision:
        if signal.signal == "NONE":
            return ExecutionIntelligenceDecision(False, "No actionable signal.", "unknown", False, False)
        if market_context is None or market_context.price_df.empty:
            decision = ExecutionIntelligenceDecision(False, "No price context for execution intelligence.", "unknown", False, False)
            logger.warning(f"ExecutionIntelligence blocked: {decision.reason}")
            return decision

        df = market_context.price_df
        consolidation = self._in_consolidation(df)
        style = self._entry_style(df, signal)
        extended = self._is_extended(df, signal)

        if consolidation and style not in {"breakout", "rejection"}:
            decision = ExecutionIntelligenceDecision(
                False,
                "Entry inside consolidation without breakout/rejection confirmation.",
                style,
                consolidation,
                extended,
            )
            logger.warning(f"ExecutionIntelligence blocked: {decision}")
            return decision
        if style not in {"breakout", "rejection"}:
            decision = ExecutionIntelligenceDecision(
                False,
                "Entry style is not breakout or rejection.",
                style,
                consolidation,
                extended,
            )
            logger.warning(f"ExecutionIntelligence blocked: {decision}")
            return decision
        if extended:
            decision = ExecutionIntelligenceDecision(
                False,
                "Move appears extended relative to local trend baseline.",
                style,
                consolidation,
                extended,
            )
            logger.warning(f"ExecutionIntelligence blocked: {decision}")
            return decision

        decision = ExecutionIntelligenceDecision(True, "Execution filters passed.", style, consolidation, extended)
        logger.info(f"ExecutionIntelligence passed: {decision}")
        return decision
