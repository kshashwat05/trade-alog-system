from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from ai_trader.agents.trigger_agent import TradeSignal
from ai_trader.data.market_data_context import MarketDataContext


@dataclass
class DataQualityDecision:
    allowed: bool
    reason: str


class DataQualityGuard:
    """Blocks trades when market data quality is degraded."""

    def evaluate(
        self,
        signal: TradeSignal,
        market_context: MarketDataContext | None,
        *,
        score_complete: bool,
    ) -> DataQualityDecision:
        if signal.signal == "NONE":
            return DataQualityDecision(False, "No actionable signal.")
        if market_context is None:
            decision = DataQualityDecision(False, "Market context missing.")
            logger.warning(f"DataQualityGuard blocked: {decision.reason}")
            return decision
        if not score_complete:
            decision = DataQualityDecision(False, "Score marked incomplete.")
            logger.warning(f"DataQualityGuard blocked: {decision.reason}")
            return decision

        quality = market_context.quality
        if not quality.critical_inputs_available:
            decision = DataQualityDecision(
                False,
                f"Critical data unavailable/freshness failed: {quality.issues or []}",
            )
            logger.warning(f"DataQualityGuard blocked: {decision.reason}")
            return decision
        if quality.used_price_fallback or quality.used_option_chain_fallback or quality.used_vix_fallback:
            decision = DataQualityDecision(False, "Fallback market data detected.")
            logger.warning(f"DataQualityGuard blocked: {decision.reason}")
            return decision
        if not signal.data_complete:
            decision = DataQualityDecision(False, "Signal data completeness is false.")
            logger.warning(f"DataQualityGuard blocked: {decision.reason}")
            return decision

        decision = DataQualityDecision(True, "Market data quality passed.")
        logger.info(f"DataQualityGuard passed: {decision.reason}")
        return decision
