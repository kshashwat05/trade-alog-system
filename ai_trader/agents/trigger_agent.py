from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from loguru import logger

from ai_trader.agents.chart_agent import ChartAnalysis
from ai_trader.agents.option_chain_agent import OptionChainAnalysis
from ai_trader.agents.news_agent import NewsMacroAnalysis
from ai_trader.agents.volatility_agent import VolatilityAnalysis
from ai_trader.agents.regime_agent import RegimeAnalysis
from ai_trader.agents.liquidity_agent import LiquidityAnalysis

SignalType = Literal["BUY_PE", "BUY_CE", "NONE"]


@dataclass
class TradeSignal:
    signal: SignalType
    entry: float
    stop_loss: float
    target: float
    confidence: float
    rationale: str


class TradeTriggerAgent:
    """Combines all analysis into a concrete trading signal."""

    MIN_CONFIDENCE = 0.55

    def generate_signal(
        self,
        chart: ChartAnalysis,
        option_chain: OptionChainAnalysis,
        news: NewsMacroAnalysis,
        vol: VolatilityAnalysis,
        regime: RegimeAnalysis,
        liquidity: LiquidityAnalysis,
        spot: Optional[float] = None,
    ) -> TradeSignal:
        # Default: no trade
        if spot is None:
            # Roughly center of expected range
            spot = sum(vol.expected_range) / 2.0
        if spot <= 0:
            logger.warning(f"TradeTriggerAgent received non-positive spot {spot}; forcing NONE.")
            return TradeSignal(
                signal="NONE",
                entry=0.0,
                stop_loss=0.0,
                target=0.0,
                confidence=0.0,
                rationale="Invalid spot price.",
            )

        base_signal: SignalType = "NONE"
        confidence = 0.0
        rationale = "Signals are not aligned."
        directional_bias = {
            "BUY_CE": "bullish",
            "BUY_PE": "bearish",
        }

        # Example heuristic from spec
        if (
            chart.trend == "bearish"
            and option_chain.bias == "bearish"
            and regime.regime == "trend_down"
            and liquidity.liquidity != "low"
        ):
            base_signal = "BUY_PE"
            rationale = "Chart, option chain, and regime all point bearish."
        elif (
            chart.trend == "bullish"
            and option_chain.bias == "bullish"
            and regime.regime == "trend_up"
            and liquidity.liquidity != "low"
        ):
            base_signal = "BUY_CE"
            rationale = "Chart, option chain, and regime all point bullish."

        if base_signal == "NONE":
            signal = TradeSignal(
                signal="NONE",
                entry=spot,
                stop_loss=spot,
                target=spot,
                confidence=0.0,
                rationale=rationale,
            )
            logger.info(f"TradeTriggerAgent signal: {signal}")
            return signal

        expected_news_bias = directional_bias[base_signal]
        if news.macro_bias not in ("neutral", expected_news_bias):
            signal = TradeSignal(
                signal="NONE",
                entry=spot,
                stop_loss=spot,
                target=spot,
                confidence=0.0,
                rationale=f"News bias {news.macro_bias} conflicts with {base_signal}.",
            )
            logger.info(f"TradeTriggerAgent signal: {signal}")
            return signal

        if vol.volatility == "high":
            signal = TradeSignal(
                signal="NONE",
                entry=spot,
                stop_loss=spot,
                target=spot,
                confidence=0.0,
                rationale="Volatility too high for a directional intraday entry.",
            )
            logger.info(f"TradeTriggerAgent signal: {signal}")
            return signal

        # Confidence as an aggregate of component confidences / buckets
        conf = 0.0
        conf += 0.3 * chart.confidence
        conf += 0.2 * regime.confidence
        conf += 0.2 * (1.0 if news.macro_bias in ("bullish", "bearish") else 0.5)
        conf += 0.1 * (1.0 if vol.volatility != "high" else 0.6)
        conf += 0.2 * (1.0 if liquidity.liquidity == "high" else 0.6)
        confidence = float(max(0.0, min(1.0, conf)))
        if confidence < self.MIN_CONFIDENCE:
            signal = TradeSignal(
                signal="NONE",
                entry=spot,
                stop_loss=spot,
                target=spot,
                confidence=confidence,
                rationale=f"Confidence {confidence:.2f} below threshold {self.MIN_CONFIDENCE:.2f}.",
            )
            logger.info(f"TradeTriggerAgent signal: {signal}")
            return signal

        range_low, range_high = vol.expected_range
        range_width = max(range_high - range_low, spot * 0.004)
        entry = float(spot)
        if base_signal == "BUY_PE":
            stop_loss = float(entry + range_width * 0.25)
            target = float(entry - range_width * 0.5)
        else:
            stop_loss = float(entry - range_width * 0.25)
            target = float(entry + range_width * 0.5)

        signal = TradeSignal(
            signal=base_signal,
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            confidence=confidence,
            rationale=rationale,
        )
        logger.info(f"TradeTriggerAgent signal: {signal}")
        return signal
