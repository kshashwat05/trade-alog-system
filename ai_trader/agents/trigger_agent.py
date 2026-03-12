from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

from loguru import logger

from ai_trader.agents.chart_agent import ChartAnalysis
from ai_trader.agents.option_chain_agent import OptionChainAnalysis
from ai_trader.agents.news_agent import NewsMacroAnalysis
from ai_trader.agents.volatility_agent import VolatilityAnalysis
from ai_trader.agents.regime_agent import RegimeAnalysis
from ai_trader.agents.liquidity_agent import LiquidityAnalysis
from ai_trader.agents.fii_positioning_agent import FiiPositioningAnalysis
from ai_trader.agents.gamma_agent import GammaAnalysis
from ai_trader.agents.liquidity_sweep_agent import LiquiditySweepAnalysis
if TYPE_CHECKING:
    from ai_trader.data.market_data_context import MarketDataContext

SignalType = Literal["BUY_PE", "BUY_CE", "NONE"]


@dataclass
class TradeSignal:
    signal: SignalType
    entry: float
    stop_loss: float
    target: float
    confidence: float
    rationale: str
    underlying_spot: float | None = None
    option_strike: int | None = None
    option_expiry: str | None = None
    instrument_symbol: str | None = None
    instrument_key: str | None = None
    price_source: str = "unknown"
    data_complete: bool = False


class TradeTriggerAgent:
    """Combines all analysis into a concrete trading signal."""

    MIN_CONFIDENCE = 0.55

    @staticmethod
    def _parse_option_contract(
        signal_type: SignalType,
        option_chain: OptionChainAnalysis,
        market_context: MarketDataContext | None,
        spot: float,
    ) -> dict[str, Any] | None:
        if market_context is None:
            return None
        rows = market_context.option_chain_raw.get("records", {}).get("data", [])
        if not rows:
            return None

        side = "PE" if signal_type == "BUY_PE" else "CE"
        target_strike = (
            option_chain.resistance if signal_type == "BUY_PE" else option_chain.support
        )
        if target_strike is None:
            target_strike = int(round(spot / 50.0) * 50)

        candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for row in rows:
            strike = row.get("strikePrice")
            contract = row.get(side) or {}
            last_price = contract.get("lastPrice")
            if strike is None or last_price in (None, 0, 0.0):
                continue
            distance = abs(float(strike) - float(target_strike))
            candidates.append((distance, row, contract))

        if not candidates:
            return None

        _, selected_row, selected_contract = min(candidates, key=lambda item: item[0])
        strike = int(selected_row["strikePrice"])
        expiry = (
            selected_contract.get("expiryDate")
            or selected_row.get("expiryDate")
            or market_context.option_chain_raw.get("records", {}).get("expiryDates", [None])[0]
        )
        instrument_symbol = (
            selected_contract.get("tradingsymbol")
            or selected_contract.get("identifier")
            or f"{side} {strike}"
        )
        return {
            "strike": strike,
            "expiry": str(expiry) if expiry is not None else None,
            "instrument_symbol": str(instrument_symbol),
            "entry_price": float(selected_contract["lastPrice"]),
        }

    def generate_signal(
        self,
        chart: ChartAnalysis,
        option_chain: OptionChainAnalysis,
        news: NewsMacroAnalysis,
        vol: VolatilityAnalysis,
        regime: RegimeAnalysis,
        liquidity: LiquidityAnalysis,
        fii: FiiPositioningAnalysis,
        gamma: GammaAnalysis,
        liquidity_sweep: LiquiditySweepAnalysis,
        spot: Optional[float] = None,
        market_context: MarketDataContext | None = None,
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
                underlying_spot=spot,
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
            and chart.structure in ("breakout", "reversal")
            and option_chain.bias == "bearish"
            and regime.regime == "trend_down"
            and fii.fii_bias == "bearish"
            and gamma.gamma_regime == "negative_gamma"
            and liquidity.liquidity == "high"
            and liquidity_sweep.event_type != "breakout_trap"
        ):
            base_signal = "BUY_PE"
            rationale = "Bearish trend, institutional positioning, and negative gamma are aligned."
        elif (
            chart.trend == "bullish"
            and chart.structure in ("breakout", "reversal")
            and option_chain.bias == "bullish"
            and regime.regime == "trend_up"
            and fii.fii_bias == "bullish"
            and gamma.gamma_regime == "negative_gamma"
            and liquidity.liquidity == "high"
            and liquidity_sweep.event_type != "breakout_trap"
        ):
            base_signal = "BUY_CE"
            rationale = "Bullish trend, institutional positioning, and negative gamma are aligned."

        if base_signal == "NONE":
            signal = TradeSignal(
                signal="NONE",
                entry=spot,
                stop_loss=spot,
                target=spot,
                confidence=0.0,
                rationale=rationale,
                underlying_spot=spot,
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
                underlying_spot=spot,
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
                underlying_spot=spot,
            )
            logger.info(f"TradeTriggerAgent signal: {signal}")
            return signal

        # Confidence as an aggregate of component confidences / buckets
        conf = 0.0
        conf += 0.3 * chart.confidence
        conf += 0.2 * regime.confidence
        conf += 0.1 * (1.0 if news.macro_bias in ("bullish", "bearish", "neutral") else 0.4)
        conf += 0.1 * (1.0 if vol.volatility == "medium" else 0.3)
        conf += 0.1 * (1.0 if liquidity.liquidity == "high" else 0.2)
        conf += 0.15 * fii.confidence
        conf += 0.05 * (1.0 if gamma.gamma_regime == "negative_gamma" else 0.2)
        conf += 0.1 * (1.0 if not liquidity_sweep.liquidity_event else 0.4)
        confidence = float(max(0.0, min(1.0, conf)))
        if confidence < self.MIN_CONFIDENCE:
            signal = TradeSignal(
                signal="NONE",
                entry=spot,
                stop_loss=spot,
                target=spot,
                confidence=confidence,
                rationale=f"Confidence {confidence:.2f} below threshold {self.MIN_CONFIDENCE:.2f}.",
                underlying_spot=spot,
            )
            logger.info(f"TradeTriggerAgent signal: {signal}")
            return signal

        option_contract = self._parse_option_contract(base_signal, option_chain, market_context, spot)
        if option_contract is None:
            signal = TradeSignal(
                signal="NONE",
                entry=spot,
                stop_loss=spot,
                target=spot,
                confidence=0.0,
                rationale="Option contract premium unavailable; refusing to emit a non-tradeable signal.",
                underlying_spot=spot,
            )
            logger.warning(f"TradeTriggerAgent signal: {signal}")
            return signal

        range_low, range_high = vol.expected_range
        range_width = max(range_high - range_low, spot * 0.004)
        entry = float(option_contract["entry_price"])
        premium_risk = max(entry * 0.18, (range_width / max(spot, 1.0)) * entry * 1.5)
        premium_reward = max(entry * 0.25, premium_risk * 1.6)
        if base_signal == "BUY_PE":
            stop_loss = float(max(entry - premium_risk, entry * 0.3))
            target = float(entry + premium_reward)
        else:
            stop_loss = float(max(entry - premium_risk, entry * 0.3))
            target = float(entry + premium_reward)

        signal = TradeSignal(
            signal=base_signal,
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            confidence=confidence,
            rationale=rationale,
            underlying_spot=spot,
            option_strike=option_contract["strike"],
            option_expiry=option_contract["expiry"],
            instrument_symbol=option_contract["instrument_symbol"],
            price_source="option_chain_last_price",
            data_complete=True,
        )
        logger.info(f"TradeTriggerAgent signal: {signal}")
        return signal
