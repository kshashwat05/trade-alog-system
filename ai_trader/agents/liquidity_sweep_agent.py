from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from ai_trader.data.market_data_context import MarketDataContext

LiquidityEventType = Literal["stop_hunt", "breakout_trap", "none"]


@dataclass
class LiquiditySweepAnalysis:
    liquidity_event: bool
    event_type: LiquidityEventType
    confidence: float


class LiquiditySweepAgent:
    def analyze(self, context: MarketDataContext) -> LiquiditySweepAnalysis:
        df = context.price_df
        if df.empty or len(df) < 5:
            analysis = LiquiditySweepAnalysis(False, "none", 0.1)
            logger.info("LiquiditySweepAgent insufficient candles; returning no event.")
            return analysis

        recent = df.tail(5).copy()
        recent["range"] = recent["high"] - recent["low"]
        recent["body"] = (recent["close"] - recent["open"]).abs()
        recent["upper_wick"] = recent["high"] - recent[["open", "close"]].max(axis=1)
        recent["lower_wick"] = recent[["open", "close"]].min(axis=1) - recent["low"]
        last = recent.iloc[-1]
        avg_volume = float(recent["volume"].mean()) if "volume" in recent else 0.0
        volume_spike = float(last.get("volume", 0.0)) > avg_volume * 1.5 if avg_volume > 0 else False
        wick_ratio = max(float(last["upper_wick"]), float(last["lower_wick"])) / max(float(last["range"]), 1e-6)

        if volume_spike and wick_ratio > 0.45:
            event_type: LiquidityEventType = "stop_hunt"
            confidence = min(1.0, 0.5 + wick_ratio / 2.0)
            analysis = LiquiditySweepAnalysis(True, event_type, confidence)
        elif volume_spike and float(last["body"]) < float(last["range"]) * 0.3:
            analysis = LiquiditySweepAnalysis(True, "breakout_trap", 0.6)
        else:
            analysis = LiquiditySweepAnalysis(False, "none", 0.2)
        logger.info(f"LiquiditySweepAgent analysis: {analysis}")
        return analysis
