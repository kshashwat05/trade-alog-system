from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from loguru import logger
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

from ai_trader.data.kite_client import KiteClient
if TYPE_CHECKING:
    from ai_trader.data.market_data_context import MarketDataContext

Trend = Literal["bullish", "bearish", "neutral"]
Structure = Literal["breakout", "reversal", "range", "unknown"]


@dataclass
class ChartAnalysis:
    trend: Trend
    structure: Structure
    confidence: float


class ChartAgent:
    """Analyzes NIFTY price data and returns a high-level chart view."""

    def __init__(self, client: KiteClient | None = None) -> None:
        self.client = client or KiteClient()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"open", "high", "low", "close", "volume"}
        if df.empty:
            return df
        if missing := required_columns.difference(df.columns):
            logger.warning(f"ChartAgent missing required columns {sorted(missing)}; returning empty frame.")
            return df

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].replace(0, np.nan)

        df = df.copy()
        df["ema_9"] = EMAIndicator(close=close, window=9).ema_indicator()
        df["ema_20"] = EMAIndicator(close=close, window=20).ema_indicator()

        # Simple VWAP approximation on intraday data
        typical_price = (high + low + close) / 3.0
        df["vwap"] = (typical_price * volume).cumsum() / volume.cumsum()

        df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
        return df

    def _detect_trend_and_structure(self, df: pd.DataFrame) -> ChartAnalysis:
        if df.empty or "ema_9" not in df or "ema_20" not in df:
            logger.warning("ChartAgent received empty data; returning neutral signal.")
            return ChartAnalysis(trend="neutral", structure="unknown", confidence=0.1)

        if len(df) < 20:
            logger.info("ChartAgent has fewer than 20 candles; downgrading structure confidence.")

        last = df.iloc[-1]
        ema_9 = last["ema_9"]
        ema_20 = last["ema_20"]
        close = last["close"]
        vwap = last.get("vwap", np.nan)

        # Trend via EMA stack
        if ema_9 > ema_20:
            trend: Trend = "bullish"
        elif ema_9 < ema_20:
            trend = "bearish"
        else:
            trend = "neutral"

        # Structure via recent highs / lows
        window_df = df.tail(20)
        highs = window_df["high"]
        lows = window_df["low"]
        rolling_high = highs.rolling(5, min_periods=1).max().iloc[-1]
        rolling_low = lows.rolling(5, min_periods=1).min().iloc[-1]
        hh = highs.iloc[-1] >= rolling_high
        ll = lows.iloc[-1] <= rolling_low

        structure: Structure = "range"
        if trend == "bullish" and hh:
            structure = "breakout"
        elif trend == "bearish" and ll:
            structure = "breakout"
        elif trend == "bullish" and close < vwap:
            structure = "reversal"
        elif trend == "bearish" and close > vwap:
            structure = "reversal"

        # Confidence from RSI and distance from VWAP
        rsi = float(last.get("rsi_14", 50.0))
        if np.isnan(rsi):
            rsi = 50.0
        rsi_conf = 1.0 - abs(rsi - 50.0) / 50.0
        vwap_conf = float(
            min(abs(close - vwap) / (close or 1.0), 0.05) * 20.0
        ) if not np.isnan(vwap) else 0.5

        confidence = float(max(0.1, min(1.0, (rsi_conf + vwap_conf) / 2.0)))
        if len(df) < 20:
            confidence = round(confidence * 0.7, 4)

        return ChartAnalysis(trend=trend, structure=structure, confidence=confidence)

    def analyze(self, context: MarketDataContext | None = None) -> ChartAnalysis:
        df = context.price_df if context is not None else self.client.fetch_nifty_intraday().df
        df = self._compute_indicators(df)
        analysis = self._detect_trend_and_structure(df)
        logger.info(f"ChartAgent analysis: {analysis}")
        return analysis
