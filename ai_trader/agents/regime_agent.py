from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from loguru import logger
from ta.trend import ADXIndicator, EMAIndicator

from ai_trader.data.kite_client import KiteClient
if TYPE_CHECKING:
    from ai_trader.data.market_data_context import MarketDataContext

Regime = Literal["trend_up", "trend_down", "range_bound", "high_volatility"]


@dataclass
class RegimeAnalysis:
    regime: Regime
    confidence: float


class RegimeAgent:
    """Determines the prevailing market regime using trend and volatility."""

    def __init__(self, client: KiteClient | None = None) -> None:
        self.client = client or KiteClient()

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"high", "low", "close"}
        if df.empty:
            return df
        if missing := required_columns.difference(df.columns):
            logger.warning(f"RegimeAgent missing required columns {sorted(missing)}; returning empty frame.")
            return df
        df = df.copy()
        df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
        df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
        df["adx_14"] = ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).adx()
        df["returns"] = df["close"].pct_change()
        df["vol_20"] = df["returns"].rolling(20).std()
        return df

    def analyze(self, context: MarketDataContext | None = None) -> RegimeAnalysis:
        df = context.price_df if context is not None else self.client.fetch_nifty_intraday().df
        df = self._compute_features(df)
        if df.empty:
            logger.warning("RegimeAgent received empty data; returning range_bound regime.")
            return RegimeAnalysis(regime="range_bound", confidence=0.1)

        last = df.iloc[-1]
        ema_20 = last.get("ema_20", np.nan)
        ema_50 = last.get("ema_50", np.nan)
        adx = last.get("adx_14", 15.0)
        vol_20 = last.get("vol_20", 0.01)

        if np.isnan(ema_20) or np.isnan(ema_50):
            logger.info("RegimeAgent lacks sufficient history for EMA features; returning low-confidence range.")
            return RegimeAnalysis(regime="range_bound", confidence=0.2)

        strong_trend = adx > 20
        high_vol = vol_20 > 0.015

        if high_vol and not strong_trend:
            regime: Regime = "high_volatility"
        elif strong_trend and ema_20 > ema_50:
            regime = "trend_up"
        elif strong_trend and ema_20 < ema_50:
            regime = "trend_down"
        else:
            regime = "range_bound"

        confidence = float(
            max(0.1, min(1.0, (abs(adx - 20) / 20.0 + (vol_20 / 0.02)) / 2.0))
        )

        analysis = RegimeAnalysis(regime=regime, confidence=confidence)
        logger.info(f"RegimeAgent analysis: {analysis}")
        return analysis
