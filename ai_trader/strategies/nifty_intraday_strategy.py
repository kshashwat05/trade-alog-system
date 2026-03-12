from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from ai_trader.agents.chart_agent import ChartAgent
from ai_trader.agents.chart_agent import ChartAnalysis


@dataclass
class StrategySignal:
    timestamp: pd.Timestamp
    direction: str


class NiftyIntradayStrategy:
    """Simple example intraday strategy used for backtesting.

    For backtests we will not hit live APIs; instead, we plug in historical
    candle data and reuse the same indicator logic as ChartAgent.
    """

    def __init__(self) -> None:
        self.chart_agent = ChartAgent()

    def generate_signals(self, candles: pd.DataFrame) -> List[StrategySignal]:
        signals: List[StrategySignal] = []
        if candles.empty:
            return signals
        required_columns = {"open", "high", "low", "close", "volume"}
        if missing := required_columns.difference(candles.columns):
            raise ValueError(f"Strategy candles missing required columns: {sorted(missing)}")

        # Rolling window analysis using ChartAgent's indicator logic in batch
        df = candles.copy()
        # Minimal re-use: treat as a single batch and call private helper
        df = self.chart_agent._compute_indicators(df)  # type: ignore[attr-defined]

        for ts, row in df.iterrows():
            mini_df = df.loc[:ts].tail(50)
            analysis: ChartAnalysis = self.chart_agent._detect_trend_and_structure(mini_df)  # type: ignore[attr-defined]
            if analysis.trend == "bullish":
                direction = "BUY_CE"
            elif analysis.trend == "bearish":
                direction = "BUY_PE"
            else:
                direction = "NONE"
            signals.append(StrategySignal(timestamp=ts, direction=direction))

        return signals
