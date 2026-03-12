from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from loguru import logger
if TYPE_CHECKING:
    from ai_trader.data.market_data_context import MarketDataContext

LiquidityBucket = Literal["high", "medium", "low"]
SlippageRisk = Literal["low", "medium", "high"]


@dataclass
class LiquidityAnalysis:
    liquidity: LiquidityBucket
    slippage_risk: SlippageRisk


class LiquidityAgent:
    """Estimates liquidity/slippage qualitatively.

    In a production system this would inspect order book depth, spreads, and volume.
    For this template, we accept pre-computed metrics or use safe defaults.
    """

    def analyze(
        self,
        avg_spread: float | None = None,
        volume_score: float | None = None,
        context: MarketDataContext | None = None,
    ) -> LiquidityAnalysis:
        # Fallbacks if not provided
        if avg_spread is None:
            avg_spread = 1.0
        if volume_score is None:
            volume_score = 0.5
        if context is not None and not context.price_df.empty:
            recent = context.price_df.tail(10)
            if "close" in recent and len(recent) > 1:
                avg_spread = float((recent["high"] - recent["low"]).tail(5).mean() / max(context.spot_price, 1.0) * 100)
            if "volume" in recent:
                baseline = float(context.price_df["volume"].median()) if "volume" in context.price_df else 0.0
                if baseline > 0:
                    volume_score = float(min(1.0, recent["volume"].tail(5).mean() / baseline))
        avg_spread = max(avg_spread, 0.0)
        volume_score = min(max(volume_score, 0.0), 1.0)

        if avg_spread <= 0.5 and volume_score >= 0.7:
            liquidity: LiquidityBucket = "high"
            slippage: SlippageRisk = "low"
        elif avg_spread <= 1.5 and volume_score >= 0.4:
            liquidity = "medium"
            slippage = "medium"
        else:
            liquidity = "low"
            slippage = "high"

        analysis = LiquidityAnalysis(liquidity=liquidity, slippage_risk=slippage)
        logger.info(f"LiquidityAgent analysis: {analysis}")
        return analysis
