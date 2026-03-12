from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loguru import logger

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
    ) -> LiquidityAnalysis:
        # Fallbacks if not provided
        if avg_spread is None:
            avg_spread = 1.0
        if volume_score is None:
            volume_score = 0.5
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
