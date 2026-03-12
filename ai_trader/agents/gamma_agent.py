from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from ai_trader.data.market_data_context import MarketDataContext

GammaRegime = Literal["positive_gamma", "negative_gamma"]
ExpectedMove = Literal["compression", "expansion"]


@dataclass
class GammaAnalysis:
    gamma_regime: GammaRegime
    gamma_flip_level: int | None
    expected_move: ExpectedMove


class GammaAgent:
    def analyze(self, context: MarketDataContext) -> GammaAnalysis:
        rows = context.option_chain_raw.get("records", {}).get("data", [])
        if not rows:
            analysis = GammaAnalysis("positive_gamma", None, "compression")
            logger.warning("GammaAgent missing option chain rows; defaulting to positive gamma.")
            return analysis

        parsed_rows = []
        for row in rows:
            ce = row.get("CE") or {}
            pe = row.get("PE") or {}
            parsed_rows.append(
                {
                    "strike": row.get("strikePrice"),
                    "total_oi": ce.get("openInterest", 0) + pe.get("openInterest", 0),
                    "net_call_minus_put": ce.get("openInterest", 0) - pe.get("openInterest", 0),
                }
            )
        df = pd.DataFrame(parsed_rows).dropna(subset=["strike"])
        if df.empty:
            return GammaAnalysis("positive_gamma", None, "compression")

        flip_row = df.iloc[(df["strike"] - context.spot_price).abs().argsort()[:1]]
        gamma_flip = int(flip_row.iloc[0]["strike"])
        aggregate = float(df["net_call_minus_put"].sum())
        if aggregate > 0:
            regime: GammaRegime = "negative_gamma"
            move: ExpectedMove = "expansion"
        else:
            regime = "positive_gamma"
            move = "compression"
        analysis = GammaAnalysis(regime, gamma_flip, move)
        logger.info(f"GammaAgent analysis: {analysis}")
        return analysis
