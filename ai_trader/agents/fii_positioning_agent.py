from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from ai_trader.data.market_data_context import MarketDataContext

InstitutionalBias = Literal["bullish", "bearish", "neutral"]


@dataclass
class FiiPositioningAnalysis:
    fii_bias: InstitutionalBias
    institutional_support: int | None
    institutional_resistance: int | None
    confidence: float


class FiiPositioningAgent:
    def analyze(self, context: MarketDataContext) -> FiiPositioningAnalysis:
        records = context.option_chain_raw.get("records", {})
        rows = records.get("data", [])
        if not rows:
            logger.warning("FIIPositioningAgent missing option chain rows; returning neutral bias.")
            return FiiPositioningAnalysis("neutral", None, None, 0.1)

        parsed_rows = []
        for row in rows:
            ce = row.get("CE") or {}
            pe = row.get("PE") or {}
            parsed_rows.append(
                {
                    "strike": row.get("strikePrice"),
                    "ce_oi": ce.get("openInterest", 0),
                    "pe_oi": pe.get("openInterest", 0),
                    "ce_oi_change": ce.get("changeinOpenInterest", 0),
                    "pe_oi_change": pe.get("changeinOpenInterest", 0),
                }
            )
        df = pd.DataFrame(parsed_rows).dropna(subset=["strike"])
        if df.empty:
            return FiiPositioningAnalysis("neutral", None, None, 0.1)

        put_write_score = float((df["pe_oi_change"].clip(lower=0)).sum())
        call_write_score = float((df["ce_oi_change"].clip(lower=0)).sum())
        futures_bias = float(context.fii_data.get("net_futures_position", 0.0))

        support = int(df.loc[df["pe_oi"].idxmax(), "strike"]) if not df.empty else None
        resistance = int(df.loc[df["ce_oi"].idxmax(), "strike"]) if not df.empty else None

        net_score = put_write_score - call_write_score + futures_bias
        if net_score > 0:
            bias: InstitutionalBias = "bullish"
        elif net_score < 0:
            bias = "bearish"
        else:
            bias = "neutral"
        magnitude = abs(net_score) / max(put_write_score + call_write_score + abs(futures_bias), 1.0)
        confidence = float(max(0.1, min(1.0, 0.4 + magnitude)))
        analysis = FiiPositioningAnalysis(bias, support, resistance, confidence)
        logger.info(f"FIIPositioningAgent analysis: {analysis}")
        return analysis
