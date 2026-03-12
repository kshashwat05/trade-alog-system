from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from ai_trader.data.nse_option_chain import NseOptionChainClient, OptionChainSummary


@dataclass
class OptionChainAnalysis:
    support: int | None
    resistance: int | None
    pcr: float
    bias: str


class OptionChainAgent:
    """Analyzes NSE NIFTY option chain for directional bias."""

    def __init__(self, client: NseOptionChainClient | None = None) -> None:
        self.client = client or NseOptionChainClient()

    def analyze(self) -> OptionChainAnalysis:
        summary: OptionChainSummary = self.client.summarize()
        analysis = OptionChainAnalysis(
            support=summary.support,
            resistance=summary.resistance,
            pcr=summary.pcr,
            bias=summary.bias,
        )
        logger.info(f"OptionChainAgent analysis: {analysis}")
        return analysis

