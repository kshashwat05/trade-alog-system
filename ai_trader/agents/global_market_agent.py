from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

import requests
from loguru import logger

from ai_trader.config.settings import settings
from ai_trader.data.http_client import build_retry_session

GlobalBias = Literal["bullish", "bearish", "neutral"]
RiskSentiment = Literal["risk_on", "risk_off", "neutral"]


@dataclass
class GlobalMarketAnalysis:
    global_bias: GlobalBias
    risk_sentiment: RiskSentiment
    confidence: float
    indicators: dict[str, float | None] = field(default_factory=dict)
    source: str = "fallback"
    data_available: bool = False
    fallback_used: bool = True


class GlobalMarketAgent:
    YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
    SYMBOL_MAP = {
        "sp500_futures": "ES=F",
        "nasdaq_futures": "NQ=F",
        "dow_futures": "YM=F",
        "brent_crude": "BZ=F",
        "gold": "GC=F",
        "usdinr": "INR=X",
        "dxy": "DX-Y.NYB",
        "us10y_yield": "^TNX",
        "nikkei": "^N225",
        "hang_seng": "^HSI",
    }

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or build_retry_session()
        self._cached_analysis: GlobalMarketAnalysis | None = None
        self._cached_at: datetime | None = None

    def _fetch_yahoo_quotes(self) -> dict[str, dict[str, float | None]]:
        params = {"symbols": ",".join(self.SYMBOL_MAP.values())}
        response = self.session.get(self.YAHOO_QUOTE_URL, params=params, timeout=8)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("quoteResponse", {}).get("result", [])
        quotes_by_symbol = {item.get("symbol"): item for item in results if item.get("symbol")}
        normalized: dict[str, dict[str, float | None]] = {}
        for name, symbol in self.SYMBOL_MAP.items():
            quote = quotes_by_symbol.get(symbol, {})
            normalized[name] = {
                "price": quote.get("regularMarketPrice"),
                "change_pct": quote.get("regularMarketChangePercent"),
            }
        return normalized

    def analyze(self) -> GlobalMarketAnalysis:
        now = datetime.utcnow()
        if self._cached_analysis is not None and self._cached_at is not None:
            age = (now - self._cached_at).total_seconds()
            if age <= settings.global_market_cache_seconds:
                return self._cached_analysis

        try:
            quotes = self._fetch_yahoo_quotes()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"GlobalMarketAgent failed to fetch quotes: {exc}")
            analysis = GlobalMarketAnalysis(
                global_bias="neutral",
                risk_sentiment="neutral",
                confidence=0.0,
                indicators={name: None for name in self.SYMBOL_MAP},
                data_available=False,
                fallback_used=True,
            )
            self._cached_analysis = analysis
            self._cached_at = now
            return analysis

        bullish_score = 0.0
        bearish_score = 0.0
        risk_off_score = 0.0
        indicators: dict[str, float | None] = {}

        for name, values in quotes.items():
            change_pct = values.get("change_pct")
            indicators[name] = change_pct
            if change_pct is None:
                continue
            if name in {"sp500_futures", "nasdaq_futures", "dow_futures", "nikkei", "hang_seng"}:
                bullish_score += max(change_pct, 0.0)
                bearish_score += abs(min(change_pct, 0.0))
            elif name in {"brent_crude", "gold", "dxy", "us10y_yield", "usdinr"}:
                if change_pct > 0:
                    risk_off_score += change_pct
                else:
                    bullish_score += abs(change_pct) * 0.5

        if bearish_score > bullish_score + 0.35:
            global_bias: GlobalBias = "bearish"
        elif bullish_score > bearish_score + 0.35:
            global_bias = "bullish"
        else:
            global_bias = "neutral"

        if risk_off_score >= 1.0 or (
            global_bias == "bearish" and bearish_score >= 0.8
        ):
            risk_sentiment: RiskSentiment = "risk_off"
        elif global_bias == "bullish" and bullish_score >= 0.8:
            risk_sentiment = "risk_on"
        else:
            risk_sentiment = "neutral"

        confidence = min(1.0, 0.35 + max(bullish_score, bearish_score, risk_off_score) * 0.3)
        analysis = GlobalMarketAnalysis(
            global_bias=global_bias,
            risk_sentiment=risk_sentiment,
            confidence=confidence,
            indicators=indicators,
            source="yahoo_finance",
            data_available=any(value is not None for value in indicators.values()),
            fallback_used=False,
        )
        self._cached_analysis = analysis
        self._cached_at = now
        return analysis
