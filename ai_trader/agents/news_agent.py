from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional

import requests
from loguru import logger

from ai_trader.config.settings import settings
from ai_trader.data.http_client import build_retry_session

MacroBias = Literal["bullish", "bearish", "neutral"]
RiskLevel = Literal["low", "medium", "high"]


@dataclass
class NewsMacroAnalysis:
    macro_bias: MacroBias
    risk_level: RiskLevel
    data_available: bool = True
    fallback_used: bool = False


class NewsAgent:
    """Fetches macro/news data and derives a coarse sentiment signal.

    This implementation intentionally uses a very simple heuristic for example purposes.
    """

    NEWS_API_URL = "https://newsapi.org/v2/top-headlines"

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None) -> None:
        self.api_key = api_key or settings.news_api_key
        self.session = session or build_retry_session()
        self._cached_analysis: NewsMacroAnalysis | None = None
        self._cached_at: datetime | None = None

    def _fetch_headlines(self) -> list[dict]:
        if not self.api_key:
            logger.warning("News API key missing; returning neutral macro sentiment.")
            return []

        params = {
            "category": "business",
            "language": "en",
            "pageSize": 50,
            "apiKey": self.api_key,
        }
        try:
            resp = self.session.get(self.NEWS_API_URL, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            if not isinstance(articles, list):
                logger.warning("News API returned a non-list articles payload; using neutral sentiment.")
                return []
            return articles
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to fetch news data: {exc}")
            return []

    def analyze(self) -> NewsMacroAnalysis:
        if self._cached_analysis is not None and self._cached_at is not None:
            age = (datetime.utcnow() - self._cached_at).total_seconds()
            if age <= settings.market_data_cache_seconds:
                logger.info(f"NewsAgent using cached analysis age={age:.1f}s.")
                return self._cached_analysis

        articles = self._fetch_headlines()
        if not articles:
            logger.info("NewsAgent analysis: neutral sentiment due to missing or unusable articles.")
            analysis = NewsMacroAnalysis(
                macro_bias="neutral",
                risk_level="high",
                data_available=False,
                fallback_used=True,
            )
            self._cached_analysis = analysis
            self._cached_at = datetime.utcnow()
            return analysis

        recent_articles: list[dict] = []
        cutoff = datetime.utcnow() - timedelta(seconds=settings.max_news_age_seconds)
        for article in articles:
            published_at = article.get("publishedAt")
            if not published_at:
                continue
            try:
                parsed = datetime.fromisoformat(str(published_at).replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                continue
            if parsed >= cutoff:
                recent_articles.append(article)

        if not recent_articles:
            logger.warning("NewsAgent received only stale articles; treating news context as unavailable.")
            analysis = NewsMacroAnalysis(
                macro_bias="neutral",
                risk_level="high",
                data_available=False,
                fallback_used=True,
            )
            self._cached_analysis = analysis
            self._cached_at = datetime.utcnow()
            return analysis

        text_blob = " ".join(
            [
                (a.get("title") or "") + " " + (a.get("description") or "")
                for a in recent_articles
            ]
        ).lower()

        negatives = sum(
            kw in text_blob
            for kw in ["sell-off", "crash", "fear", "inflation", "rate hike", "war"]
        )
        positives = sum(
            kw in text_blob
            for kw in ["rally", "record high", "optimism", "easing", "cut", "stimulus"]
        )

        if positives > negatives + 1:
            macro_bias: MacroBias = "bullish"
            risk_level: RiskLevel = "medium"
        elif negatives > positives + 1:
            macro_bias = "bearish"
            risk_level = "high"
        else:
            macro_bias = "neutral"
            risk_level = "medium"

        analysis = NewsMacroAnalysis(
            macro_bias=macro_bias,
            risk_level=risk_level,
            data_available=True,
            fallback_used=False,
        )
        self._cached_analysis = analysis
        self._cached_at = datetime.utcnow()
        logger.info(f"NewsAgent analysis: {analysis}")
        return analysis
