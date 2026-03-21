from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Literal, Optional
from xml.etree import ElementTree

import requests
from loguru import logger

from ai_trader.analysis.news_impact_engine import NewsImpactEngine
from ai_trader.config.settings import settings
from ai_trader.data.http_client import build_retry_session

MacroBias = Literal["bullish", "bearish", "neutral"]
RiskLevel = Literal["low", "medium", "high"]


@dataclass
class NewsMacroAnalysis:
    macro_bias: MacroBias
    risk_level: RiskLevel
    impact_level: str = "low"
    confidence: float = 0.0
    article_count: int = 0
    top_headlines: list[dict[str, str]] | None = None
    sources: list[str] | None = None
    data_available: bool = True
    fallback_used: bool = False


class NewsAgent:
    """Aggregates macro news from APIs and RSS feeds and derives macro sentiment."""

    NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
    NEWSDATA_URL = "https://newsdata.io/api/1/news"
    MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
    RSS_FEEDS = (
        ("reuters", "https://feeds.reuters.com/reuters/businessNews"),
        ("economic_times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("moneycontrol", "https://www.moneycontrol.com/rss/business.xml"),
    )

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None) -> None:
        self.api_key = api_key or settings.news_api_key
        self.session = session or build_retry_session()
        self.newsdata_api_key = settings.newsdata_api_key
        self.marketaux_api_key = settings.marketaux_api_key
        self.impact_engine = NewsImpactEngine()
        self._cached_analysis: NewsMacroAnalysis | None = None
        self._cached_at: datetime | None = None

    @staticmethod
    def _normalize_headline(value: str) -> str:
        return " ".join(value.lower().split())

    def _request_json(self, url: str, *, params: dict[str, str]) -> dict:
        response = self.session.get(url, params=params, timeout=6)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def _fetch_newsapi(self) -> list[dict[str, str]]:
        if not self.api_key:
            return []

        params = {
            "category": "business",
            "language": "en",
            "pageSize": 50,
            "apiKey": self.api_key,
        }
        data = self._request_json(self.NEWS_API_URL, params=params)
        articles = data.get("articles", [])
        if not isinstance(articles, list):
            return []
        return [
            {
                "title": str(article.get("title") or ""),
                "description": str(article.get("description") or ""),
                "published_at": str(article.get("publishedAt") or ""),
                "source": str((article.get("source") or {}).get("name") or "newsapi"),
                "url": str(article.get("url") or ""),
            }
            for article in articles
        ]

    def _fetch_newsdata(self) -> list[dict[str, str]]:
        if not self.newsdata_api_key:
            return []
        params = {
            "apikey": self.newsdata_api_key,
            "category": "business",
            "language": "en",
            "size": "20",
        }
        data = self._request_json(self.NEWSDATA_URL, params=params)
        results = data.get("results", [])
        if not isinstance(results, list):
            return []
        return [
            {
                "title": str(article.get("title") or ""),
                "description": str(article.get("description") or ""),
                "published_at": str(article.get("pubDate") or ""),
                "source": str(article.get("source_id") or "newsdata"),
                "url": str(article.get("link") or ""),
            }
            for article in results
        ]

    def _fetch_marketaux(self) -> list[dict[str, str]]:
        if not self.marketaux_api_key:
            return []
        params = {
            "api_token": self.marketaux_api_key,
            "language": "en",
            "limit": "20",
        }
        data = self._request_json(self.MARKETAUX_URL, params=params)
        results = data.get("data", [])
        if not isinstance(results, list):
            return []
        return [
            {
                "title": str(article.get("title") or ""),
                "description": str(article.get("description") or ""),
                "published_at": str(article.get("published_at") or ""),
                "source": str(article.get("source") or "marketaux"),
                "url": str(article.get("url") or ""),
            }
            for article in results
        ]

    def _fetch_rss_feed(self, name: str, url: str) -> list[dict[str, str]]:
        response = self.session.get(url, timeout=6)
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        items = root.findall(".//item")
        normalized: list[dict[str, str]] = []
        for item in items[:20]:
            title = item.findtext("title") or ""
            description = item.findtext("description") or ""
            pub_date = item.findtext("pubDate") or ""
            if pub_date:
                try:
                    pub_date = parsedate_to_datetime(pub_date).isoformat()
                except (TypeError, ValueError):
                    pub_date = ""
            normalized.append(
                {
                    "title": title,
                    "description": description,
                    "published_at": pub_date,
                    "source": name,
                    "url": item.findtext("link") or "",
                }
            )
        return normalized

    def _fetch_headlines(self) -> list[dict[str, str]]:
        sources = [
            self._fetch_newsapi,
            self._fetch_newsdata,
            self._fetch_marketaux,
        ]
        articles: list[dict[str, str]] = []
        for fetcher in sources:
            try:
                articles.extend(fetcher())
            except Exception as exc:  # noqa: BLE001
                logger.error(f"News source fetch failed: {exc}")
        for name, url in self.RSS_FEEDS:
            try:
                articles.extend(self._fetch_rss_feed(name, url))
            except Exception as exc:  # noqa: BLE001
                logger.error(f"RSS fetch failed for {name}: {exc}")

        deduped: dict[str, dict[str, str]] = {}
        for article in articles:
            title = (article.get("title") or "").strip()
            if not title:
                continue
            key = self._normalize_headline(title)
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = article
                continue
            current_time = article.get("published_at") or ""
            existing_time = existing.get("published_at") or ""
            if current_time > existing_time:
                deduped[key] = article

        return sorted(
            deduped.values(),
            key=lambda item: item.get("published_at") or "",
            reverse=True,
        )

    def analyze(self) -> NewsMacroAnalysis:
        if self._cached_analysis is not None and self._cached_at is not None:
            age = (datetime.utcnow() - self._cached_at).total_seconds()
            if age <= settings.news_feed_cache_seconds:
                logger.info(f"NewsAgent using cached analysis age={age:.1f}s.")
                return self._cached_analysis

        articles = self._fetch_headlines()
        if not articles:
            logger.info("NewsAgent analysis: neutral sentiment due to missing or unusable articles.")
            analysis = NewsMacroAnalysis(
                macro_bias="neutral",
                risk_level="high",
                impact_level="high",
                data_available=False,
                fallback_used=True,
                top_headlines=[],
                sources=[],
            )
            self._cached_analysis = analysis
            self._cached_at = datetime.utcnow()
            return analysis

        recent_articles: list[dict[str, str]] = []
        cutoff = datetime.utcnow() - timedelta(seconds=settings.max_news_age_seconds)
        for article in articles:
            published_at = article.get("published_at")
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
                impact_level="high",
                data_available=False,
                fallback_used=True,
                top_headlines=[],
                sources=[],
            )
            self._cached_analysis = analysis
            self._cached_at = datetime.utcnow()
            return analysis

        impact = self.impact_engine.analyze(recent_articles)
        if impact.impact_level == "high" and impact.macro_bias == "bearish":
            risk_level: RiskLevel = "high"
        elif impact.impact_level == "high":
            risk_level = "medium"
        elif impact.impact_level == "medium":
            risk_level = "medium"
        else:
            risk_level = "low"

        analysis = NewsMacroAnalysis(
            macro_bias=impact.macro_bias,
            risk_level=risk_level,
            impact_level=impact.impact_level,
            confidence=impact.confidence,
            article_count=len(recent_articles),
            top_headlines=recent_articles[:5],
            sources=sorted({str(article.get("source") or "unknown") for article in recent_articles}),
            data_available=True,
            fallback_used=False,
        )
        self._cached_analysis = analysis
        self._cached_at = datetime.utcnow()
        logger.info(f"NewsAgent analysis: {analysis}")
        return analysis
