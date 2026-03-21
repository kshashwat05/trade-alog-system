from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import requests
from loguru import logger

from ai_trader.config.settings import settings
from ai_trader.data.http_client import build_retry_session

EventRisk = Literal["low", "medium", "high"]
MarketImpact = Literal["bullish", "bearish", "neutral"]


@dataclass
class MacroCalendarAnalysis:
    event_risk: EventRisk
    event_type: str
    expected_market_impact: MarketImpact
    upcoming_events: list[dict[str, str]] = field(default_factory=list)
    data_available: bool = False
    fallback_used: bool = True


class MacroCalendarAgent:
    TRADING_ECONOMICS_URL = "https://api.tradingeconomics.com/calendar"
    KEYWORD_IMPACT = {
        "rbi": ("medium", "neutral"),
        "inflation": ("high", "bearish"),
        "cpi": ("high", "bearish"),
        "gdp": ("medium", "neutral"),
        "fed": ("high", "bearish"),
        "fomc": ("high", "bearish"),
        "oil": ("high", "bearish"),
        "crude": ("high", "bearish"),
        "war": ("high", "bearish"),
        "geopolitical": ("high", "bearish"),
    }

    def __init__(self, api_key: str | None = None, session: Optional[requests.Session] = None) -> None:
        self.api_key = api_key or settings.tradingeconomics_api_key
        self.session = session or build_retry_session()
        self._cached_analysis: MacroCalendarAnalysis | None = None
        self._cached_at: datetime | None = None

    def _fetch_events(self) -> list[dict[str, str]]:
        if not self.api_key:
            return []
        params = {"c": self.api_key}
        response = self.session.get(self.TRADING_ECONOMICS_URL, params=params, timeout=8)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    def analyze(self) -> MacroCalendarAnalysis:
        now = datetime.utcnow()
        if self._cached_analysis is not None and self._cached_at is not None:
            age = (now - self._cached_at).total_seconds()
            if age <= settings.macro_calendar_cache_seconds:
                return self._cached_analysis

        try:
            events = self._fetch_events()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"MacroCalendarAgent failed to fetch events: {exc}")
            events = []

        horizon = datetime.now(timezone.utc) + timedelta(days=1)
        parsed_events: list[dict[str, str]] = []
        highest_risk: EventRisk = "low"
        event_type = "none"
        impact: MarketImpact = "neutral"

        for event in events:
            category = str(event.get("Category") or event.get("category") or "")
            country = str(event.get("Country") or event.get("country") or "")
            title = str(event.get("Event") or event.get("event") or category)
            date_value = str(event.get("Date") or event.get("date") or "")
            if not title:
                continue
            event_dt: datetime | None = None
            if date_value:
                try:
                    event_dt = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                except ValueError:
                    event_dt = None
            if event_dt is not None and event_dt > horizon:
                continue
            text = f"{title} {category} {country}".lower()
            parsed_events.append(
                {
                    "title": title,
                    "category": category,
                    "country": country,
                    "date": date_value,
                }
            )
            for keyword, (risk_level, inferred_impact) in self.KEYWORD_IMPACT.items():
                if keyword in text:
                    if risk_level == "high":
                        highest_risk = "high"
                    elif highest_risk != "high" and risk_level == "medium":
                        highest_risk = "medium"
                    event_type = keyword
                    impact = inferred_impact  # type: ignore[assignment]
                    break

        analysis = MacroCalendarAnalysis(
            event_risk=highest_risk if parsed_events else "low",
            event_type=event_type if parsed_events else "none",
            expected_market_impact=impact if parsed_events else "neutral",
            upcoming_events=parsed_events[:5],
            data_available=bool(parsed_events),
            fallback_used=not bool(parsed_events),
        )
        self._cached_analysis = analysis
        self._cached_at = now
        return analysis
