from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_trader.agents.global_market_agent import GlobalMarketAnalysis
from ai_trader.agents.macro_calendar_agent import MacroCalendarAnalysis
from ai_trader.analysis.news_impact_engine import NewsImpactAnalysis

Sentiment = Literal["bullish", "bearish", "neutral"]


@dataclass
class GlobalSentimentAnalysis:
    market_sentiment: Sentiment
    confidence: float
    risk_blocked: bool
    rationale: str


class GlobalSentimentEngine:
    def combine(
        self,
        global_market: GlobalMarketAnalysis,
        macro_calendar: MacroCalendarAnalysis,
        news_impact: NewsImpactAnalysis,
    ) -> GlobalSentimentAnalysis:
        score = 0.0
        notes: list[str] = []

        if global_market.global_bias == "bullish":
            score += 2.0 * max(global_market.confidence, 0.4)
            notes.append("global tape supportive")
        elif global_market.global_bias == "bearish":
            score -= 2.0 * max(global_market.confidence, 0.4)
            notes.append("global tape defensive")

        if global_market.risk_sentiment == "risk_off":
            score -= 1.5
            notes.append("risk-off conditions detected")
        elif global_market.risk_sentiment == "risk_on":
            score += 1.0
            notes.append("risk appetite is healthy")

        if news_impact.macro_bias == "bullish":
            score += 1.5 * max(news_impact.confidence, 0.3)
            notes.append("macro headlines skew constructive")
        elif news_impact.macro_bias == "bearish":
            score -= 1.5 * max(news_impact.confidence, 0.3)
            notes.append("macro headlines skew defensive")

        if macro_calendar.event_risk == "high":
            score -= 2.0
            notes.append(f"high-risk event ahead: {macro_calendar.event_type}")
        elif macro_calendar.event_risk == "medium":
            score -= 0.5
            notes.append(f"watch event calendar: {macro_calendar.event_type}")

        if score >= 1.5:
            market_sentiment: Sentiment = "bullish"
        elif score <= -1.5:
            market_sentiment = "bearish"
        else:
            market_sentiment = "neutral"

        confidence = min(1.0, 0.3 + abs(score) * 0.15)
        risk_blocked = macro_calendar.event_risk == "high" or (
            global_market.risk_sentiment == "risk_off" and global_market.confidence >= 0.8
        )
        rationale = "; ".join(notes) if notes else "insufficient macro context"
        return GlobalSentimentAnalysis(
            market_sentiment=market_sentiment,
            confidence=confidence,
            risk_blocked=risk_blocked,
            rationale=rationale,
        )
