from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

MacroBias = Literal["bullish", "bearish", "neutral"]
ImpactLevel = Literal["low", "medium", "high"]


@dataclass
class NewsImpactAnalysis:
    macro_bias: MacroBias
    impact_level: ImpactLevel
    confidence: float
    impact_score: float
    top_headlines: list[str]


class NewsImpactEngine:
    POSITIVE_KEYWORDS = {
        "stimulus": 1.2,
        "rate cut": 1.5,
        "easing": 1.0,
        "fii buying": 1.4,
        "soft landing": 1.2,
        "gdp growth": 1.1,
        "lower inflation": 1.3,
        "cooling inflation": 1.3,
    }
    NEGATIVE_KEYWORDS = {
        "rbi": 0.9,
        "inflation": 1.4,
        "interest rates": 1.3,
        "rate hike": 1.6,
        "crude oil": 1.1,
        "geopolitics": 1.4,
        "war": 1.6,
        "sanctions": 1.4,
        "fii selling": 1.5,
        "rupee": 0.8,
        "bond yields": 1.1,
        "fed": 1.2,
        "conflict": 1.3,
        "tariff": 1.1,
    }

    def analyze(self, headlines: Iterable[dict[str, str]]) -> NewsImpactAnalysis:
        normalized = []
        positive_score = 0.0
        negative_score = 0.0
        impact_score = 0.0
        for article in headlines:
            title = (article.get("title") or "").strip()
            description = (article.get("description") or "").strip()
            text = f"{title} {description}".lower()
            if not text.strip():
                continue
            normalized.append(title or description)
            for keyword, weight in self.POSITIVE_KEYWORDS.items():
                if keyword in text:
                    positive_score += weight
                    impact_score += weight
            for keyword, weight in self.NEGATIVE_KEYWORDS.items():
                if keyword in text:
                    negative_score += weight
                    impact_score += weight

        if negative_score > positive_score + 0.75:
            macro_bias: MacroBias = "bearish"
        elif positive_score > negative_score + 0.75:
            macro_bias = "bullish"
        else:
            macro_bias = "neutral"

        if impact_score >= 4.0:
            impact_level: ImpactLevel = "high"
        elif impact_score >= 1.75:
            impact_level = "medium"
        else:
            impact_level = "low"

        confidence = min(1.0, 0.35 + min(impact_score, 5.0) * 0.1)
        return NewsImpactAnalysis(
            macro_bias=macro_bias,
            impact_level=impact_level,
            confidence=confidence if normalized else 0.0,
            impact_score=impact_score,
            top_headlines=normalized[:5],
        )
