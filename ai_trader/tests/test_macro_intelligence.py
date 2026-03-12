from __future__ import annotations

from datetime import datetime, timedelta

from ai_trader.agents.global_market_agent import GlobalMarketAgent
from ai_trader.agents.macro_calendar_agent import MacroCalendarAgent
from ai_trader.agents.news_agent import NewsAgent
from ai_trader.analysis.global_sentiment_engine import GlobalSentimentEngine
from ai_trader.analysis.news_impact_engine import NewsImpactEngine
from ai_trader.orchestrator.decision_engine import DecisionEngine


class JsonResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TextResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


def test_news_impact_engine_detects_bearish_macro_shock():
    analysis = NewsImpactEngine().analyze(
        [
            {"title": "RBI inflation shock raises war fears", "description": "crude oil spikes on conflict"},
            {"title": "Bond yields and rupee volatility jump", "description": "fii selling intensifies"},
        ]
    )
    assert analysis.macro_bias == "bearish"
    assert analysis.impact_level == "high"


def test_global_sentiment_engine_blocks_high_risk_event():
    from ai_trader.agents.global_market_agent import GlobalMarketAnalysis
    from ai_trader.agents.macro_calendar_agent import MacroCalendarAnalysis
    from ai_trader.analysis.news_impact_engine import NewsImpactAnalysis

    result = GlobalSentimentEngine().combine(
        global_market=GlobalMarketAnalysis(
            global_bias="bearish",
            risk_sentiment="risk_off",
            confidence=0.9,
            data_available=True,
            fallback_used=False,
        ),
        macro_calendar=MacroCalendarAnalysis(
            event_risk="high",
            event_type="fed",
            expected_market_impact="bearish",
            data_available=True,
            fallback_used=False,
        ),
        news_impact=NewsImpactAnalysis(
            macro_bias="bearish",
            impact_level="high",
            confidence=0.8,
            impact_score=5.0,
            top_headlines=["Fed event tonight"],
        ),
    )
    assert result.market_sentiment == "bearish"
    assert result.risk_blocked is True


def test_global_market_agent_parses_yahoo_quotes():
    class DummySession:
        def get(self, url, params=None, timeout=8):
            return JsonResponse(
                {
                    "quoteResponse": {
                        "result": [
                            {"symbol": "ES=F", "regularMarketChangePercent": -0.8, "regularMarketPrice": 5200},
                            {"symbol": "NQ=F", "regularMarketChangePercent": -1.1, "regularMarketPrice": 18200},
                            {"symbol": "YM=F", "regularMarketChangePercent": -0.6, "regularMarketPrice": 38800},
                            {"symbol": "BZ=F", "regularMarketChangePercent": 1.2, "regularMarketPrice": 82},
                            {"symbol": "GC=F", "regularMarketChangePercent": 0.7, "regularMarketPrice": 2400},
                            {"symbol": "INR=X", "regularMarketChangePercent": 0.5, "regularMarketPrice": 83.2},
                            {"symbol": "DX-Y.NYB", "regularMarketChangePercent": 0.4, "regularMarketPrice": 104.1},
                            {"symbol": "^TNX", "regularMarketChangePercent": 0.3, "regularMarketPrice": 4.2},
                            {"symbol": "^N225", "regularMarketChangePercent": -0.4, "regularMarketPrice": 39000},
                            {"symbol": "^HSI", "regularMarketChangePercent": -0.3, "regularMarketPrice": 17000},
                        ]
                    }
                }
            )

    analysis = GlobalMarketAgent(session=DummySession()).analyze()  # type: ignore[arg-type]
    assert analysis.global_bias == "bearish"
    assert analysis.risk_sentiment == "risk_off"
    assert analysis.data_available is True


def test_macro_calendar_agent_flags_fed_event():
    near_event = (datetime.utcnow() + timedelta(hours=6)).isoformat() + "Z"

    class DummySession:
        def get(self, url, params=None, timeout=8):
            return JsonResponse(
                [
                    {
                        "Event": "Fed Interest Rate Decision",
                        "Category": "Interest Rate",
                        "Country": "United States",
                        "Date": near_event,
                    }
                ]
            )

    analysis = MacroCalendarAgent(api_key="key", session=DummySession()).analyze()  # type: ignore[arg-type]
    assert analysis.event_risk == "high"
    assert analysis.expected_market_impact == "bearish"


def test_news_agent_merges_and_deduplicates_sources():
    fresh = datetime.utcnow().isoformat() + "Z"

    class DummySession:
        def get(self, url, params=None, timeout=6):
            if "newsapi" in url:
                return JsonResponse(
                    {
                        "articles": [
                            {
                                "title": "RBI inflation watch",
                                "description": "Rates may stay high",
                                "publishedAt": fresh,
                                "source": {"name": "newsapi"},
                                "url": "https://example.com/a",
                            }
                        ]
                    }
                )
            if "newsdata" in url:
                return JsonResponse(
                    {
                        "results": [
                            {
                                "title": "RBI inflation watch",
                                "description": "Duplicate title from another source",
                                "pubDate": fresh,
                                "source_id": "newsdata",
                                "link": "https://example.com/b",
                            }
                        ]
                    }
                )
            return TextResponse(
                f"""
                <rss><channel>
                  <item>
                    <title>Crude oil jumps on war fears</title>
                    <description>Geopolitical shock keeps traders defensive</description>
                    <pubDate>{datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')}</pubDate>
                    <link>https://example.com/rss</link>
                  </item>
                </channel></rss>
                """
            )

    agent = NewsAgent(api_key="newsapi-key", session=DummySession())  # type: ignore[arg-type]
    agent.newsdata_api_key = "newsdata-key"
    result = agent.analyze()
    assert result.data_available is True
    assert result.article_count >= 2
    assert result.impact_level in {"medium", "high"}
    assert len(result.top_headlines or []) >= 2


def test_decision_engine_scores_global_sentiment_alignment():
    engine = DecisionEngine()
    output = {}

    class DummyContext:
        spot_price = 24000
        option_chain_raw = {
            "records": {
                "data": [
                    {
                        "strikePrice": 24000,
                        "PE": {"lastPrice": 220.0, "identifier": "NIFTY24APR24000PE"},
                        "CE": {"lastPrice": 180.0, "identifier": "NIFTY24APR24000CE"},
                    }
                ]
            }
        }
        quality = type(
            "Quality",
            (),
            {
                "price_data_available": True,
                "option_chain_available": True,
                "vix_available": True,
                "fii_data_available": True,
                "price_fresh": True,
                "option_chain_fresh": True,
                "vix_fresh": True,
                "critical_inputs_available": True,
                "to_dict": staticmethod(lambda: {}),
            },
        )()

    from ai_trader.agents.chart_agent import ChartAnalysis
    from ai_trader.agents.option_chain_agent import OptionChainAnalysis
    from ai_trader.agents.news_agent import NewsMacroAnalysis
    from ai_trader.agents.volatility_agent import VolatilityAnalysis
    from ai_trader.agents.regime_agent import RegimeAnalysis
    from ai_trader.agents.liquidity_agent import LiquidityAnalysis
    from ai_trader.agents.fii_positioning_agent import FiiPositioningAnalysis
    from ai_trader.agents.gamma_agent import GammaAnalysis
    from ai_trader.agents.liquidity_sweep_agent import LiquiditySweepAnalysis
    from ai_trader.agents.global_market_agent import GlobalMarketAnalysis
    from ai_trader.agents.macro_calendar_agent import MacroCalendarAnalysis

    engine.market_data_provider.build = lambda: DummyContext()  # type: ignore[method-assign]
    engine.chart_agent.analyze = lambda context=None: ChartAnalysis("bearish", "breakout", 0.8)  # type: ignore[method-assign]
    engine.option_agent.analyze = lambda context=None: OptionChainAnalysis(23900, 24000, 0.8, "bearish")  # type: ignore[method-assign]
    engine.news_agent.analyze = lambda: NewsMacroAnalysis("bearish", "medium", impact_level="medium", confidence=0.7, article_count=3, top_headlines=[])  # type: ignore[method-assign]
    engine.vol_agent.analyze = lambda spot=None, context=None: VolatilityAnalysis("medium", (23850, 24150), True, False)  # type: ignore[method-assign]
    engine.regime_agent.analyze = lambda context=None: RegimeAnalysis("trend_down", 0.8)  # type: ignore[method-assign]
    engine.liquidity_agent.analyze = lambda avg_spread=None, volume_score=None, context=None: LiquidityAnalysis("high", "low")  # type: ignore[method-assign]
    engine.fii_agent.analyze = lambda context: FiiPositioningAnalysis("bearish", 23850, 24000, 0.8)  # type: ignore[method-assign]
    engine.gamma_agent.analyze = lambda context: GammaAnalysis("negative_gamma", 23950, "expansion")  # type: ignore[method-assign]
    engine.liquidity_sweep_agent.analyze = lambda context: LiquiditySweepAnalysis(False, "none", 0.7)  # type: ignore[method-assign]
    engine.global_market_agent.analyze = lambda: GlobalMarketAnalysis("bearish", "neutral", 0.8, data_available=True, fallback_used=False)  # type: ignore[method-assign]
    engine.macro_calendar_agent.analyze = lambda: MacroCalendarAnalysis("low", "none", "neutral", data_available=True, fallback_used=False)  # type: ignore[method-assign]
    result = engine.run_once()
    output = result.state["score_breakdown"]
    assert output["global_sentiment"] == 2
    assert result.decision_score >= 7
