from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from ai_trader.agents.fii_positioning_agent import FiiPositioningAgent
from ai_trader.agents.gamma_agent import GammaAgent
from ai_trader.agents.liquidity_sweep_agent import LiquiditySweepAgent
from ai_trader.agents.llm_validator_agent import LlmValidatorAgent
from ai_trader.agents.news_agent import NewsAgent
from ai_trader.config.settings import settings
from ai_trader.data.market_data_context import MarketDataContext, MarketDataQuality
from ai_trader.data.nse_option_chain import OptionChainSummary


def _build_context(price_df: pd.DataFrame | None = None, option_chain_raw: dict | None = None) -> MarketDataContext:
    if price_df is None:
        price_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-03-13 09:15", periods=10, freq="5min"),
                "open": [24000, 24010, 24020, 24025, 24040, 24035, 24030, 24045, 24050, 24060],
                "high": [24010, 24020, 24035, 24050, 24055, 24042, 24048, 24060, 24068, 24075],
                "low": [23990, 24000, 24010, 24020, 24015, 24010, 24005, 24030, 24035, 24040],
                "close": [24005, 24015, 24030, 24045, 24020, 24018, 24040, 24058, 24062, 24070],
                "volume": [100, 110, 120, 130, 180, 160, 170, 190, 210, 260],
            }
        )
    if option_chain_raw is None:
        option_chain_raw = {
            "records": {
                "data": [
                    {
                        "strikePrice": 23900,
                        "CE": {"openInterest": 500, "changeinOpenInterest": 150},
                        "PE": {"openInterest": 200, "changeinOpenInterest": 50},
                    },
                    {
                        "strikePrice": 24000,
                        "CE": {"openInterest": 400, "changeinOpenInterest": 125},
                        "PE": {"openInterest": 650, "changeinOpenInterest": 300},
                    },
                ]
            }
        }
    return MarketDataContext(
        fetched_at=datetime.utcnow(),
        price_df=price_df,
        option_chain_raw=option_chain_raw,
        option_chain_summary=OptionChainSummary(
            support=24000,
            resistance=23900,
            pcr=1.2,
            bias="bullish",
        ),
        vix_value=14.0,
        spot_price=24050.0,
        fii_data={"net_futures_position": 100.0},
        quality=MarketDataQuality(
            price_data_available=True,
            option_chain_available=True,
            vix_available=True,
            fii_data_available=True,
            used_price_fallback=False,
            used_option_chain_fallback=False,
            used_vix_fallback=False,
            issues=[],
        ),
    )


def test_fii_positioning_agent_detects_bullish_bias():
    analysis = FiiPositioningAgent().analyze(_build_context())
    assert analysis.fii_bias == "bullish"
    assert analysis.institutional_support == 24000


def test_gamma_agent_detects_negative_gamma():
    analysis = GammaAgent().analyze(_build_context())
    assert analysis.gamma_regime == "negative_gamma"
    assert analysis.expected_move == "expansion"


def test_liquidity_sweep_agent_detects_sweep():
    price_df = pd.DataFrame(
        {
            "date": pd.date_range("2026-03-13 09:15", periods=6, freq="5min"),
            "open": [24000, 24010, 24020, 24015, 24018, 24060],
            "high": [24010, 24020, 24030, 24025, 24090, 24120],
            "low": [23995, 24000, 24010, 24005, 23980, 24020],
            "close": [24005, 24015, 24018, 24020, 24010, 24040],
            "volume": [100, 110, 115, 120, 300, 320],
        }
    )
    analysis = LiquiditySweepAgent().analyze(_build_context(price_df=price_df))
    assert analysis.liquidity_event is True


def test_llm_validator_uses_openai_structured_response():
    class DummyResponse:
        class Choice:
            class Message:
                content = (
                    '{"validation":"rejected","confidence_adjustment":-0.1,'
                    '"reasoning":"Liquidity sweep indicates breakout trap."}'
                )

            message = Message()

        choices = [Choice()]

    class DummyClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    return DummyResponse()

            completions = Completions()

        chat = Chat()

    validator = LlmValidatorAgent(
        validation_enabled=True,
        provider="openai",
        api_key="test-key",
        client=DummyClient(),
    )
    result = validator.validate(
        {
            "signal": "BUY_CE",
            "decision_score": 8,
            "liquidity_sweep": {"event_type": "breakout_trap"},
            "gamma_analysis": {"gamma_regime": "negative_gamma"},
            "fii_positioning": {"fii_bias": "bullish"},
            "regime": {"regime": "trend_up"},
        }
    )
    assert result.validation == "rejected"
    assert "breakout trap" in result.reasoning
    assert result.source == "openai"


def test_llm_validator_falls_back_to_deterministic_logic_on_api_failure():
    class FailingClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("timeout")

            completions = Completions()

        chat = Chat()

    validator = LlmValidatorAgent(
        validation_enabled=True,
        provider="openai",
        api_key="test-key",
        client=FailingClient(),
    )
    result = validator.validate(
        {
            "signal": "BUY_CE",
            "decision_score": 8,
            "liquidity_sweep": {"event_type": "breakout_trap"},
            "gamma_analysis": {"gamma_regime": "negative_gamma"},
            "fii_positioning": {"fii_bias": "bullish"},
            "regime": {"regime": "trend_up"},
        }
    )
    assert result.validation == "approved"
    assert result.confidence_adjustment == 0.0
    assert result.fallback_used is True
    assert "deterministic signal" in result.reasoning


def test_llm_validator_disabled_uses_deterministic_signal():
    validator = LlmValidatorAgent(validation_enabled=False)
    result = validator.validate({"signal": "BUY_PE"})
    assert result.validation == "approved"
    assert result.source == "deterministic_fallback"


def test_llm_validator_uses_gemini_structured_response():
    class DummyGeminiResponse:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"validation":"approved","confidence_adjustment":0.05,"reasoning":"Institutional alignment confirmed."}'
                                }
                            ]
                        }
                    }
                ]
            }

    class DummyGeminiClient:
        @staticmethod
        def post(url, json=None, timeout=8.0):
            return DummyGeminiResponse()

    validator = LlmValidatorAgent(
        validation_enabled=True,
        provider="gemini",
        api_key="gemini-key",
        client=DummyGeminiClient(),
    )
    result = validator.validate({"signal": "BUY_CE", "decision_score": 8})
    assert result.validation == "approved"
    assert result.source == "gemini"


def test_llm_validator_uses_provider_default_model_when_config_default_is_for_other_provider(monkeypatch):
    captured: dict[str, str] = {}

    class DummyResponse:
        output_text = '{"validation":"approved","confidence_adjustment":0.01,"reasoning":"Validated."}'

    class DummyClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    captured["model"] = kwargs["model"]
                    return DummyResponse()

            completions = Completions()

        chat = Chat()

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "llm_model", "gemini-2.0-flash")
    validator = LlmValidatorAgent(
        validation_enabled=True,
        provider="openai",
        api_key="test-key",
        client=DummyClient(),
    )
    result = validator.validate({"signal": "BUY_CE", "decision_score": 8})
    assert result.source == "openai"
    assert captured["model"] == "gpt-4.1-mini"


def test_news_agent_marks_missing_articles_as_unavailable():
    agent = NewsAgent(api_key=None)
    result = agent.analyze()
    assert result.data_available is False
    assert result.fallback_used is True
    assert result.risk_level == "high"


def test_news_agent_marks_stale_articles_as_unavailable():
    stale_time = (datetime.utcnow() - timedelta(hours=3)).isoformat() + "Z"

    class DummySession:
        def get(self, url, params=None, timeout=5):
            class DummyResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"articles": [{"title": "Old headline", "description": "Old", "publishedAt": stale_time}]}

            return DummyResponse()

    agent = NewsAgent(api_key="key", session=DummySession())  # type: ignore[arg-type]
    result = agent.analyze()
    assert result.data_available is False
    assert result.fallback_used is True
