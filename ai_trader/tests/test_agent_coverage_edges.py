from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trader.agents.fii_positioning_agent import FiiPositioningAgent
from ai_trader.agents.gamma_agent import GammaAgent
from ai_trader.agents.liquidity_agent import LiquidityAgent
from ai_trader.agents.liquidity_sweep_agent import LiquiditySweepAgent
from ai_trader.agents.llm_validator_agent import LlmValidatorAgent
from ai_trader.agents.news_agent import NewsAgent
from ai_trader.agents.position_monitor_agent import PositionMonitorAgent
from ai_trader.agents.regime_agent import RegimeAgent
from ai_trader.agents.trigger_agent import TradeSignal
from ai_trader.alerts.whatsapp_alert import WhatsAppAlerter
from ai_trader.config.settings import settings
from ai_trader.data.kite_client import KiteClient, PriceData
from ai_trader.data.market_data_context import MarketDataContext, MarketDataQuality
from ai_trader.data.nse_option_chain import NseOptionChainClient
from ai_trader.data.trade_journal import (
    STATUS_EXECUTED,
    STATUS_REVERSAL_EXIT,
    STATUS_STOP_LOSS_HIT,
    STATUS_TARGET_HIT,
    TradeJournal,
)
from ai_trader.simulation.missed_trade_analyzer import MissedTradeAnalyzer


def _quality() -> MarketDataQuality:
    return MarketDataQuality(
        price_data_available=True,
        option_chain_available=True,
        vix_available=True,
        fii_data_available=True,
        price_fresh=True,
        option_chain_fresh=True,
        vix_fresh=True,
        used_price_fallback=False,
        used_option_chain_fallback=False,
        used_vix_fallback=False,
        issues=[],
    )


def _context(
    *,
    price_df: pd.DataFrame | None = None,
    option_chain_raw: dict | None = None,
    spot_price: float = 24000.0,
    fii_data: dict | None = None,
) -> MarketDataContext:
    if price_df is None:
        price_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-03-13 09:15", periods=10, freq="5min"),
                "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
            }
        )
    if option_chain_raw is None:
        option_chain_raw = {"records": {"data": []}}
    return MarketDataContext(
        fetched_at=datetime.utcnow(),
        price_df=price_df,
        option_chain_raw=option_chain_raw,
        option_chain_summary=SimpleNamespace(support=None, resistance=None, pcr=1.0, bias="neutral"),
        vix_value=14.0,
        spot_price=spot_price,
        fii_data=fii_data or {"net_futures_position": 0.0},
        quality=_quality(),
    )


def _record_signal(journal: TradeJournal, **metadata: object) -> int:
    return journal.record_signal(
        timestamp=datetime(2026, 3, 13, 9, 15),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=7,
        rationale="test signal",
        metadata=dict(metadata),
    )


def test_fii_positioning_agent_returns_neutral_without_rows():
    analysis = FiiPositioningAgent().analyze(_context())
    assert analysis.fii_bias == "neutral"
    assert analysis.institutional_support is None
    assert analysis.confidence == 0.1


def test_fii_positioning_agent_returns_neutral_for_invalid_rows():
    analysis = FiiPositioningAgent().analyze(
        _context(option_chain_raw={"records": {"data": [{"strikePrice": None, "CE": {}, "PE": {}}]}})
    )
    assert analysis.fii_bias == "neutral"
    assert analysis.institutional_resistance is None


def test_fii_positioning_agent_detects_bearish_and_neutral_bias():
    bearish_context = _context(
        option_chain_raw={
            "records": {
                "data": [
                    {
                        "strikePrice": 23900,
                        "CE": {"openInterest": 700, "changeinOpenInterest": 300},
                        "PE": {"openInterest": 200, "changeinOpenInterest": 50},
                    },
                    {
                        "strikePrice": 24000,
                        "CE": {"openInterest": 900, "changeinOpenInterest": 200},
                        "PE": {"openInterest": 300, "changeinOpenInterest": 25},
                    },
                ]
            }
        },
        fii_data={"net_futures_position": -50.0},
    )
    neutral_context = _context(
        option_chain_raw={
            "records": {
                "data": [
                    {
                        "strikePrice": 24000,
                        "CE": {"openInterest": 500, "changeinOpenInterest": 100},
                        "PE": {"openInterest": 550, "changeinOpenInterest": 100},
                    }
                ]
            }
        },
        fii_data={"net_futures_position": 0.0},
    )
    assert FiiPositioningAgent().analyze(bearish_context).fii_bias == "bearish"
    assert FiiPositioningAgent().analyze(neutral_context).fii_bias == "neutral"


def test_gamma_agent_handles_empty_and_invalid_rows():
    agent = GammaAgent()
    assert agent.analyze(_context()).gamma_regime == "positive_gamma"
    invalid = _context(option_chain_raw={"records": {"data": [{"strikePrice": None, "CE": {}, "PE": {}}]}})
    analysis = agent.analyze(invalid)
    assert analysis.gamma_regime == "positive_gamma"
    assert analysis.gamma_flip_level is None


def test_gamma_agent_detects_positive_gamma_when_put_oi_dominates():
    context = _context(
        option_chain_raw={
            "records": {
                "data": [
                    {
                        "strikePrice": 23900,
                        "CE": {"openInterest": 100},
                        "PE": {"openInterest": 300},
                    },
                    {
                        "strikePrice": 24000,
                        "CE": {"openInterest": 150},
                        "PE": {"openInterest": 500},
                    },
                ]
            }
        },
        spot_price=23980.0,
    )
    analysis = GammaAgent().analyze(context)
    assert analysis.gamma_regime == "positive_gamma"
    assert analysis.expected_move == "compression"
    assert analysis.gamma_flip_level == 24000


def test_liquidity_sweep_agent_handles_short_series_and_no_event():
    short_df = pd.DataFrame({"open": [1, 2], "high": [2, 3], "low": [0, 1], "close": [1, 2], "volume": [1, 2]})
    assert LiquiditySweepAgent().analyze(_context(price_df=short_df)).event_type == "none"

    flat_df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [100, 105, 110, 115, 118],
        }
    )
    analysis = LiquiditySweepAgent().analyze(_context(price_df=flat_df))
    assert analysis.liquidity_event is False
    assert analysis.event_type == "none"


def test_liquidity_sweep_agent_detects_breakout_trap():
    price_df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [102, 103, 104, 105, 109],
            "low": [99, 100, 101, 102, 100],
            "close": [101, 102, 103, 104, 105],
            "volume": [100, 100, 100, 100, 220],
        }
    )
    analysis = LiquiditySweepAgent().analyze(_context(price_df=price_df))
    assert analysis.liquidity_event is True
    assert analysis.event_type == "breakout_trap"


def test_llm_validator_handles_none_signal_and_disabled_mode():
    validator = LlmValidatorAgent(validation_enabled=False)
    none_result = validator.validate({"signal": "NONE"})
    disabled_result = validator.validate({"signal": "BUY_CE"})
    assert none_result.validation == "rejected"
    assert disabled_result.validation == "approved"
    assert disabled_result.source == "deterministic_fallback"


def test_llm_validator_uses_output_text_and_missing_client_fallback():
    class DummyResponse:
        output_text = '{"validation":"approved","confidence_adjustment":0.05,"reasoning":"Aligned setup."}'

    class DummyClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    return DummyResponse()

            completions = Completions()

        chat = Chat()

    validator = LlmValidatorAgent(validation_enabled=True, api_key="x", client=DummyClient())
    result = validator.validate({"signal": "BUY_CE"})
    assert result.validation == "approved"
    assert result.source == "openai"

    fallback = LlmValidatorAgent(validation_enabled=True, api_key=None, client=None).validate({"signal": "BUY_CE"})
    assert fallback.validation == "approved"
    assert fallback.fallback_used is True


def test_llm_validator_falls_back_for_invalid_payload_and_unstructured_response():
    class InvalidJsonClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(output_text='{"validation":"approved","confidence_adjustment":0.8}')

            completions = Completions()

        chat = Chat()

    invalid_result = LlmValidatorAgent(
        validation_enabled=True,
        api_key="x",
        client=InvalidJsonClient(),
    ).validate({"signal": "BUY_CE"})
    assert invalid_result.source == "deterministic_fallback"
    assert invalid_result.fallback_used is True

    with pytest.raises(ValueError):
        LlmValidatorAgent._extract_response_text(SimpleNamespace())


def test_position_monitor_handles_stop_loss_reversal_and_missing_price(tmp_path):
    journal = TradeJournal(tmp_path / "monitor.db")
    stop_id = _record_signal(journal, instrument_key="NFO:STOP")
    reversal_id = _record_signal(journal, instrument_key="NFO:REV")
    no_price_id = _record_signal(journal, instrument_key="NFO:NOPRICE")

    journal.record_execution(stop_id, execution_price=100.0, quantity=1, instrument_key="NFO:STOP")
    journal.record_execution(reversal_id, execution_price=100.0, quantity=1, instrument_key="NFO:REV")
    journal.record_execution(no_price_id, execution_price=100.0, quantity=1, instrument_key="NFO:NOPRICE")

    prices = {"NFO:STOP": 89.0, "NFO:REV": 94.0, "NFO:NOPRICE": None}
    monitor = PositionMonitorAgent(journal, price_fetcher=lambda trade: prices[trade.metadata["instrument_key"]])
    results = monitor.monitor_once()
    statuses = {result.signal_id: result.status for result in results}

    assert statuses[stop_id] == STATUS_STOP_LOSS_HIT
    assert statuses[reversal_id] == STATUS_REVERSAL_EXIT
    assert journal.get_trade(no_price_id).status == STATUS_EXECUTED  # type: ignore[union-attr]


def test_position_monitor_target_message_contains_profit_text(tmp_path):
    journal = TradeJournal(tmp_path / "target.db")
    signal_id = _record_signal(journal)
    journal.record_execution(signal_id, execution_price=100.0, quantity=2, instrument_key="NFO:TARGET")
    result = PositionMonitorAgent(journal, price_fetcher=lambda trade: 121.0).monitor_once()[0]
    assert result.status == STATUS_TARGET_HIT
    assert "Book profit now." in result.message
    assert result.pnl == 42.0


def test_regime_agent_handles_missing_columns_and_short_history():
    agent = RegimeAgent()
    df_missing = pd.DataFrame({"close": [1, 2, 3]})
    assert agent._compute_features(df_missing).equals(df_missing)

    short_features = pd.DataFrame([{"ema_20": float("nan"), "ema_50": float("nan"), "adx_14": 15.0, "vol_20": 0.01}])
    agent._compute_features = lambda df: short_features  # type: ignore[method-assign]
    analysis = agent.analyze(_context(price_df=pd.DataFrame({"high": [1], "low": [0.5], "close": [0.8]})))
    assert analysis.regime == "range_bound"
    assert analysis.confidence == 0.2


def test_regime_agent_detects_trend_and_high_volatility():
    agent = RegimeAgent()
    trend_up_df = pd.DataFrame(
        {
            "high": list(range(101, 171)),
            "low": list(range(99, 169)),
            "close": list(range(100, 170)),
        }
    )
    down_close = list(range(170, 100, -1))
    trend_down_df = pd.DataFrame(
        {
            "high": [value + 1 for value in down_close],
            "low": [value - 1 for value in down_close],
            "close": down_close,
        }
    )
    alternating = [100 + (8 if idx % 2 else -8) + idx * 0.3 for idx in range(70)]
    high_vol_df = pd.DataFrame(
        {
            "high": [value + 1 for value in alternating],
            "low": [value - 1 for value in alternating],
            "close": alternating,
        }
    )

    assert agent.analyze(_context(price_df=trend_up_df)).regime == "trend_up"
    assert agent.analyze(_context(price_df=trend_down_df)).regime == "trend_down"
    assert agent.analyze(_context(price_df=high_vol_df)).regime == "high_volatility"


def test_liquidity_agent_uses_context_to_detect_high_and_low_liquidity():
    agent = LiquidityAgent()
    high_df = pd.DataFrame(
        {
            "high": [100.2] * 12,
            "low": [99.8] * 12,
            "close": [100.0] * 12,
            "volume": [100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 230, 240],
        }
    )
    low_df = pd.DataFrame(
        {
            "high": [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131],
            "low": [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
            "close": [100] * 12,
            "volume": [100, 100, 100, 100, 100, 100, 20, 20, 20, 20, 20, 20],
        }
    )

    assert agent.analyze(context=_context(price_df=high_df, spot_price=100.0)).liquidity == "high"
    assert agent.analyze(context=_context(price_df=low_df, spot_price=100.0)).liquidity == "low"


def test_news_agent_handles_non_list_payload_and_positive_sentiment_cache():
    class NonListSession:
        def get(self, url, params=None, timeout=5):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"articles": "unexpected"},
            )

    fallback = NewsAgent(api_key="key", session=NonListSession()).analyze()  # type: ignore[arg-type]
    assert fallback.data_available is False

    fresh_time = (datetime.utcnow() - timedelta(minutes=5)).isoformat() + "Z"

    class PositiveSession:
        calls = 0

        def get(self, url, params=None, timeout=5):
            self.calls += 1
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "articles": [
                        {
                            "title": "Market rally on optimism and stimulus",
                            "description": "Record high on easing and rate cut hopes",
                            "publishedAt": fresh_time,
                        }
                    ]
                },
            )

    session = PositiveSession()
    agent = NewsAgent(api_key="key", session=session)  # type: ignore[arg-type]
    first = agent.analyze()
    second = agent.analyze()
    assert first.macro_bias == "bullish"
    assert first.data_available is True
    assert second.macro_bias == "bullish"
    assert session.calls == 1


def test_nse_option_chain_fetch_and_summarize_handles_success_and_edge_cases(monkeypatch):
    class SuccessSession:
        def get(self, url, timeout=5):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "records": {
                        "data": [
                            {"strikePrice": 23900, "CE": {"openInterest": 300}, "PE": {"openInterest": 700}},
                            {"strikePrice": 24000, "CE": {"openInterest": 200}, "PE": {"openInterest": 600}},
                        ]
                    }
                },
            )

    monkeypatch.setattr("ai_trader.data.nse_option_chain.prime_nse_session", lambda session: None)
    client = NseOptionChainClient(session=SuccessSession())  # type: ignore[arg-type]
    raw = client.fetch_raw()
    summary = client.summarize(raw)
    assert "_meta" in raw
    assert summary.bias == "bullish"
    assert summary.support == 23900

    neutral = client.summarize({"records": {"data": [{"strikePrice": None, "CE": {}, "PE": {}}]}})
    assert neutral.bias == "neutral"
    assert neutral.support is None

    class FailingSession:
        def get(self, url, timeout=5):
            raise RuntimeError("nse down")

    failing = NseOptionChainClient(session=FailingSession())  # type: ignore[arg-type]
    assert failing.fetch_raw() == {}


def test_missed_trade_analyzer_handles_missing_trade_and_empty_series(tmp_path):
    journal = TradeJournal(tmp_path / "missed.db")
    with pytest.raises(ValueError):
        MissedTradeAnalyzer(journal).analyze_trade(999)

    signal_id = _record_signal(journal, instrument_key="NFO:EMPTY")

    class EmptyClient:
        def fetch_intraday_by_instrument_key(self, instrument_key, days=1):
            return PriceData(df=pd.DataFrame(columns=["date", "close"]))

    analysis = MissedTradeAnalyzer(journal, client=EmptyClient()).analyze_trade(signal_id)  # type: ignore[arg-type]
    assert analysis.available is False
    assert analysis.reason == "No pricing series available for simulation."


def test_whatsapp_alerter_handles_none_signal_and_delivery_failures(monkeypatch):
    monkeypatch.setattr(settings, "twilio_account_sid", None)
    monkeypatch.setattr(settings, "twilio_auth_token", None)
    monkeypatch.setattr(settings, "twilio_whatsapp_from", None)
    monkeypatch.setattr(settings, "whatsapp_to", None)
    alerter = WhatsAppAlerter()
    alerter.send_trade_signal(TradeSignal("NONE", 0.0, 0.0, 0.0, 0.0, "none"))
    alerter.send_exit_alert("exit")

    sent_bodies: list[str] = []

    class DummyMessages:
        def create(self, *, body, from_, to):
            sent_bodies.append(body)
            if "EXIT" in body:
                raise Exception("twilio failure")

    class DummyClient:
        messages = DummyMessages()

    monkeypatch.setattr(settings, "twilio_account_sid", "sid")
    monkeypatch.setattr(settings, "twilio_auth_token", "token")
    monkeypatch.setattr(settings, "twilio_whatsapp_from", "whatsapp:+111")
    monkeypatch.setattr(settings, "whatsapp_to", "whatsapp:+222")
    monkeypatch.setattr("ai_trader.alerts.whatsapp_alert.Client", lambda sid, token: DummyClient())
    monkeypatch.setattr("ai_trader.alerts.whatsapp_alert.TwilioRestException", Exception)

    live = WhatsAppAlerter()
    live.send_trade_signal(
        TradeSignal("BUY_CE", 100.0, 90.0, 120.0, 0.8, "Aligned setup"),
        institutional_bias="bullish",
        gamma_regime="negative_gamma",
    )
    live.send_exit_alert("NIFTY TRADE EXIT")
    assert any("NIFTY TRADE SIGNAL" in body for body in sent_bodies)


def test_kite_client_helpers_cover_resolution_and_error_paths():
    class FakeKite:
        def __init__(self):
            self.access_token = None

        def set_access_token(self, token):
            self.access_token = token

        def instruments(self, exchange):
            if exchange == "NSE":
                return [
                    {"tradingsymbol": "NIFTY 50", "name": "NIFTY 50", "instrument_token": 123, "segment": "INDICES", "exchange": "NSE"},
                    {"tradingsymbol": "ABC", "instrument_token": 456, "segment": "NFO-OPT", "exchange": "NFO"},
                ]
            if exchange == "NFO":
                return [{"tradingsymbol": "NIFTY24MAR23700PE", "instrument_token": 789}]
            return []

        def historical_data(self, **kwargs):
            raise RuntimeError("historical failure")

        def ltp(self, instrument_key):
            return {instrument_key: {"last_price": 212.5}}

    client = KiteClient(kite=FakeKite())  # type: ignore[arg-type]
    assert client.resolve_instrument_key("NIFTY24MAR23700PE") == "NFO:NIFTY24MAR23700PE"
    assert client.resolve_instrument_token_by_key("NFO:NIFTY24MAR23700PE") == 789
    assert client.resolve_instrument_token_by_key("INVALID") is None
    assert client.fetch_nifty_intraday(days=0).df.empty
    assert client.fetch_intraday_by_instrument_key("NFO:NIFTY24MAR23700PE").df.empty
    assert client.fetch_ltp_by_instrument_key("NFO:NIFTY24MAR23700PE") == 212.5
    assert KiteClient._normalize_dataframe([]).empty
    with pytest.raises(ValueError):
        KiteClient._normalize_dataframe([{"date": "2026-03-13"}])
