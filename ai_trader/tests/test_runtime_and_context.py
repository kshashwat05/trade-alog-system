from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import pytest

from ai_trader.agents.llm_validator_agent import LlmValidationResult
from ai_trader.agents.risk_agent import RiskCheckResult
from ai_trader.agents.trigger_agent import TradeSignal
from ai_trader.agents.position_monitor_agent import PositionMonitorResult
from ai_trader.data.kite_client import KiteClient, PriceData
from ai_trader.data.market_data_context import MarketDataProvider, MarketDataQuality
from ai_trader.data.nse_option_chain import OptionChainSummary
from ai_trader.data.trade_journal import TradeJournal
from ai_trader.orchestrator.decision_engine import OrchestratorOutput


def test_market_data_provider_builds_context():
    class DummyKiteClient:
        def fetch_nifty_intraday(self):
            return PriceData(
                df=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-03-13 09:15:00"]),
                        "open": [24000],
                        "high": [24010],
                        "low": [23990],
                        "close": [24005],
                        "volume": [1000],
                    }
                )
            )

    class DummyOptionClient:
        def fetch_raw(self):
            return {"records": {"data": []}}

        def summarize(self, data=None):
            return OptionChainSummary(None, None, 1.0, "neutral")

    class DummyVolAgent:
        def fetch_vix(self):
            return 14.0

    provider = MarketDataProvider(
        kite_client=DummyKiteClient(),  # type: ignore[arg-type]
        option_chain_client=DummyOptionClient(),  # type: ignore[arg-type]
        vol_agent=DummyVolAgent(),  # type: ignore[arg-type]
    )
    context = provider.build()
    assert context.spot_price == 24005
    assert context.vix_value == 14.0
    assert context.quality.critical_inputs_available is False


def test_market_data_provider_uses_ttl_cache():
    calls = {"price": 0, "option": 0, "vix": 0}

    class DummyKiteClient:
        def fetch_nifty_intraday(self):
            calls["price"] += 1
            return PriceData(
                df=pd.DataFrame(
                    {
                        "date": pd.to_datetime([datetime.utcnow()]),
                        "open": [24000],
                        "high": [24010],
                        "low": [23990],
                        "close": [24005],
                        "volume": [1000],
                    }
                ),
                fetched_at=datetime.utcnow(),
            )

    class DummyOptionClient:
        def fetch_raw(self):
            calls["option"] += 1
            return {
                "_meta": {"fetched_at": datetime.utcnow().isoformat()},
                "records": {"data": [{"strikePrice": 24000, "PE": {"openInterest": 10}, "CE": {"openInterest": 10}}]},
            }

        def summarize(self, data=None):
            return OptionChainSummary(24000, 24000, 1.0, "neutral")

    class DummyVolAgent:
        def fetch_vix(self):
            calls["vix"] += 1
            return 14.0

    provider = MarketDataProvider(
        kite_client=DummyKiteClient(),  # type: ignore[arg-type]
        option_chain_client=DummyOptionClient(),  # type: ignore[arg-type]
        vol_agent=DummyVolAgent(),  # type: ignore[arg-type]
    )
    first = provider.build()
    second = provider.build()
    assert first is second
    assert calls == {"price": 1, "option": 1, "vix": 1}


def test_market_data_provider_marks_stale_price_data():
    class DummyKiteClient:
        def fetch_nifty_intraday(self):
            return PriceData(
                df=pd.DataFrame(
                    {
                        "date": pd.to_datetime([datetime.utcnow() - timedelta(hours=2)]),
                        "open": [24000],
                        "high": [24010],
                        "low": [23990],
                        "close": [24005],
                        "volume": [1000],
                    }
                ),
                fetched_at=datetime.utcnow(),
            )

    class DummyOptionClient:
        def fetch_raw(self):
            return {
                "_meta": {"fetched_at": datetime.utcnow().isoformat()},
                "records": {"data": [{"strikePrice": 24000, "PE": {"openInterest": 10}, "CE": {"openInterest": 10}}]},
            }

        def summarize(self, data=None):
            return OptionChainSummary(24000, 24000, 1.0, "neutral")

    class DummyVolAgent:
        def fetch_vix(self):
            return 14.0

    provider = MarketDataProvider(
        kite_client=DummyKiteClient(),  # type: ignore[arg-type]
        option_chain_client=DummyOptionClient(),  # type: ignore[arg-type]
        vol_agent=DummyVolAgent(),  # type: ignore[arg-type]
    )
    context = provider.build()
    assert context.quality.price_fresh is False
    assert context.quality.critical_inputs_available is False


def test_kite_client_resolves_instrument_key():
    class DummyKite:
        def instruments(self, exchange=None):
            return [{"tradingsymbol": "NIFTY24MAR23700PE"}]

    client = KiteClient(kite=DummyKite())  # type: ignore[arg-type]
    assert client.resolve_instrument_key("NIFTY24MAR23700PE") == "NFO:NIFTY24MAR23700PE"


def test_kite_client_fetch_ltp_by_instrument_key():
    class DummyKite:
        def ltp(self, instrument_key):
            return {instrument_key: {"last_price": 212.5}}

    client = KiteClient(kite=DummyKite())  # type: ignore[arg-type]
    assert client.fetch_ltp_by_instrument_key("NFO:NIFTY24MAR23700PE") == 212.5


def test_main_runtime_helpers(monkeypatch, tmp_path):
    import ai_trader.main as main_module

    live_state_path = tmp_path / "live_state.json"
    monkeypatch.setattr(main_module, "_live_state_path", live_state_path)
    monkeypatch.setattr(main_module, "is_market_open", lambda now=None: True)

    @dataclass
    class DummyState:
        fii_bias: str = "bearish"
        gamma_regime: str = "negative_gamma"
        event_type: str = "none"

    output = OrchestratorOutput(
        timestamp=datetime.utcnow().isoformat(),
        signal=TradeSignal(
            signal="BUY_PE",
            entry=210.0,
            stop_loss=175.0,
            target=280.0,
            confidence=0.78,
            rationale="Institutional setup",
        ),
        risk=RiskCheckResult(True, None),
        llm_validation=LlmValidationResult("approved", 0.05, "Approved by heuristic validator."),
        decision_score=8,
        state={
            "fii_positioning": type("Fii", (), {"fii_bias": "bearish"})(),
            "gamma_analysis": type("Gamma", (), {"gamma_regime": "negative_gamma"})(),
            "liquidity_sweep": type("Sweep", (), {"event_type": "none"})(),
            "chart": DummyState(),
        },
    )

    sent = {"trade": 0, "exit": 0}
    monkeypatch.setattr(main_module.engine, "run_once", lambda open_trades=0: output)
    monkeypatch.setattr(main_module.journal, "get_open_trades", lambda: [])
    monkeypatch.setattr(main_module.journal, "record_signal", lambda **kwargs: 101)
    monkeypatch.setattr(main_module.engine.risk_agent, "record_signal_emitted", lambda emitted_at=None: None)
    monkeypatch.setattr(main_module.engine.risk_agent, "record_trade_open_today", lambda: None)
    monkeypatch.setattr(
        main_module.alerter,
        "send_trade_signal",
        lambda signal, institutional_bias=None, gamma_regime=None: sent.__setitem__("trade", sent["trade"] + 1),
    )
    main_module.run_trading_cycle()
    assert sent["trade"] == 1
    assert json.loads(live_state_path.read_text())["decision_score"] == 8

    monkeypatch.setattr(
        main_module.position_monitor,
        "monitor_once",
        lambda: [type("Exit", (), {"message": "exit now"})()],
    )
    monkeypatch.setattr(
        main_module.alerter,
        "send_exit_alert",
        lambda message: sent.__setitem__("exit", sent["exit"] + 1),
    )
    main_module.run_position_monitor_cycle()
    assert sent["exit"] == 1

    analyzed = []
    monkeypatch.setattr(
        main_module.journal,
        "get_pending_simulation_trades",
        lambda: [type("Trade", (), {"id": 1, "trade_executed": False})()],
    )
    monkeypatch.setattr(
        main_module.missed_trade_analyzer,
        "analyze_trade",
        lambda signal_id: analyzed.append(signal_id),
    )
    main_module.run_missed_trade_analysis_cycle()
    assert analyzed == [1]


def test_market_data_provider_handles_corrupted_option_chain_payload():
    class DummyKiteClient:
        def fetch_nifty_intraday(self):
            return PriceData(
                df=pd.DataFrame(
                    {
                        "date": pd.to_datetime([datetime.utcnow()]),
                        "open": [24000],
                        "high": [24010],
                        "low": [23990],
                        "close": [24005],
                        "volume": [1000],
                    }
                ),
                fetched_at=datetime.utcnow(),
            )

    class CorruptedOptionClient:
        def fetch_raw(self):
            return "not-json"

        def summarize(self, data=None):
            raise AssertionError("summarize should not be called with malformed payload")

    class DummyVolAgent:
        def fetch_vix(self):
            return 14.0

    provider = MarketDataProvider(
        kite_client=DummyKiteClient(),  # type: ignore[arg-type]
        option_chain_client=CorruptedOptionClient(),  # type: ignore[arg-type]
        vol_agent=DummyVolAgent(),  # type: ignore[arg-type]
    )
    context = provider.build()
    assert context.option_chain_raw == {}
    assert context.quality.option_chain_available is False
    assert context.quality.critical_inputs_available is False


def test_market_data_provider_handles_missing_vix_data():
    class DummyKiteClient:
        def fetch_nifty_intraday(self):
            return PriceData(
                df=pd.DataFrame(
                    {
                        "date": pd.to_datetime([datetime.utcnow()]),
                        "open": [24000],
                        "high": [24010],
                        "low": [23990],
                        "close": [24005],
                        "volume": [1000],
                    }
                ),
                fetched_at=datetime.utcnow(),
            )

    class DummyOptionClient:
        def fetch_raw(self):
            return {
                "_meta": {"fetched_at": datetime.utcnow().isoformat()},
                "records": {"data": [{"strikePrice": 24000, "PE": {"openInterest": 10}, "CE": {"openInterest": 10}}]},
            }

        def summarize(self, data=None):
            return OptionChainSummary(24000, 24000, 1.0, "neutral")

    class MissingVixAgent:
        def fetch_vix(self):
            return None

    provider = MarketDataProvider(
        kite_client=DummyKiteClient(),  # type: ignore[arg-type]
        option_chain_client=DummyOptionClient(),  # type: ignore[arg-type]
        vol_agent=MissingVixAgent(),  # type: ignore[arg-type]
    )
    context = provider.build()
    assert context.vix_value is None
    assert context.quality.vix_available is False
    assert context.quality.critical_inputs_available is False


def test_engine_handles_nse_api_down_without_crashing():
    from ai_trader.orchestrator.decision_engine import DecisionEngine

    engine = DecisionEngine()

    class DummyKiteClient:
        def fetch_nifty_intraday(self):
            return PriceData(
                df=pd.DataFrame(
                    {
                        "date": pd.to_datetime([datetime.utcnow()]),
                        "open": [24000],
                        "high": [24010],
                        "low": [23990],
                        "close": [24005],
                        "volume": [1000],
                    }
                ),
                fetched_at=datetime.utcnow(),
            )

    class DownOptionClient:
        def fetch_raw(self):
            raise RuntimeError("NSE API down")

        def summarize(self, data=None):
            return OptionChainSummary(None, None, 1.0, "neutral")

    class DummyVolAgent:
        def fetch_vix(self):
            return 14.0

        def analyze(self, spot=None, context=None):
            return type("Vol", (), {"volatility": "medium", "expected_range": (23800, 24200), "data_available": True, "fallback_used": False})()

    engine.market_data_provider = MarketDataProvider(
        kite_client=DummyKiteClient(),  # type: ignore[arg-type]
        option_chain_client=DownOptionClient(),  # type: ignore[arg-type]
        vol_agent=DummyVolAgent(),  # type: ignore[arg-type]
    )
    output = engine.run_once()
    assert output.signal.signal == "NONE"
    assert output.state["market_context"].quality.option_chain_available is False


def test_engine_handles_zerodha_timeout_without_crashing():
    from ai_trader.orchestrator.decision_engine import DecisionEngine

    engine = DecisionEngine()

    class TimeoutKiteClient:
        def fetch_nifty_intraday(self):
            raise TimeoutError("zerodha timeout")

    class DummyOptionClient:
        def fetch_raw(self):
            return {
                "_meta": {"fetched_at": datetime.utcnow().isoformat()},
                "records": {"data": [{"strikePrice": 24000, "PE": {"openInterest": 10}, "CE": {"openInterest": 10}}]},
            }

        def summarize(self, data=None):
            return OptionChainSummary(24000, 24000, 1.0, "neutral")

    class DummyVolAgent:
        def fetch_vix(self):
            return 14.0

        def analyze(self, spot=None, context=None):
            return type("Vol", (), {"volatility": "medium", "expected_range": (23800, 24200), "data_available": True, "fallback_used": False})()

    engine.market_data_provider = MarketDataProvider(
        kite_client=TimeoutKiteClient(),  # type: ignore[arg-type]
        option_chain_client=DummyOptionClient(),  # type: ignore[arg-type]
        vol_agent=DummyVolAgent(),  # type: ignore[arg-type]
    )
    output = engine.run_once()
    assert output.signal.signal == "NONE"
    assert output.state["market_context"].quality.price_data_available is False


def test_engine_handles_network_outage_without_crashing():
    from ai_trader.orchestrator.decision_engine import DecisionEngine

    engine = DecisionEngine()

    def fail(*args, **kwargs):
        raise ConnectionError("network outage")

    engine.market_data_provider.build = fail  # type: ignore[method-assign]
    import ai_trader.main as main_module

    called = {"persisted": False}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main_module, "is_market_open", lambda now=None: True)
    monkeypatch.setattr(main_module.engine, "run_once", lambda open_trades=0: (_ for _ in ()).throw(ConnectionError("network outage")))
    monkeypatch.setattr(main_module, "persist_live_state", lambda result: called.__setitem__("persisted", True))
    main_module.run_trading_cycle()
    assert called["persisted"] is False
    monkeypatch.undo()


def test_main_runtime_survives_trade_and_exit_alert_failures(monkeypatch, tmp_path):
    import ai_trader.main as main_module

    live_state_path = tmp_path / "live_state.json"
    monkeypatch.setattr(main_module, "_live_state_path", live_state_path)
    monkeypatch.setattr(main_module, "is_market_open", lambda now=None: True)
    monkeypatch.setattr(main_module.journal, "get_open_trades", lambda: [])
    monkeypatch.setattr(main_module.journal, "record_signal", lambda **kwargs: 101)

    output = OrchestratorOutput(
        timestamp=datetime.utcnow().isoformat(),
        signal=TradeSignal(
            signal="BUY_PE",
            entry=210.0,
            stop_loss=175.0,
            target=280.0,
            confidence=0.78,
            rationale="Institutional setup",
        ),
        risk=RiskCheckResult(True, None),
        llm_validation=LlmValidationResult("approved", 0.0, "Approved."),
        decision_score=8,
        state={
            "fii_positioning": type("Fii", (), {"fii_bias": "bearish"})(),
            "gamma_analysis": type("Gamma", (), {"gamma_regime": "negative_gamma"})(),
            "liquidity_sweep": type("Sweep", (), {"event_type": "none"})(),
            "chart": type("Chart", (), {"trend": "bearish"})(),
        },
    )
    monkeypatch.setattr(main_module.engine, "run_once", lambda open_trades=0: output)
    monkeypatch.setattr(main_module.alerter, "send_trade_signal", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("twilio down")))
    main_module.run_trading_cycle()
    assert live_state_path.exists()

    monkeypatch.setattr(
        main_module.position_monitor,
        "monitor_once",
        lambda: [PositionMonitorResult(signal_id=1, status="target_hit", current_price=120.0, pnl=20.0, message="EXIT")],
    )
    monkeypatch.setattr(main_module.alerter, "send_exit_alert", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("twilio down")))
    main_module.run_position_monitor_cycle()


def test_nse_option_chain_client_handles_corrupted_payload(monkeypatch):
    from ai_trader.data.nse_option_chain import NseOptionChainClient

    monkeypatch.setattr("ai_trader.data.nse_option_chain.prime_nse_session", lambda session: None)

    class CorruptedSession:
        def get(self, url, timeout=5):
            return type(
                "Resp",
                (),
                {
                    "raise_for_status": staticmethod(lambda: None),
                    "json": staticmethod(lambda: "corrupted"),
                },
            )()

    client = NseOptionChainClient(session=CorruptedSession())  # type: ignore[arg-type]
    assert client.fetch_raw() == {}
    summary = client.summarize({"records": "bad"})
    assert summary.bias == "neutral"


def test_trade_journal_enforces_persistent_signal_limits(tmp_path):
    journal = TradeJournal(tmp_path / "journal.db")
    ts = datetime(2026, 3, 13, 9, 15)
    signal_id = journal.record_signal(
        timestamp=ts,
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=8,
        rationale="first",
        enforce_limits=True,
        cooldown_minutes=15,
        max_trades_per_day=3,
    )
    assert signal_id > 0
    with pytest.raises(ValueError):
        journal.record_signal(
            timestamp=ts + timedelta(minutes=10),
            signal_type="BUY_PE",
            entry_price=100.0,
            stop_loss=90.0,
            target=120.0,
            confidence=0.8,
            decision_score=8,
            rationale="duplicate",
            enforce_limits=True,
            cooldown_minutes=15,
            max_trades_per_day=3,
        )


def test_startup_preflight_persists_runtime_health(monkeypatch, tmp_path):
    import ai_trader.main as main_module

    runtime_health_path = tmp_path / "runtime_health.json"
    live_state_path = tmp_path / "live_state.json"
    monkeypatch.setattr(main_module, "_runtime_health_path", runtime_health_path)
    monkeypatch.setattr(main_module, "_live_state_path", live_state_path)
    monkeypatch.setattr(main_module.settings, "llm_validation_enabled", True)
    monkeypatch.setattr(main_module.settings, "openai_api_key", None)
    monkeypatch.setattr(main_module.settings, "strict_startup_checks", False)

    report = main_module.run_startup_preflight()
    assert report["llm_configured"] is False
    assert runtime_health_path.exists()
    persisted = json.loads(runtime_health_path.read_text())
    assert "startup_checks" in persisted
    assert persisted["startup_checks"]["errors"]
