from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import pandas as pd
import pytest

from ai_trader.agents.chart_agent import ChartAgent, ChartAnalysis
from ai_trader.agents.option_chain_agent import OptionChainAgent, OptionChainAnalysis
from ai_trader.agents.news_agent import NewsAgent, NewsMacroAnalysis
from ai_trader.agents.volatility_agent import VolatilityAgent, VolatilityAnalysis
from ai_trader.agents.regime_agent import RegimeAgent, RegimeAnalysis
from ai_trader.agents.liquidity_agent import LiquidityAgent, LiquidityAnalysis
from ai_trader.agents.trigger_agent import TradeTriggerAgent, TradeSignal
from ai_trader.agents.fii_positioning_agent import FiiPositioningAnalysis
from ai_trader.agents.gamma_agent import GammaAnalysis
from ai_trader.agents.liquidity_sweep_agent import LiquiditySweepAnalysis
from ai_trader.agents.risk_agent import RiskManagerAgent, RiskCheckResult
from ai_trader.data.market_data_context import MarketDataQuality
from ai_trader.orchestrator.decision_engine import DecisionEngine, OrchestratorOutput
from ai_trader.main import is_market_open


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_chart_agent_returns_analysis(monkeypatch):
    # Force ChartAgent to work with empty data but still return a valid object
    agent = ChartAgent()
    analysis = agent.analyze()
    assert isinstance(analysis, ChartAnalysis)
    assert analysis.trend in ("bullish", "bearish", "neutral")


def test_option_chain_agent_works_with_empty_data(monkeypatch):
    class DummyClient:
        def summarize(self):
            return type(
                "S", (),
                {"support": 23000, "resistance": 24000, "pcr": 1.0, "bias": "neutral"},
            )()

    agent = OptionChainAgent(client=DummyClient())  # type: ignore[arg-type]
    res = agent.analyze()
    assert isinstance(res, OptionChainAnalysis)
    assert res.support == 23000
    assert res.resistance == 24000


def test_news_agent_neutral_without_key(monkeypatch):
    agent = NewsAgent(api_key=None)
    res = agent.analyze()
    assert isinstance(res, NewsMacroAnalysis)
    assert res.macro_bias in ("bullish", "bearish", "neutral")


def test_volatility_agent_basic():
    class DummySession:
        def get(self, url, timeout=5):
            if "quote-indices" in url:
                return DummyResponse({"data": [{"last": 14.5}]})
            return DummyResponse({})

    agent = VolatilityAgent(session=DummySession())  # type: ignore[arg-type]
    res = agent.analyze(spot=24000)
    assert isinstance(res, VolatilityAnalysis)
    assert len(res.expected_range) == 2


def test_regime_agent_handles_empty_data():
    agent = RegimeAgent()
    res = agent.analyze()
    assert isinstance(res, RegimeAnalysis)
    assert res.regime in ("trend_up", "trend_down", "range_bound", "high_volatility")


def test_liquidity_agent():
    agent = LiquidityAgent()
    res = agent.analyze(avg_spread=1.0, volume_score=0.8)
    assert isinstance(res, LiquidityAnalysis)
    assert res.liquidity in ("high", "medium", "low")


def test_trigger_and_risk_and_orchestrator_integration():
    # Build synthetic inputs
    chart = ChartAnalysis(trend="bearish", structure="breakout", confidence=0.8)
    option_chain = OptionChainAnalysis(
        support=23000, resistance=24000, pcr=0.8, bias="bearish"
    )
    news = NewsMacroAnalysis(macro_bias="bearish", risk_level="high")
    vol = VolatilityAnalysis(volatility="medium", expected_range=(23800, 24200))
    regime = RegimeAnalysis(regime="trend_down", confidence=0.9)
    liq = LiquidityAnalysis(liquidity="high", slippage_risk="low")
    fii = FiiPositioningAnalysis(
        fii_bias="bearish",
        institutional_support=23600,
        institutional_resistance=23800,
        confidence=0.8,
    )
    gamma = GammaAnalysis(
        gamma_regime="negative_gamma",
        gamma_flip_level=23700,
        expected_move="expansion",
    )
    liquidity_sweep = LiquiditySweepAnalysis(
        liquidity_event=False,
        event_type="none",
        confidence=0.8,
    )

    trigger = TradeTriggerAgent()
    signal = trigger.generate_signal(
        chart=chart,
        option_chain=option_chain,
        news=news,
        vol=vol,
        regime=regime,
        liquidity=liq,
        fii=fii,
        gamma=gamma,
        liquidity_sweep=liquidity_sweep,
        spot=24000,
        market_context=type(
            "Ctx",
            (),
            {
                "option_chain_raw": {
                    "records": {
                        "data": [
                            {
                                "strikePrice": 24000,
                                "PE": {"lastPrice": 210.0, "identifier": "NIFTY24MAR24000PE"},
                                "CE": {"lastPrice": 190.0, "identifier": "NIFTY24MAR24000CE"},
                            }
                        ]
                    }
                }
            },
        )(),
    )
    assert isinstance(signal, TradeSignal)
    assert signal.signal in ("BUY_PE", "BUY_CE", "NONE")
    assert isinstance(signal.rationale, str)

    risk = RiskManagerAgent()
    check = risk.check(signal, lots=1)
    assert isinstance(check, RiskCheckResult)

    # Orchestrator smoke test
    engine = DecisionEngine()
    engine.market_data_provider.build = lambda: type(  # type: ignore[method-assign]
        "Ctx",
        (),
        {
            "spot_price": 24000,
            "option_chain_raw": {
                "records": {
                    "data": [
                        {
                            "strikePrice": 24000,
                            "PE": {"lastPrice": 210.0, "identifier": "NIFTY24MAR24000PE"},
                            "CE": {"lastPrice": 190.0, "identifier": "NIFTY24MAR24000CE"},
                        }
                    ]
                }
            },
            "quality": MarketDataQuality(
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
            ),
        },
    )()
    engine.chart_agent.analyze = lambda context=None: chart  # type: ignore[method-assign]
    engine.option_agent.analyze = lambda context=None: option_chain  # type: ignore[method-assign]
    engine.news_agent.analyze = lambda: news  # type: ignore[method-assign]
    engine.vol_agent.analyze = lambda spot=None, context=None: vol  # type: ignore[method-assign]
    engine.regime_agent.analyze = lambda context=None: regime  # type: ignore[method-assign]
    engine.liquidity_agent.analyze = lambda avg_spread=None, volume_score=None, context=None: liq  # type: ignore[method-assign]
    engine.fii_agent.analyze = lambda context: fii  # type: ignore[method-assign]
    engine.gamma_agent.analyze = lambda context: gamma  # type: ignore[method-assign]
    engine.liquidity_sweep_agent.analyze = lambda context: liquidity_sweep  # type: ignore[method-assign]
    out = engine.run_once()
    assert isinstance(out, OrchestratorOutput)
    assert isinstance(out.decision_score, int)
    assert "chart" in out.state


def test_risk_manager_enforces_max_trades_per_day():
    risk = RiskManagerAgent()
    signal = TradeSignal(
        signal="BUY_CE",
        entry=1.0,
        stop_loss=0.8,
        target=1.2,
        confidence=0.9,
        rationale="test",
    )

    for _ in range(5):
        risk.record_trade_open(date.today())

    check = risk.check(signal, lots=1)
    assert isinstance(check, RiskCheckResult)
    assert check.allowed is False
    assert check.reason is not None


def test_risk_manager_enforces_cooldown():
    risk = RiskManagerAgent()
    signal = TradeSignal(
        signal="BUY_CE",
        entry=1.0,
        stop_loss=0.8,
        target=1.2,
        confidence=0.9,
        rationale="test",
    )
    risk.record_signal_emitted()
    check = risk.check(signal, lots=1)
    assert check.allowed is False
    assert check.reason is not None


def test_risk_manager_authorize_signal_is_atomic():
    risk = RiskManagerAgent()
    signal = TradeSignal(
        signal="BUY_CE",
        entry=100.0,
        stop_loss=80.0,
        target=140.0,
        confidence=0.9,
        rationale="test",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(risk.authorize_signal, signal) for _ in range(2)]
        results = [future.result() for future in futures]

    allowed_count = sum(1 for result in results if result.allowed)
    assert allowed_count == 1
    assert any(result.reason and "cooldown" in result.reason.lower() for result in results if not result.allowed)


def test_is_market_open_uses_configured_market_window():
    inside = datetime(2026, 3, 12, 10, 0)
    outside = datetime(2026, 3, 12, 16, 0)

    assert is_market_open(inside) is True
    assert is_market_open(outside) is False


def test_trigger_blocks_conflicting_news():
    trigger = TradeTriggerAgent()
    signal = trigger.generate_signal(
        chart=ChartAnalysis(trend="bullish", structure="breakout", confidence=0.9),
        option_chain=OptionChainAnalysis(
            support=23000, resistance=24000, pcr=1.2, bias="bullish"
        ),
        news=NewsMacroAnalysis(macro_bias="bearish", risk_level="high"),
        vol=VolatilityAnalysis(volatility="medium", expected_range=(23800, 24200)),
        regime=RegimeAnalysis(regime="trend_up", confidence=0.9),
        liquidity=LiquidityAnalysis(liquidity="high", slippage_risk="low"),
        fii=FiiPositioningAnalysis("bullish", 23600, 23800, 0.8),
        gamma=GammaAnalysis("negative_gamma", 23700, "expansion"),
        liquidity_sweep=LiquiditySweepAnalysis(False, "none", 0.8),
        spot=24000,
        market_context=type(
            "Ctx",
            (),
            {
                "option_chain_raw": {
                    "records": {
                        "data": [
                            {
                                "strikePrice": 24000,
                                "PE": {"lastPrice": 210.0, "identifier": "NIFTY24MAR24000PE"},
                                "CE": {"lastPrice": 190.0, "identifier": "NIFTY24MAR24000CE"},
                            }
                        ]
                    }
                }
            },
        )(),
    )
    assert signal.signal == "NONE"
    assert "conflicts" in signal.rationale


def test_decision_engine_masks_signal_when_risk_rejects():
    chart = ChartAnalysis(trend="bearish", structure="breakout", confidence=0.8)
    option_chain = OptionChainAnalysis(
        support=23000, resistance=24000, pcr=0.8, bias="bearish"
    )
    news = NewsMacroAnalysis(macro_bias="bearish", risk_level="high")
    vol = VolatilityAnalysis(volatility="medium", expected_range=(23800, 24200))
    regime = RegimeAnalysis(regime="trend_down", confidence=0.9)
    liq = LiquidityAnalysis(liquidity="high", slippage_risk="low")
    fii = FiiPositioningAnalysis("bearish", 23600, 23800, 0.8)
    gamma = GammaAnalysis("negative_gamma", 23700, "expansion")
    liquidity_sweep = LiquiditySweepAnalysis(False, "none", 0.8)

    engine = DecisionEngine()
    engine.market_data_provider.build = lambda: type(  # type: ignore[method-assign]
        "Ctx",
        (),
        {
            "spot_price": 24000,
            "option_chain_raw": {
                "records": {
                    "data": [
                        {
                            "strikePrice": 24000,
                            "PE": {"lastPrice": 210.0, "identifier": "NIFTY24MAR24000PE"},
                            "CE": {"lastPrice": 190.0, "identifier": "NIFTY24MAR24000CE"},
                        }
                    ]
                }
            },
            "quality": MarketDataQuality(
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
            ),
        },
    )()
    engine.chart_agent.analyze = lambda context=None: chart  # type: ignore[method-assign]
    engine.option_agent.analyze = lambda context=None: option_chain  # type: ignore[method-assign]
    engine.news_agent.analyze = lambda: news  # type: ignore[method-assign]
    engine.vol_agent.analyze = lambda spot=None, context=None: vol  # type: ignore[method-assign]
    engine.regime_agent.analyze = lambda context=None: regime  # type: ignore[method-assign]
    engine.liquidity_agent.analyze = lambda avg_spread=None, volume_score=None, context=None: liq  # type: ignore[method-assign]
    engine.fii_agent.analyze = lambda context: fii  # type: ignore[method-assign]
    engine.gamma_agent.analyze = lambda context: gamma  # type: ignore[method-assign]
    engine.liquidity_sweep_agent.analyze = lambda context: liquidity_sweep  # type: ignore[method-assign]

    out = engine.run_once()
    assert out.signal.signal == "NONE"
    assert out.risk.allowed is False
    assert out.risk.reason == "News risk too high."


def test_decision_engine_defaults_to_deterministic_signal_on_llm_failure():
    chart = ChartAnalysis(trend="bearish", structure="breakout", confidence=0.8)
    option_chain = OptionChainAnalysis(
        support=23000, resistance=24000, pcr=0.8, bias="bearish"
    )
    news = NewsMacroAnalysis(macro_bias="bearish", risk_level="medium")
    vol = VolatilityAnalysis(volatility="medium", expected_range=(23800, 24200))
    regime = RegimeAnalysis(regime="trend_down", confidence=0.9)
    liq = LiquidityAnalysis(liquidity="high", slippage_risk="low")
    fii = FiiPositioningAnalysis("bearish", 23600, 23800, 0.8)
    gamma = GammaAnalysis("negative_gamma", 23700, "expansion")
    liquidity_sweep = LiquiditySweepAnalysis(False, "none", 0.8)

    engine = DecisionEngine()
    engine.market_data_provider.build = lambda: type(  # type: ignore[method-assign]
        "Ctx",
        (),
        {
            "spot_price": 24000,
            "option_chain_raw": {
                "records": {
                    "data": [
                        {
                            "strikePrice": 24000,
                            "PE": {"lastPrice": 210.0, "identifier": "NIFTY24MAR24000PE"},
                            "CE": {"lastPrice": 190.0, "identifier": "NIFTY24MAR24000CE"},
                        }
                    ]
                }
            },
            "quality": MarketDataQuality(
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
            ),
        },
    )()
    engine.chart_agent.analyze = lambda context=None: chart  # type: ignore[method-assign]
    engine.option_agent.analyze = lambda context=None: option_chain  # type: ignore[method-assign]
    engine.news_agent.analyze = lambda: news  # type: ignore[method-assign]
    engine.vol_agent.analyze = lambda spot=None, context=None: vol  # type: ignore[method-assign]
    engine.regime_agent.analyze = lambda context=None: regime  # type: ignore[method-assign]
    engine.liquidity_agent.analyze = lambda avg_spread=None, volume_score=None, context=None: liq  # type: ignore[method-assign]
    engine.fii_agent.analyze = lambda context: fii  # type: ignore[method-assign]
    engine.gamma_agent.analyze = lambda context: gamma  # type: ignore[method-assign]
    engine.liquidity_sweep_agent.analyze = lambda context: liquidity_sweep  # type: ignore[method-assign]
    engine.llm_validator.validate = lambda payload: (_ for _ in ()).throw(RuntimeError("llm timeout"))  # type: ignore[method-assign]

    out = engine.run_once()
    assert out.signal.signal == "BUY_PE"
    assert out.risk.allowed is True
    assert out.llm_validation.fallback_used is True


def test_strategy_rejects_missing_columns():
    from ai_trader.strategies.nifty_intraday_strategy import NiftyIntradayStrategy

    strategy = NiftyIntradayStrategy()
    with pytest.raises(ValueError):
        strategy.generate_signals(pd.DataFrame({"close": [1, 2, 3]}))
