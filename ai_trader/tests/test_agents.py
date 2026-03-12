from __future__ import annotations

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
from ai_trader.agents.risk_agent import RiskManagerAgent, RiskCheckResult
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
    liq = LiquidityAnalysis(liquidity="medium", slippage_risk="medium")

    trigger = TradeTriggerAgent()
    signal = trigger.generate_signal(
        chart=chart,
        option_chain=option_chain,
        news=news,
        vol=vol,
        regime=regime,
        liquidity=liq,
        spot=24000,
    )
    assert isinstance(signal, TradeSignal)
    assert signal.signal in ("BUY_PE", "BUY_CE", "NONE")
    assert isinstance(signal.rationale, str)

    risk = RiskManagerAgent()
    check = risk.check(signal, lots=1)
    assert isinstance(check, RiskCheckResult)

    # Orchestrator smoke test
    engine = DecisionEngine()
    engine.chart_agent.analyze = lambda: chart  # type: ignore[method-assign]
    engine.option_agent.analyze = lambda: option_chain  # type: ignore[method-assign]
    engine.news_agent.analyze = lambda: news  # type: ignore[method-assign]
    engine.vol_agent.analyze = lambda: vol  # type: ignore[method-assign]
    engine.regime_agent.analyze = lambda: regime  # type: ignore[method-assign]
    engine.liquidity_agent.analyze = lambda: liq  # type: ignore[method-assign]
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
        spot=24000,
    )
    assert signal.signal == "NONE"
    assert "conflicts" in signal.rationale


def test_strategy_rejects_missing_columns():
    from ai_trader.strategies.nifty_intraday_strategy import NiftyIntradayStrategy

    strategy = NiftyIntradayStrategy()
    with pytest.raises(ValueError):
        strategy.generate_signals(pd.DataFrame({"close": [1, 2, 3]}))
