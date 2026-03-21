from __future__ import annotations

from datetime import datetime

from ai_trader.agents.data_quality_guard import DataQualityDecision
from ai_trader.agents.exit_intelligence_agent import ExitIntelligenceAgent
from ai_trader.agents.kill_switch_agent import KillSwitchAgent
from ai_trader.agents.position_monitor_agent import PositionMonitorAgent
from ai_trader.agents.risk_agent import RiskCheckResult
from ai_trader.agents.trigger_agent import TradeSignal
from ai_trader.agents.position_tracker import PositionTracker
from ai_trader.agents.llm_validator_agent import LlmValidationResult
from ai_trader.data.trade_journal import TradeJournal, TradeJournalEntry
from ai_trader.orchestrator.decision_engine import OrchestratorOutput


def _build_trade_entry(*, trade_id: int = 1) -> TradeJournalEntry:
    return TradeJournalEntry(
        id=trade_id,
        timestamp=datetime.utcnow().isoformat(),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        trade_executed=True,
        execution_price=100.0,
        exit_price=None,
        closed_at=None,
        pnl=None,
        quantity=1,
        status="executed",
        decision_score=8,
        rationale="test",
        metadata={"instrument_key": "NFO:NIFTY24MAR24000CE", "instrument_symbol": "NIFTY24MAR24000CE"},
        max_profit=None,
        max_drawdown=None,
        target_hit=False,
        stop_loss_hit=False,
        llm_reasoning=None,
        institutional_bias=None,
        gamma_regime=None,
        liquidity_event=None,
    )


def _build_output(signal: TradeSignal) -> OrchestratorOutput:
    return OrchestratorOutput(
        timestamp=datetime.utcnow().isoformat(),
        signal=signal,
        risk=RiskCheckResult(True, None),
        llm_validation=LlmValidationResult("approved", 0.0, "approved"),
        decision_score=8,
        state={
            "market_context": object(),
            "score_complete": True,
            "liquidity": type("Liquidity", (), {"liquidity": "high"})(),
            "vol": type("Vol", (), {"volatility": "medium"})(),
            "news": type("News", (), {"risk_level": "medium"})(),
            "macro_calendar": type("Macro", (), {"event_risk": "low"})(),
            "global_market": type("Global", (), {"risk_sentiment": "neutral", "confidence": 0.0})(),
            "fii_positioning": type("Fii", (), {"fii_bias": "bullish"})(),
            "gamma_analysis": type("Gamma", (), {"gamma_regime": "negative_gamma"})(),
            "liquidity_sweep": type("Sweep", (), {"event_type": "none"})(),
        },
    )


def test_exit_intelligence_deduplicates_repeated_advisories():
    agent = ExitIntelligenceAgent()
    trade = _build_trade_entry()

    for _ in range(5):
        agent.observe_trade(trade, 100.0)
    first = agent.observe_trade(trade, 100.0)
    agent.mark_advisory_sent(trade.id, first)
    second = agent.observe_trade(trade, 100.0)

    assert first.action == "PARTIAL_EXIT"
    assert second.action == "NONE"
    assert second.reason == "Advisory cooldown active."


def test_position_monitor_skips_advisory_callback_when_hard_exit_hits(tmp_path):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    signal_id = journal.record_signal(
        timestamp=datetime(2026, 3, 13, 9, 15),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="Aligned setup",
        metadata={"instrument_symbol": "NIFTY24MAR24000CE"},
    )
    journal.record_execution(signal_id, execution_price=100.0, quantity=1, instrument_key="NFO:NIFTY24MAR24000CE")
    advisory_calls = {"count": 0}

    def _advisory(trade, current_price):
        advisory_calls["count"] += 1
        raise AssertionError("advisory callback should not run for hard exits")

    monitor = PositionMonitorAgent(journal=journal, price_fetcher=lambda trade: 121.0, exit_intelligence_cb=_advisory)
    results = monitor.monitor_once()

    assert len(results) == 1
    assert results[0].advisory_only is False
    assert advisory_calls["count"] == 0


def test_kill_switch_counts_consecutive_losses_only_for_latest_trading_day(tmp_path, monkeypatch):
    from ai_trader.config.settings import settings

    monkeypatch.setattr(settings, "kill_switch_consecutive_losses", 2)
    journal = TradeJournal(tmp_path / "trade_journal.db")
    yesterday = datetime(2026, 3, 12, 9, 15)
    today = datetime(2026, 3, 13, 9, 15)

    first = journal.record_signal(
        timestamp=yesterday,
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="loss 1",
    )
    journal.record_execution(first, execution_price=100.0, quantity=1, instrument_key="NFO:NIFTY24MAR24000CE")
    journal.close_trade(first, status="stop_loss_hit", exit_price=90.0, pnl=-10.0)
    journal._conn.execute("UPDATE trades SET closed_at = ? WHERE id = ?", ("2026-03-12T15:25:00", first))

    second = journal.record_signal(
        timestamp=today,
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="loss 2",
    )
    journal.record_execution(second, execution_price=100.0, quantity=1, instrument_key="NFO:NIFTY24MAR24000CE")
    journal.close_trade(second, status="stop_loss_hit", exit_price=90.0, pnl=-10.0)
    journal._conn.execute("UPDATE trades SET closed_at = ? WHERE id = ?", ("2026-03-13T09:25:00", second))
    journal._conn.commit()

    decision = KillSwitchAgent().evaluate(journal)
    assert decision.blocked is False
    assert decision.consecutive_losses == 1


def test_kill_switch_uses_realized_close_day_for_overnight_losses(tmp_path, monkeypatch):
    from ai_trader.config.settings import settings

    monkeypatch.setattr(settings, "kill_switch_consecutive_losses", 2)
    journal = TradeJournal(tmp_path / "trade_journal.db")

    first = journal.record_signal(
        timestamp=datetime(2026, 3, 12, 15, 20),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="overnight loss 1",
    )
    journal.record_execution(first, execution_price=100.0, quantity=1, instrument_key="NFO:NIFTY24MAR24000CE")
    journal.close_trade(first, status="stop_loss_hit", exit_price=90.0, pnl=-10.0)
    journal._conn.execute("UPDATE trades SET closed_at = ? WHERE id = ?", ("2026-03-13T09:10:00", first))

    second = journal.record_signal(
        timestamp=datetime(2026, 3, 13, 9, 20),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="same-day loss 2",
    )
    journal.record_execution(second, execution_price=100.0, quantity=1, instrument_key="NFO:NIFTY24MAR24000CE")
    journal.close_trade(second, status="stop_loss_hit", exit_price=90.0, pnl=-10.0)
    journal._conn.execute("UPDATE trades SET closed_at = ? WHERE id = ?", ("2026-03-13T09:30:00", second))
    journal._conn.commit()

    trade = journal.get_trade(first)
    assert trade is not None
    assert trade.closed_at is not None

    decision = KillSwitchAgent().evaluate(journal)
    assert decision.blocked is True
    assert decision.consecutive_losses == 2


def test_position_tracker_scans_all_executed_trades_not_only_open(tmp_path):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    signal_id = journal.record_signal(
        timestamp=datetime(2026, 3, 13, 9, 15),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="Aligned setup",
        metadata={"instrument_symbol": "NIFTY24MAR24000CE"},
    )
    journal.record_execution(signal_id, execution_price=100.0, quantity=1, instrument_key="NFO:NIFTY24MAR24000PE")
    journal.close_trade(signal_id, status="target_hit", exit_price=120.0, pnl=20.0)

    report = PositionTracker().sync_with_journal(journal)
    assert report.mismatches == 1


def test_run_trading_cycle_does_not_authorize_when_data_quality_blocks(monkeypatch, tmp_path):
    import ai_trader.main as main_module

    signal = TradeSignal(
        signal="BUY_CE",
        entry=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        rationale="test",
    )
    output = _build_output(signal)
    authorization_calls = {"count": 0}

    monkeypatch.setattr(main_module, "_live_state_path", tmp_path / "live_state.json")
    monkeypatch.setattr(main_module, "_runtime_health_path", tmp_path / "runtime_health.json")
    monkeypatch.setattr(main_module, "is_market_open", lambda now=None: True)
    monkeypatch.setattr(main_module.engine, "run_once", lambda open_trades=0: output)
    monkeypatch.setattr(main_module.journal, "get_open_trades", lambda: [])
    monkeypatch.setattr(main_module.data_quality_guard, "evaluate", lambda *args, **kwargs: DataQualityDecision(False, "bad data"))
    monkeypatch.setattr(
        main_module.engine.risk_agent,
        "authorize_signal",
        lambda *args, **kwargs: authorization_calls.__setitem__("count", authorization_calls["count"] + 1),
    )
    monkeypatch.setattr(main_module.settings, "data_quality_guard_enabled", True)
    monkeypatch.setattr(main_module.settings, "execution_intelligence_enabled", False)
    monkeypatch.setattr(main_module.settings, "kill_switch_enabled", False)

    main_module.run_trading_cycle()
    assert authorization_calls["count"] == 0


def test_run_trading_cycle_does_not_mark_alert_delivered_on_alert_failure(monkeypatch, tmp_path):
    import ai_trader.main as main_module

    signal = TradeSignal(
        signal="BUY_CE",
        entry=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        rationale="test",
    )
    output = _build_output(signal)
    alert_delivery_calls = {"count": 0}

    monkeypatch.setattr(main_module, "_live_state_path", tmp_path / "live_state.json")
    monkeypatch.setattr(main_module, "_runtime_health_path", tmp_path / "runtime_health.json")
    monkeypatch.setattr(main_module, "is_market_open", lambda now=None: True)
    monkeypatch.setattr(main_module.engine, "run_once", lambda open_trades=0: output)
    monkeypatch.setattr(main_module.journal, "get_open_trades", lambda: [])
    monkeypatch.setattr(main_module.journal, "record_signal", lambda **kwargs: 42)
    monkeypatch.setattr(main_module.engine.risk_agent, "authorize_signal", lambda *args, **kwargs: RiskCheckResult(True, None))
    monkeypatch.setattr(main_module.alerter, "send_trade_signal", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("twilio down")))
    monkeypatch.setattr(
        main_module.position_tracker,
        "record_alert_delivery",
        lambda *args, **kwargs: alert_delivery_calls.__setitem__("count", alert_delivery_calls["count"] + 1),
    )
    monkeypatch.setattr(main_module.settings, "data_quality_guard_enabled", False)
    monkeypatch.setattr(main_module.settings, "execution_intelligence_enabled", False)
    monkeypatch.setattr(main_module.settings, "kill_switch_enabled", False)
    monkeypatch.setattr(main_module.settings, "position_tracker_enabled", True)

    main_module.run_trading_cycle()
    assert alert_delivery_calls["count"] == 0
