from __future__ import annotations

import json
import sys
from datetime import datetime

import pandas as pd
import pytest

from ai_trader.agents.position_monitor_agent import PositionMonitorAgent
from ai_trader.analytics.performance_metrics import summarize_by_outcome
from ai_trader.config.settings import settings
from ai_trader.data.kite_client import PriceData
from ai_trader.data.trade_journal import (
    STATUS_EXECUTED,
    STATUS_TARGET_HIT,
    TradeJournal,
)
from ai_trader.simulation.missed_trade_analyzer import MissedTradeAnalyzer


def test_trade_journal_records_signal_and_execution(tmp_path):
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
        metadata={"foo": "bar"},
    )
    journal.record_execution(
        signal_id,
        execution_price=101.0,
        quantity=2,
        instrument_key="NFO:NIFTY24MAR23700CE",
    )
    entry = journal.get_trade(signal_id)
    assert entry is not None
    assert entry.trade_executed is True
    assert entry.status == STATUS_EXECUTED
    assert entry.quantity == 2


def test_trade_journal_metadata_merge(tmp_path):
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
        metadata={"foo": "bar"},
    )
    journal.merge_metadata(signal_id, {"instrument_key": "NFO:NIFTY24MAR23700PE"})
    entry = journal.get_trade(signal_id)
    assert entry is not None
    assert entry.metadata["foo"] == "bar"
    assert entry.metadata["instrument_key"] == "NFO:NIFTY24MAR23700PE"


def test_position_monitor_closes_target_hit(tmp_path):
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
    )
    journal.record_execution(
        signal_id,
        execution_price=100.0,
        quantity=1,
        instrument_key="NFO:NIFTY24MAR23700PE",
    )
    monitor = PositionMonitorAgent(journal=journal, price_fetcher=lambda trade: 121.0)
    results = monitor.monitor_once()
    assert len(results) == 1
    assert results[0].status == STATUS_TARGET_HIT
    assert journal.get_trade(signal_id).status == STATUS_TARGET_HIT  # type: ignore[union-attr]


def test_position_monitor_skips_trade_without_instrument_token(tmp_path):
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
    )
    journal.merge_metadata(signal_id, {"note": "missing instrument key"})
    journal.record_execution(
        signal_id,
        execution_price=100.0,
        quantity=1,
        instrument_key="NFO:NIFTY24MAR23700PE",
    )
    journal.merge_metadata(signal_id, {"instrument_key": None})
    monitor = PositionMonitorAgent(journal=journal, price_fetcher=lambda trade: 121.0)
    assert monitor.monitor_once() == []
    assert journal.get_trade(signal_id).status == STATUS_EXECUTED  # type: ignore[union-attr]


def test_missed_trade_analyzer_updates_journal(tmp_path):
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
        metadata={"simulation_prices": [100.0, 110.0, 125.0]},
    )

    class DummyClient:
        def fetch_nifty_intraday(self, days=1):
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        ["2026-03-13 09:15:00", "2026-03-13 09:20:00", "2026-03-13 09:25:00"]
                    ),
                    "close": [100.0, 110.0, 125.0],
                }
            )
            return PriceData(df=df)

    analyzer = MissedTradeAnalyzer(journal=journal, client=DummyClient())  # type: ignore[arg-type]
    analysis = analyzer.analyze_trade(signal_id)
    updated = journal.get_trade(signal_id)
    assert analysis.target_hit is True
    assert updated is not None
    assert updated.max_profit is not None
    assert updated.target_hit is True


def test_performance_metrics_split_executed_and_missed(tmp_path):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    executed_id = journal.record_signal(
        timestamp=datetime(2026, 3, 13, 9, 15),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="Aligned setup",
    )
    missed_id = journal.record_signal(
        timestamp=datetime(2026, 3, 13, 9, 17),
        signal_type="BUY_PE",
        entry_price=100.0,
        stop_loss=110.0,
        target=85.0,
        confidence=0.7,
        decision_score=5,
        rationale="Bearish setup",
    )
    journal.record_execution(
        executed_id,
        100.0,
        1,
        instrument_key="NFO:NIFTY24MAR23700CE",
    )
    journal.close_trade(executed_id, status=STATUS_TARGET_HIT, exit_price=120.0, pnl=20.0)
    journal.mark_trade_missed(missed_id)
    journal.update_simulation(
        missed_id,
        max_profit=10.0,
        max_drawdown=-5.0,
        target_hit=False,
        stop_loss_hit=False,
        pnl=3.0,
    )
    summaries = summarize_by_outcome(journal.get_all_trades())
    assert summaries["executed"]["trade_count"] == 1
    assert summaries["missed"]["trade_count"] == 1


def test_dashboard_endpoints(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ai_trader.dashboard.app import create_dashboard_app

    db_path = tmp_path / "trade_journal.db"
    live_state_path = tmp_path / "live_state.json"
    runtime_health_path = tmp_path / "runtime_health.json"
    journal = TradeJournal(db_path)
    journal.record_signal(
        timestamp=datetime(2026, 3, 13, 9, 15),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="Aligned setup",
    )
    live_state_path.write_text(json.dumps({"decision_score": 6}))
    runtime_health_path.write_text(json.dumps({"startup_checks": {"errors": [], "warnings": []}}))
    monkeypatch.setattr(settings, "trade_journal_path", str(db_path))
    monkeypatch.setattr(settings, "live_state_path", str(live_state_path))
    monkeypatch.setattr(settings, "runtime_health_path", str(runtime_health_path))
    app = create_dashboard_app()
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/readiness").json()["ready"] is True
    assert client.get("/api/signals").status_code == 200
    live_state = client.get("/api/live-state").json()
    assert live_state["decision_score"] == 6
    assert "runtime_health" in live_state
    signal_id = client.get("/api/signals").json()[0]["id"]
    assert client.post(
        f"/api/trades/{signal_id}/execute",
        params={"execution_price": 101, "quantity": 1, "instrument_key": "NFO:NIFTY24MAR23700PE"},
    ).status_code == 200


def test_dashboard_executes_trade_via_symbol_resolution(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ai_trader.dashboard.app import create_dashboard_app

    db_path = tmp_path / "trade_journal.db"
    live_state_path = tmp_path / "live_state.json"
    runtime_health_path = tmp_path / "runtime_health.json"
    journal = TradeJournal(db_path)
    signal_id = journal.record_signal(
        timestamp=datetime(2026, 3, 13, 9, 15),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="Aligned setup",
    )
    live_state_path.write_text(json.dumps({"decision_score": 6}))
    runtime_health_path.write_text(json.dumps({"startup_checks": {"errors": [], "warnings": []}}))
    monkeypatch.setattr(settings, "trade_journal_path", str(db_path))
    monkeypatch.setattr(settings, "live_state_path", str(live_state_path))
    monkeypatch.setattr(settings, "runtime_health_path", str(runtime_health_path))
    monkeypatch.setattr(
        "ai_trader.dashboard.app.KiteClient.resolve_instrument_key",
        lambda self, symbol, exchange="NFO": f"{exchange}:{symbol.upper()}",
    )
    client = TestClient(create_dashboard_app())
    response = client.post(
        f"/api/trades/{signal_id}/execute",
        params={"execution_price": 101, "quantity": 1, "instrument_symbol": "nifty24mar23700pe"},
    )
    assert response.status_code == 200
    assert journal.get_trade(signal_id).metadata["instrument_key"] == "NFO:NIFTY24MAR23700PE"  # type: ignore[union-attr]


def test_dashboard_exposes_replay_reports(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ai_trader.dashboard.app import create_dashboard_app

    db_path = tmp_path / "trade_journal.db"
    live_state_path = tmp_path / "live_state.json"
    runtime_health_path = tmp_path / "runtime_health.json"
    replay_reports_path = tmp_path / "replay_reports"
    replay_reports_path.mkdir()
    report_path = replay_reports_path / "replay_2026-03-10.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "replay_date": "2026-03-10",
                    "signals_generated": 2,
                    "wrong_signals": 1,
                },
                "signals": [{"signal_timestamp": "2026-03-10T10:00:00", "signal_type": "BUY_CE"}],
            }
        )
    )
    live_state_path.write_text(json.dumps({"decision_score": 6}))
    runtime_health_path.write_text(json.dumps({"startup_checks": {"errors": [], "warnings": []}}))
    monkeypatch.setattr(settings, "trade_journal_path", str(db_path))
    monkeypatch.setattr(settings, "live_state_path", str(live_state_path))
    monkeypatch.setattr(settings, "runtime_health_path", str(runtime_health_path))
    monkeypatch.setattr(settings, "replay_reports_path", str(replay_reports_path))

    client = TestClient(create_dashboard_app())
    replays = client.get("/api/replays")
    assert replays.status_code == 200
    payload = replays.json()
    assert payload[0]["summary"]["replay_date"] == "2026-03-10"
    detail = client.get("/api/replays/replay_2026-03-10.json")
    assert detail.status_code == 200
    assert detail.json()["summary"]["signals_generated"] == 2


def test_dashboard_returns_not_found_for_missing_replay_report(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ai_trader.dashboard.app import create_dashboard_app

    replay_reports_path = tmp_path / "replay_reports"
    replay_reports_path.mkdir()
    monkeypatch.setattr(settings, "trade_journal_path", str(tmp_path / "trade_journal.db"))
    monkeypatch.setattr(settings, "live_state_path", str(tmp_path / "live_state.json"))
    monkeypatch.setattr(settings, "runtime_health_path", str(tmp_path / "runtime_health.json"))
    monkeypatch.setattr(settings, "replay_reports_path", str(replay_reports_path))
    (tmp_path / "runtime_health.json").write_text(json.dumps({"startup_checks": {"errors": [], "warnings": []}}))
    client = TestClient(create_dashboard_app())
    response = client.get("/api/replays/replay_missing.json")
    assert response.status_code == 200
    assert response.json()["status"] == "not_found"


def test_manual_trade_cli_record_trade(tmp_path, monkeypatch, capsys):
    from ai_trader.manual_trade_cli import main

    db_path = tmp_path / "trade_journal.db"
    monkeypatch.setattr(settings, "trade_journal_path", str(db_path))
    journal = TradeJournal(db_path)
    signal_id = journal.record_signal(
        timestamp=datetime(2026, 3, 13, 9, 15),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="Aligned setup",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manual_trade_cli",
            "record_trade",
            "--signal_id",
            str(signal_id),
            "--price",
            "101",
            "--lots",
            "1",
            "--instrument-key",
            "NFO:NIFTY24MAR23700PE",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert f"signal_id={signal_id}" in out
    updated = journal.get_trade(signal_id)
    assert updated is not None
    assert updated.trade_executed is True
    assert updated.metadata["instrument_key"] == "NFO:NIFTY24MAR23700PE"


def test_manual_trade_cli_record_trade_via_symbol(tmp_path, monkeypatch, capsys):
    from ai_trader.manual_trade_cli import main

    db_path = tmp_path / "trade_journal.db"
    monkeypatch.setattr(settings, "trade_journal_path", str(db_path))
    journal = TradeJournal(db_path)
    signal_id = journal.record_signal(
        timestamp=datetime(2026, 3, 13, 9, 15),
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        decision_score=6,
        rationale="Aligned setup",
    )
    monkeypatch.setattr(
        "ai_trader.manual_trade_cli.KiteClient.resolve_instrument_key",
        lambda self, symbol, exchange="NFO": f"{exchange}:{symbol.upper()}",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manual_trade_cli",
            "record_trade",
            "--signal_id",
            str(signal_id),
            "--price",
            "101",
            "--lots",
            "1",
            "--instrument-symbol",
            "nifty24mar23700pe",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert f"signal_id={signal_id}" in out
    assert journal.get_trade(signal_id).metadata["instrument_key"] == "NFO:NIFTY24MAR23700PE"  # type: ignore[union-attr]


def test_main_price_fetcher_skips_without_instrument_key():
    from ai_trader.main import _fetch_trade_price_from_signal
    from ai_trader.data.trade_journal import TradeJournalEntry

    trade = TradeJournalEntry(
        id=1,
        timestamp="2026-03-13T09:15:00",
        signal_type="BUY_CE",
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        trade_executed=True,
        execution_price=101.0,
        exit_price=None,
        closed_at=None,
        pnl=None,
        quantity=1,
        status="executed",
        decision_score=6,
        rationale="Aligned setup",
        metadata={},
        max_profit=None,
        max_drawdown=None,
        target_hit=False,
        stop_loss_hit=False,
        llm_reasoning=None,
        institutional_bias=None,
        gamma_regime=None,
        liquidity_event=None,
    )
    assert _fetch_trade_price_from_signal(trade) is None
