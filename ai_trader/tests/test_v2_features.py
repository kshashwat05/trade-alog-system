from __future__ import annotations

import json
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
    journal.record_execution(signal_id, execution_price=101.0, quantity=2)
    entry = journal.get_trade(signal_id)
    assert entry is not None
    assert entry.trade_executed is True
    assert entry.status == STATUS_EXECUTED
    assert entry.quantity == 2


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
    journal.record_execution(signal_id, execution_price=100.0, quantity=1)
    monitor = PositionMonitorAgent(journal=journal, price_fetcher=lambda trade: 121.0)
    results = monitor.monitor_once()
    assert len(results) == 1
    assert results[0].status == STATUS_TARGET_HIT
    assert journal.get_trade(signal_id).status == STATUS_TARGET_HIT  # type: ignore[union-attr]


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
    journal.record_execution(executed_id, 100.0, 1)
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
    monkeypatch.setattr(settings, "trade_journal_path", str(db_path))
    monkeypatch.setattr(settings, "live_state_path", str(live_state_path))
    app = create_dashboard_app()
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/signals").status_code == 200
    assert client.get("/api/live-state").json()["decision_score"] == 6
    signal_id = client.get("/api/signals").json()[0]["id"]
    assert client.post(f"/api/trades/{signal_id}/execute", params={"execution_price": 101, "quantity": 1}).status_code == 200
