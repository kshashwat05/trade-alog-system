from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from ai_trader.analytics.performance_metrics import summarize_by_outcome
from ai_trader.config.settings import settings
from ai_trader.data.trade_journal import (
    STATUS_EXECUTED,
    STATUS_MISSED,
    STATUS_SIGNAL_GENERATED,
    TradeJournal,
)
from ai_trader.simulation.missed_trade_analyzer import MissedTradeAnalyzer


def create_dashboard_app() -> FastAPI:
    app = FastAPI(title="AI Trader Dashboard")
    journal = TradeJournal(settings.trade_journal_path)
    analyzer = MissedTradeAnalyzer(journal=journal)
    live_state_path = Path(settings.live_state_path)

    def _read_live_state() -> dict[str, Any]:
        if not live_state_path.exists():
            return {}
        return json.loads(live_state_path.read_text())

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>AI Trader Dashboard</title>
          <meta http-equiv="refresh" content="30">
          <style>
            body { font-family: Arial, sans-serif; margin: 24px; background: #f7f8fb; color: #1c2430; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
            .card { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
            table { width: 100%; border-collapse: collapse; font-size: 14px; }
            th, td { padding: 8px; border-bottom: 1px solid #e6e9ef; text-align: left; }
            h1, h2 { margin-top: 0; }
          </style>
        </head>
        <body>
          <h1>AI Trader Dashboard</h1>
          <div id="app">Loading...</div>
          <script>
            async function load() {
              const [summary, signals, analytics, liveState] = await Promise.all([
                fetch('/api/summary').then(r => r.json()),
                fetch('/api/signals').then(r => r.json()),
                fetch('/api/analytics').then(r => r.json()),
                fetch('/api/live-state').then(r => r.json())
              ]);
              document.getElementById('app').innerHTML = `
                <div class="grid">
                  <div class="card"><h2>PnL Summary</h2><pre>${JSON.stringify(summary, null, 2)}</pre></div>
                  <div class="card"><h2>Live Agent Outputs</h2><pre>${JSON.stringify(liveState, null, 2)}</pre></div>
                  <div class="card"><h2>Analytics</h2><pre>${JSON.stringify(analytics, null, 2)}</pre></div>
                  <div class="card"><h2>Recent Signals</h2><pre>${JSON.stringify(signals, null, 2)}</pre></div>
                </div>`;
            }
            load();
          </script>
        </body>
        </html>
        """

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/live-state")
    def live_state() -> dict[str, Any]:
        return _read_live_state()

    @app.get("/api/signals")
    def signals() -> list[dict[str, Any]]:
        return TradeJournal.serialize_entries(journal.get_recent_signals(limit=20))

    @app.get("/api/trades")
    def trades() -> list[dict[str, Any]]:
        return TradeJournal.serialize_entries(journal.get_all_trades())

    @app.get("/api/executed-trades")
    def executed_trades() -> list[dict[str, Any]]:
        return TradeJournal.serialize_entries(journal.get_trades_by_status([STATUS_EXECUTED]))

    @app.get("/api/missed-trades")
    def missed_trades() -> list[dict[str, Any]]:
        return TradeJournal.serialize_entries(
            journal.get_trades_by_status([STATUS_MISSED, STATUS_SIGNAL_GENERATED])
        )

    @app.get("/api/analytics")
    def analytics() -> dict[str, Any]:
        return summarize_by_outcome(journal.get_all_trades())

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        trades = journal.get_all_trades()
        executed = [trade for trade in trades if trade.trade_executed]
        missed = [
            trade for trade in trades
            if trade.status in {STATUS_MISSED, STATUS_SIGNAL_GENERATED} and not trade.trade_executed
        ]
        return {
            "total_trades": len(trades),
            "executed_trades": len(executed),
            "missed_trades": len(missed),
        }

    @app.post("/api/trades/{signal_id}/execute")
    def execute_trade(signal_id: int, execution_price: float, quantity: int) -> dict[str, Any]:
        journal.record_execution(signal_id, execution_price, quantity)
        return {"status": "ok", "signal_id": signal_id}

    @app.post("/api/trades/{signal_id}/missed")
    def mark_missed(signal_id: int) -> dict[str, Any]:
        journal.mark_trade_missed(signal_id)
        analysis = analyzer.analyze_trade(signal_id)
        return {"status": "ok", "signal_id": signal_id, "analysis": analysis.__dict__}

    return app
