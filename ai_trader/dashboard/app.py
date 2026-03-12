from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from ai_trader.analytics.performance_metrics import summarize_by_outcome
from ai_trader.config.settings import settings
from ai_trader.data.kite_client import KiteClient
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
    kite_client = KiteClient()
    analyzer = MissedTradeAnalyzer(journal=journal)
    live_state_path = Path(settings.live_state_path)
    runtime_health_path = Path(settings.runtime_health_path)
    replay_reports_path = Path(settings.replay_reports_path)

    def _read_live_state() -> dict[str, Any]:
        if not live_state_path.exists():
            return {}
        return json.loads(live_state_path.read_text())

    def _read_runtime_health() -> dict[str, Any]:
        if not runtime_health_path.exists():
            return {}
        return json.loads(runtime_health_path.read_text())

    def _list_replay_reports() -> list[dict[str, Any]]:
        if not replay_reports_path.exists():
            return []
        reports: list[dict[str, Any]] = []
        for report_path in sorted(replay_reports_path.glob("replay_*.json"), reverse=True):
            try:
                payload = json.loads(report_path.read_text())
            except Exception:  # noqa: BLE001
                continue
            summary = payload.get("summary", {})
            reports.append(
                {
                    "report_name": report_path.name,
                    "report_path": str(report_path),
                    "summary": summary,
                    "signal_count": len(payload.get("signals", [])),
                }
            )
        return reports

    def _read_replay_report(report_name: str) -> dict[str, Any]:
        report_path = replay_reports_path / report_name
        if not report_path.exists():
            return {}
        return json.loads(report_path.read_text())

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
              const [summary, signals, analytics, liveState, replays] = await Promise.all([
                fetch('/api/summary').then(r => r.json()),
                fetch('/api/signals').then(r => r.json()),
                fetch('/api/analytics').then(r => r.json()),
                fetch('/api/live-state').then(r => r.json()),
                fetch('/api/replays').then(r => r.json())
              ]);
              const institutional = liveState.state || {};
              const latestReplay = replays.length > 0 ? replays[0] : {};
              document.getElementById('app').innerHTML = `
                <div class="grid">
                  <div class="card"><h2>PnL Summary</h2><pre>${JSON.stringify(summary, null, 2)}</pre></div>
                  <div class="card"><h2>Live Agent Outputs</h2><pre>${JSON.stringify(liveState, null, 2)}</pre></div>
                  <div class="card"><h2>Institutional Panel</h2><pre>${JSON.stringify({
                    fii_positioning: institutional.fii_positioning,
                    gamma_analysis: institutional.gamma_analysis
                  }, null, 2)}</pre></div>
                  <div class="card"><h2>Market Intelligence</h2><pre>${JSON.stringify({
                    liquidity_sweep: institutional.liquidity_sweep,
                    regime: institutional.regime,
                    volatility: institutional.vol
                  }, null, 2)}</pre></div>
                  <div class="card"><h2>LLM Reasoning</h2><pre>${JSON.stringify(liveState.llm_validation || {}, null, 2)}</pre></div>
                  <div class="card"><h2>Analytics</h2><pre>${JSON.stringify(analytics, null, 2)}</pre></div>
                  <div class="card"><h2>Recent Signals</h2><pre>${JSON.stringify(signals, null, 2)}</pre></div>
                  <div class="card"><h2>Latest Replay</h2><pre>${JSON.stringify(latestReplay, null, 2)}</pre></div>
                </div>`;
            }
            load();
          </script>
        </body>
        </html>
        """

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        runtime_health = _read_runtime_health()
        startup_checks = runtime_health.get("startup_checks", {})
        healthy = not startup_checks.get("errors")
        return {
            "status": "ok" if healthy else "degraded",
            "startup_checks": startup_checks,
            "runtime": runtime_health,
        }

    @app.get("/api/readiness")
    def readiness() -> dict[str, Any]:
        runtime_health = _read_runtime_health()
        startup_checks = runtime_health.get("startup_checks", {})
        ready = not startup_checks.get("errors")
        return {
            "ready": ready,
            "errors": startup_checks.get("errors", []),
            "warnings": startup_checks.get("warnings", []),
        }

    @app.get("/api/live-state")
    def live_state() -> dict[str, Any]:
        live_state = _read_live_state()
        live_state["runtime_health"] = _read_runtime_health()
        return live_state

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

    @app.get("/api/replays")
    def replays() -> list[dict[str, Any]]:
        return _list_replay_reports()

    @app.get("/api/replays/{report_name}")
    def replay_report(report_name: str) -> dict[str, Any]:
        report = _read_replay_report(report_name)
        if not report:
            return {"status": "not_found", "report_name": report_name}
        return report

    @app.get("/api/institutional")
    def institutional() -> dict[str, Any]:
        live_state = _read_live_state()
        state = live_state.get("state", {})
        return {
            "fii_positioning": state.get("fii_positioning"),
            "gamma_analysis": state.get("gamma_analysis"),
            "liquidity_sweep": state.get("liquidity_sweep"),
            "llm_validation": live_state.get("llm_validation"),
        }

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
    def execute_trade(
        signal_id: int,
        execution_price: float,
        quantity: int,
        instrument_key: Optional[str] = None,
        instrument_symbol: Optional[str] = None,
        exchange: str = "NFO",
    ) -> dict[str, Any]:
        resolved_instrument_key = instrument_key
        if resolved_instrument_key is None and instrument_symbol is not None:
            resolved_instrument_key = kite_client.resolve_instrument_key(instrument_symbol, exchange)
        if resolved_instrument_key is None:
            raise ValueError("execute_trade requires instrument_key or a resolvable instrument_symbol")
        journal.record_execution(
            signal_id,
            execution_price,
            quantity,
            instrument_key=resolved_instrument_key,
        )
        return {"status": "ok", "signal_id": signal_id}

    @app.post("/api/trades/{signal_id}/missed")
    def mark_missed(signal_id: int) -> dict[str, Any]:
        journal.mark_trade_missed(signal_id)
        analysis = analyzer.analyze_trade(signal_id)
        return {"status": "ok", "signal_id": signal_id, "analysis": analysis.__dict__}

    return app
