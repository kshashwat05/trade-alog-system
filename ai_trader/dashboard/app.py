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
            :root {
              --bg: #f4f1ea;
              --panel: #fffdf8;
              --ink: #1d2a33;
              --muted: #667784;
              --line: #e3ddd1;
              --accent: #0d6b57;
              --accent-soft: #dcefe8;
              --warn: #9c5a00;
              --warn-soft: #fff1d8;
              --danger: #9e2f1d;
              --danger-soft: #ffe4de;
            }
            * { box-sizing: border-box; }
            body {
              margin: 0;
              padding: 32px;
              font-family: Georgia, "Times New Roman", serif;
              background:
                radial-gradient(circle at top right, rgba(13,107,87,0.10), transparent 24%),
                linear-gradient(180deg, #f7f4ee 0%, var(--bg) 100%);
              color: var(--ink);
            }
            .shell {
              max-width: 1440px;
              margin: 0 auto;
            }
            .hero {
              display: flex;
              justify-content: space-between;
              align-items: flex-start;
              gap: 20px;
              margin-bottom: 22px;
            }
            .hero h1 {
              margin: 0 0 8px 0;
              font-size: 40px;
              line-height: 1;
            }
            .hero p {
              margin: 0;
              max-width: 720px;
              color: var(--muted);
              font-size: 16px;
            }
            .stamp {
              font-size: 13px;
              color: var(--muted);
              background: rgba(255,255,255,0.65);
              border: 1px solid var(--line);
              border-radius: 999px;
              padding: 10px 14px;
              white-space: nowrap;
            }
            .status-grid,
            .grid {
              display: grid;
              gap: 16px;
            }
            .status-grid {
              grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              margin-bottom: 16px;
            }
            .grid {
              grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            }
            .card {
              background: var(--panel);
              border: 1px solid rgba(125, 109, 81, 0.14);
              border-radius: 18px;
              padding: 18px;
              box-shadow: 0 14px 30px rgba(72, 52, 17, 0.06);
            }
            .metric-card {
              padding: 18px;
              border-radius: 16px;
              background: rgba(255,255,255,0.75);
              border: 1px solid var(--line);
            }
            .eyebrow {
              margin: 0 0 6px 0;
              color: var(--muted);
              font-size: 12px;
              letter-spacing: 0.08em;
              text-transform: uppercase;
            }
            .big-number {
              font-size: 30px;
              font-weight: bold;
              line-height: 1;
              margin: 0;
            }
            .small-note {
              margin: 8px 0 0 0;
              font-size: 13px;
              color: var(--muted);
            }
            .card h2 {
              margin: 0 0 12px 0;
              font-size: 22px;
            }
            .section-copy {
              margin: 0 0 14px 0;
              color: var(--muted);
              font-size: 14px;
            }
            .pill {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              border-radius: 999px;
              padding: 6px 12px;
              font-size: 13px;
              margin: 0 8px 8px 0;
              border: 1px solid transparent;
            }
            .ok { background: var(--accent-soft); color: var(--accent); border-color: rgba(13,107,87,0.20); }
            .warn { background: var(--warn-soft); color: var(--warn); border-color: rgba(156,90,0,0.18); }
            .bad { background: var(--danger-soft); color: var(--danger); border-color: rgba(158,47,29,0.18); }
            .neutral { background: #f3eee5; color: #6a5d49; border-color: rgba(106,93,73,0.18); }
            .split {
              display: grid;
              grid-template-columns: 1.2fr 0.8fr;
              gap: 16px;
            }
            .summary-list {
              margin: 0;
              padding-left: 18px;
            }
            .summary-list li {
              margin-bottom: 10px;
              line-height: 1.4;
            }
            .kv {
              display: grid;
              grid-template-columns: 1fr auto;
              gap: 8px 16px;
              font-size: 14px;
            }
            .kv div:nth-child(odd) {
              color: var(--muted);
            }
            .reason-box {
              background: #f8f4ec;
              border: 1px solid var(--line);
              border-radius: 14px;
              padding: 14px;
              line-height: 1.5;
              font-size: 14px;
            }
            table {
              width: 100%;
              border-collapse: collapse;
              font-size: 14px;
            }
            th, td {
              padding: 10px 8px;
              border-bottom: 1px solid var(--line);
              text-align: left;
              vertical-align: top;
            }
            th {
              color: var(--muted);
              font-weight: normal;
              font-size: 12px;
              text-transform: uppercase;
              letter-spacing: 0.06em;
            }
            .empty {
              color: var(--muted);
              font-style: italic;
            }
            .mono {
              font-family: "SFMono-Regular", Menlo, monospace;
              font-size: 12px;
            }
            @media (max-width: 900px) {
              body { padding: 18px; }
              .hero { flex-direction: column; }
              .split { grid-template-columns: 1fr; }
            }
          </style>
        </head>
        <body>
          <div class="shell">
            <div class="hero">
              <div>
                <h1>AI Trader Dashboard</h1>
                <p>
                  A readable view of system readiness, live market intelligence, recent signals,
                  and replay outcomes. Refreshes automatically every 30 seconds.
                </p>
              </div>
              <div class="stamp" id="updatedAt">Loading latest snapshot...</div>
            </div>
            <div id="app">Loading...</div>
          </div>
          <script>
            function escapeHtml(value) {
              return String(value ?? '')
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#39;');
            }

            function fmtNumber(value, digits = 2) {
              const num = Number(value);
              return Number.isFinite(num) ? num.toFixed(digits) : 'n/a';
            }

            function fmtPct(value) {
              const num = Number(value);
              return Number.isFinite(num) ? `${(num * 100).toFixed(0)}%` : 'n/a';
            }

            function fmtDate(value) {
              if (!value) return 'n/a';
              const date = new Date(value);
              return Number.isNaN(date.getTime()) ? escapeHtml(value) : date.toLocaleString();
            }

            function statusPill(label, tone = 'neutral') {
              return `<span class="pill ${tone}">${escapeHtml(label)}</span>`;
            }

            function narrative(summary, readiness, liveState) {
              const score = liveState.decision_score ?? 'n/a';
              const signal = liveState.signal?.signal ?? 'NONE';
              const llm = liveState.llm_validation || {};
              const warnings = readiness.warnings || [];
              const notes = [];

              if (readiness.ready) {
                notes.push(`System checks are passing. The engine is ready to operate with a current decision score of ${score}.`);
              } else {
                notes.push('Startup checks are failing, so the engine should not be treated as market-ready.');
              }

              if (signal === 'NONE') {
                notes.push('There is no active actionable setup right now. The engine is filtering for high-conviction trades only.');
              } else {
                notes.push(`The latest actionable setup is ${signal} with confidence ${fmtPct(liveState.signal?.confidence)}.`);
              }

              if (llm.reasoning) {
                notes.push(`The LLM validation layer says: ${llm.reasoning}`);
              }

              if (warnings.length > 0) {
                notes.push(`There ${warnings.length === 1 ? 'is' : 'are'} ${warnings.length} readiness warning${warnings.length === 1 ? '' : 's'} that may reduce signal availability.`);
              }

              if ((summary.executed_trades || 0) === 0) {
                notes.push('No executed trades are recorded yet, so performance numbers should be treated as setup metrics rather than live performance evidence.');
              }

              return `<ul class="summary-list">${notes.map(note => `<li>${escapeHtml(note)}</li>`).join('')}</ul>`;
            }

            function renderMetrics(summary, analytics, readiness) {
              const overall = analytics.overall || {};
              return `
                <div class="status-grid">
                  <div class="metric-card">
                    <p class="eyebrow">System Status</p>
                    <p class="big-number">${readiness.ready ? 'Ready' : 'Blocked'}</p>
                    <p class="small-note">${readiness.errors?.length || 0} errors, ${readiness.warnings?.length || 0} warnings</p>
                  </div>
                  <div class="metric-card">
                    <p class="eyebrow">Signals Logged</p>
                    <p class="big-number">${summary.total_trades ?? 0}</p>
                    <p class="small-note">${summary.executed_trades ?? 0} executed, ${summary.missed_trades ?? 0} missed</p>
                  </div>
                  <div class="metric-card">
                    <p class="eyebrow">Win Rate</p>
                    <p class="big-number">${fmtPct(overall.win_rate)}</p>
                    <p class="small-note">Across ${overall.trade_count ?? 0} scored trades</p>
                  </div>
                  <div class="metric-card">
                    <p class="eyebrow">Total PnL</p>
                    <p class="big-number">${fmtNumber(overall.total_pnl)}</p>
                    <p class="small-note">Expectancy ${fmtNumber(overall.expectancy)} per trade</p>
                  </div>
                </div>
              `;
            }

            function renderReadiness(health, readiness) {
              const warnings = readiness.warnings || [];
              const errors = readiness.errors || [];
              const startup = health.startup_checks || {};
              const checks = Object.entries(startup)
                .filter(([key, value]) => typeof value === 'boolean')
                .map(([key, value]) => `
                  <div>${escapeHtml(key.replaceAll('_', ' '))}</div>
                  <div>${value ? statusPill('Configured', 'ok') : statusPill('Missing', 'warn')}</div>
                `)
                .join('');

              return `
                <div class="card">
                  <h2>Operational Readiness</h2>
                  <p class="section-copy">This panel translates startup checks into plain language for the trading day.</p>
                  <div>
                    ${readiness.ready ? statusPill('Trading engine ready', 'ok') : statusPill('Do not start trading', 'bad')}
                    ${(warnings.length === 0 && errors.length === 0) ? statusPill('No active warnings', 'ok') : ''}
                    ${warnings.length > 0 ? statusPill(`${warnings.length} warning${warnings.length === 1 ? '' : 's'}`, 'warn') : ''}
                    ${errors.length > 0 ? statusPill(`${errors.length} error${errors.length === 1 ? '' : 's'}`, 'bad') : ''}
                  </div>
                  <div class="split" style="margin-top: 14px;">
                    <div>
                      <div class="reason-box">
                        ${errors.length > 0 ? `<strong>Errors:</strong> ${errors.map(escapeHtml).join('<br>')}` : '<strong>No blocking errors detected.</strong>'}
                        ${warnings.length > 0 ? `<br><br><strong>Warnings:</strong> ${warnings.map(escapeHtml).join('<br>')}` : ''}
                      </div>
                    </div>
                    <div class="kv">${checks || '<div class="empty">No startup checks available.</div>'}</div>
                  </div>
                </div>
              `;
            }

            function renderLiveDecision(liveState) {
              const signal = liveState.signal || {};
              const llm = liveState.llm_validation || {};
              const scoreBreakdown = liveState.score_breakdown || {};
              const scoreRows = Object.entries(scoreBreakdown).map(([key, value]) => `
                <div>${escapeHtml(key.replaceAll('_', ' '))}</div>
                <div>${escapeHtml(value)}</div>
              `).join('');

              return `
                <div class="card">
                  <h2>Live Decision Summary</h2>
                  <p class="section-copy">The latest engine decision, translated into trader-facing language.</p>
                  <div>
                    ${statusPill(`Signal: ${signal.signal || 'NONE'}`, signal.signal && signal.signal !== 'NONE' ? 'ok' : 'neutral')}
                    ${statusPill(`Score: ${liveState.decision_score ?? 'n/a'}`, 'neutral')}
                    ${statusPill(`Risk: ${(liveState.risk?.allowed ?? false) ? 'Approved' : 'Blocked'}`, (liveState.risk?.allowed ?? false) ? 'ok' : 'warn')}
                  </div>
                  <div class="split" style="margin-top: 14px;">
                    <div class="reason-box">
                      <strong>Trading plan</strong><br>
                      Entry ${fmtNumber(signal.entry)} | Stop ${fmtNumber(signal.stop_loss)} | Target ${fmtNumber(signal.target)}<br><br>
                      <strong>Why this setup exists</strong><br>
                      ${escapeHtml(signal.rationale || 'No rationale recorded.')}
                    </div>
                    <div class="kv">
                      <div>Confidence</div><div>${fmtPct(signal.confidence)}</div>
                      <div>LLM validation</div><div>${escapeHtml(llm.validation || 'n/a')}</div>
                      <div>LLM source</div><div>${escapeHtml(llm.source || 'n/a')}</div>
                      <div>Fallback used</div><div>${llm.fallback_used ? 'Yes' : 'No'}</div>
                      ${scoreRows}
                    </div>
                  </div>
                </div>
              `;
            }

            function renderInstitutional(liveState) {
              const state = liveState.state || {};
              const fii = state.fii_positioning || {};
              const gamma = state.gamma_analysis || {};
              const sweep = state.liquidity_sweep || {};
              const regime = state.regime || {};
              const vol = state.vol || {};

              return `
                <div class="card">
                  <h2>Market Intelligence</h2>
                  <p class="section-copy">A concise reading of institutional posture and current market texture.</p>
                  <div>
                    ${statusPill(`FII bias: ${fii.fii_bias || 'n/a'}`, fii.fii_bias === 'bullish' || fii.fii_bias === 'bearish' ? 'ok' : 'neutral')}
                    ${statusPill(`Gamma: ${gamma.gamma_regime || 'n/a'}`, gamma.gamma_regime === 'negative_gamma' ? 'warn' : 'neutral')}
                    ${statusPill(`Sweep: ${sweep.event_type || 'none'}`, sweep.liquidity_event ? 'warn' : 'ok')}
                    ${statusPill(`Regime: ${regime.regime || 'n/a'}`, 'neutral')}
                    ${statusPill(`Volatility: ${vol.volatility_state || vol.volatility || 'n/a'}`, 'neutral')}
                  </div>
                  <div class="split" style="margin-top: 14px;">
                    <div class="reason-box">
                      <strong>Institutional map</strong><br>
                      Support near ${escapeHtml(fii.institutional_support ?? 'n/a')} and resistance near ${escapeHtml(fii.institutional_resistance ?? 'n/a')}.<br><br>
                      Dealers appear to be in <strong>${escapeHtml(gamma.gamma_regime || 'unknown')}</strong> with a flip level near <strong>${escapeHtml(gamma.gamma_flip_level ?? 'n/a')}</strong>.<br><br>
                      Liquidity sweep status: <strong>${escapeHtml(sweep.event_type || 'none')}</strong>.
                    </div>
                    <div class="kv">
                      <div>FII confidence</div><div>${fmtPct(fii.confidence)}</div>
                      <div>Expected move</div><div>${escapeHtml(gamma.expected_move || 'n/a')}</div>
                      <div>Sweep confidence</div><div>${fmtPct(sweep.confidence)}</div>
                      <div>Volatility source</div><div>${escapeHtml(vol.source || 'n/a')}</div>
                    </div>
                  </div>
                </div>
              `;
            }

            function renderPerformance(analytics) {
              const overall = analytics.overall || {};
              const executed = analytics.executed || {};
              const missed = analytics.missed || {};
              const signalAccuracy = analytics.signal_accuracy || {};
              const institutional = analytics.institutional_alignment_success_rate || {};
              const llmImpact = analytics.llm_approval_impact || {};

              return `
                <div class="card">
                  <h2>Performance Snapshot</h2>
                  <p class="section-copy">Performance translated into plain-language quality indicators.</p>
                  <div class="split">
                    <div class="reason-box">
                      <strong>What the numbers say</strong><br>
                      The system has a ${fmtPct(overall.win_rate)} win rate, profit factor of ${fmtNumber(overall.profit_factor)},
                      and max drawdown of ${fmtNumber(overall.max_drawdown)}. Executed trades show expectancy of
                      ${fmtNumber(executed.expectancy)}, while missed trades show expectancy of ${fmtNumber(missed.expectancy)}.
                    </div>
                    <div class="kv">
                      <div>Average profit</div><div>${fmtNumber(overall.average_profit)}</div>
                      <div>Average loss</div><div>${fmtNumber(overall.average_loss)}</div>
                      <div>Correct signals</div><div>${signalAccuracy.correct_signals ?? 0} / ${signalAccuracy.total_scored_signals ?? 0}</div>
                      <div>Institutional wins</div><div>${institutional.aligned_wins ?? 0} / ${institutional.aligned_signals ?? 0}</div>
                      <div>LLM-reviewed trades</div><div>${llmImpact.with_llm_reasoning ?? 0}</div>
                      <div>Positive with LLM</div><div>${llmImpact.llm_positive_pnl ?? 0}</div>
                    </div>
                  </div>
                </div>
              `;
            }

            function renderSignals(signals) {
              if (!signals.length) {
                return `<div class="card"><h2>Recent Signals</h2><p class="empty">No signals have been recorded yet.</p></div>`;
              }

              return `
                <div class="card">
                  <h2>Recent Signals</h2>
                  <p class="section-copy">Latest engine outputs with trade plan and status.</p>
                  <table>
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Signal</th>
                        <th>Plan</th>
                        <th>Confidence</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      ${signals.slice(0, 8).map(signal => `
                        <tr>
                          <td>${fmtDate(signal.timestamp)}</td>
                          <td>${escapeHtml(signal.signal_type || 'n/a')}</td>
                          <td>Entry ${fmtNumber(signal.entry_price)}<br>SL ${fmtNumber(signal.stop_loss)} / Tgt ${fmtNumber(signal.target)}</td>
                          <td>${fmtPct(signal.confidence)}</td>
                          <td>${escapeHtml(signal.status || 'n/a')}</td>
                        </tr>
                      `).join('')}
                    </tbody>
                  </table>
                </div>
              `;
            }

            function renderReplay(latestReplay) {
              if (!latestReplay || !latestReplay.summary) {
                return `<div class="card"><h2>Replay Review</h2><p class="empty">No replay reports are available yet.</p></div>`;
              }

              const summary = latestReplay.summary || {};
              return `
                <div class="card">
                  <h2>Replay Review</h2>
                  <p class="section-copy">Historical replay summary for the latest stored session.</p>
                  <div class="reason-box">
                    Replay <strong>${escapeHtml(latestReplay.report_name || 'latest')}</strong> processed
                    ${escapeHtml(summary.minute_count ?? 'n/a')} minutes and produced
                    <strong>${escapeHtml(summary.signals_generated ?? latestReplay.signal_count ?? 0)}</strong> signals.
                    Wrong-signal detections: <strong>${escapeHtml(summary.wrong_signals ?? 0)}</strong>.
                  </div>
                </div>
              `;
            }

            async function load() {
              const [summary, signals, analytics, liveState, replays, health, readiness] = await Promise.all([
                fetch('/api/summary').then(r => r.json()),
                fetch('/api/signals').then(r => r.json()),
                fetch('/api/analytics').then(r => r.json()),
                fetch('/api/live-state').then(r => r.json()),
                fetch('/api/replays').then(r => r.json()),
                fetch('/api/health').then(r => r.json()),
                fetch('/api/readiness').then(r => r.json())
              ]);
              const latestReplay = replays.length > 0 ? replays[0] : {};
              document.getElementById('updatedAt').textContent = `Last refreshed: ${new Date().toLocaleString()}`;
              document.getElementById('app').innerHTML = `
                ${renderMetrics(summary, analytics, readiness)}
                <div class="grid">
                  <div class="card">
                    <h2>Trading Day Narrative</h2>
                    <p class="section-copy">A plain-language summary of what the engine is seeing right now.</p>
                    ${narrative(summary, readiness, liveState)}
                  </div>
                  ${renderReadiness(health, readiness)}
                  ${renderLiveDecision(liveState)}
                  ${renderInstitutional(liveState)}
                  ${renderPerformance(analytics)}
                  <div class="card">
                    <h2>LLM Reasoning</h2>
                    <p class="section-copy">Human-readable validation layer output for the latest decision.</p>
                    <div class="reason-box">
                      ${escapeHtml(liveState.llm_validation?.reasoning || 'No LLM reasoning is available yet.')}
                    </div>
                    <p class="small-note mono" style="margin-top: 12px;">
                      Source: ${escapeHtml(liveState.llm_validation?.source || 'n/a')} |
                      Validation: ${escapeHtml(liveState.llm_validation?.validation || 'n/a')} |
                      Fallback: ${liveState.llm_validation?.fallback_used ? 'yes' : 'no'}
                    </p>
                  </div>
                  ${renderSignals(signals)}
                  ${renderReplay(latestReplay)}
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
