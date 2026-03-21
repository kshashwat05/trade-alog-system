from __future__ import annotations

import json
import time
from datetime import datetime, time as dtime
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import schedule
from loguru import logger

from ai_trader.agents.position_monitor_agent import PositionMonitorAgent
from ai_trader.agents.data_quality_guard import DataQualityGuard
from ai_trader.agents.execution_intelligence_agent import ExecutionIntelligenceAgent
from ai_trader.agents.exit_intelligence_agent import ExitIntelligenceAgent, ExitIntelligenceSuggestion
from ai_trader.agents.kill_switch_agent import KillSwitchAgent
from ai_trader.agents.position_tracker import PositionTracker
from ai_trader.alerts.whatsapp_alert import WhatsAppAlerter
from ai_trader.auth.token_manager import get_authenticated_kite_client
from ai_trader.config.settings import settings
from ai_trader.data.kite_client import KiteClient
from ai_trader.data.trade_journal import TradeJournal
from ai_trader.orchestrator.decision_engine import DecisionEngine
from ai_trader.simulation.missed_trade_analyzer import MissedTradeAnalyzer
from kiteconnect import KiteConnect


engine = DecisionEngine()
alerter = WhatsAppAlerter()
journal = TradeJournal(settings.trade_journal_path)
_cycle_lock = Lock()
_live_state_path = Path(settings.live_state_path)
_live_state_path.parent.mkdir(parents=True, exist_ok=True)
_runtime_health_path = Path(settings.runtime_health_path)
_runtime_health_path.parent.mkdir(parents=True, exist_ok=True)
_price_client = KiteClient()
data_quality_guard = DataQualityGuard()
execution_intelligence = ExecutionIntelligenceAgent()
exit_intelligence = ExitIntelligenceAgent()
kill_switch = KillSwitchAgent()
position_tracker = PositionTracker()
_runtime_health: dict[str, object] = {
    "service_started_at": datetime.utcnow().isoformat(),
    "last_cycle_started_at": None,
    "last_cycle_completed_at": None,
    "last_monitor_completed_at": None,
    "last_missed_trade_analysis_at": None,
    "last_error": None,
    "cycle_failures": 0,
    "monitor_failures": 0,
    "missed_trade_failures": 0,
    "trade_alert_failures": 0,
    "exit_alert_failures": 0,
    "execution_intelligence_blocks": 0,
    "data_quality_blocks": 0,
    "kill_switch_blocks": 0,
    "position_mismatch_count": 0,
    "startup_checks": {},
}


def _fetch_trade_price_from_signal(trade) -> float | None:
    try:
        instrument_key = trade.metadata.get("instrument_key")
        if instrument_key is None:
            logger.info(f"No tradable instrument key for trade_id={trade.id}; skipping live price fetch.")
            return None
        return _price_client.fetch_ltp_by_instrument_key(str(instrument_key))
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to fetch current trade price for monitoring: {exc}")
        return None


def _exit_intelligence_callback(trade, current_price: float) -> ExitIntelligenceSuggestion:
    if not settings.exit_intelligence_enabled:
        return ExitIntelligenceSuggestion(action="NONE", reason="Exit intelligence disabled.", confidence=0.0)
    return exit_intelligence.observe_trade(trade, current_price)


position_monitor = PositionMonitorAgent(
    journal=journal,
    price_fetcher=_fetch_trade_price_from_signal,
    exit_intelligence_cb=_exit_intelligence_callback if settings.exit_intelligence_enabled else None,
    exit_cleanup_cb=exit_intelligence.clear_trade if settings.exit_intelligence_enabled else None,
)
missed_trade_analyzer = MissedTradeAnalyzer(journal=journal)


def _hydrate_risk_state_from_journal() -> None:
    latest_signal_at: datetime | None = None
    for trade in journal.get_all_trades():
        timestamp = datetime.fromisoformat(trade.timestamp)
        engine.risk_agent.record_trade_open(timestamp.date())
        if trade.pnl is not None:
            engine.risk_agent.record_trade_result(timestamp.date(), float(trade.pnl))
        if latest_signal_at is None or timestamp > latest_signal_at:
            latest_signal_at = timestamp
    if latest_signal_at is not None:
        engine.risk_agent.record_signal_emitted(latest_signal_at)


_hydrate_risk_state_from_journal()


def configure_authenticated_runtime(kite: KiteConnect) -> KiteClient:
    authenticated_client = KiteClient(kite=kite)
    engine.chart_agent.client = authenticated_client
    engine.regime_agent.client = authenticated_client
    engine.market_data_provider.kite_client = authenticated_client
    engine.market_data_provider._cached_context = None
    engine.market_data_provider._cached_at = None
    missed_trade_analyzer.client = authenticated_client
    global _price_client
    _price_client = authenticated_client
    logger.info("Injected authenticated Kite client into runtime components.")
    return authenticated_client


def _persist_runtime_health() -> None:
    stats = journal.get_runtime_stats()
    payload = {
        **_runtime_health,
        "journal": stats,
        "market_timezone": settings.market_timezone,
    }
    tmp_path = _runtime_health_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, default=str, indent=2))
    tmp_path.replace(_runtime_health_path)


def run_startup_preflight() -> dict[str, object]:
    checks: dict[str, object] = {
        "kite_configured": bool(settings.kite_api_key and settings.kite_api_secret and settings.kite_redirect_url),
        "kite_access_token_present": bool(settings.kite_access_token),
        "twilio_configured": bool(
            settings.twilio_account_sid
            and settings.twilio_auth_token
            and settings.twilio_whatsapp_from
            and settings.whatsapp_to
        ),
        "news_configured": bool(
            settings.news_api_key
            or settings.newsdata_api_key
            or settings.marketaux_api_key
        ),
        "llm_configured": (not settings.llm_validation_enabled)
        or (
            bool(settings.openai_api_key)
            if settings.llm_provider.lower() == "openai"
            else bool(settings.gemini_api_key)
        ),
        "journal_writable": True,
        "live_state_writable": True,
        "runtime_health_writable": True,
        "errors": [],
        "warnings": [],
    }
    try:
        journal.db_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path = journal.db_path.parent / ".write_probe"
        probe_path.write_text("ok")
        probe_path.unlink()
    except Exception as exc:  # noqa: BLE001
        checks["journal_writable"] = False
        checks["errors"].append(f"journal_path_unwritable: {exc}")  # type: ignore[index]
    for key, path in (
        ("live_state_writable", _live_state_path),
        ("runtime_health_writable", _runtime_health_path),
    ):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            probe = path.parent / f".{path.stem}.probe"
            probe.write_text("ok")
            probe.unlink()
        except Exception as exc:  # noqa: BLE001
            checks[key] = False
            checks["errors"].append(f"{key}: {exc}")  # type: ignore[index]
    if not checks["kite_configured"]:
        checks["warnings"].append("Kite API credentials missing; automatic broker authentication is unavailable.")  # type: ignore[index]
    if not checks["twilio_configured"]:
        checks["warnings"].append("Twilio credentials incomplete; alerts will run in mock mode.")  # type: ignore[index]
    if not checks["news_configured"]:
        checks["warnings"].append(
            "No API-backed news source configured; the system will rely on RSS feeds only for macro news."
        )  # type: ignore[index]
    if settings.llm_validation_enabled and not checks["llm_configured"]:
        missing_key_name = "OPENAI_API_KEY" if settings.llm_provider.lower() == "openai" else "GEMINI_API_KEY"
        checks["errors"].append(f"LLM validation enabled but {missing_key_name} is missing.")  # type: ignore[index]
    _runtime_health["startup_checks"] = checks
    _persist_runtime_health()
    if settings.strict_startup_checks and checks["errors"]:  # type: ignore[index]
        raise SystemExit(f"Startup preflight failed: {checks['errors']}")
    return checks


def get_market_now() -> datetime:
    try:
        return datetime.now(ZoneInfo(settings.market_timezone))
    except ZoneInfoNotFoundError:
        logger.warning(
            f"Unknown market timezone {settings.market_timezone!r}; falling back to local time."
        )
        return datetime.now()


def is_market_open(now: datetime | None = None) -> bool:
    now = now or get_market_now()
    start = dtime(settings.market_start_hour, settings.market_start_minute)
    end = dtime(settings.market_end_hour, settings.market_end_minute)
    return start <= now.time() <= end


def persist_live_state(result) -> None:
    market_context = result.state.get("market_context")
    market_quality = getattr(market_context, "quality", None)
    signal_payload = {
        "signal": result.signal.signal,
        "entry": result.signal.entry,
        "stop_loss": result.signal.stop_loss,
        "target": result.signal.target,
        "confidence": result.signal.confidence,
        "rationale": result.signal.rationale,
        "underlying_spot": result.signal.underlying_spot,
        "option_strike": result.signal.option_strike,
        "option_expiry": result.signal.option_expiry,
        "instrument_symbol": result.signal.instrument_symbol,
        "instrument_key": result.signal.instrument_key,
        "price_source": result.signal.price_source,
        "data_complete": result.signal.data_complete,
    }
    snapshot = {
        "timestamp": result.timestamp,
        "decision_score": result.decision_score,
        "signal": signal_payload,
        "risk": result.risk.__dict__,
        "llm_validation": result.llm_validation.__dict__,
        "state": {key: getattr(value, "__dict__", value) for key, value in result.state.items()},
        "score_breakdown": result.state.get("score_breakdown", {}),
        "score_complete": result.state.get("score_complete", False),
        "agent_health": result.state.get("agent_health", {}),
        "market_data_quality": getattr(market_quality, "__dict__", {}),
        "cycle_runtime_health": {
            "last_cycle_started_at": _runtime_health.get("last_cycle_started_at"),
            "last_cycle_completed_at": _runtime_health.get("last_cycle_completed_at"),
            "last_error": _runtime_health.get("last_error"),
        },
    }
    tmp_path = _live_state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(snapshot, default=str, indent=2))
    tmp_path.replace(_live_state_path)


def run_trading_cycle() -> None:
    if not _cycle_lock.acquire(blocking=False):
        logger.warning("Trading cycle skipped because a previous cycle is still running.")
        return

    try:
        _run_trading_cycle()
    finally:
        _cycle_lock.release()


def _run_trading_cycle() -> None:
    if not is_market_open():
        logger.info("Market closed, skipping trading cycle.")
        return

    cycle_started_at = datetime.utcnow()
    _runtime_health["last_cycle_started_at"] = cycle_started_at.isoformat()
    logger.info("Starting trading cycle.")
    try:
        result = engine.run_once(open_trades=len(journal.get_open_trades()))
    except Exception as exc:  # noqa: BLE001
        _runtime_health["cycle_failures"] = int(_runtime_health.get("cycle_failures", 0)) + 1
        _runtime_health["last_error"] = f"trading_cycle_failed: {exc}"
        _persist_runtime_health()
        logger.error(f"Trading cycle failed: {exc}")
        return

    if result.signal.signal != "NONE" and result.risk.allowed:
        if settings.data_quality_guard_enabled:
            dq_decision = data_quality_guard.evaluate(
                result.signal,
                result.state.get("market_context"),
                score_complete=bool(result.state.get("score_complete", False)),
            )
            if not dq_decision.allowed:
                _runtime_health["data_quality_blocks"] = int(_runtime_health.get("data_quality_blocks", 0)) + 1
                result.risk.allowed = False
                result.risk.reason = dq_decision.reason
                result.signal.signal = "NONE"
                result.signal.rationale = dq_decision.reason
                result.signal.confidence = 0.0

    if result.signal.signal != "NONE" and result.risk.allowed:
        if settings.execution_intelligence_enabled:
            exec_decision = execution_intelligence.evaluate(
                result.signal,
                result.state.get("market_context"),
            )
            if not exec_decision.allowed:
                _runtime_health["execution_intelligence_blocks"] = int(
                    _runtime_health.get("execution_intelligence_blocks", 0)
                ) + 1
                result.risk.allowed = False
                result.risk.reason = exec_decision.reason
                result.signal.signal = "NONE"
                result.signal.rationale = exec_decision.reason
                result.signal.confidence = 0.0

    if result.signal.signal != "NONE" and result.risk.allowed:
        if settings.kill_switch_enabled:
            kill_decision = kill_switch.evaluate(journal)
            if kill_decision.blocked:
                _runtime_health["kill_switch_blocks"] = int(_runtime_health.get("kill_switch_blocks", 0)) + 1
                result.risk.allowed = False
                result.risk.reason = kill_decision.reason
                result.signal.signal = "NONE"
                result.signal.rationale = kill_decision.reason
                result.signal.confidence = 0.0

    if result.signal.signal != "NONE" and result.risk.allowed:
        liquidity_state = result.state.get("liquidity")
        vol_state = result.state.get("vol")
        news_state = result.state.get("news")
        macro_state = result.state.get("macro_calendar")
        global_state = result.state.get("global_market")
        authorization = engine.risk_agent.authorize_signal(
            result.signal,
            open_trades=len(journal.get_open_trades()),
            liquidity=getattr(liquidity_state, "liquidity", "medium"),
            volatility=getattr(vol_state, "volatility", "medium"),
            news_risk=getattr(news_state, "risk_level", "medium"),
            macro_event_risk=getattr(macro_state, "event_risk", "low"),
            global_risk_sentiment=getattr(global_state, "risk_sentiment", "neutral")
            if getattr(global_state, "confidence", 0.0) >= 0.8
            else "neutral",
        )
        if not authorization.allowed:
            result.risk.allowed = False
            result.risk.reason = authorization.reason
            result.signal.signal = "NONE"
            result.signal.rationale = authorization.reason or "Rejected by risk manager."
            result.signal.confidence = 0.0

    persist_live_state(result)
    _runtime_health["last_cycle_completed_at"] = datetime.utcnow().isoformat()
    _runtime_health["last_error"] = None

    trade_id: int | None = None
    if result.signal.signal != "NONE" and result.risk.allowed:
        market_context = result.state.get("market_context")
        market_quality = getattr(market_context, "quality", None)
        try:
            trade_id = journal.record_signal(
                timestamp=datetime.fromisoformat(result.timestamp),
                signal_type=result.signal.signal,
                entry_price=result.signal.entry,
                stop_loss=result.signal.stop_loss,
                target=result.signal.target,
                confidence=result.signal.confidence,
                decision_score=result.decision_score,
                rationale=result.signal.rationale,
                llm_reasoning=result.llm_validation.reasoning,
                institutional_bias=getattr(result.state["fii_positioning"], "fii_bias", None),
                gamma_regime=getattr(result.state["gamma_analysis"], "gamma_regime", None),
                liquidity_event=getattr(result.state["liquidity_sweep"], "event_type", None),
                metadata={
                    "state": {key: getattr(value, "__dict__", value) for key, value in result.state.items()},
                    "underlying_spot": result.signal.underlying_spot,
                    "option_strike": result.signal.option_strike,
                    "option_expiry": result.signal.option_expiry,
                    "instrument_symbol": result.signal.instrument_symbol,
                    "instrument_key": result.signal.instrument_key,
                    "price_source": result.signal.price_source,
                    "data_complete": result.signal.data_complete,
                    "score_breakdown": result.state.get("score_breakdown", {}),
                    "score_complete": result.state.get("score_complete", False),
                    "agent_health": result.state.get("agent_health", {}),
                    "market_data_quality": getattr(market_quality, "__dict__", {}),
                    "source_timestamp": result.timestamp,
                    "llm_validation_source": result.llm_validation.source,
                    "llm_validation_fallback": result.llm_validation.fallback_used,
                },
                enforce_limits=True,
                cooldown_minutes=settings.signal_cooldown_minutes,
                max_trades_per_day=settings.max_trades_per_day,
            )
        except ValueError as exc:
            _runtime_health["last_error"] = f"signal_reservation_blocked: {exc}"
            logger.warning(f"Signal journal reservation blocked: {exc}")
            result.risk.allowed = False
            result.risk.reason = str(exc)

    if result.signal.signal != "NONE" and result.risk.allowed:
        alert_sent = False
        try:
            alerter.send_trade_signal(
                result.signal,
                institutional_bias=getattr(result.state["fii_positioning"], "fii_bias", None),
                gamma_regime=getattr(result.state["gamma_analysis"], "gamma_regime", None),
            )
            alert_sent = True
        except Exception as exc:  # noqa: BLE001
            _runtime_health["trade_alert_failures"] = int(_runtime_health.get("trade_alert_failures", 0)) + 1
            _runtime_health["last_error"] = f"trade_alert_failed: {exc}"
            logger.error(f"Trade alert delivery failed: {exc}")
        if settings.position_tracker_enabled and alert_sent:
            position_tracker.record_alert_delivery(journal, trade_id)
            sync_report = position_tracker.sync_with_journal(journal)
            _runtime_health["position_mismatch_count"] = sync_report.mismatches
            if sync_report.mismatches > 0:
                logger.warning(f"PositionTracker mismatch details: {sync_report.details}")
        if alert_sent:
            logger.info(f"Actionable signal sent and recorded with trade_id={trade_id}.")
        else:
            logger.warning(f"Trade recorded without alert delivery trade_id={trade_id}.")
    else:
        logger.info(
            f"No actionable signal. decision_score={result.decision_score}, "
            f"risk_allowed={result.risk.allowed}, rationale={result.signal.rationale}, "
            f"risk_reason={result.risk.reason}",
        )
    _persist_runtime_health()


def run_position_monitor_cycle() -> None:
    if not is_market_open():
        return

    try:
        exits = position_monitor.monitor_once()
    except Exception as exc:  # noqa: BLE001
        _runtime_health["monitor_failures"] = int(_runtime_health.get("monitor_failures", 0)) + 1
        _runtime_health["last_error"] = f"position_monitor_failed: {exc}"
        _persist_runtime_health()
        logger.error(f"Position monitor failed: {exc}")
        return

    for exit_result in exits:
        try:
            alerter.send_exit_alert(exit_result.message)
            if exit_result.advisory_only:
                if exit_result.exit_suggestion is not None:
                    exit_intelligence.mark_advisory_sent(exit_result.signal_id, exit_result.exit_suggestion)
                journal.merge_metadata(
                    exit_result.signal_id,
                    {
                        "exit_intelligence": {
                            "action": exit_result.status,
                            "reason": exit_result.exit_suggestion.reason if exit_result.exit_suggestion else None,
                            "confidence": (
                                exit_result.exit_suggestion.confidence if exit_result.exit_suggestion else None
                            ),
                            "sent_at": datetime.utcnow().isoformat(),
                        }
                    },
                )
        except Exception as exc:  # noqa: BLE001
            _runtime_health["exit_alert_failures"] = int(_runtime_health.get("exit_alert_failures", 0)) + 1
            _runtime_health["last_error"] = f"exit_alert_failed: {exc}"
            logger.error(f"Exit alert delivery failed for trade_id={exit_result.signal_id}: {exc}")
    if settings.position_tracker_enabled:
        sync_report = position_tracker.sync_with_journal(journal)
        _runtime_health["position_mismatch_count"] = sync_report.mismatches
    _runtime_health["last_monitor_completed_at"] = datetime.utcnow().isoformat()
    _persist_runtime_health()


def run_missed_trade_analysis_cycle() -> None:
    missed_candidates = journal.get_pending_simulation_trades()
    for trade in missed_candidates:
        if trade.trade_executed:
            continue
        try:
            missed_trade_analyzer.analyze_trade(trade.id)
        except Exception as exc:  # noqa: BLE001
            _runtime_health["missed_trade_failures"] = int(_runtime_health.get("missed_trade_failures", 0)) + 1
            _runtime_health["last_error"] = f"missed_trade_analysis_failed: {exc}"
            logger.error(f"Missed trade analysis failed for trade_id={trade.id}: {exc}")
    _runtime_health["last_missed_trade_analysis_at"] = datetime.utcnow().isoformat()
    _persist_runtime_health()


def main() -> None:
    logger.add("logs/trading_engine.log", rotation="1 day", retention="7 days", serialize=True)
    preflight = run_startup_preflight()
    logger.info("Starting AI Trader main loop.")
    logger.info(f"Startup preflight: {preflight}")
    if preflight["kite_configured"]:
        try:
            configure_authenticated_runtime(get_authenticated_kite_client(auto_login=True))
            _runtime_health["startup_checks"]["kite_session_valid"] = True  # type: ignore[index]
            _runtime_health["last_error"] = None
            _persist_runtime_health()
        except Exception as exc:  # noqa: BLE001
            _runtime_health["startup_checks"]["kite_session_valid"] = False  # type: ignore[index]
            _runtime_health["startup_checks"]["errors"].append(f"kite_authentication_failed: {exc}")  # type: ignore[index]
            _runtime_health["last_error"] = f"kite_authentication_failed: {exc}"
            _persist_runtime_health()
            raise SystemExit(f"Kite authentication failed: {exc}")

    # Run immediately once, then every 2 minutes
    run_trading_cycle()
    run_position_monitor_cycle()
    schedule.every(2).minutes.do(run_trading_cycle)
    schedule.every(settings.position_monitor_interval_seconds).seconds.do(run_position_monitor_cycle)
    schedule.every(5).minutes.do(run_missed_trade_analysis_cycle)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
