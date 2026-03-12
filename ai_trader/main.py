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
from ai_trader.alerts.whatsapp_alert import WhatsAppAlerter
from ai_trader.config.settings import settings
from ai_trader.data.kite_client import KiteClient
from ai_trader.data.trade_journal import TradeJournal
from ai_trader.orchestrator.decision_engine import DecisionEngine
from ai_trader.simulation.missed_trade_analyzer import MissedTradeAnalyzer


engine = DecisionEngine()
alerter = WhatsAppAlerter()
journal = TradeJournal(settings.trade_journal_path)
_cycle_lock = Lock()
_live_state_path = Path(settings.live_state_path)
_live_state_path.parent.mkdir(parents=True, exist_ok=True)
_price_client = KiteClient()


def _fetch_trade_price_from_signal(trade) -> float | None:
    try:
        candles = _price_client.fetch_nifty_intraday().df
        if candles.empty:
            return None
        return float(candles.iloc[-1]["close"])
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to fetch current trade price for monitoring: {exc}")
        return None


position_monitor = PositionMonitorAgent(journal=journal, price_fetcher=_fetch_trade_price_from_signal)
missed_trade_analyzer = MissedTradeAnalyzer(journal=journal)


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
    snapshot = {
        "timestamp": result.timestamp,
        "decision_score": result.decision_score,
        "signal": result.signal.__dict__,
        "risk": result.risk.__dict__,
        "state": {key: getattr(value, "__dict__", value) for key, value in result.state.items()},
    }
    _live_state_path.write_text(json.dumps(snapshot, default=str, indent=2))


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

    logger.info("Starting trading cycle.")
    try:
        result = engine.run_once()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Trading cycle failed: {exc}")
        return

    persist_live_state(result)

    trade_id: int | None = None
    if result.signal.signal != "NONE":
        trade_id = journal.record_signal(
            timestamp=datetime.fromisoformat(result.timestamp),
            signal_type=result.signal.signal,
            entry_price=result.signal.entry,
            stop_loss=result.signal.stop_loss,
            target=result.signal.target,
            confidence=result.signal.confidence,
            decision_score=result.decision_score,
            rationale=result.signal.rationale,
            metadata={"state": {key: getattr(value, "__dict__", value) for key, value in result.state.items()}},
        )

    if result.signal.signal != "NONE" and result.risk.allowed:
        engine.risk_agent.record_signal_emitted()
        engine.risk_agent.record_trade_open_today()
        alerter.send_trade_signal(result.signal)
        logger.info(f"Actionable signal sent and recorded with trade_id={trade_id}.")
    else:
        logger.info(
            f"No actionable signal. decision_score={result.decision_score}, "
            f"risk_allowed={result.risk.allowed}, rationale={result.signal.rationale}, "
            f"risk_reason={result.risk.reason}",
        )


def run_position_monitor_cycle() -> None:
    if not is_market_open():
        return

    try:
        exits = position_monitor.monitor_once()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Position monitor failed: {exc}")
        return

    for exit_result in exits:
        alerter.send_exit_alert(exit_result.message)


def run_missed_trade_analysis_cycle() -> None:
    missed_candidates = journal.get_trades_by_status(["signal_generated", "missed"])
    for trade in missed_candidates:
        if trade.trade_executed:
            continue
        try:
            missed_trade_analyzer.analyze_trade(trade.id)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Missed trade analysis failed for trade_id={trade.id}: {exc}")


def main() -> None:
    logger.add("logs/trading_engine.log", rotation="1 day", retention="7 days", serialize=True)
    logger.info("Starting AI Trader main loop.")

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
