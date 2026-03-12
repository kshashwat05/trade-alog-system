from __future__ import annotations

import time
from datetime import datetime, time as dtime
from threading import Lock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import schedule
from loguru import logger

from ai_trader.alerts.whatsapp_alert import WhatsAppAlerter
from ai_trader.config.settings import settings
from ai_trader.orchestrator.decision_engine import DecisionEngine


engine = DecisionEngine()
alerter = WhatsAppAlerter()
_cycle_lock = Lock()


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

    if result.signal.signal != "NONE" and result.risk.allowed:
        engine.risk_agent.record_trade_open_today()
        alerter.send_trade_signal(result.signal)
    else:
        logger.info(
            f"No actionable signal. decision_score={result.decision_score}, "
            f"risk_allowed={result.risk.allowed}, rationale={result.signal.rationale}, "
            f"risk_reason={result.risk.reason}",
        )


def main() -> None:
    logger.add("logs/ai_trader.log", rotation="1 day", retention="7 days")
    logger.info("Starting AI Trader main loop.")

    # Run immediately once, then every 2 minutes
    run_trading_cycle()
    schedule.every(2).minutes.do(run_trading_cycle)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
