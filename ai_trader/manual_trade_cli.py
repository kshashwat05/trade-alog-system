from __future__ import annotations

import argparse

from ai_trader.config.settings import settings
from ai_trader.agents.position_tracker import PositionTracker
from ai_trader.data.kite_client import KiteClient
from ai_trader.data.trade_journal import TradeJournal


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual trade tracking CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_trade = subparsers.add_parser("record_trade")
    record_trade.add_argument("--signal_id", type=int, required=True)
    record_trade.add_argument("--price", type=float, required=True)
    record_trade.add_argument("--lots", type=int, required=True)
    record_trade.add_argument("--instrument-key", type=str, required=False)
    record_trade.add_argument("--instrument-symbol", type=str, required=False)
    record_trade.add_argument("--exchange", type=str, default="NFO")

    mark_missed = subparsers.add_parser("mark_missed")
    mark_missed.add_argument("--signal_id", type=int, required=True)

    args = parser.parse_args()
    journal = TradeJournal(settings.trade_journal_path)
    tracker = PositionTracker()
    kite_client = KiteClient()
    if args.command == "record_trade":
        instrument_key = args.instrument_key
        if instrument_key is None and args.instrument_symbol:
            instrument_key = kite_client.resolve_instrument_key(args.instrument_symbol, args.exchange)
        if instrument_key is None:
            raise ValueError("record_trade requires --instrument-key or a resolvable --instrument-symbol")
        journal.record_execution(
            args.signal_id,
            args.price,
            args.lots,
            instrument_key=instrument_key,
        )
        if settings.position_tracker_enabled:
            report = tracker.sync_with_journal(journal)
            if report.mismatches:
                print(f"Position mismatches detected: {report.details}")
        print(f"Recorded executed trade for signal_id={args.signal_id}")
    elif args.command == "mark_missed":
        journal.mark_trade_missed(args.signal_id)
        print(f"Marked signal_id={args.signal_id} as missed")


if __name__ == "__main__":
    main()
