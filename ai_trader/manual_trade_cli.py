from __future__ import annotations

import argparse

from ai_trader.config.settings import settings
from ai_trader.data.trade_journal import TradeJournal


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual trade tracking CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_trade = subparsers.add_parser("record_trade")
    record_trade.add_argument("--signal_id", type=int, required=True)
    record_trade.add_argument("--price", type=float, required=True)
    record_trade.add_argument("--lots", type=int, required=True)

    mark_missed = subparsers.add_parser("mark_missed")
    mark_missed.add_argument("--signal_id", type=int, required=True)

    args = parser.parse_args()
    journal = TradeJournal(settings.trade_journal_path)
    if args.command == "record_trade":
        journal.record_execution(args.signal_id, args.price, args.lots)
        print(f"Recorded executed trade for signal_id={args.signal_id}")
    elif args.command == "mark_missed":
        journal.mark_trade_missed(args.signal_id)
        print(f"Marked signal_id={args.signal_id} as missed")


if __name__ == "__main__":
    main()
