from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger

from ai_trader.auth.token_manager import TokenManager


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click daily startup for ai_trader.")
    parser.add_argument(
        "--with-dashboard",
        action="store_true",
        help="Start the local dashboard in a background subprocess before launching the trading engine.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not auto-open the Zerodha login page if a fresh session is required.",
    )
    args = parser.parse_args()

    logger.info("Starting ai_trader daily startup flow.")
    manager = TokenManager()
    manager.get_authenticated_kite_client(auto_login=True, open_browser=not args.no_open_browser)

    if args.with_dashboard:
        dashboard_script = Path(__file__).resolve().parents[1] / "run_dashboard.py"
        subprocess.Popen([sys.executable, str(dashboard_script)])
        logger.info("Started dashboard subprocess.")

    from ai_trader.main import main as trading_main

    trading_main()


if __name__ == "__main__":
    main()
