from __future__ import annotations

import argparse

from ai_trader.auth.token_manager import TokenManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Zerodha Kite Connect login flow.")
    parser.add_argument(
        "--print-url-only",
        action="store_true",
        help="Only print the login URL; do not start the callback server or exchange the token.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not auto-open the Zerodha login URL in the default browser.",
    )
    args = parser.parse_args()

    manager = TokenManager()
    if args.print_url_only:
        print(manager.get_login_url())
        return

    manager.authenticate_interactively(open_browser=not args.no_open_browser)
    print("Kite login completed successfully. Access token stored in local environment file.")


if __name__ == "__main__":
    main()
