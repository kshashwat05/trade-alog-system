from __future__ import annotations

import requests

from ai_trader.data.http_client import build_retry_session


NSE_HOME_URL = "https://www.nseindia.com/"
NSE_BASE_HEADERS = {
    "accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "referer": NSE_HOME_URL,
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


def build_nse_session() -> requests.Session:
    session = build_retry_session()
    session.headers.update(NSE_BASE_HEADERS)
    return session


def prime_nse_session(session: requests.Session) -> None:
    session.get(NSE_HOME_URL, timeout=5)
