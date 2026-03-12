from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import requests
from loguru import logger

from ai_trader.data.nse_session import build_nse_session, prime_nse_session

VolBucket = Literal["low", "medium", "high"]


@dataclass
class VolatilityAnalysis:
    volatility: VolBucket
    expected_range: Tuple[float, float]


class VolatilityAgent:
    """Estimates intraday range based on India VIX."""

    INDIA_VIX_URL = "https://www.nseindia.com/api/quote-indices?index=INDIA%20VIX"

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or build_nse_session()
        self._session_primed = False

    def _prime_session(self) -> None:
        if self._session_primed:
            return
        prime_nse_session(self.session)
        self._session_primed = True

    def _fetch_vix(self) -> float | None:
        try:
            self._prime_session()
            resp = self.session.get(self.INDIA_VIX_URL, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            vix = data.get("data", [{}])[0].get("last", None)
            return float(vix) if vix is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to fetch India VIX: {exc}")
            return None

    def analyze(self, spot: float | None = None) -> VolatilityAnalysis:
        vix = self._fetch_vix()
        if vix is None:
            logger.warning("No VIX data; assuming medium volatility and +/- 1% range.")
            vix = 14.0

        if vix < 12:
            bucket: VolBucket = "low"
        elif vix > 20:
            bucket = "high"
        else:
            bucket = "medium"

        # Convert annualized VIX to rough intraday range; very approximate.
        if spot is None:
            spot = 24000.0

        daily_vol = (vix / 100.0) / (252 ** 0.5)
        expected_move = spot * daily_vol
        lower = spot - expected_move
        upper = spot + expected_move

        analysis = VolatilityAnalysis(volatility=bucket, expected_range=(lower, upper))
        logger.info(f"VolatilityAgent analysis: {analysis}")
        return analysis
