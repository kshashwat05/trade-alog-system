from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests
from loguru import logger

from ai_trader.data.nse_session import build_nse_session, prime_nse_session


@dataclass
class OptionChainSummary:
    support: Optional[int]
    resistance: Optional[int]
    pcr: float
    bias: str


class NseOptionChainClient:
    """Client for fetching and summarizing NSE option chain data."""

    NSE_OPTION_CHAIN_URL = (
        "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    )

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or build_nse_session()
        self._session_primed = False

    def _prime_session(self) -> None:
        if self._session_primed:
            return
        prime_nse_session(self.session)
        self._session_primed = True

    def fetch_raw(self) -> Dict[str, Any]:
        try:
            self._prime_session()
            resp = self.session.get(self.NSE_OPTION_CHAIN_URL, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                logger.warning("NSE option chain returned a non-dict payload; using empty fallback.")
                return {}
            data.setdefault("_meta", {})
            data["_meta"]["fetched_at"] = datetime.utcnow().isoformat()
            return data
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to fetch NSE option chain: {exc}")
            return {}

    def summarize(self, data: Optional[Dict[str, Any]] = None) -> OptionChainSummary:
        if data is None:
            data = self.fetch_raw()
        if not isinstance(data, dict):
            logger.warning("Malformed option chain payload; returning neutral summary.")
            return OptionChainSummary(
                support=None,
                resistance=None,
                pcr=1.0,
                bias="neutral",
            )

        records = data.get("records", {})
        if not isinstance(records, dict):
            logger.warning("Malformed option chain records payload; returning neutral summary.")
            return OptionChainSummary(
                support=None,
                resistance=None,
                pcr=1.0,
                bias="neutral",
            )
        oi_data = records.get("data", [])
        if not isinstance(oi_data, list):
            logger.warning("Malformed option chain data rows; returning neutral summary.")
            return OptionChainSummary(
                support=None,
                resistance=None,
                pcr=1.0,
                bias="neutral",
            )
        if not oi_data:
            logger.warning("Empty option chain data; returning neutral summary.")
            return OptionChainSummary(
                support=None,
                resistance=None,
                pcr=1.0,
                bias="neutral",
            )

        rows = []
        for row in oi_data:
            ce = row.get("CE") or {}
            pe = row.get("PE") or {}
            rows.append(
                {
                    "strikePrice": row.get("strikePrice"),
                    "CE_oi": ce.get("openInterest", 0),
                    "PE_oi": pe.get("openInterest", 0),
                }
            )

        df = pd.DataFrame(rows).dropna(subset=["strikePrice"])
        if df.empty:
            logger.warning("Option chain rows were present but no valid strikes were parsed.")
            return OptionChainSummary(
                support=None,
                resistance=None,
                pcr=1.0,
                bias="neutral",
            )
        total_put_oi = df["PE_oi"].sum()
        total_call_oi = df["CE_oi"].sum()

        pcr = float(total_put_oi) / float(total_call_oi or 1)
        bias = "bullish" if pcr > 1.1 else "bearish" if pcr < 0.9 else "neutral"

        # Support = strike with highest PE OI, Resistance = highest CE OI
        support_row = df.loc[df["PE_oi"].idxmax()] if not df.empty else None
        resistance_row = df.loc[df["CE_oi"].idxmax()] if not df.empty else None

        support = int(support_row["strikePrice"]) if support_row is not None else None
        resistance = (
            int(resistance_row["strikePrice"]) if resistance_row is not None else None
        )

        summary = OptionChainSummary(
            support=support,
            resistance=resistance,
            pcr=pcr,
            bias=bias,
        )
        logger.info(f"NSE option chain summary: {summary}")
        return summary
