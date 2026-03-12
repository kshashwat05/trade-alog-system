from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from kiteconnect import KiteConnect
from loguru import logger

from ai_trader.config.settings import settings


@dataclass
class PriceData:
    df: pd.DataFrame
    fetched_at: datetime | None = None


class KiteClient:
    """Thin wrapper around KiteConnect for fetching NIFTY data.

    In tests, this class can be monkeypatched or replaced with a fake implementation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        kite: Optional[KiteConnect] = None,
    ) -> None:
        self._instrument_token: Optional[int] = settings.kite_instrument_token
        self._instrument_books: dict[str, list[dict[str, Any]]] = {}
        if kite is not None:
            self._kite = kite
            return

        api_key = api_key or settings.kite_api_key
        access_token = access_token or settings.kite_access_token

        if not (api_key and access_token):
            logger.warning("Kite credentials missing; KiteClient will run in mock mode.")
            self._kite: Optional[KiteConnect] = None
            return

        self._kite = KiteConnect(api_key=api_key)
        self._kite.set_access_token(access_token)

    @staticmethod
    def _empty_price_data() -> PriceData:
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        return PriceData(df=df, fetched_at=datetime.now())

    def is_mock(self) -> bool:
        return self._kite is None

    @staticmethod
    def _normalize_dataframe(candles: list[dict[str, Any]]) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles)
        required_columns = ["date", "open", "high", "low", "close", "volume"]
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Kite historical data missing columns: {missing}")

        df = df[required_columns + [column for column in df.columns if column not in required_columns]]
        df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _matches_index_symbol(instrument: dict[str, Any], symbol: str) -> bool:
        candidates = {
            str(instrument.get("tradingsymbol") or "").strip().upper(),
            str(instrument.get("name") or "").strip().upper(),
        }
        normalized = symbol.strip().upper()
        aliases = {normalized, normalized.replace(" ", "")}
        if normalized == "NIFTY 50":
            aliases.update({"NIFTY", "NIFTY50"})
        return any(candidate in aliases for candidate in candidates if candidate)

    def _get_instruments(self, exchange: str) -> list[dict[str, Any]]:
        if self._kite is None:
            return []
        exchange = exchange.upper()
        if exchange in self._instrument_books:
            return self._instrument_books[exchange]
        try:
            instruments = self._kite.instruments(exchange)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to fetch Kite instruments for exchange={exchange}: {exc}")
            return []
        self._instrument_books[exchange] = instruments
        return instruments

    @lru_cache(maxsize=4)
    def _resolve_instrument_token(self, symbol: str) -> Optional[int]:
        if self._kite is None:
            return None

        try:
            instruments = self._kite.instruments("NSE")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to fetch Kite instruments for symbol resolution: {exc}")
            return None

        preferred_match: Optional[dict[str, Any]] = None
        fallback_match: Optional[dict[str, Any]] = None
        for instrument in instruments:
            if not self._matches_index_symbol(instrument, symbol):
                continue
            segment = str(instrument.get("segment") or "").upper()
            exchange = str(instrument.get("exchange") or "").upper()
            if segment == "INDICES" or exchange == "NSE":
                preferred_match = instrument
                break
            fallback_match = instrument

        chosen = preferred_match or fallback_match
        if chosen is None:
            logger.error(f"Unable to resolve instrument token for symbol {symbol!r}.")
            return None

        token = int(chosen["instrument_token"])
        logger.info(
            f"Resolved Kite instrument token {token} for symbol {symbol!r} using "
            f"tradingsymbol={chosen.get('tradingsymbol')!r} segment={chosen.get('segment')!r}."
        )
        return token

    def _get_nifty_instrument_token(self) -> Optional[int]:
        if self._instrument_token is not None:
            return self._instrument_token
        token = self._resolve_instrument_token(settings.index_symbol)
        if token is not None:
            self._instrument_token = token
        return token

    @lru_cache(maxsize=128)
    def resolve_instrument_key(
        self,
        tradingsymbol: str,
        exchange: str = "NFO",
    ) -> str | None:
        if ":" in tradingsymbol:
            return tradingsymbol
        if self._kite is None:
            logger.warning("Cannot resolve instrument key without a live Kite connection.")
            return None

        exchange = exchange.upper()
        symbol = tradingsymbol.strip().upper()
        instruments = self._get_instruments(exchange)
        if not instruments:
            return None

        for instrument in instruments:
            candidate = str(instrument.get("tradingsymbol") or "").strip().upper()
            if candidate == symbol:
                return f"{exchange}:{candidate}"

        logger.error(f"Unable to resolve instrument key for exchange={exchange} tradingsymbol={symbol!r}.")
        return None

    def resolve_instrument_token_by_key(self, instrument_key: str) -> int | None:
        if ":" not in instrument_key:
            logger.error(f"Invalid instrument_key={instrument_key!r}; expected 'EXCHANGE:SYMBOL'.")
            return None
        exchange, tradingsymbol = instrument_key.split(":", 1)
        instruments = self._get_instruments(exchange)
        if not instruments:
            return None
        symbol = tradingsymbol.strip().upper()
        for instrument in instruments:
            candidate = str(instrument.get("tradingsymbol") or "").strip().upper()
            if candidate == symbol:
                token = instrument.get("instrument_token")
                return int(token) if token is not None else None
        logger.error(f"Unable to resolve instrument token for instrument_key={instrument_key!r}.")
        return None

    def fetch_nifty_intraday(self, interval: str = "5minute", days: int = 1) -> PriceData:
        """Fetch recent intraday candles for NIFTY.
        """
        if self._kite is None:
            logger.info("Using mock intraday data for NIFTY (no Kite connection).")
            return self._empty_price_data()

        if days <= 0:
            logger.warning(f"Invalid days={days}; expected positive integer.")
            return self._empty_price_data()

        instrument_token = self._get_nifty_instrument_token()
        if instrument_token is None:
            return self._empty_price_data()

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        try:
            candles = self._kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )
            df = self._normalize_dataframe(candles)
            logger.info(
                f"Fetched {len(df)} NIFTY candles from Kite for interval={interval}, days={days}, "
                f"instrument_token={instrument_token}."
            )
            return PriceData(df=df, fetched_at=to_date)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Failed to fetch Kite historical data for interval={interval}, days={days}, "
                f"instrument_token={instrument_token}: {exc}"
            )
            return self._empty_price_data()

    def fetch_ltp_by_instrument_key(self, instrument_key: str) -> float | None:
        if self._kite is None:
            logger.warning("Cannot fetch LTP without a live Kite connection.")
            return None
        try:
            data = self._kite.ltp(instrument_key)
            payload = data.get(instrument_key, {})
            ltp = payload.get("last_price")
            return float(ltp) if ltp is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to fetch LTP for instrument_key={instrument_key}: {exc}")
            return None

    def fetch_intraday_by_instrument_key(
        self,
        instrument_key: str,
        *,
        interval: str = "5minute",
        days: int = 1,
    ) -> PriceData:
        if self._kite is None:
            logger.warning("Cannot fetch instrument candles without a live Kite connection.")
            return self._empty_price_data()
        if days <= 0:
            logger.warning(f"Invalid days={days}; expected positive integer.")
            return self._empty_price_data()
        instrument_token = self.resolve_instrument_token_by_key(instrument_key)
        if instrument_token is None:
            return self._empty_price_data()

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        try:
            candles = self._kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )
            df = self._normalize_dataframe(candles)
            logger.info(
                f"Fetched {len(df)} candles for instrument_key={instrument_key} interval={interval}, days={days}."
            )
            return PriceData(df=df, fetched_at=to_date)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Failed to fetch historical data for instrument_key={instrument_key}, "
                f"interval={interval}, days={days}: {exc}"
            )
            return self._empty_price_data()
