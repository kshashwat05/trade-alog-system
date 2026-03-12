from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from ai_trader.agents.volatility_agent import VolatilityAgent
from ai_trader.config.settings import settings
from ai_trader.data.kite_client import KiteClient, PriceData
from ai_trader.data.nse_option_chain import NseOptionChainClient, OptionChainSummary


@dataclass
class MarketDataQuality:
    price_data_available: bool
    option_chain_available: bool
    vix_available: bool
    fii_data_available: bool
    price_fresh: bool = False
    option_chain_fresh: bool = False
    vix_fresh: bool = False
    used_price_fallback: bool = False
    used_option_chain_fallback: bool = False
    used_vix_fallback: bool = False
    issues: list[str] | None = None

    @property
    def critical_inputs_available(self) -> bool:
        return (
            self.price_data_available
            and self.option_chain_available
            and self.vix_available
            and self.price_fresh
            and self.option_chain_fresh
            and self.vix_fresh
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = self.issues or []
        return payload


@dataclass
class MarketDataContext:
    fetched_at: datetime
    price_df: pd.DataFrame
    option_chain_raw: dict[str, Any]
    option_chain_summary: OptionChainSummary
    vix_value: float | None
    spot_price: float
    fii_data: dict[str, Any]
    quality: MarketDataQuality


class MarketDataProvider:
    def __init__(
        self,
        kite_client: KiteClient | None = None,
        option_chain_client: NseOptionChainClient | None = None,
        vol_agent: VolatilityAgent | None = None,
    ) -> None:
        self.kite_client = kite_client or KiteClient()
        self.option_chain_client = option_chain_client or NseOptionChainClient()
        self.vol_agent = vol_agent or VolatilityAgent()
        self._cached_context: MarketDataContext | None = None
        self._cached_at: datetime | None = None

    def _fetch_fii_data(self) -> dict[str, Any]:
        # Placeholder until a live NSE FII source is added. The agent degrades safely from this.
        logger.info("Using fallback neutral FII positioning data.")
        return {
            "net_futures_position": 0.0,
            "put_write_score": 0.0,
            "call_write_score": 0.0,
            "_fallback": True,
        }

    @staticmethod
    def _is_recent(timestamp: datetime | None, *, max_age_seconds: int) -> bool:
        if timestamp is None:
            return False
        return (datetime.utcnow() - timestamp) <= timedelta(seconds=max_age_seconds)

    def build(self) -> MarketDataContext:
        now = datetime.utcnow()
        if self._cached_context is not None and self._cached_at is not None:
            cache_age = (now - self._cached_at).total_seconds()
            if cache_age <= settings.market_data_cache_seconds:
                logger.info(f"Using cached market data context age={cache_age:.1f}s.")
                return self._cached_context

        with ThreadPoolExecutor(max_workers=3) as executor:
            price_future = executor.submit(self.kite_client.fetch_nifty_intraday)
            option_future = executor.submit(self.option_chain_client.fetch_raw)
            vix_future = executor.submit(self.vol_agent.fetch_vix)
            try:
                price_data = price_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to fetch price data: {exc}")
                price_data = PriceData(
                    df=pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]),
                    fetched_at=datetime.utcnow(),
                )
            try:
                option_chain_raw = option_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to fetch option chain data: {exc}")
                option_chain_raw = {}
            try:
                vix_value = vix_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to fetch VIX data: {exc}")
                vix_value = None

        price_df = price_data.df
        if not isinstance(option_chain_raw, dict):
            logger.warning("MarketDataProvider received malformed option chain payload; using empty fallback.")
            option_chain_raw = {}
        try:
            option_chain_summary = self.option_chain_client.summarize(option_chain_raw)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to summarize option chain payload: {exc}")
            option_chain_raw = {}
            option_chain_summary = OptionChainSummary(
                support=None,
                resistance=None,
                pcr=1.0,
                bias="neutral",
            )
        issues: list[str] = []
        price_data_available = not price_df.empty and "close" in price_df.columns
        option_chain_rows = option_chain_raw.get("records", {}).get("data", [])
        if not isinstance(option_chain_rows, list):
            option_chain_rows = []
        option_chain_available = bool(option_chain_rows)
        latest_candle_at: datetime | None = None
        if price_data_available and "date" in price_df.columns:
            latest_value = pd.to_datetime(price_df.iloc[-1]["date"], errors="coerce")
            latest_candle_at = None if pd.isna(latest_value) else latest_value.to_pydatetime()
        price_fresh = price_data_available and self._is_recent(
            latest_candle_at,
            max_age_seconds=settings.max_price_candle_age_seconds,
        )
        if price_data_available and not price_fresh:
            issues.append("price_data_stale")
        spot_price = (
            float(price_df.iloc[-1]["close"])
            if price_data_available
            else 24000.0
        )
        if not price_data_available:
            issues.append("price_data_unavailable")
        if not option_chain_available:
            issues.append("option_chain_unavailable")
        option_chain_fetched_at_raw = option_chain_raw.get("_meta", {}).get("fetched_at")
        option_chain_fetched_at: datetime | None = None
        if option_chain_fetched_at_raw:
            try:
                option_chain_fetched_at = datetime.fromisoformat(str(option_chain_fetched_at_raw))
            except ValueError:
                option_chain_fetched_at = None
        option_chain_fresh = option_chain_available and self._is_recent(
            option_chain_fetched_at,
            max_age_seconds=settings.max_option_chain_age_seconds,
        )
        if option_chain_available and not option_chain_fresh:
            issues.append("option_chain_stale")
        vix_fresh = vix_value is not None
        if vix_value is None:
            issues.append("vix_unavailable")
        fii_data = self._fetch_fii_data()
        fii_data_available = not bool(fii_data.get("_fallback", False))
        if not fii_data_available:
            issues.append("fii_data_fallback")
        quality = MarketDataQuality(
            price_data_available=price_data_available,
            option_chain_available=option_chain_available,
            vix_available=vix_value is not None,
            fii_data_available=fii_data_available,
            price_fresh=price_fresh,
            option_chain_fresh=option_chain_fresh,
            vix_fresh=vix_fresh,
            used_price_fallback=not price_data_available,
            used_option_chain_fallback=not option_chain_available,
            used_vix_fallback=vix_value is None,
            issues=issues,
        )
        context = MarketDataContext(
            fetched_at=datetime.utcnow(),
            price_df=price_df,
            option_chain_raw=option_chain_raw,
            option_chain_summary=option_chain_summary,
            vix_value=vix_value,
            spot_price=spot_price,
            fii_data=fii_data,
            quality=quality,
        )
        logger.info(
            f"Built market data context at {context.fetched_at.isoformat()} with "
            f"{len(price_df)} price rows and spot={context.spot_price:.2f}. "
            f"quality={context.quality.to_dict()}"
        )
        self._cached_context = context
        self._cached_at = now
        return context
