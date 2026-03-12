from __future__ import annotations

import pandas as pd

from ai_trader.data.kite_client import KiteClient, PriceData
from ai_trader.data.nse_option_chain import NseOptionChainClient, OptionChainSummary


def test_option_chain_summary_from_mock_data():
    # Example of minimal NSE option chain-like structure
    raw = {
        "records": {
            "data": [
                {
                    "strikePrice": 23800,
                    "CE": {"openInterest": 100},
                    "PE": {"openInterest": 250},
                },
                {
                    "strikePrice": 23900,
                    "CE": {"openInterest": 300},
                    "PE": {"openInterest": 150},
                },
            ]
        }
    }

    client = NseOptionChainClient()
    summary: OptionChainSummary = client.summarize(data=raw)
    assert summary.support in (23800, 23900)
    assert summary.resistance in (23800, 23900)
    assert summary.pcr > 0


def test_option_chain_client_handles_empty():
    client = NseOptionChainClient()
    summary = client.summarize(data={})
    assert summary.bias == "neutral"
    assert summary.pcr == 1.0


def test_option_chain_client_handles_invalid_rows():
    client = NseOptionChainClient()
    summary = client.summarize(data={"records": {"data": [{"strikePrice": None}]}})
    assert summary.bias == "neutral"
    assert summary.support is None


def test_kite_client_uses_configured_instrument_token():
    class DummyKite:
        def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False, oi=False):
            assert instrument_token == 12345
            assert interval == "5minute"
            return [
                {
                    "date": "2026-03-12 09:15:00",
                    "open": 24000,
                    "high": 24010,
                    "low": 23995,
                    "close": 24005,
                    "volume": 1000,
                }
            ]

    client = KiteClient(kite=DummyKite())  # type: ignore[arg-type]
    client._instrument_token = 12345
    price_data: PriceData = client.fetch_nifty_intraday()
    assert isinstance(price_data.df, pd.DataFrame)
    assert list(price_data.df.columns[:6]) == ["date", "open", "high", "low", "close", "volume"]
    assert len(price_data.df) == 1


def test_kite_client_resolves_symbol_when_token_missing():
    class DummyKite:
        def instruments(self, exchange=None):
            assert exchange == "NSE"
            return [
                {
                    "instrument_token": 256265,
                    "tradingsymbol": "NIFTY 50",
                    "name": "NIFTY 50",
                    "segment": "INDICES",
                    "exchange": "NSE",
                }
            ]

        def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False, oi=False):
            assert instrument_token == 256265
            return [
                {
                    "date": "2026-03-12 09:15:00",
                    "open": 24000,
                    "high": 24010,
                    "low": 23995,
                    "close": 24005,
                    "volume": 1000,
                }
            ]

    client = KiteClient(kite=DummyKite())  # type: ignore[arg-type]
    client._instrument_token = None
    price_data = client.fetch_nifty_intraday()
    assert len(price_data.df) == 1
    assert client._instrument_token == 256265


def test_kite_client_returns_empty_frame_on_fetch_failure():
    class DummyKite:
        def instruments(self, exchange=None):
            return [
                {
                    "instrument_token": 256265,
                    "tradingsymbol": "NIFTY 50",
                    "name": "NIFTY 50",
                    "segment": "INDICES",
                    "exchange": "NSE",
                }
            ]

        def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False, oi=False):
            raise RuntimeError("boom")

    client = KiteClient(kite=DummyKite())  # type: ignore[arg-type]
    client._instrument_token = None
    price_data = client.fetch_nifty_intraday()
    assert price_data.df.empty
