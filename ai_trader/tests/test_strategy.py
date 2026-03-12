from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai_trader.backtesting.backtester import Backtester, BacktestResult
from ai_trader.strategies.nifty_intraday_strategy import (
    NiftyIntradayStrategy,
    StrategySignal,
)


def test_nifty_intraday_strategy_generates_signals():
    # Create minimal mock intraday data
    idx = pd.date_range("2024-01-01 09:15", periods=20, freq="5min")
    df = pd.DataFrame(
        {
            "open": [24000 + i for i in range(20)],
            "high": [24010 + i for i in range(20)],
            "low": [23990 + i for i in range(20)],
            "close": [24005 + i for i in range(20)],
            "volume": [1000 + i * 10 for i in range(20)],
        },
        index=idx,
    )

    strat = NiftyIntradayStrategy()
    signals = strat.generate_signals(df)
    assert len(signals) == len(df)
    assert all(isinstance(s, StrategySignal) for s in signals)


def test_backtester_runs_against_mock_dataset():
    data_path = Path("ai_trader/tests/mock_data/nifty_intraday.csv")
    backtester = Backtester(data_path=str(data_path))
    result = backtester.run()

    assert isinstance(result, BacktestResult)
    assert result.total_trades >= 0


def test_nifty_intraday_strategy_requires_ohlcv_columns():
    df = pd.DataFrame({"close": [1, 2, 3]})
    strat = NiftyIntradayStrategy()
    with pytest.raises(ValueError):
        strat.generate_signals(df)
