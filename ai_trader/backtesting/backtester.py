from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from ai_trader.config.settings import settings
from ai_trader.strategies.nifty_intraday_strategy import (
    NiftyIntradayStrategy,
    StrategySignal,
)


@dataclass
class BacktestResult:
    total_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float


class Backtester:
    """Runs a simple backtest on historical NIFTY data."""

    def __init__(self, data_path: str | None = None) -> None:
        self.data_path = data_path or settings.backtest_data_path
        self.strategy = NiftyIntradayStrategy()

    def _load_data(self) -> pd.DataFrame:
        path = Path(self.data_path)
        if not path.exists():
            logger.error(f"Backtest data not found at {path}; returning empty DataFrame.")
            return pd.DataFrame()
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return df

    def run(self) -> BacktestResult:
        df = self._load_data()
        if df.empty:
            logger.warning("Backtester has no data; returning empty result.")
            return BacktestResult(
                total_trades=0, win_rate=0.0, profit_factor=0.0, total_pnl=0.0
            )

        signals: List[StrategySignal] = self.strategy.generate_signals(df)
        next_closes = df["close"].shift(-1)

        pnl_list: List[float] = []
        for sig in signals:
            if sig.direction == "NONE":
                continue

            if sig.timestamp not in df.index:
                continue

            entry = float(df.loc[sig.timestamp, "close"])
            exit_price = next_closes.get(sig.timestamp)
            if pd.isna(exit_price):
                continue

            move = float(exit_price) - entry
            pnl = move if sig.direction == "BUY_CE" else -move
            pnl_list.append(pnl)

        if not pnl_list:
            return BacktestResult(
                total_trades=0, win_rate=0.0, profit_factor=0.0, total_pnl=0.0
            )

        pnl_arr = np.array(pnl_list)
        wins = pnl_arr[pnl_arr > 0]
        losses = -pnl_arr[pnl_arr < 0]

        total_trades = len(pnl_arr)
        win_rate = float(len(wins) / total_trades)
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        total_pnl = float(pnl_arr.sum())

        result = BacktestResult(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
        )
        logger.info(f"Backtest result: {result}")
        return result


if __name__ == "__main__":
    bt = Backtester()
    res = bt.run()
    print(res)
