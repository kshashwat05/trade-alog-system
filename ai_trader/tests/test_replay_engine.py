from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ai_trader.agents.trigger_agent import TradeSignal
from ai_trader.simulation.replay_engine import HistoricalReplayEngine


def _write_dataset(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def test_replay_engine_resamples_to_minutes(tmp_path):
    data_path = tmp_path / "replay.csv"
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-03-10 09:15:00",
                    "2026-03-10 09:20:00",
                    "2026-03-10 09:25:00",
                ]
            ),
            "open": [100, 105, 110],
            "high": [105, 110, 115],
            "low": [95, 100, 105],
            "close": [104, 109, 114],
            "volume": [1000, 1100, 1200],
        }
    )
    _write_dataset(data_path, frame)
    engine = HistoricalReplayEngine(data_path=data_path, output_dir=tmp_path)
    minute_df = engine._resample_to_minutes(frame)
    assert len(minute_df) == 11
    assert minute_df.iloc[0]["date"].minute == 15
    assert minute_df.iloc[-1]["date"].minute == 25


def test_replay_engine_runs_full_day_and_persists_report(tmp_path):
    data_path = tmp_path / "trend_day.csv"
    idx = pd.date_range("2026-03-10 09:15", periods=120, freq="1min")
    close = [24000 + i * 2.5 for i in range(len(idx))]
    frame = pd.DataFrame(
        {
            "date": idx,
            "open": close,
            "high": [value + 4 for value in close],
            "low": [value - 4 for value in close],
            "close": close,
            "volume": [1000 + i * 3 for i in range(len(idx))],
        }
    )
    _write_dataset(data_path, frame)
    engine = HistoricalReplayEngine(data_path=data_path, output_dir=tmp_path)
    result = engine.replay_day("2026-03-10")

    assert result.summary.minute_count == 120
    assert result.summary.evaluated_cycles == 120
    assert Path(result.report_path).exists()

    report = json.loads(Path(result.report_path).read_text())
    assert report["summary"]["replay_date"] == "2026-03-10"
    assert "signals" in report


def test_replay_engine_detects_wrong_signal_on_reversal():
    engine = HistoricalReplayEngine(data_path="ai_trader/tests/mock_data/nifty_intraday.csv")
    signal = TradeSignal(
        signal="BUY_CE",
        entry=100.0,
        stop_loss=90.0,
        target=120.0,
        confidence=0.8,
        rationale="Bullish breakout",
        underlying_spot=24000.0,
        option_strike=24000,
        instrument_symbol="NIFTY_REPLAY_24000CE",
        price_source="replay",
        data_complete=True,
    )
    future_df = pd.DataFrame(
        {
            "date": pd.date_range("2026-03-10 10:01", periods=4, freq="1min"),
            "close": [23980.0, 23960.0, 23940.0, 23920.0],
        }
    )
    outcome = engine._evaluate_signal_outcome(signal, datetime(2026, 3, 10, 10, 0), future_df, 8)
    assert outcome.wrong_signal is True
    assert outcome.outcome == "stop_loss_hit"
    assert outcome.wrong_reason is not None
