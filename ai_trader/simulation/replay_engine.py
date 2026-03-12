from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ai_trader.agents.news_agent import NewsMacroAnalysis
from ai_trader.agents.trigger_agent import TradeSignal
from ai_trader.config.settings import settings
from ai_trader.data.market_data_context import MarketDataContext, MarketDataQuality
from ai_trader.data.nse_option_chain import OptionChainSummary
from ai_trader.orchestrator.decision_engine import DecisionEngine


@dataclass
class ReplaySignalOutcome:
    signal_timestamp: str
    signal_type: str
    entry: float
    stop_loss: float
    target: float
    decision_score: int
    confidence: float
    rationale: str
    outcome: str
    exit_timestamp: str | None
    exit_price: float | None
    pnl: float | None
    correct: bool
    wrong_signal: bool
    wrong_reason: str | None
    metadata: dict[str, Any]


@dataclass
class ReplaySummary:
    replay_date: str
    minute_count: int
    evaluated_cycles: int
    signals_generated: int
    correct_signals: int
    wrong_signals: int
    target_hits: int
    stop_loss_hits: int
    expired_signals: int
    signal_accuracy: float
    total_pnl: float


@dataclass
class ReplayDayResult:
    summary: ReplaySummary
    signals: list[ReplaySignalOutcome]
    report_path: str


class ReplayMarketDataProvider:
    def __init__(self, day_df: pd.DataFrame) -> None:
        self.day_df = day_df.reset_index(drop=True).copy()
        self.cursor = 0

    def set_cursor(self, cursor: int) -> None:
        self.cursor = cursor

    @staticmethod
    def _estimate_vix(price_df: pd.DataFrame) -> float:
        if len(price_df) < 10:
            return 14.0
        realized = price_df["close"].pct_change().tail(30).std()
        if pd.isna(realized) or realized <= 0:
            return 14.0
        annualized = float(realized * (252 ** 0.5) * 100 * 5.0)
        return float(max(10.0, min(28.0, annualized)))

    @staticmethod
    def _estimate_fii_data(price_df: pd.DataFrame) -> dict[str, Any]:
        if len(price_df) < 6:
            return {"net_futures_position": 0.0, "put_write_score": 0.0, "call_write_score": 0.0}
        momentum = float(price_df["close"].iloc[-1] - price_df["close"].iloc[-6])
        futures_position = max(-800.0, min(800.0, momentum * 10.0))
        if momentum > 0:
            return {
                "net_futures_position": futures_position,
                "put_write_score": abs(futures_position) * 0.8,
                "call_write_score": abs(futures_position) * 0.2,
            }
        if momentum < 0:
            return {
                "net_futures_position": futures_position,
                "put_write_score": abs(futures_position) * 0.2,
                "call_write_score": abs(futures_position) * 0.8,
            }
        return {"net_futures_position": 0.0, "put_write_score": 0.0, "call_write_score": 0.0}

    @staticmethod
    def _option_premium(spot: float, strike: int, side: str, minutes_left: int) -> float:
        intrinsic = max(0.0, spot - strike) if side == "CE" else max(0.0, strike - spot)
        time_value = max(8.0, spot * 0.003 * max(minutes_left, 1) / 375.0)
        distance = abs(spot - strike)
        extrinsic = max(4.0, time_value - distance * 0.18)
        return round(intrinsic + extrinsic, 2)

    def _build_option_chain_raw(self, price_df: pd.DataFrame) -> dict[str, Any]:
        spot = float(price_df.iloc[-1]["close"])
        current_time = pd.Timestamp(price_df.iloc[-1]["date"]).to_pydatetime()
        minutes_left = max(1, int((self.day_df.iloc[-1]["date"] - price_df.iloc[-1]["date"]).total_seconds() // 60))
        rounded_spot = int(round(spot / 50.0) * 50)
        strikes = [rounded_spot + offset for offset in range(-150, 151, 50)]
        momentum = 0.0
        if len(price_df) >= 5:
            momentum = float(price_df["close"].iloc[-1] - price_df["close"].iloc[-5])

        rows: list[dict[str, Any]] = []
        for strike in strikes:
            call_bias = max(25.0, 120.0 - max(momentum, 0.0) * 6.0 + max(strike - spot, 0.0) * 1.5)
            put_bias = max(25.0, 120.0 + min(momentum, 0.0) * -6.0 + max(spot - strike, 0.0) * 1.5)
            ce_oi = int(350 + call_bias * 4)
            pe_oi = int(350 + put_bias * 4)
            ce_change = int(max(5.0, call_bias if momentum < 0 else call_bias * 0.35))
            pe_change = int(max(5.0, put_bias if momentum > 0 else put_bias * 0.35))
            expiry = current_time.strftime("%Y-%m-%d")
            rows.append(
                {
                    "strikePrice": strike,
                    "expiryDate": expiry,
                    "CE": {
                        "openInterest": ce_oi,
                        "changeinOpenInterest": ce_change,
                        "lastPrice": self._option_premium(spot, strike, "CE", minutes_left),
                        "identifier": f"NIFTY_REPLAY_{expiry}_{strike}CE",
                    },
                    "PE": {
                        "openInterest": pe_oi,
                        "changeinOpenInterest": pe_change,
                        "lastPrice": self._option_premium(spot, strike, "PE", minutes_left),
                        "identifier": f"NIFTY_REPLAY_{expiry}_{strike}PE",
                    },
                }
            )
        return {
            "_meta": {"fetched_at": current_time.isoformat(), "source": "historical_replay"},
            "records": {"data": rows, "expiryDates": [current_time.strftime("%Y-%m-%d")]},
        }

    @staticmethod
    def _summarize_option_chain(option_chain_raw: dict[str, Any]) -> OptionChainSummary:
        rows = option_chain_raw.get("records", {}).get("data", [])
        if not rows:
            return OptionChainSummary(None, None, 1.0, "neutral")
        frame = pd.DataFrame(
            [
                {
                    "strike": row.get("strikePrice"),
                    "ce_oi": (row.get("CE") or {}).get("openInterest", 0),
                    "pe_oi": (row.get("PE") or {}).get("openInterest", 0),
                }
                for row in rows
            ]
        ).dropna(subset=["strike"])
        if frame.empty:
            return OptionChainSummary(None, None, 1.0, "neutral")
        total_put_oi = float(frame["pe_oi"].sum())
        total_call_oi = float(frame["ce_oi"].sum())
        pcr = total_put_oi / max(total_call_oi, 1.0)
        bias = "bullish" if pcr > 1.1 else "bearish" if pcr < 0.9 else "neutral"
        support = int(frame.loc[frame["pe_oi"].idxmax(), "strike"])
        resistance = int(frame.loc[frame["ce_oi"].idxmax(), "strike"])
        return OptionChainSummary(support=support, resistance=resistance, pcr=float(pcr), bias=bias)

    def build(self) -> MarketDataContext:
        price_df = self.day_df.iloc[: self.cursor + 1].copy()
        current_time = pd.Timestamp(price_df.iloc[-1]["date"]).to_pydatetime()
        option_chain_raw = self._build_option_chain_raw(price_df)
        vix = self._estimate_vix(price_df)
        fii_data = self._estimate_fii_data(price_df)
        quality = MarketDataQuality(
            price_data_available=not price_df.empty,
            option_chain_available=True,
            vix_available=True,
            fii_data_available=True,
            price_fresh=True,
            option_chain_fresh=True,
            vix_fresh=True,
            used_price_fallback=False,
            used_option_chain_fallback=False,
            used_vix_fallback=False,
            issues=[],
        )
        return MarketDataContext(
            fetched_at=current_time,
            price_df=price_df,
            option_chain_raw=option_chain_raw,
            option_chain_summary=self._summarize_option_chain(option_chain_raw),
            vix_value=vix,
            spot_price=float(price_df.iloc[-1]["close"]),
            fii_data=fii_data,
            quality=quality,
        )


class _ReplayNewsAgent:
    def analyze(self) -> NewsMacroAnalysis:
        return NewsMacroAnalysis(macro_bias="neutral", risk_level="medium", data_available=True, fallback_used=False)


class HistoricalReplayEngine:
    def __init__(self, data_path: str | Path | None = None, output_dir: str | Path = "ai_trader/data/replay_reports") -> None:
        self.data_path = Path(data_path or settings.backtest_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Replay data not found: {self.data_path}")
        df = pd.read_csv(self.data_path, parse_dates=["date"])
        required_columns = {"date", "open", "high", "low", "close", "volume"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Replay data missing required columns: {sorted(missing)}")
        return df.sort_values("date").reset_index(drop=True)

    @staticmethod
    def _resample_to_minutes(day_df: pd.DataFrame) -> pd.DataFrame:
        day_df = day_df.copy().set_index("date").sort_index()
        minute_index = pd.date_range(day_df.index.min(), day_df.index.max(), freq="1min")
        base = pd.DataFrame(index=minute_index)
        base["close"] = day_df["close"].reindex(minute_index).interpolate(method="time").ffill().bfill()
        base["open"] = base["close"].shift(1).fillna(base["close"])
        base["high"] = base[["open", "close"]].max(axis=1)
        base["low"] = base[["open", "close"]].min(axis=1)
        volume = day_df["volume"].resample("1min").sum().reindex(minute_index, fill_value=0.0)
        base["volume"] = volume
        base = base.reset_index().rename(columns={"index": "date"})
        return base[["date", "open", "high", "low", "close", "volume"]]

    @staticmethod
    def _simulate_option_price(signal: TradeSignal, spot: float, minutes_left: int) -> float:
        if signal.option_strike is None:
            return signal.entry
        side = "PE" if signal.signal == "BUY_PE" else "CE"
        intrinsic = max(0.0, spot - signal.option_strike) if side == "CE" else max(0.0, signal.option_strike - spot)
        base_time = max(4.0, (signal.entry * 0.45) * max(minutes_left, 1) / 375.0)
        premium = intrinsic + base_time
        if signal.underlying_spot is not None:
            directional_move = (spot - signal.underlying_spot) if side == "CE" else (signal.underlying_spot - spot)
            premium = max(premium, signal.entry + directional_move * 0.45)
        return round(max(0.05, premium), 2)

    def _evaluate_signal_outcome(self, signal: TradeSignal, timestamp: datetime, future_df: pd.DataFrame, decision_score: int) -> ReplaySignalOutcome:
        if future_df.empty:
            return ReplaySignalOutcome(
                signal_timestamp=timestamp.isoformat(),
                signal_type=signal.signal,
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                target=signal.target,
                decision_score=decision_score,
                confidence=signal.confidence,
                rationale=signal.rationale,
                outcome="expired",
                exit_timestamp=None,
                exit_price=signal.entry,
                pnl=0.0,
                correct=False,
                wrong_signal=True,
                wrong_reason="No future market data available after signal.",
                metadata={"instrument_symbol": signal.instrument_symbol, "price_source": signal.price_source},
            )

        for idx, row in future_df.reset_index(drop=True).iterrows():
            minutes_left = max(1, len(future_df) - idx)
            current_price = self._simulate_option_price(signal, float(row["close"]), minutes_left)
            exit_timestamp = pd.Timestamp(row["date"]).to_pydatetime().isoformat()
            if current_price >= signal.target:
                return ReplaySignalOutcome(
                    signal_timestamp=timestamp.isoformat(),
                    signal_type=signal.signal,
                    entry=signal.entry,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    decision_score=decision_score,
                    confidence=signal.confidence,
                    rationale=signal.rationale,
                    outcome="target_hit",
                    exit_timestamp=exit_timestamp,
                    exit_price=current_price,
                    pnl=round(current_price - signal.entry, 2),
                    correct=True,
                    wrong_signal=False,
                    wrong_reason=None,
                    metadata={"instrument_symbol": signal.instrument_symbol, "price_source": signal.price_source},
                )
            if current_price <= signal.stop_loss:
                return ReplaySignalOutcome(
                    signal_timestamp=timestamp.isoformat(),
                    signal_type=signal.signal,
                    entry=signal.entry,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    decision_score=decision_score,
                    confidence=signal.confidence,
                    rationale=signal.rationale,
                    outcome="stop_loss_hit",
                    exit_timestamp=exit_timestamp,
                    exit_price=current_price,
                    pnl=round(current_price - signal.entry, 2),
                    correct=False,
                    wrong_signal=True,
                    wrong_reason="Stop loss would have been hit before reaching target.",
                    metadata={"instrument_symbol": signal.instrument_symbol, "price_source": signal.price_source},
                )

        final_row = future_df.iloc[-1]
        final_price = self._simulate_option_price(signal, float(final_row["close"]), 1)
        pnl = round(final_price - signal.entry, 2)
        wrong_signal = pnl <= 0
        return ReplaySignalOutcome(
            signal_timestamp=timestamp.isoformat(),
            signal_type=signal.signal,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            target=signal.target,
            decision_score=decision_score,
            confidence=signal.confidence,
            rationale=signal.rationale,
            outcome="expired",
            exit_timestamp=pd.Timestamp(final_row["date"]).to_pydatetime().isoformat(),
            exit_price=final_price,
            pnl=pnl,
            correct=not wrong_signal,
            wrong_signal=wrong_signal,
            wrong_reason="Trade expired negative or flat by session end." if wrong_signal else None,
            metadata={"instrument_symbol": signal.instrument_symbol, "price_source": signal.price_source},
        )

    def replay_day(self, replay_date: str | datetime | None = None) -> ReplayDayResult:
        df = self._load_data()
        if replay_date is None:
            selected_date = pd.Timestamp(df["date"].dt.date.iloc[0]).date()
        else:
            selected_date = pd.Timestamp(replay_date).date()
        day_df = df[df["date"].dt.date == selected_date]
        if day_df.empty:
            raise ValueError(f"No replay data found for date {selected_date.isoformat()}")

        minute_df = self._resample_to_minutes(day_df)
        provider = ReplayMarketDataProvider(minute_df)
        engine = DecisionEngine()
        engine.market_data_provider = provider
        engine.news_agent = _ReplayNewsAgent()  # type: ignore[assignment]
        engine.llm_validator.validation_enabled = False
        engine.risk_agent.set_now_provider(lambda: pd.Timestamp(minute_df.iloc[provider.cursor]["date"]).to_pydatetime())

        signal_records: list[ReplaySignalOutcome] = []
        for cursor in range(len(minute_df)):
            provider.set_cursor(cursor)
            output = engine.run_once(open_trades=0)
            if output.signal.signal == "NONE" or not output.risk.allowed:
                continue
            signal_time = pd.Timestamp(minute_df.iloc[cursor]["date"]).to_pydatetime()
            future_df = minute_df.iloc[cursor + 1 :].copy()
            signal_records.append(
                self._evaluate_signal_outcome(output.signal, signal_time, future_df, output.decision_score)
            )

        total_pnl = round(sum(record.pnl or 0.0 for record in signal_records), 2)
        correct_count = sum(1 for record in signal_records if record.correct)
        wrong_count = sum(1 for record in signal_records if record.wrong_signal)
        target_hits = sum(1 for record in signal_records if record.outcome == "target_hit")
        stop_loss_hits = sum(1 for record in signal_records if record.outcome == "stop_loss_hit")
        expired = sum(1 for record in signal_records if record.outcome == "expired")
        summary = ReplaySummary(
            replay_date=selected_date.isoformat(),
            minute_count=len(minute_df),
            evaluated_cycles=len(minute_df),
            signals_generated=len(signal_records),
            correct_signals=correct_count,
            wrong_signals=wrong_count,
            target_hits=target_hits,
            stop_loss_hits=stop_loss_hits,
            expired_signals=expired,
            signal_accuracy=(correct_count / len(signal_records)) if signal_records else 0.0,
            total_pnl=total_pnl,
        )
        report_path = self.output_dir / f"replay_{selected_date.isoformat()}.json"
        report = {
            "summary": asdict(summary),
            "signals": [asdict(record) for record in signal_records],
        }
        report_path.write_text(json.dumps(report, indent=2))
        logger.info(f"Historical replay completed for {selected_date.isoformat()}: {summary}")
        return ReplayDayResult(summary=summary, signals=signal_records, report_path=str(report_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay historical market data through the live trading engine.")
    parser.add_argument("--data-path", default=settings.backtest_data_path, help="CSV with date, OHLCV columns.")
    parser.add_argument("--date", default=None, help="Replay date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default="ai_trader/data/replay_reports", help="Directory for replay reports.")
    args = parser.parse_args()

    result = HistoricalReplayEngine(data_path=args.data_path, output_dir=args.output_dir).replay_day(args.date)
    print(result.summary)
    print(result.report_path)
