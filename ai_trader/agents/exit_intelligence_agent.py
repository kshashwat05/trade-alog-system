from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import time

from loguru import logger

from ai_trader.config.settings import settings
from ai_trader.data.trade_journal import TradeJournalEntry


@dataclass
class ExitIntelligenceSuggestion:
    action: str
    reason: str
    confidence: float
    dedupe_key: str = ""


class ExitIntelligenceAgent:
    """Suggests partial exits / trailing stops before hard exits trigger."""

    def __init__(self) -> None:
        self._price_windows: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=max(3, settings.exit_intel_window_points))
        )
        self._last_advisories: dict[int, tuple[str, str, float]] = {}

    def clear_trade(self, trade_id: int) -> None:
        self._price_windows.pop(trade_id, None)
        self._last_advisories.pop(trade_id, None)

    def _should_emit(self, trade_id: int, suggestion: ExitIntelligenceSuggestion) -> bool:
        previous = self._last_advisories.get(trade_id)
        cooldown = max(0, settings.exit_intel_advisory_cooldown_seconds)
        if previous is not None:
            prev_action, prev_key, prev_at = previous
            suggestion_key = suggestion.dedupe_key or suggestion.action
            if prev_action == suggestion.action and prev_key == suggestion_key and (time.monotonic() - prev_at) < cooldown:
                return False
        return True

    def mark_advisory_sent(self, trade_id: int, suggestion: ExitIntelligenceSuggestion) -> None:
        if suggestion.action == "NONE":
            return
        self._last_advisories[trade_id] = (
            suggestion.action,
            suggestion.dedupe_key or suggestion.action,
            time.monotonic(),
        )

    def observe_trade(
        self,
        trade: TradeJournalEntry,
        current_price: float,
    ) -> ExitIntelligenceSuggestion:
        if current_price <= 0:
            return ExitIntelligenceSuggestion(action="NONE", reason="Invalid current price.", confidence=0.0)

        window = self._price_windows[trade.id]
        window.append(float(current_price))
        if len(window) < 3:
            return ExitIntelligenceSuggestion(action="NONE", reason="Insufficient price history.", confidence=0.1)

        entry_price = float(trade.execution_price or trade.entry_price)
        pnl_pct = ((current_price - entry_price) / max(entry_price, 1e-6)) * 100.0
        min_move = max(0.5, settings.exit_intel_stagnation_threshold_pct)
        stagnating = abs(pnl_pct) <= min_move and len(window) >= window.maxlen

        lookback = max(2, settings.exit_intel_momentum_lookback)
        recent = list(window)[-lookback:]
        deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        slowdown = all(delta <= 0 for delta in deltas)

        if stagnating:
            suggestion = ExitIntelligenceSuggestion(
                action="PARTIAL_EXIT",
                reason=f"Premium stagnating near entry ({pnl_pct:.2f}%).",
                confidence=0.65,
                dedupe_key="stagnation_near_entry",
            )
            if self._should_emit(trade.id, suggestion):
                logger.info(f"ExitIntelligence suggestion trade_id={trade.id}: {suggestion}")
                return suggestion
            return ExitIntelligenceSuggestion(action="NONE", reason="Advisory cooldown active.", confidence=0.0)
        if slowdown and pnl_pct > 0:
            suggestion = ExitIntelligenceSuggestion(
                action="TRAILING_STOP",
                reason="Momentum slowdown detected after favorable move.",
                confidence=0.7,
                dedupe_key="momentum_slowdown",
            )
            if self._should_emit(trade.id, suggestion):
                logger.info(f"ExitIntelligence suggestion trade_id={trade.id}: {suggestion}")
                return suggestion
            return ExitIntelligenceSuggestion(action="NONE", reason="Advisory cooldown active.", confidence=0.0)

        return ExitIntelligenceSuggestion(action="NONE", reason="No exit-intel action.", confidence=0.2)
